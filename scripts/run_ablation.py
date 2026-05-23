#!/usr/bin/env python3
"""Ablation study orchestration script.

Trains and evaluates the full experiment grid (G1-G6), saving results
for each configuration.  The grid is designed so each step adds exactly
one component, enabling isolated measurement of each contribution.

Pipeline stages:
  - Stage A: DINOv2 LoRA + dual classifier (FF++)
  - Stage 2: VLM fine-tuning (DD-VQA)
  - Stage 3: Localization (DD-VQA with bboxes)

Grid:
  G1: Baseline SigLIP only                           [trains Stage 2]
  G2: + DINOv2 frozen (I-MoF)                        [trains Stage 2]
  G3: + DINOv2 LoRA (deepfake-aware, end-to-end)     [trains Stage 2]
  G4: + Localization (end-to-end, Stage 3 from G3)    [trains Stage 3]
  G5: + Classifier verdict (reuses G3, eval-only)     [eval only]
  G6: + Classifier + Localization (reuses G4, eval)   [eval only]

Each experiment has its own output directory — nothing is ever overwritten.
G5 and G6 reuse trained models from G3 and G4 respectively, evaluating
them with the external DINOv2 classifier verdict instead of re-training.

Usage::

    # Full run
    python scripts/run_ablation.py

    # Dry-run (few samples, 1 epoch — validate pipeline)
    python scripts/run_ablation.py --dry-run

    # Specific experiments only
    python scripts/run_ablation.py --experiments G3 G4

    # Skip Stage A (reuse existing checkpoint)
    python scripts/run_ablation.py --skip-stage-a
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import yaml

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
)
logger = logging.getLogger("ablation")

ROOT = Path(__file__).resolve().parent.parent

# ── Checkpoint paths ─────────────────────────────────────────────────

DINO_LORA_CKPT = "outputs/dino_lora_classifier/best.pt"
ADAPTER_FROZEN_CKPT = "outputs/tinyllava_stage1_adapter_full/checkpoint-final.pt"

COOLDOWN_SECS = 120

# ── Experiment grid ──────────────────────────────────────────────────
#
# Two experiment types:
#   - "train_config": trains a model using this config (has its own output_dir)
#   - "reuses": eval-only — evaluates the model from another experiment
#     with a different setting (e.g. classifier verdict)

EXPERIMENTS = {
    "G1": {
        "description": "Baseline: SigLIP only, end-to-end, no DINOv2",
        "train_config": "configs/ablation/g1_baseline.yaml",
        "classifier_ckpt": None,
        "needs_stage_a": False,
    },
    "G2": {
        "description": "+ DINOv2 frozen (I-MoF), end-to-end",
        "train_config": "configs/ablation/g2_imof.yaml",
        "classifier_ckpt": None,
        "needs_stage_a": False,
    },
    "G3": {
        "description": "+ DINOv2 LoRA (single), deepfake-aware features, end-to-end",
        "train_config": "configs/ablation/g3_lora.yaml",
        "classifier_ckpt": None,
        "needs_stage_a": True,
    },
    "G4": {
        "description": "+ Localization (end-to-end verdict, Stage 3 from G3)",
        "train_config": "configs/ablation/g4_lora_loc.yaml",
        "classifier_ckpt": None,
        "needs_stage_a": True,
    },
    "G5": {
        "description": "+ Classifier verdict (G3 model, eval-only with classifier)",
        "reuses": "G3",
        "eval_output_dir": "outputs/ablation/g5_classifier/evaluation",
        "classifier_ckpt": DINO_LORA_CKPT,
        "needs_stage_a": True,
    },
    "G6": {
        "description": "+ Classifier + Localization (G4 model, eval-only with classifier)",
        "reuses": "G4",
        "eval_output_dir": "outputs/ablation/g6_full/evaluation",
        "classifier_ckpt": DINO_LORA_CKPT,
        "needs_stage_a": True,
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="Run ablation study grid")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Quick pipeline validation with minimal samples and 1 epoch",
    )
    p.add_argument(
        "--experiments", nargs="*", default=None,
        help="Specific experiments to run (e.g. G1 G3 G4). Default: all.",
    )
    p.add_argument(
        "--skip-stage-a", action="store_true",
        help="Skip Stage A (LoRA training). Reuse existing checkpoint.",
    )
    p.add_argument(
        "--cooldown", type=int, default=COOLDOWN_SECS,
        help=f"Seconds to wait between experiments (default: {COOLDOWN_SECS})",
    )
    return p.parse_args()


def _apply_dryrun_overrides(cfg: dict) -> dict:
    cfg = deepcopy(cfg)
    cfg["max_train_samples"] = 32
    cfg["max_val_samples"] = 32
    cfg["max_epochs"] = 1
    cfg["batch_size"] = 4
    cfg["gradient_accumulation_steps"] = 1
    cfg["save_interval"] = 5
    cfg["val_interval"] = 5
    cfg["sample_interval"] = 5
    cfg["log_interval"] = 2
    cfg["num_workers"] = 0
    cfg["early_stopping_patience"] = 0
    cfg["warmup_epochs"] = 0
    return cfg


def _write_temp_config(base_config_path: str, dry_run: bool,
                       overrides: dict | None = None) -> str:
    with open(ROOT / base_config_path) as f:
        cfg = yaml.safe_load(f)
    if dry_run:
        cfg = _apply_dryrun_overrides(cfg)
    if overrides:
        cfg.update(overrides)
    tmp_path = ROOT / "configs" / ".tmp_ablation_config.yaml"
    with open(tmp_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return str(tmp_path)


def run_command(cmd: list[str], description: str) -> int:
    logger.info("=" * 60)
    logger.info("RUNNING: %s", description)
    logger.info("CMD: %s", " ".join(cmd))
    logger.info("=" * 60)
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        logger.error("FAILED (exit %d): %s", result.returncode, description)
    else:
        logger.info("SUCCESS: %s", description)
    return result.returncode


# ── Stage A: DINOv2 LoRA Training ────────────────────────────────────

def train_stage_a(dry_run: bool) -> bool:
    """Train DINOv2 LoRA classifier. Returns True on success."""
    tmp = _write_temp_config("configs/dino_lora_classifier.yaml", dry_run)
    rc = run_command(
        [sys.executable, "training/train_dino_lora.py", "--config", tmp],
        "Stage A: DINOv2 LoRA (single)",
    )
    return rc == 0


# ── Training ─────────────────────────────────────────────────────────

def train_model(config_path: str, description: str, dry_run: bool) -> bool:
    """Train a model using train_stage2.py. Returns True on success."""
    tmp = _write_temp_config(config_path, dry_run)
    rc = run_command(
        [sys.executable, "training/train_stage2.py", "--config", tmp],
        description,
    )
    return rc == 0


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate_model(config_path: str, checkpoint: str, split: str,
                   output_dir: str, dry_run: bool,
                   classifier_ckpt: str | None = None) -> dict | None:
    if not os.path.isfile(ROOT / checkpoint):
        logger.warning("Checkpoint not found: %s (skipping eval)", checkpoint)
        return None

    tmp = _write_temp_config(config_path, dry_run=False)

    cmd = [
        sys.executable, "evaluation/evaluate.py",
        "--config", tmp,
        "--checkpoint", checkpoint,
        "--split", split,
        "--output-dir", output_dir,
    ]
    if classifier_ckpt and os.path.isfile(ROOT / classifier_ckpt):
        cmd.extend(["--classifier-checkpoint", classifier_ckpt])

    if dry_run:
        cmd.extend(["--batch-size", "4", "--max-eval-samples", "32"])

    rc = run_command(cmd, f"Evaluate {split} ({os.path.basename(checkpoint)})")

    results_path = os.path.join(output_dir, "results.json")
    if rc == 0 and os.path.isfile(results_path):
        with open(results_path) as f:
            return json.load(f)
    return None


def _evaluate_checkpoints(config_path: str, output_dir: str, eval_base: str,
                          dry_run: bool, classifier_ckpt: str | None,
                          results: dict, prefix: str = "") -> None:
    """Evaluate best and final checkpoints on val and test splits."""
    for ckpt_name in ("checkpoint-best.pt", "checkpoint-final.pt"):
        tag = "best" if "best" in ckpt_name else "last"
        ckpt_path = f"{output_dir}/{ckpt_name}"
        for split in ("val", "test"):
            eval_dir = f"{eval_base}/{prefix}{tag}_{split}"
            r = evaluate_model(
                config_path, ckpt_path, split, eval_dir, dry_run,
                classifier_ckpt,
            )
            if r:
                results[f"{prefix}{tag}_{split}"] = r


# ── Experiment runner ────────────────────────────────────────────────

def run_experiment(name: str, spec: dict, dry_run: bool, cooldown: int,
                   skip_stage_a: bool) -> dict:
    logger.info("#" * 60)
    logger.info("EXPERIMENT %s: %s", name, spec["description"])
    logger.info("#" * 60)

    results = {"name": name, "description": spec["description"]}
    classifier_ckpt = spec.get("classifier_ckpt")

    # ── Stage A (if needed) ──
    if spec.get("needs_stage_a") and not skip_stage_a:
        if not os.path.isfile(ROOT / DINO_LORA_CKPT):
            logger.info("Stage A checkpoint not found — training now.")
            ok = train_stage_a(dry_run)
            if not ok:
                results["error"] = "Stage A training failed"
                return results
            time.sleep(cooldown)
        else:
            logger.info("Stage A checkpoint found: %s", DINO_LORA_CKPT)

    # ── Eval-only experiment (reuses another experiment's model) ──
    reuses = spec.get("reuses")
    if reuses:
        ref_spec = EXPERIMENTS[reuses]
        ref_config = ref_spec["train_config"]
        with open(ROOT / ref_config) as f:
            ref_cfg = yaml.safe_load(f)
        ref_output = ref_cfg["output_dir"]
        eval_base = spec["eval_output_dir"]

        logger.info(
            "Eval-only: reusing model from %s (%s), evaluating with classifier",
            reuses, ref_output,
        )
        _evaluate_checkpoints(
            ref_config, ref_output, eval_base, dry_run, classifier_ckpt,
            results,
        )
        return results

    # ── Training experiment ──
    train_config = spec["train_config"]
    ok = train_model(train_config, f"{name} training", dry_run)
    if not ok:
        results["error"] = "Training failed"
        return results

    with open(ROOT / train_config) as f:
        cfg = yaml.safe_load(f)
    output_dir = cfg["output_dir"]
    eval_base = str(Path(output_dir).parent / "evaluation")

    _evaluate_checkpoints(
        train_config, output_dir, eval_base, dry_run, classifier_ckpt,
        results,
    )

    return results


# ── Summary ──────────────────────────────────────────────────────────

def write_summary(all_results: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Summary saved to %s", output_path)

    logger.info("\n" + "=" * 80)
    logger.info("ABLATION STUDY RESULTS SUMMARY")
    logger.info("=" * 80)
    header = (
        f"{'Exp':<5} {'Ckpt':<8} {'Split':<5} "
        f"{'Acc':>7} {'F1':>7} {'BLEU4':>7} {'ROUGE':>7} {'CIDEr':>7}"
    )
    logger.info(header)
    logger.info("-" * 80)

    for res in all_results:
        name = res["name"]
        for key in sorted(res.keys()):
            if "_test" in key or "_val" in key:
                parts = key.rsplit("_", 1)
                tag = parts[0]
                split = parts[1]
                metrics = res[key]
                acc = metrics.get("accuracy", 0)
                f1 = metrics.get("f1", 0)
                bleu4 = metrics.get("bleu4", 0)
                rouge = metrics.get("rouge_l", 0)
                cider = metrics.get("cider", 0)
                logger.info(
                    f"{name:<5} {tag:<8} {split:<5} {acc:>7.4f} {f1:>7.4f} "
                    f"{bleu4:>7.4f} {rouge:>7.4f} {cider:>7.4f}"
                )
    logger.info("=" * 80)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.dry_run:
        logger.info("*** DRY-RUN MODE: minimal samples and epochs ***")

    exp_names = args.experiments or list(EXPERIMENTS.keys())
    for name in exp_names:
        if name not in EXPERIMENTS:
            logger.error(
                "Unknown experiment: %s (available: %s)",
                name, list(EXPERIMENTS.keys()),
            )
            sys.exit(1)

    # Validate that eval-only experiments have their dependencies
    for name in exp_names:
        spec = EXPERIMENTS[name]
        reuses = spec.get("reuses")
        if reuses and reuses not in exp_names:
            ref_config = EXPERIMENTS[reuses]["train_config"]
            with open(ROOT / ref_config) as f:
                ref_cfg = yaml.safe_load(f)
            ref_ckpt = f"{ref_cfg['output_dir']}/checkpoint-best.pt"
            if not os.path.isfile(ROOT / ref_ckpt):
                logger.error(
                    "%s reuses %s but checkpoint not found: %s. "
                    "Run %s first or include it in --experiments.",
                    name, reuses, ref_ckpt, reuses,
                )
                sys.exit(1)

    # ── Pre-train Stage A if any experiment needs it ──
    needs_stage_a = any(EXPERIMENTS[n].get("needs_stage_a") for n in exp_names)

    if needs_stage_a and not args.skip_stage_a:
        if not os.path.isfile(ROOT / DINO_LORA_CKPT):
            logger.info("Stage A checkpoint not found — training now.")
            ok = train_stage_a(args.dry_run)
            if not ok:
                logger.error("Stage A training failed! Aborting.")
                sys.exit(1)
            time.sleep(args.cooldown)
        else:
            logger.info("Stage A checkpoint found: %s", DINO_LORA_CKPT)

    # ── Run experiments ──
    all_results: list[dict] = []
    for i, name in enumerate(exp_names):
        spec = EXPERIMENTS[name]
        results = run_experiment(
            name, spec, args.dry_run, args.cooldown,
            args.skip_stage_a,
        )
        all_results.append(results)

        if i < len(exp_names) - 1:
            logger.info("Cooling down for %d seconds...", args.cooldown)
            time.sleep(args.cooldown)

    # ── Write summary ──
    suffix = "_dryrun" if args.dry_run else ""
    summary_path = ROOT / f"outputs/ablation/summary{suffix}.json"
    write_summary(all_results, summary_path)

    # Cleanup temp configs
    for tmp_name in (".tmp_ablation_config.yaml", ".tmp_classifier_config.yaml"):
        tmp = ROOT / "configs" / tmp_name
        if tmp.exists():
            tmp.unlink()

    logger.info("Ablation study complete!")


if __name__ == "__main__":
    main()
