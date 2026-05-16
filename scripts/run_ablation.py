#!/usr/bin/env python3
"""Ablation study orchestration script.

Trains and evaluates the full experiment grid, saving results for each
configuration. Supports dry-run mode to validate the pipeline before
committing to long training runs.

Usage::

    # Full run
    python scripts/run_ablation.py

    # Dry-run (few samples, 1 epoch — validate pipeline)
    python scripts/run_ablation.py --dry-run

    # Specific experiments only
    python scripts/run_ablation.py --experiments G1 G4 G5

    # Skip classifier training (reuse existing)
    python scripts/run_ablation.py --skip-classifier
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
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

# ── Experiment grid ──────────────────────────────────────────────────

EXPERIMENTS = {
    "G1": {
        "description": "Baseline: SigLIP only, end-to-end, no DINOv2, no aug",
        "stage2_config": "configs/ablation/g1_baseline.yaml",
        "stage3_config": None,
        "use_classifier": False,
    },
    "G2": {
        "description": "+ DINOv2 I-MoF, end-to-end, no aug",
        "stage2_config": "configs/ablation/g2_imof.yaml",
        "stage3_config": None,
        "use_classifier": False,
    },
    "G4": {
        "description": "+ DINOv2 classifier (verdict desacoplado), no aug",
        "stage2_config": "configs/ablation/g4_classifier.yaml",
        "stage3_config": None,
        "use_classifier": True,
    },
    "G5": {
        "description": "+ Classifier + augmentation",
        "stage2_config": "configs/ablation/g5_classifier_aug.yaml",
        "stage3_config": None,
        "use_classifier": True,
    },
    "G6": {
        "description": "Full pipeline: classifier + aug + localization",
        "stage2_config": "configs/ablation/g5_classifier_aug.yaml",
        "stage3_config": "configs/ablation/g6_full_stage3.yaml",
        "use_classifier": True,
    },
}

CLASSIFIER_CONFIG = "configs/dino_classifier.yaml"
CLASSIFIER_CHECKPOINT = "outputs/ablation/classifier/best.pt"
COOLDOWN_SECS = 120


def parse_args():
    p = argparse.ArgumentParser(description="Run ablation study grid")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Quick pipeline validation with minimal samples and 1 epoch",
    )
    p.add_argument(
        "--experiments", nargs="*", default=None,
        help="Specific experiments to run (e.g. G1 G4 G5). Default: all.",
    )
    p.add_argument(
        "--skip-classifier", action="store_true",
        help="Skip classifier training (reuse existing checkpoint)",
    )
    p.add_argument(
        "--cooldown", type=int, default=COOLDOWN_SECS,
        help=f"Seconds to wait between experiments (default: {COOLDOWN_SECS})",
    )
    return p.parse_args()


def _apply_dryrun_overrides(cfg: dict) -> dict:
    """Override config values for a fast dry-run."""
    cfg = deepcopy(cfg)
    cfg["max_train_samples"] = 32
    cfg["max_epochs"] = 1
    cfg["batch_size"] = 4
    cfg["gradient_accumulation_steps"] = 1
    cfg["save_interval"] = 5
    cfg["val_interval"] = 5
    cfg["sample_interval"] = 5
    cfg["log_interval"] = 2
    cfg["num_workers"] = 0
    return cfg


def _write_temp_config(base_config_path: str, dry_run: bool, output_dir_override: str | None = None) -> str:
    """Load a config, optionally apply dry-run overrides, write to a temp file."""
    with open(ROOT / base_config_path) as f:
        cfg = yaml.safe_load(f)
    if dry_run:
        cfg = _apply_dryrun_overrides(cfg)
    if output_dir_override:
        cfg["output_dir"] = output_dir_override
    tmp_path = ROOT / "configs" / ".tmp_ablation_config.yaml"
    with open(tmp_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return str(tmp_path)


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and return exit code."""
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


def train_classifier(dry_run: bool) -> bool:
    """Train the DINOv2 binary classifier. Returns True on success."""
    with open(ROOT / CLASSIFIER_CONFIG) as f:
        cfg = yaml.safe_load(f)

    cfg["output_dir"] = "outputs/ablation/classifier"
    cfg["augmentation"] = True
    if dry_run:
        cfg["max_train_samples"] = 32
        cfg["max_epochs"] = 2
        cfg["batch_size"] = 8
        cfg["num_workers"] = 0

    tmp = ROOT / "configs" / ".tmp_classifier_config.yaml"
    with open(tmp, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    rc = run_command(
        [sys.executable, "training/train_classifier.py", "--config", str(tmp)],
        "Train DINOv2 classifier (Stage A)",
    )
    return rc == 0


def train_stage(config_path: str, description: str, dry_run: bool,
                output_dir_override: str | None = None) -> bool:
    """Train a Stage 2 or 3 model. Returns True on success."""
    tmp = _write_temp_config(config_path, dry_run, output_dir_override)
    rc = run_command(
        [sys.executable, "training/train_stage2.py", "--config", tmp],
        description,
    )
    return rc == 0


def evaluate_model(config_path: str, checkpoint: str, split: str,
                   output_dir: str, dry_run: bool,
                   classifier_ckpt: str | None = None) -> dict | None:
    """Evaluate a checkpoint on a given split. Returns results dict or None."""
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
        cmd.extend(["--batch-size", "4"])

    rc = run_command(cmd, f"Evaluate {split} ({os.path.basename(checkpoint)})")

    results_path = os.path.join(output_dir, "results.json")
    if rc == 0 and os.path.isfile(results_path):
        with open(results_path) as f:
            return json.load(f)
    return None


def run_experiment(name: str, spec: dict, dry_run: bool, cooldown: int) -> dict:
    """Run a single experiment (train + evaluate). Returns results summary."""
    logger.info("#" * 60)
    logger.info("EXPERIMENT %s: %s", name, spec["description"])
    logger.info("#" * 60)

    results = {"name": name, "description": spec["description"]}
    s2_config = spec["stage2_config"]
    s3_config = spec["stage3_config"]
    use_classifier = spec["use_classifier"]
    classifier_ckpt = CLASSIFIER_CHECKPOINT if use_classifier else None

    # ── Stage 2 Training ──
    ok = train_stage(s2_config, f"{name} Stage 2 training", dry_run)
    if not ok:
        results["error"] = "Stage 2 training failed"
        return results

    # Read the output_dir from the config
    with open(ROOT / s2_config) as f:
        s2_cfg = yaml.safe_load(f)
    s2_out = s2_cfg["output_dir"]
    eval_base = str(Path(s2_out).parent / "evaluation")

    # ── Stage 2 Evaluation ──
    for ckpt_name in ("checkpoint-best.pt", "checkpoint-final.pt"):
        tag = "best" if "best" in ckpt_name else "last"
        ckpt_path = f"{s2_out}/{ckpt_name}"
        for split in ("val", "test"):
            eval_dir = f"{eval_base}/s2_{tag}_{split}"
            r = evaluate_model(
                s2_config, ckpt_path, split, eval_dir, dry_run, classifier_ckpt,
            )
            if r:
                results[f"s2_{tag}_{split}"] = r

    # ── Stage 3 (if applicable) ──
    if s3_config:
        time.sleep(cooldown)
        ok = train_stage(s3_config, f"{name} Stage 3 training", dry_run)
        if ok:
            with open(ROOT / s3_config) as f:
                s3_cfg = yaml.safe_load(f)
            s3_out = s3_cfg["output_dir"]
            eval_base_s3 = str(Path(s3_out).parent / "evaluation")

            for ckpt_name in ("checkpoint-best.pt", "checkpoint-final.pt"):
                tag = "best" if "best" in ckpt_name else "last"
                ckpt_path = f"{s3_out}/{ckpt_name}"
                for split in ("val", "test"):
                    eval_dir = f"{eval_base_s3}/s3_{tag}_{split}"
                    r = evaluate_model(
                        s3_config, ckpt_path, split, eval_dir, dry_run, classifier_ckpt,
                    )
                    if r:
                        results[f"s3_{tag}_{split}"] = r
        else:
            results["error_s3"] = "Stage 3 training failed"

    return results


def write_summary(all_results: list[dict], output_path: Path) -> None:
    """Write a summary JSON with all experiment results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Summary saved to %s", output_path)

    # Also print a compact table
    logger.info("\n" + "=" * 80)
    logger.info("ABLATION STUDY RESULTS SUMMARY")
    logger.info("=" * 80)
    header = f"{'Exp':<5} {'Ckpt':<6} {'Split':<5} {'Acc':>7} {'F1':>7} {'BLEU4':>7} {'ROUGE':>7} {'CIDEr':>7}"
    logger.info(header)
    logger.info("-" * 80)

    for res in all_results:
        name = res["name"]
        for key in sorted(res.keys()):
            if key.startswith("s2_") or key.startswith("s3_"):
                parts = key.split("_")
                stage = parts[0]
                tag = parts[1]
                split = parts[2]
                metrics = res[key]
                acc = metrics.get("accuracy", 0)
                f1 = metrics.get("f1", 0)
                bleu4 = metrics.get("bleu4", 0)
                rouge = metrics.get("rouge_l", 0)
                cider = metrics.get("cider", 0)
                label = f"{stage}_{tag}"
                logger.info(
                    f"{name:<5} {label:<6} {split:<5} {acc:>7.4f} {f1:>7.4f} "
                    f"{bleu4:>7.4f} {rouge:>7.4f} {cider:>7.4f}"
                )
    logger.info("=" * 80)


def main():
    args = parse_args()

    if args.dry_run:
        logger.info("*** DRY-RUN MODE: minimal samples and epochs ***")

    # Determine which experiments to run
    exp_names = args.experiments or list(EXPERIMENTS.keys())
    for name in exp_names:
        if name not in EXPERIMENTS:
            logger.error("Unknown experiment: %s (available: %s)", name, list(EXPERIMENTS.keys()))
            sys.exit(1)

    # Check if any experiment needs the classifier
    needs_classifier = any(
        EXPERIMENTS[n]["use_classifier"] for n in exp_names
    )

    # ── Train classifier if needed ──
    if needs_classifier and not args.skip_classifier:
        if not os.path.isfile(ROOT / CLASSIFIER_CHECKPOINT):
            logger.info("Classifier checkpoint not found — training now.")
            ok = train_classifier(args.dry_run)
            if not ok:
                logger.error("Classifier training failed! Aborting.")
                sys.exit(1)
            time.sleep(args.cooldown)
        else:
            logger.info("Classifier checkpoint found: %s", CLASSIFIER_CHECKPOINT)
    elif needs_classifier and args.skip_classifier:
        if not os.path.isfile(ROOT / CLASSIFIER_CHECKPOINT):
            logger.warning(
                "--skip-classifier set but checkpoint not found at %s. "
                "Experiments using classifier may fail.", CLASSIFIER_CHECKPOINT,
            )

    # ── Run experiments ──
    all_results: list[dict] = []
    for i, name in enumerate(exp_names):
        spec = EXPERIMENTS[name]
        results = run_experiment(name, spec, args.dry_run, args.cooldown)
        all_results.append(results)

        if i < len(exp_names) - 1:
            logger.info("Cooling down for %d seconds...", args.cooldown)
            time.sleep(args.cooldown)

    # ── Write summary ──
    suffix = "_dryrun" if args.dry_run else ""
    summary_path = ROOT / f"outputs/ablation/summary{suffix}.json"
    write_summary(all_results, summary_path)

    # Cleanup temp config
    tmp_cfg = ROOT / "configs" / ".tmp_ablation_config.yaml"
    if tmp_cfg.exists():
        tmp_cfg.unlink()
    tmp_cls = ROOT / "configs" / ".tmp_classifier_config.yaml"
    if tmp_cls.exists():
        tmp_cls.unlink()

    logger.info("Ablation study complete!")


if __name__ == "__main__":
    main()
