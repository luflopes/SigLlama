"""Evaluate FaceGroundVLM on DD-VQA (or any VQA split).

Metrics follow TruthLens: detection accuracy, BLEU-3, BLEU-4, ROUGE-L, CIDEr.

Usage::

    python evaluation/evaluate.py \
        --config configs/stage2_finetune.yaml \
        --checkpoint outputs/stage2_finetune/checkpoint-final.pt \
        --split val
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import torch
import yaml
from tqdm import tqdm
from transformers import LogitsProcessor, LogitsProcessorList

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.ddvqa_dataset import DDVQADataset, collate_ddvqa  # noqa: E402
from data.prompt_formats import build_generation_inputs  # noqa: E402
from evaluation.metrics import parse_real_fake, parse_real_fake_legacy  # noqa: E402
from models import build_model  # noqa: E402
from training.factory import build_processor_and_transforms  # noqa: E402

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", level=logging.INFO)
logger = logging.getLogger("evaluate")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True, help="Checkpoint .pt with adapter + lora")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output-dir", default=None)
    p.add_argument(
        "--repetition-penalty", type=float, default=None,
        help="Override generation_config.repetition_penalty (default: use model's).",
    )
    p.add_argument(
        "--no-repeat-ngram-size", type=int, default=None,
        help="Override generation_config.no_repeat_ngram_size (default: use model's).",
    )
    p.add_argument(
        "--constrained-first-token",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Restrict the first generated token to a Real/Fake candidate "
            "(use --no-constrained-first-token to disable)."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Constrained decoding — multi-token verdict prefix
# ---------------------------------------------------------------------------
#
# TinyLlama's SentencePiece vocab includes single-token encodings for
# ``▁Real`` / ``▁real`` / ``▁fake`` but *not* for ``▁Fake`` (which
# tokenises as ``[▁F, ake]``). Restricting only the first generated token
# to a set of single-token verdicts therefore *excluded* "Fake" entirely
# and the model collapsed onto "Real" via prior. The fix is a small
# state-machine processor that allows whichever verdict path has been
# entered and forces the remaining tokens of that path until the verdict
# is fully emitted.

_VERDICT_SURFACE_FORMS = (" Real", " Fake", " real", " fake")


def _build_verdict_paths(tokenizer) -> list[list[int]]:
    """Return the list of token-id sequences spelling out each verdict.

    Encoding is done with ``add_special_tokens=False`` and a leading
    whitespace so the path matches what the model sees during training
    (where the answer text is `` Real./ Fake.`` after the assistant turn
    marker). Duplicate paths (e.g. when the slow vs fast tokenizer emit
    the same ids for two surface forms) are collapsed.
    """
    paths: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for s in _VERDICT_SURFACE_FORMS:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if not ids:
            continue
        key = tuple(int(i) for i in ids)
        if key in seen:
            continue
        seen.add(key)
        paths.append(list(key))
    return paths


class VerdictPrefixProcessor(LogitsProcessor):
    """Force generation to emit one of the verdict id-sequences first.

    Each row is constrained until it has emitted a full verdict path. At
    every generation step the processor:
      * looks at the row's already-generated tokens
      * keeps the verdict paths still consistent with that prefix
      * if all consistent paths are still incomplete, masks out every
        token id that isn't the next legal token of any consistent path
      * if no consistent path exists (path completed or diverged), leaves
        the row's logits untouched

    Because we feed the prompt via ``inputs_embeds`` to ``generate``, the
    ``input_ids`` tensor passed to ``__call__`` only contains the tokens
    produced so far. ``input_ids.shape[-1] == 0`` therefore marks the
    very first generation step.
    """

    def __init__(self, verdict_paths: list[list[int]]):
        super().__init__()
        self.paths: list[tuple[int, ...]] = [tuple(p) for p in verdict_paths if p]
        self.max_len = max((len(p) for p in self.paths), default=0)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        gen_len = input_ids.shape[-1]
        if gen_len >= self.max_len:
            return scores

        out = scores.clone()
        for b in range(input_ids.shape[0]):
            if gen_len == 0:
                allowed = {p[0] for p in self.paths}
            else:
                prev = tuple(int(t) for t in input_ids[b, :gen_len].tolist())
                consistent = [
                    p for p in self.paths
                    if len(p) > gen_len and p[:gen_len] == prev
                ]
                if not consistent:
                    continue
                allowed = {p[gen_len] for p in consistent}

            mask = torch.full_like(out[b], float("-inf"))
            allowed_t = torch.tensor(sorted(allowed), device=out.device, dtype=torch.long)
            mask[allowed_t] = out[b, allowed_t]
            out[b] = mask
        return out


def load_model(cfg: dict, checkpoint_path: str, device: torch.device):
    model = build_model(cfg, use_lora=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if model.dino_adapter is not None and "adapter" in ckpt:
        model.dino_adapter.load_state_dict(ckpt["adapter"])
    if "connector" in ckpt and hasattr(model, "connector"):
        model.connector.load_state_dict(ckpt["connector"])
        logger.info("Loaded fine-tuned connector weights from checkpoint")
    if "lora" in ckpt:
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model.language_model, ckpt["lora"])
    logger.info("Loaded checkpoint from %s (step %s)", checkpoint_path, ckpt.get("step", "?"))

    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, args.checkpoint, device)

    proc = build_processor_and_transforms(cfg)
    tokenizer = proc["tokenizer"]
    image_processor = proc["image_processor"]
    dino_transform = proc["dino_transform"]

    meta_key = "val_metadata" if args.split == "val" else "test_metadata"
    metadata_path = cfg.get(meta_key, cfg.get("val_metadata"))

    dataset = DDVQADataset(
        metadata_path=metadata_path,
        image_root=cfg["image_root"],
        tokenizer=tokenizer,
        image_processor=image_processor,
        dino_transform=dino_transform,
        backbone=cfg.get("backbone", "paligemma"),
        max_length=cfg.get("max_text_length", 256),
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_ddvqa,
    )

    predictions: list[str] = []
    references: list[str] = []
    pred_labels: list[str] = []
    pred_labels_legacy: list[str] = []
    true_labels: list[str] = []
    per_sample_rows: list[dict] = []

    backbone = cfg.get("backbone", "paligemma")

    first_token_processor: LogitsProcessorList | None = None
    if args.constrained_first_token:
        verdict_paths = _build_verdict_paths(tokenizer)
        if verdict_paths:
            first_token_processor = LogitsProcessorList(
                [VerdictPrefixProcessor(verdict_paths)]
            )
            paths_dump = [
                {
                    "ids": p,
                    "tokens": tokenizer.convert_ids_to_tokens(p),
                    "decoded": tokenizer.decode(p),
                }
                for p in verdict_paths
            ]
            logger.info(
                "Constrained verdict-prefix decoding enabled | %d paths: %s",
                len(verdict_paths),
                paths_dump,
            )
        else:
            logger.warning(
                "Constrained verdict-prefix decoding requested but no "
                "Real/Fake token paths resolved — falling back to "
                "unconstrained decoding."
            )

    logger.info("Running inference on %d samples...", len(dataset))
    for batch in tqdm(loader, desc="Evaluating"):
        if batch is None:
            continue

        # Rebuild *query-only* inputs (left-padded) for generation so the
        # model is never fed the reference answer.
        questions = batch.get("question", [])
        gen_inputs = build_generation_inputs(
            questions, tokenizer, backbone,
            max_length=cfg.get("max_text_length", 256),
        )
        pv_sig = batch["pixel_values_siglip"].to(device)
        pv_din = batch["pixel_values_dino"].to(device)
        inp_ids = gen_inputs["input_ids"].to(device)
        attn = gen_inputs["attention_mask"].to(device)

        gen_overrides: dict = {}
        if args.repetition_penalty is not None:
            gen_overrides["repetition_penalty"] = args.repetition_penalty
        if args.no_repeat_ngram_size is not None:
            gen_overrides["no_repeat_ngram_size"] = args.no_repeat_ngram_size
        if first_token_processor is not None:
            gen_overrides["logits_processor"] = first_token_processor

        with torch.no_grad():
            gen_ids = model.generate(
                pixel_values_siglip=pv_sig,
                pixel_values_dino=pv_din,
                input_ids=inp_ids,
                attention_mask=attn,
                max_new_tokens=args.max_new_tokens,
                **gen_overrides,
            )

        gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        # Reference strings come from the original (untruncated) answer field.
        ref_texts = list(batch.get("answer", []))

        images = batch.get("image", [""] * len(gen_texts))
        questions = batch.get("question", [""] * len(gen_texts))
        answers = batch.get("answer", [""] * len(gen_texts))
        methods = batch.get("method", ["unknown"] * len(gen_texts))

        for gen, ref, lbl, img, q, a, meth in zip(
            gen_texts, ref_texts, batch["label_str"],
            images, questions, answers, methods,
        ):
            gen_str = gen.strip()
            ref_str = ref.strip()
            pred_lbl = parse_real_fake(gen_str)
            pred_lbl_legacy = parse_real_fake_legacy(gen_str)

            predictions.append(gen_str)
            references.append(ref_str)
            pred_labels.append(pred_lbl)
            pred_labels_legacy.append(pred_lbl_legacy)
            true_labels.append(lbl)

            per_sample_rows.append({
                "image": img,
                "question": q,
                "reference_answer": a or ref_str,
                "generated": gen_str,
                "true_label": lbl,
                "pred_label": pred_lbl,
                "pred_label_legacy": pred_lbl_legacy,
                "correct": pred_lbl == lbl,
                "correct_legacy": pred_lbl_legacy == lbl,
                "method": meth,
            })

    from evaluation.metrics import (
        compute_bleu,
        compute_rouge,
        compute_cider,
        compute_detection_f1,
    )

    bleu = compute_bleu(predictions, references)
    rouge = compute_rouge(predictions, references)
    detection = compute_detection_f1(pred_labels, true_labels)
    detection_legacy = compute_detection_f1(pred_labels_legacy, true_labels)

    try:
        cider_score = compute_cider(predictions, references)
    except Exception as e:
        logger.warning("CIDEr computation failed: %s", e)
        cider_score = 0.0

    results = {
        **bleu,
        **rouge,
        "cider": cider_score,
        **detection,
        "detection_legacy": detection_legacy,
        "constrained_first_token": bool(first_token_processor is not None),
    }

    output_dir = args.output_dir or os.path.join(cfg["output_dir"], "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    predictions_path = os.path.join(output_dir, "predictions.jsonl")
    with open(predictions_path, "w", encoding="utf-8") as f:
        for row in per_sample_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    import csv
    predictions_csv = os.path.join(output_dir, "predictions.csv")
    fieldnames = [
        "image", "method", "true_label",
        "pred_label", "pred_label_legacy",
        "correct", "correct_legacy",
        "question", "reference_answer", "generated",
    ]
    with open(predictions_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in per_sample_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    logger.info("Results saved to %s/results.json", output_dir)
    logger.info(
        "Per-sample predictions saved: %s (%d rows) and %s",
        predictions_path, len(per_sample_rows), predictions_csv,
    )
    for k, v in sorted(results.items()):
        if isinstance(v, float):
            logger.info("  %s: %.4f", k, v)
        elif isinstance(v, (int, bool, str)):
            logger.info("  %s: %s", k, v)
        elif isinstance(v, dict):
            compact = {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
            logger.info("  %s: %s", k, compact)


if __name__ == "__main__":
    main()
