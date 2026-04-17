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
import re
import sys

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.ddvqa_dataset import DDVQADataset, collate_ddvqa  # noqa: E402
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
    return p.parse_args()


def parse_real_fake(text: str) -> str:
    t = text.lower().strip()
    if re.search(r"\breal\b", t):
        if not re.search(r"\bfake\b", t):
            return "real"
    if re.search(r"\bfake\b", t):
        return "fake"
    return "fake"


def load_model(cfg: dict, checkpoint_path: str, device: torch.device):
    model = build_model(cfg, use_lora=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if model.dino_adapter is not None and "adapter" in ckpt:
        model.dino_adapter.load_state_dict(ckpt["adapter"])
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
    true_labels: list[str] = []
    per_sample_rows: list[dict] = []

    logger.info("Running inference on %d samples...", len(dataset))
    for batch in tqdm(loader, desc="Evaluating"):
        if batch is None:
            continue

        pv_sig = batch["pixel_values_siglip"].to(device)
        pv_din = batch["pixel_values_dino"].to(device)
        inp_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                pixel_values_siglip=pv_sig,
                pixel_values_dino=pv_din,
                input_ids=inp_ids,
                attention_mask=attn,
                max_new_tokens=args.max_new_tokens,
            )

        gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        ref_ids = batch["labels"]
        ref_ids[ref_ids == -100] = tokenizer.pad_token_id or 0
        ref_texts = tokenizer.batch_decode(ref_ids, skip_special_tokens=True)

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

            predictions.append(gen_str)
            references.append(ref_str)
            pred_labels.append(pred_lbl)
            true_labels.append(lbl)

            per_sample_rows.append({
                "image": img,
                "question": q,
                "reference_answer": a or ref_str,
                "generated": gen_str,
                "true_label": lbl,
                "pred_label": pred_lbl,
                "correct": pred_lbl == lbl,
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
        "image", "method", "true_label", "pred_label", "correct",
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
        logger.info("  %s: %.4f", k, v)


if __name__ == "__main__":
    main()
