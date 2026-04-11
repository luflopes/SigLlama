"""Evaluation script for SigLlama Stage 2.

Loads a fine-tuned checkpoint, generates answers for a test split,
and computes text-generation quality + detection metrics.

Usage::

    python evaluation/evaluate.py \
        --config configs/finetuning.yaml \
        --checkpoint outputs/finetuning/best_checkpoint \
        --split test \
        --output-dir outputs/evaluation
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.deepfake_vqa_dataset import DeepfakeVQADataset, collate_vqa  # noqa: E402
from models.sigllama_finetune import SigLlamaForFinetune  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    compute_bertscore,
    compute_bleu,
    compute_cider,
    compute_detection_f1,
    compute_meteor,
    compute_rouge,
    compute_sentence_bert,
    parse_real_fake,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SigLlama on deepfake VQA")
    p.add_argument("--config", required=True, help="Path to finetuning YAML config")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--output-dir", default="outputs/evaluation")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--skip-generation", action="store_true",
                    help="Skip generation, only compute metrics from existing results")
    return p.parse_args()


def load_model(cfg: dict, checkpoint_dir: str, device: torch.device) -> SigLlamaForFinetune:
    """Load model from checkpoint with adapter + LoRA weights."""
    lora_target = cfg.get("lora_target_modules", ["q_proj", "v_proj"])
    if isinstance(lora_target, str):
        lora_target = [m.strip() for m in lora_target.split(",")]

    model = SigLlamaForFinetune(
        siglip_model=cfg["siglip_model"],
        llm_model=cfg["llm_model"],
        visual_dim=cfg.get("visual_dim", 768),
        llm_dim=cfg.get("llm_dim", 2048),
        adapter_checkpoint=None,
        lora_rank=cfg.get("lora_rank", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_target_modules=lora_target,
        lora_dropout=0.0,
    )

    adapter_path = os.path.join(checkpoint_dir, "adapter.pt")
    if os.path.isfile(adapter_path):
        ckpt = torch.load(adapter_path, map_location="cpu", weights_only=True)
        state = ckpt["adapter"] if isinstance(ckpt, dict) and "adapter" in ckpt else ckpt
        model.adapter.load_state_dict(state)
        logger.info("Loaded adapter from %s", adapter_path)

    lora_path = os.path.join(checkpoint_dir, "lora")
    if os.path.isdir(lora_path):
        from peft import PeftModel
        model.llm = PeftModel.from_pretrained(
            model.llm.get_base_model(), lora_path
        )
        logger.info("Loaded LoRA from %s", lora_path)

    model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.json")
    predictions_path = os.path.join(args.output_dir, "predictions.jsonl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg["llm_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    processor = AutoImageProcessor.from_pretrained(cfg["siglip_model"])

    # ---- Dataset ----
    if args.split == "test":
        meta_path = cfg.get("test_metadata", cfg["val_metadata"].replace("val.", "test."))
    else:
        meta_path = cfg["val_metadata"]

    dataset = DeepfakeVQADataset(
        metadata_path=meta_path,
        image_root=cfg["image_root"],
        processor=processor,
        tokenizer=tokenizer,
        max_length=cfg.get("max_text_length", 256),
    )

    if not args.skip_generation:
        model = load_model(cfg, args.checkpoint, device)

        logger.info("Generating predictions for %d samples...", len(dataset))

        all_predictions = []
        all_references = []
        all_questions = []
        all_is_real = []

        for i in tqdm(range(len(dataset)), desc="Generating"):
            sample = dataset[i]
            if sample is None:
                continue

            raw = dataset.samples[i]
            pv = sample["pixel_values"].unsqueeze(0).to(device)
            question = raw["question"]

            with torch.no_grad():
                gen = model.generate(
                    pv, tokenizer, prompt=question,
                    max_new_tokens=args.max_new_tokens,
                )

            prediction = gen[0]
            reference = raw["answer"]

            all_predictions.append(prediction)
            all_references.append(reference)
            all_questions.append(question)
            all_is_real.append(raw.get("is_real", False))

        # Save predictions
        with open(predictions_path, "w") as f:
            for q, pred, ref, is_real in zip(
                all_questions, all_predictions, all_references, all_is_real
            ):
                f.write(json.dumps({
                    "question": q,
                    "prediction": pred,
                    "reference": ref,
                    "is_real": is_real,
                }) + "\n")
        logger.info("Saved predictions to %s", predictions_path)

    else:
        logger.info("Loading existing predictions from %s", predictions_path)
        all_predictions, all_references, all_questions, all_is_real = [], [], [], []
        with open(predictions_path) as f:
            for line in f:
                item = json.loads(line)
                all_predictions.append(item["prediction"])
                all_references.append(item["reference"])
                all_questions.append(item["question"])
                all_is_real.append(item["is_real"])

    # ---- Compute Metrics ----
    logger.info("Computing metrics on %d samples...", len(all_predictions))
    metrics: dict[str, float] = {}

    logger.info("  BLEU...")
    metrics.update(compute_bleu(all_predictions, all_references))

    logger.info("  ROUGE...")
    metrics.update(compute_rouge(all_predictions, all_references))

    logger.info("  BERTScore...")
    metrics.update(compute_bertscore(all_predictions, all_references))

    logger.info("  Sentence-BERT...")
    metrics["sentence_bert"] = compute_sentence_bert(all_predictions, all_references)

    logger.info("  METEOR...")
    metrics["meteor"] = compute_meteor(all_predictions, all_references)

    logger.info("  CIDEr...")
    metrics["cider"] = compute_cider(all_predictions, all_references)

    # Detection accuracy from general questions
    general_preds = []
    general_labels = []
    for q, pred, is_real in zip(all_questions, all_predictions, all_is_real):
        if "image" in q.lower() and ("real" in q.lower() or "fake" in q.lower()):
            parsed = parse_real_fake(pred)
            if parsed != "unknown":
                general_preds.append(parsed)
                general_labels.append("real" if is_real else "fake")

    if general_preds:
        logger.info("  Detection F1 (%d samples)...", len(general_preds))
        metrics.update(compute_detection_f1(general_preds, general_labels))
    else:
        logger.warning("  No general questions found for detection metrics")

    # ---- Save results ----
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("\n=== Results ===")
    for k, v in sorted(metrics.items()):
        logger.info("  %-20s: %.4f", k, v)
    logger.info("Saved to %s", results_path)


if __name__ == "__main__":
    main()
