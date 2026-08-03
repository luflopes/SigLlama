#!/bin/bash

# G3 / G4 — verdict_score no DD-VQA (val e test)
for split in val test; do
  python evaluation/evaluate.py --config configs/ablation/g3_lora.yaml \
    --checkpoint outputs/ablation/g3_lora/stage2/checkpoint-best.pt \
    --split $split --export-scores \
    --output-dir outputs/ablation/g3_lora/evaluation/scores_$split

  python evaluation/evaluate.py --config configs/ablation/g4_lora_loc.yaml \
    --checkpoint outputs/ablation/g4_lora_loc/stage3/checkpoint-best.pt \
    --split $split --export-scores \
    --output-dir outputs/ablation/g4_lora_loc/evaluation/scores_$split
done

# DINOv2 LoRA — P(fake) no FF++ (val e test)
for split in val test; do
  python scripts/export_dino_scores.py \
    --checkpoint outputs/dino_lora_classifier/best.pt \
    --metadata /datasets/deepfake/ff_classification/$split.jsonl \
    --images-dir /datasets/deepfake/ff_classification/frames \
    --output outputs/dino_lora_classifier/scores_$split.jsonl
done