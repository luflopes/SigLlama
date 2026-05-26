#!/bin/bash
# Cross-dataset evaluation: G3 and G4 on Celeb-DF-v2
#
# Prerequisites:
#   1. Transfer the zip to the server:
#      scp /home/lucas/Downloads/Celeb-DF-v2.zip servidor:/datasets/deepfake/
#
#   2. On the server, unzip:
#      cd /datasets/deepfake/ && unzip Celeb-DF-v2.zip
#
# Then run this script from the project root on the server.
set -euo pipefail

CELEBDF_ROOT="/datasets/deepfake/Celeb-DF-v2"
OUTPUT_DIR="/datasets/deepfake/celebdf_prepared"
NUM_FRAMES="${1:-32}"

echo "=== Step 1: Extract frames (num_frames=${NUM_FRAMES}) ==="
python scripts/prepare_celebdf.py \
    --celebdf-root "$CELEBDF_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --num-frames "$NUM_FRAMES" \
    --skip-existing

echo ""
echo "=== Step 2: Evaluate G3 (DINOv2 LoRA, end-to-end verdict) ==="
python evaluation/evaluate.py \
    --config configs/cross_dataset/celebdf_g3_lora.yaml \
    --checkpoint outputs/ablation/g3_lora/stage2/checkpoint-final.pt \
    --split test \
    --export-scores \
    --batch-size 8

echo ""
echo "=== Step 3: Evaluate G4 (DINOv2 LoRA + Localization) ==="
python evaluation/evaluate.py \
    --config configs/cross_dataset/celebdf_g4_lora_loc.yaml \
    --checkpoint outputs/ablation/g4_lora_loc/stage3/checkpoint-final.pt \
    --split test \
    --export-scores \
    --batch-size 8

echo ""
echo "=== Step 4: Aggregate results (frame + video level) ==="
python scripts/aggregate_video_predictions.py \
    --predictions outputs/cross_dataset/celebdf_g3_lora/evaluation/predictions.jsonl

python scripts/aggregate_video_predictions.py \
    --predictions outputs/cross_dataset/celebdf_g4_lora_loc/evaluation/predictions.jsonl

echo ""
echo "=== Done! Results at: ==="
echo "  G3: outputs/cross_dataset/celebdf_g3_lora/evaluation/"
echo "  G4: outputs/cross_dataset/celebdf_g4_lora_loc/evaluation/"
