#!/bin/bash
# Cross-dataset evaluation: G3 and G4 on WildDeepfake
#
# Prerequisites:
#   pip install datasets  (huggingface datasets library)
#
# This script downloads the WildDeepfake test split from HuggingFace,
# prepares the metadata, and runs inference with G3 and G4 models.
#
# Run from the project root on the server:
#   bash scripts/run_wilddeepfake_eval.sh
#
# Optional: limit samples for a quick test:
#   bash scripts/run_wilddeepfake_eval.sh 5000
set -euo pipefail

PYTHON="/home/lucasfl/repos/SigLlama/venv/bin/python"
OUTPUT_DIR="/datasets/deepfake/wilddeepfake_prepared"
MAX_SAMPLES="${1:-0}"

echo "=== Step 1: Download WildDeepfake and prepare metadata ==="
if [ -f "${OUTPUT_DIR}/test.jsonl" ]; then
    echo "  test.jsonl already exists, skipping download (use --skip-existing for images)."
    echo "  Delete ${OUTPUT_DIR}/test.jsonl to force re-download."
else
    $PYTHON scripts/prepare_wilddeepfake.py \
        --output-dir "$OUTPUT_DIR" \
        --max-samples "$MAX_SAMPLES" \
        --skip-existing
fi

echo ""
echo "=== Step 2: Evaluate G3 (DINOv2 LoRA, end-to-end verdict) ==="
$PYTHON evaluation/evaluate.py \
    --config configs/cross_dataset/wild_g3_lora.yaml \
    --checkpoint outputs/ablation/g3_lora/stage2/checkpoint-final.pt \
    --split test \
    --export-scores \
    --batch-size 8

echo ""
echo "=== Step 3: Evaluate G4 (DINOv2 LoRA + Localization) ==="
$PYTHON evaluation/evaluate.py \
    --config configs/cross_dataset/wild_g4_lora_loc.yaml \
    --checkpoint outputs/ablation/g4_lora_loc/stage3/checkpoint-final.pt \
    --split test \
    --export-scores \
    --batch-size 8

echo ""
echo "=== Step 4: Aggregate results (frame + video level) ==="
$PYTHON scripts/aggregate_video_predictions.py \
    --predictions outputs/cross_dataset/wild_g3_lora/evaluation/predictions.jsonl

$PYTHON scripts/aggregate_video_predictions.py \
    --predictions outputs/cross_dataset/wild_g4_lora_loc/evaluation/predictions.jsonl

echo ""
echo "=== Done! Results at: ==="
echo "  G3: outputs/cross_dataset/wild_g3_lora/evaluation/"
echo "  G4: outputs/cross_dataset/wild_g4_lora_loc/evaluation/"
