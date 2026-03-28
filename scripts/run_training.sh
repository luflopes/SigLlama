#!/usr/bin/env bash
# =============================================================
# SigLlama — Launch training script
# =============================================================
# Usage:
#   bash scripts/run_training.sh pretrain   # Stage 1
#   bash scripts/run_training.sh finetune   # Stage 2

set -euo pipefail

STAGE="${1:-pretrain}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

case "$STAGE" in
    pretrain)
        echo "[SigLlama] Stage 1: Adapter pre-training on LCS-558K"
        python "$PROJECT_ROOT/training/pretrain.py" \
            --config "$PROJECT_ROOT/configs/pretraining.yaml"
        ;;
    finetune)
        echo "[SigLlama] Stage 2: Fine-tuning with MoE"
        python "$PROJECT_ROOT/training/finetune.py" \
            --config "$PROJECT_ROOT/configs/finetuning.yaml"
        ;;
    *)
        echo "Usage: $0 {pretrain|finetune}"
        exit 1
        ;;
esac
