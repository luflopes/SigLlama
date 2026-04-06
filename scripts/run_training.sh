#!/usr/bin/env bash
# =============================================================
# SigLlama — Launch training script
# =============================================================
# Usage:
#   bash scripts/run_training.sh pretrain              # single GPU
#   bash scripts/run_training.sh pretrain --multi-gpu   # multi-GPU via accelerate
#   bash scripts/run_training.sh finetune

set -euo pipefail

STAGE="${1:-pretrain}"
MULTI_GPU=false
shift || true
for arg in "$@"; do
    case "$arg" in
        --multi-gpu) MULTI_GPU=true ;;
    esac
done

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

case "$STAGE" in
    pretrain)
        echo "[SigLlama] Stage 1: Adapter pre-training on LCS-558K"
        if [ "$MULTI_GPU" = true ]; then
            accelerate launch "$PROJECT_ROOT/training/pretrain.py" \
                --config "$PROJECT_ROOT/configs/pretraining.yaml"
        else
            python "$PROJECT_ROOT/training/pretrain.py" \
                --config "$PROJECT_ROOT/configs/pretraining.yaml"
        fi
        ;;
    finetune)
        echo "[SigLlama] Stage 2: Fine-tuning with MoE"
        python "$PROJECT_ROOT/training/finetune.py" \
            --config "$PROJECT_ROOT/configs/finetuning.yaml"
        ;;
    *)
        echo "Usage: $0 {pretrain|finetune} [--multi-gpu]"
        exit 1
        ;;
esac
