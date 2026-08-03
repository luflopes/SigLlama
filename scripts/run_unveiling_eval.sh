#!/usr/bin/env bash
# Avaliação cross-dataset do modelo Unveiling_Deepfake (Xception, treinado em FF++ c23)
# nos test sets do DD-VQA, Celeb-DF-v2 e WildDeepfake.
#
# Requisitos no ambiente: torch (CUDA), torchvision, albumentations (+albumentations.pytorch),
# opencv-python, scikit-learn, numpy. Repo clonado em ./Unveiling_Deepfake.
#
# Uso: bash scripts/run_unveiling_eval.sh
set -e

CKPT="unvealing_deepfake_models/ffpp_c23.pth"
REPO="Unveiling_Deepfake"
OUT="outputs/unveiling"
mkdir -p "$OUT"

# --- DD-VQA (FF++ in-domain via DD-VQA): múltiplas QA por imagem -> dedup ---
python scripts/eval_unveiling.py \
  --checkpoint "$CKPT" --repo "$REPO" --dataset-name "DD-VQA" \
  --metadata /datasets/deepfake/ddvqa_prepared/test.jsonl \
  --image-key image --images-root /datasets/deepfake/ddvqa_prepared/frames \
  --is-real-key is_real --video-key video_id --dedup-by-image \
  --output "$OUT/ddvqa_test_scores.jsonl" 2>&1 | tee "$OUT/ddvqa_test.log"

# --- Celeb-DF-v2 (test.jsonl oficial: 518 vídeos x 32 frames = 16.576) ---
python scripts/eval_unveiling.py \
  --checkpoint "$CKPT" --repo "$REPO" --dataset-name "Celeb-DF-v2" \
  --metadata /datasets/deepfake/celebdf_prepared/test.jsonl \
  --image-key image --images-root /datasets/deepfake/celebdf_prepared/frames \
  --is-real-key is_real --video-key video_id --dedup-by-image \
  --output "$OUT/celebdf_test_scores.jsonl" 2>&1 | tee "$OUT/celebdf_test.log"

# --- WildDeepfake (test_sampled.jsonl: 32 frames/vídeo = 25.792, usado nos experimentos) ---
python scripts/eval_unveiling.py \
  --checkpoint "$CKPT" --repo "$REPO" --dataset-name "WildDeepfake" \
  --metadata /datasets/deepfake/wilddeepfake_prepared/test_sampled.jsonl \
  --image-key image --images-root /datasets/deepfake/wilddeepfake_prepared/frames \
  --is-real-key is_real --video-key video_id --dedup-by-image \
  --output "$OUT/wilddeepfake_test_scores.jsonl" 2>&1 | tee "$OUT/wilddeepfake_test.log"

echo "Concluído. Scores e logs em $OUT/"
