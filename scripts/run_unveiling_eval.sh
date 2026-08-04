#!/usr/bin/env bash
# Avaliação cross-dataset do Xception_Net (rede tripla completa do Unveiling_Deepfake),
# treinado em FF++ c23 (faces limpas), nos test sets do DD-VQA, Celeb-DF-v2 e WildDeepfake.
#
# Requisitos no ambiente (venv): torch (CUDA), torchvision, torchjpeg, albumentations
# (+albumentations.pytorch), opencv-python, scikit-learn, numpy. Repo em ./Unveiling_Deepfake.
#
# Uso: CKPT=outputs/unveiling_train/<timestamp>/best.pkl bash scripts/run_unveiling_eval.sh
set -e

# Checkpoint treinado (rede tripla). Sobrescreva via variável de ambiente CKPT.
CKPT="${CKPT:-outputs/unveiling_train/best.pkl}"
ARCH="${ARCH:-full}"
XCEPTION_PRETRAINED="${XCEPTION_PRETRAINED:-unvealing_deepfake_models/xception-b5690688.pth}"
REPO="Unveiling_Deepfake"
OUT="outputs/unveiling"
mkdir -p "$OUT"

if [ ! -f "$CKPT" ]; then
  echo "[ERRO] checkpoint não encontrado: $CKPT"
  echo "       Treine primeiro (scripts/run_unveiling_train.sh) ou aponte CKPT=<caminho>."
  exit 1
fi

COMMON=( --repo "$REPO" --arch "$ARCH" --xception-pretrained "$XCEPTION_PRETRAINED" \
         --image-key image --is-real-key is_real --video-key video_id --dedup-by-image )

# --- FF++ (c23) in-domain TEST: mesmo split usado no treino/val (ff_classification) ---
python scripts/eval_unveiling.py \
  --checkpoint "$CKPT" --dataset-name "FF++ (c23) test" \
  --metadata /datasets/deepfake/ff_classification/test.jsonl \
  --images-root /datasets/deepfake/ff_classification/frames \
  "${COMMON[@]}" \
  --output "$OUT/ffpp_test_scores.jsonl" 2>&1 | tee "$OUT/ffpp_test.log"

# --- DD-VQA (FF++ in-domain via DD-VQA): múltiplas QA por imagem -> dedup ---
python scripts/eval_unveiling.py \
  --checkpoint "$CKPT" --dataset-name "DD-VQA" \
  --metadata /datasets/deepfake/ddvqa_prepared/test.jsonl \
  --images-root /datasets/deepfake/ddvqa_prepared/frames \
  "${COMMON[@]}" \
  --output "$OUT/ddvqa_test_scores.jsonl" 2>&1 | tee "$OUT/ddvqa_test.log"

# --- Celeb-DF-v2 (test.jsonl oficial) ---
python scripts/eval_unveiling.py \
  --checkpoint "$CKPT" --dataset-name "Celeb-DF-v2" \
  --metadata /datasets/deepfake/celebdf_prepared/test.jsonl \
  --images-root /datasets/deepfake/celebdf_prepared/frames \
  "${COMMON[@]}" \
  --output "$OUT/celebdf_test_scores.jsonl" 2>&1 | tee "$OUT/celebdf_test.log"

# --- WildDeepfake (test_sampled.jsonl) ---
python scripts/eval_unveiling.py \
  --checkpoint "$CKPT" --dataset-name "WildDeepfake" \
  --metadata /datasets/deepfake/wilddeepfake_prepared/test_sampled.jsonl \
  --images-root /datasets/deepfake/wilddeepfake_prepared/frames \
  "${COMMON[@]}" \
  --output "$OUT/wilddeepfake_test_scores.jsonl" 2>&1 | tee "$OUT/wilddeepfake_test.log"

echo "Concluído. Scores e logs em $OUT/"
