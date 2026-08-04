#!/usr/bin/env bash
# Treino do Xception_Net completo (Unveiling_Deepfake) no FF++ c23 (faces limpas),
# em 1 GPU. Rodar no tarkin, dentro do venv.
#
# Uso:
#   bash scripts/run_unveiling_train.sh                 # treino completo (15 épocas)
#   SANITY=1 bash scripts/run_unveiling_train.sh        # sanity check rápido (subamostra)
set -e

REPO="Unveiling_Deepfake"
DATA_ROOT="${DATA_ROOT:-/datasets/deepfake/ff_classification}"
XCEPTION_PRETRAINED="${XCEPTION_PRETRAINED:-unvealing_deepfake_models/xception-b5690688.pth}"
OUT="${OUT:-outputs/unveiling_train}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
EPOCHS="${EPOCHS:-15}"
GPU="${GPU:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# --- dependência do ramo DCT (torchjpeg) ---
# Descomente para instalar (precisa casar com a versão do torch instalada):
#   pip install torchjpeg albumentations
python -c "import torchjpeg" 2>/dev/null || {
  echo "[AVISO] torchjpeg não instalado. Instale com: pip install torchjpeg albumentations"
  echo "        (a versão do torchjpeg precisa ser compatível com o torch do venv)"
  exit 1
}

EXTRA=()
if [ "${SANITY:-0}" = "1" ]; then
  echo "== SANITY CHECK: subamostra de frames + 2 épocas =="
  EXTRA=( --frames-per-video 5 --max-val 2000 --epochs 2 )
else
  EXTRA=( --epochs "$EPOCHS" )
fi

python scripts/train_unveiling.py \
  --repo "$REPO" \
  --data-root "$DATA_ROOT" \
  --xception-pretrained "$XCEPTION_PRETRAINED" \
  --batch-size "$BATCH_SIZE" --grad-accum "$GRAD_ACCUM" \
  --gpu "$GPU" --num-workers "$NUM_WORKERS" \
  --output-dir "$OUT" \
  "${EXTRA[@]}"

echo "Treino concluído. Aponte CKPT=$OUT/<timestamp>/best.pkl para scripts/run_unveiling_eval.sh"
