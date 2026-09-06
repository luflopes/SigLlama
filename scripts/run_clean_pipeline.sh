#!/usr/bin/env bash
# Pipeline revisado (protocolo automático + gold). Rode na tarkin.
#
# PRÉ-REQUISITO (uma vez): copie os JSONL gerados localmente para a tarkin,
# mantendo o mesmo image_root (frames):
#   rsync -av data/ddvqa_clean/  tarkin:~/repos/SigLlama/... e depois para
#   /datasets/deepfake/ddvqa_prepared/clean/
#   rsync -av data/ddvqa_gold/   -> /datasets/deepfake/ddvqa_prepared/gold/
#
# Passos:
#   1) A4-auto-clean  (substitui o A4-auto do ablation)  -> g4_auto_clean.yaml
#   2) A3-clean+gold  (texto, do zero)                   -> g3_clean_gold.yaml
#   3) A4-final       (parte do A3 do passo 2)           -> g4_final.yaml
#
# Como o A4-auto-clean SUBSTITUI o A4 do ablation, os testes cross-dataset
# (WildDeepfake + Celeb-DF-v2) são reexecutados para os A4 novos
# (g4_auto_clean e g4_final), reusando os configs cross existentes e apenas
# trocando --checkpoint/--output-dir. Use CROSS=0 para pular.
#
# Uso:
#   bash scripts/run_clean_pipeline.sh            # tudo (treinos + in-domain + cross)
#   STEP=1 bash scripts/run_clean_pipeline.sh     # só o passo 1 (+ seu cross)
#   STEP=cross bash scripts/run_clean_pipeline.sh # só cross dos A4 já treinados
#   CROSS=0 bash scripts/run_clean_pipeline.sh    # treinos + in-domain, sem cross
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PY="${PYTHON:-python}"; MAXNEW="${MAXNEW:-320}"
STEP="${STEP:-all}"
CROSS="${CROSS:-1}"
CLEAN=/datasets/deepfake/ddvqa_prepared/clean
GOLD=/datasets/deepfake/ddvqa_prepared/gold

run_step () { [ "$STEP" = "all" ] || [ "$STEP" = "$1" ]; }

# Cross-dataset (WildDeepfake + Celeb-DF-v2) para um modelo A4.
#   $1 = nome do modelo (ex.: g4_auto_clean)   $2 = caminho do checkpoint
cross_eval () {
  local model="$1" ckpt="$2"
  [ "$CROSS" = "1" ] || { echo "==> cross desligado (CROSS=0)"; return 0; }
  if [ ! -f "$ckpt" ]; then echo "!! checkpoint ausente p/ cross: $ckpt"; return 0; fi
  for ds in wild celebdf; do
    local out="outputs/cross_dataset/${ds}_${model}/evaluation"
    echo "==> [cross] $model em $ds -> $out"
    "$PY" evaluation/evaluate.py \
      --config "configs/cross_dataset/${ds}_g4_lora_loc.yaml" \
      --checkpoint "$ckpt" --split test --export-scores --batch-size 8 \
      --max-new-tokens "$MAXNEW" --output-dir "$out"
    "$PY" scripts/aggregate_video_predictions.py --predictions "$out/predictions.jsonl"
  done
}

# ---------- PASSO 1: A4-auto-clean ----------
if run_step 1; then
  echo "==> [1] Treino A4-auto-clean"
  "$PY" training/train_stage2.py --config configs/ablation/g4_auto_clean.yaml
  CK=outputs/ablation/g4_auto_clean/stage3/checkpoint-best.pt
  echo "==> [1] Eval: gold test (fino) + clean test (n-grande)"
  "$PY" evaluation/evaluate.py --config configs/ablation/g4_auto_clean.yaml --checkpoint "$CK" \
    --split test --max-new-tokens "$MAXNEW" --test-metadata "$GOLD/test_loc_gold.jsonl" \
    --output-dir outputs/ablation/g4_auto_clean/evaluation/goldtest
  "$PY" evaluation/evaluate.py --config configs/ablation/g4_auto_clean.yaml --checkpoint "$CK" \
    --split test --max-new-tokens "$MAXNEW" \
    --output-dir outputs/ablation/g4_auto_clean/evaluation/cleantest
  # cross-dataset usa o checkpoint FINAL (mesmo protocolo do ablation original)
  cross_eval g4_auto_clean outputs/ablation/g4_auto_clean/stage3/checkpoint-final.pt
fi

# ---------- PASSO 2: A3-clean+gold (texto) ----------
if run_step 2; then
  echo "==> [2] Treino A3-clean+gold (texto, do zero)"
  "$PY" training/train_stage2.py --config configs/ablation/g3_clean_gold.yaml
  CK=outputs/ablation/g3_clean_gold/stage2/checkpoint-best.pt
  echo "==> [2] Eval: gold test (texto) + DD-VQA clean test (n-grande p/ métricas de texto)"
  "$PY" evaluation/evaluate.py --config configs/ablation/g3_clean_gold.yaml --checkpoint "$CK" \
    --split test --max-new-tokens "$MAXNEW" \
    --output-dir outputs/ablation/g3_clean_gold/evaluation/goldtest
  "$PY" evaluation/evaluate.py --config configs/ablation/g3_clean_gold.yaml --checkpoint "$CK" \
    --split test --max-new-tokens "$MAXNEW" --test-metadata "$CLEAN/test_loc_clean.jsonl" \
    --output-dir outputs/ablation/g3_clean_gold/evaluation/cleantest
fi

# ---------- PASSO 3: A4-final ----------
if run_step 3; then
  echo "==> [3] Treino A4-final (warm-start do A3 do passo 2)"
  "$PY" training/train_stage2.py --config configs/ablation/g4_final.yaml
  CK=outputs/ablation/g4_final/stage3/checkpoint-best.pt
  echo "==> [3] Eval: gold test (fino) + clean test (n-grande)"
  "$PY" evaluation/evaluate.py --config configs/ablation/g4_final.yaml --checkpoint "$CK" \
    --split test --max-new-tokens "$MAXNEW" \
    --output-dir outputs/ablation/g4_final/evaluation/goldtest
  "$PY" evaluation/evaluate.py --config configs/ablation/g4_final.yaml --checkpoint "$CK" \
    --split test --max-new-tokens "$MAXNEW" --test-metadata "$CLEAN/test_loc_clean.jsonl" \
    --output-dir outputs/ablation/g4_final/evaluation/cleantest
  cross_eval g4_final outputs/ablation/g4_final/stage3/checkpoint-final.pt
fi

# ---------- CROSS-ONLY: reexecuta cross dos A4 já treinados ----------
if [ "$STEP" = "cross" ]; then
  echo "==> [cross] reexecutando cross-dataset dos A4 novos"
  cross_eval g4_auto_clean outputs/ablation/g4_auto_clean/stage3/checkpoint-final.pt
  cross_eval g4_final      outputs/ablation/g4_final/stage3/checkpoint-final.pt
fi

cat <<EOF

==> Concluído. Traga os predictions.jsonl para a máquina local e rode:

# Texto (n-grande, DD-VQA clean test): A3-clean+gold vs A3 do ablation
python scripts/eval_text_quality.py \\
  --pred outputs/ablation/g3_clean_gold/evaluation/cleantest/predictions.jsonl \\
  --compare outputs/ablation/g3_lora/evaluation/last_test/predictions.jsonl \\
  --labels A3-clean A3-ablation

# Localização (gold test, fino, com IC bootstrap): A4-final vs A4-auto-clean vs A4-auto
python scripts/eval_localization.py \\
  --pred    outputs/ablation/g4_final/evaluation/goldtest/predictions.jsonl \\
  --compare outputs/ablation/g4_auto_clean/evaluation/goldtest/predictions.jsonl \\
  --labels A4-final A4-auto-clean --exclude-methods Original --bootstrap 2000

# Calibração + texto (n-grande, clean test): A4-final vs A4-auto-clean
python scripts/eval_text_quality.py \\
  --pred    outputs/ablation/g4_final/evaluation/cleantest/predictions.jsonl \\
  --compare outputs/ablation/g4_auto_clean/evaluation/cleantest/predictions.jsonl \\
  --labels A4-final A4-auto-clean

# Cross-dataset (WildDeepfake / Celeb-DF-v2) — resultados agregados (frame+vídeo) em:
#   outputs/cross_dataset/{wild,celebdf}_g4_auto_clean/evaluation/
#   outputs/cross_dataset/{wild,celebdf}_g4_final/evaluation/
# Compare com o A4 antigo do ablation (outputs/cross_dataset/{wild,celebdf}_g4_lora_loc/).
EOF
