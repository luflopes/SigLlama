#!/usr/bin/env bash
# Avalia o A4-manual (run g4_a4_gold, warm-start do A4-auto) no test_gold e
# val_gold, testando OS DOIS checkpoints:
#   - checkpoint-best.pt  (menor val_loss; pode ser um ponto ANTES de o modelo
#     estabilizar a emissão de caixas)
#   - checkpoint-final.pt (fim do treino; costuma emitir mais caixas)
# Assim decidimos empiricamente qual usar para a localização.
#
# Rode na tarkin:
#   bash scripts/eval_a4_gold.sh
#   RUN=g4_a4_gold CKPTS="checkpoint-final.pt" bash scripts/eval_a4_gold.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/ablation/g4_gold.yaml}"
RUN="${RUN:-g4_a4_gold}"
STAGE="outputs/ablation/$RUN/stage3"
EVAL="outputs/ablation/$RUN/evaluation"
MAXNEW="${MAXNEW:-320}"
CKPTS="${CKPTS:-checkpoint-best.pt checkpoint-final.pt}"

echo "==> RUN=$RUN  STAGE=$STAGE  CONFIG=$CONFIG  MAXNEW=$MAXNEW"

for CK in $CKPTS; do
  if [ ! -f "$STAGE/$CK" ]; then
    echo "!! ausente, pulando: $STAGE/$CK"
    continue
  fi
  name="${CK%.pt}"; name="${name#checkpoint-}"   # -> best / final
  for SPLIT in test val; do
    echo "==> Avaliando $CK em ${SPLIT}_gold -> $EVAL/${name}_${SPLIT}"
    "$PYTHON" evaluation/evaluate.py \
      --config "$CONFIG" \
      --checkpoint "$STAGE/$CK" \
      --split "$SPLIT" \
      --max-new-tokens "$MAXNEW" \
      --output-dir "$EVAL/${name}_${SPLIT}"
  done
done

cat <<EOF

==> Concluído. Predições (test) em:
    $EVAL/best_test/predictions.jsonl
    $EVAL/final_test/predictions.jsonl

Traga para a máquina local e rode as métricas de localização, ex.:
    python scripts/eval_localization.py --pred $EVAL/final_test/predictions.jsonl
    python scripts/eval_localization.py --pred $EVAL/best_test/predictions.jsonl
EOF
