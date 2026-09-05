#!/usr/bin/env bash
# Treina o A4-manual (G4-gold) em cima do A3 (G3) usando o gold anotado,
# depois avalia no test_gold e val_gold. Rode na máquina de treino (tarkin).
#
#   bash scripts/train_a4_gold.sh                # treina + avalia
#   SKIP_TRAIN=1 bash scripts/train_a4_gold.sh   # só avalia (checkpoint existente)
#
# Pré-requisitos:
#   1) A3 treinado: outputs/ablation/g3_lora/stage2/checkpoint-final.pt
#   2) Gold copiado para: /datasets/deepfake/ddvqa_prepared/gold/{train,val,test}_loc_gold.jsonl
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
CONFIG="configs/ablation/g4_gold.yaml"
OUT="outputs/ablation/g4_gold/stage3"
CKPT="${CKPT:-$OUT/checkpoint-best.pt}"

echo "==> Config: $CONFIG"

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  echo "==> Treinando A4-manual (warm-start do A3/G3)"
  "$PYTHON" training/train_stage2.py --config "$CONFIG"
fi

for SPLIT in test val; do
  echo "==> Avaliando A4-manual em ${SPLIT}_gold"
  "$PYTHON" evaluation/evaluate.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --split "$SPLIT" \
    --output-dir "outputs/ablation/g4_gold/evaluation/best_${SPLIT}"
done

cat <<EOF

==> Pronto. Predições em:
    outputs/ablation/g4_gold/evaluation/best_test/predictions.jsonl
    outputs/ablation/g4_gold/evaluation/best_val/predictions.jsonl

Compare com o A4-auto (g4_lora_loc) no MESMO test_gold para a tese
(IoU + artifact-hit-rate + texto).

Loop de pré-anotação (voltar sugestões do A4 para ajuste no Label Studio):
  1) Inferência no pool (na tarkin):
       cp configs/ablation/g4_gold.yaml configs/ablation/g4_gold_pool.yaml
       # edite test_metadata: .../gold/pool.jsonl  e output_dir do eval
       python evaluation/evaluate.py --config configs/ablation/g4_gold_pool.yaml \\
         --checkpoint $CKPT --split test \\
         --output-dir outputs/ablation/g4_gold/evaluation/pool
  2) Traga predictions.jsonl para a máquina do Label Studio e injete:
       python scripts/inject_model_predictions.py \\
         --predictions outputs/ablation/g4_gold/evaluation/pool/predictions.jsonl \\
         --sqlite ~/.local/share/label-studio/label_studio.sqlite3 \\
         --project-title DD-VQA-Loc --model-version a4_gold
EOF
