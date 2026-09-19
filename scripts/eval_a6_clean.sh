#!/usr/bin/env bash
# Reprocessa a avaliação da A6 (Completo) de forma correta.
#
# Contexto: o antigo `outputs/ablation/g6_full/evaluation/best_test` reusava o
# A4 SUJO (g4_lora_loc, protocolo com caixas em faces reais) + veredito do
# classificador. Como aquele gerador é um MODELO DISTINTO do A4 limpo
# (g4_auto_clean), o texto/localização da A6 não coincidia com o da A4 — não se
# pode afirmar que as métricas textuais são iguais partindo de modelos diferentes.
#
# Correção: a A6 é, por definição, o MESMO gerador da A4 (veredito desacoplado
# vindo do classificador). Portanto reavaliamos o modelo da A4 LIMPA
# (g4_auto_clean) com --classifier-checkpoint, no protocolo limpo e com o mesmo
# checkpoint/metadata/max-new-tokens usados na A4. Assim texto e localização
# partem do mesmo modelo, diferindo apenas onde o classificador troca o veredito.
#
# Uso (na tarkin, a partir da raiz do repo):
#   bash scripts/eval_a6_clean.sh
#   GOLD=0 bash scripts/eval_a6_clean.sh   # pula o gold test (só clean test)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"

PY="${PYTHON:-python}"
MAXNEW="${MAXNEW:-320}"                 # mesmo default do run_clean_pipeline.sh
BATCH="${BATCH:-8}"
GOLD_DIR="${GOLD_DIR:-/datasets/deepfake/ddvqa_prepared/gold}"
DO_GOLD="${GOLD:-1}"                     # GOLD=0 desliga o gold test

# A6 = gerador da A4 limpa + veredito do classificador DINOv2.
CFG=configs/ablation/g4_auto_clean.yaml
CK=outputs/ablation/g4_auto_clean/stage3/checkpoint-best.pt   # idêntico ao usado na A4 cleantest
CLF=outputs/dino_lora_classifier/best.pt                      # mesmo classificador da A5/A6

for f in "$CFG" "$CK" "$CLF"; do
  [ -e "$f" ] || { echo "!! arquivo ausente: $f"; exit 1; }
done

echo "==> [A6-clean] clean test  (modelo A4 limpo + veredito do classificador)"
"$PY" evaluation/evaluate.py \
  --config "$CFG" --checkpoint "$CK" \
  --classifier-checkpoint "$CLF" \
  --split test --max-new-tokens "$MAXNEW" --batch-size "$BATCH" \
  --output-dir outputs/ablation/g6_full/evaluation/cleantest

if [ "$DO_GOLD" = "1" ] && [ -f "$GOLD_DIR/test_loc_gold.jsonl" ]; then
  echo "==> [A6-clean] gold test  (para a coincidência de localização A6 ~ A4 no gold)"
  "$PY" evaluation/evaluate.py \
    --config "$CFG" --checkpoint "$CK" \
    --classifier-checkpoint "$CLF" \
    --split test --max-new-tokens "$MAXNEW" --batch-size "$BATCH" \
    --test-metadata "$GOLD_DIR/test_loc_gold.jsonl" \
    --output-dir outputs/ablation/g6_full/evaluation/goldtest
else
  echo "==> gold test pulado (GOLD=0 ou metadata ausente)"
fi

cat <<EOF

==> Concluido. Traga de volta para a maquina local (mesmos caminhos relativos):
    outputs/ablation/g6_full/evaluation/cleantest/{results.json,predictions.jsonl,predictions.csv}
    outputs/ablation/g6_full/evaluation/goldtest/   (se gerado)

Depois, localmente, recompute as metricas texto-only e regenere o notebook:
    python scripts/recompute_text_metrics.py
    # e reexecute notebooks/analyze_evaluation.ipynb (G6 ja aponta para cleantest)
EOF
