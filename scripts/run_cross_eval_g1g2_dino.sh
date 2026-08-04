#!/bin/bash
# Generalização cross-dataset em Celeb-DF-v2 e WildDeepfake:
#   - Configs de ablation ainda NÃO avaliadas cross-dataset: G1 (baseline SigLIP)
#     e G2 (+DINOv2 congelado). G3/G4 já foram feitos em run_celebdf_eval.sh /
#     run_wilddeepfake_eval.sh e NÃO são repetidos aqui.
#   - Classificador DINOv2 binário (cabeça de classificação) nos dois datasets.
#
# Pré-requisitos: /datasets/deepfake/celebdf_prepared e wilddeepfake_prepared já
# preparados (rode antes run_celebdf_eval.sh / run_wilddeepfake_eval.sh, que também
# fazem a extração de frames, ou os scripts prepare_*.py).
#
# Uso (na raiz do projeto, no servidor):
#   bash scripts/run_cross_eval_g1g2_dino.sh
#   PYTHON=python BATCH_SIZE=8 bash scripts/run_cross_eval_g1g2_dino.sh
set -euo pipefail

PYTHON="${PYTHON:-/home/lucasfl/repos/SigLlama/venv/bin/python}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DINO_BATCH="${DINO_BATCH:-32}"

CELEB_META="/datasets/deepfake/celebdf_prepared/test.jsonl"
CELEB_FRAMES="/datasets/deepfake/celebdf_prepared/frames"
WILD_META="/datasets/deepfake/wilddeepfake_prepared/test_sampled.jsonl"
WILD_FRAMES="/datasets/deepfake/wilddeepfake_prepared/frames"
DINO_CKPT="outputs/dino_lora_classifier/best.pt"

# ------------------------------------------------------------------
# 1) VLM (ablation): G1 e G2 em Celeb-DF-v2 e WildDeepfake
#    evaluate.py exporta predictions.jsonl com pred_label + verdict_score.
#    aggregate_video_predictions.py gera métricas frame/video + por método.
# ------------------------------------------------------------------
run_vlm () {
  local ds="$1" cfgname="$2" ckpt="$3"
  echo ""
  echo "=== ${ds} : ${cfgname} (VLM: veredito + verdict_score) ==="
  $PYTHON evaluation/evaluate.py \
    --config "configs/cross_dataset/${ds}_${cfgname}.yaml" \
    --checkpoint "$ckpt" \
    --split test --export-scores --batch-size "$BATCH_SIZE"
  $PYTHON scripts/aggregate_video_predictions.py \
    --predictions "outputs/cross_dataset/${ds}_${cfgname}/evaluation/predictions.jsonl"
}

run_vlm celebdf g1_baseline "outputs/ablation/g1_baseline/stage2/checkpoint-final.pt"
run_vlm wild    g1_baseline "outputs/ablation/g1_baseline/stage2/checkpoint-final.pt"
run_vlm celebdf g2_imof     "outputs/ablation/g2_imof/stage2/checkpoint-final.pt"
run_vlm wild    g2_imof     "outputs/ablation/g2_imof/stage2/checkpoint-final.pt"

# ------------------------------------------------------------------
# 2) DINOv2 (cabeça binária) em Celeb-DF-v2 e WildDeepfake
#    export_dino_scores.py aceita o schema is_real; score = softmax P(fake).
# ------------------------------------------------------------------
echo ""
echo "=== Celeb-DF-v2 : DINOv2 (cabeça binária) ==="
$PYTHON scripts/export_dino_scores.py \
  --checkpoint "$DINO_CKPT" \
  --metadata "$CELEB_META" --images-dir "$CELEB_FRAMES" \
  --format ff \
  --output outputs/dino_lora_classifier/scores_celebdf_test.jsonl \
  --batch-size "$DINO_BATCH"

echo ""
echo "=== WildDeepfake : DINOv2 (cabeça binária) ==="
$PYTHON scripts/export_dino_scores.py \
  --checkpoint "$DINO_CKPT" \
  --metadata "$WILD_META" --images-dir "$WILD_FRAMES" \
  --format ff \
  --output outputs/dino_lora_classifier/scores_wild_test.jsonl \
  --batch-size "$DINO_BATCH"

echo ""
echo "=== Concluído. Resultados em: ==="
echo "  VLM  : outputs/cross_dataset/{celebdf,wild}_{g1_baseline,g2_imof}/evaluation/"
echo "  DINO : outputs/dino_lora_classifier/scores_{celebdf,wild}_test.jsonl"
