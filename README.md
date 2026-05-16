# FaceGroundVLM

**Explainable and Localizing Deepfake Detection through Grounded Vision-Language Modeling**

FaceGroundVLM is a multimodal deepfake detection model that combines TinyLLaVA-1.5B (SigLIP-So400m + TinyLlama-1.1B) with DINOv2-Large through an Interleave Mixture-of-Features (I-MoF) strategy. The model generates natural-language explanations of its predictions and grounds them to specific facial regions using textual bounding boxes `[y1,x1,y2,x2]` derived from MediaPipe facial landmarks.

A dedicated **DINOv2 classification head** predicts the Real/Fake verdict independently, decoupling detection accuracy from explanation quality. The LLM receives the verdict as a text prefix and focuses on generating coherent, grounded explanations.

## Architecture

```
                          ┌─── DINOv2-Large (frozen) ──> CLS token ──> MLP Head ──> "Fake"
                          │                                                          │
Image ──> DINOv2-Large ───┤                                                   (inference only)
          (frozen)        │                                                          │
                          └─── Patch tokens ──> Adapter (trainable) ──> [B, 729, 2048]
                                                                              │
Image ──> SigLIP-So400m/14 ──> MLP Connector ────────────────────> [B, 729, 2048]
          (frozen)              (trainable)                               │
                                                                    I-MoF interleave
                                                                          │
                                                              visual_embeds [B, 1458, 2048]
                                                                          │
"USER: Is this real? ASSISTANT: Fake." ──> embed_tokens ───────> text_embeds
                                                                          │
                                                              TinyLlama-1.1B (LoRA)
                                                                          │
                                                                      LM Head
                                                                          │
                                              "The person's eyes [416,344,469,655] look..."
```

## Training Stages

| Stage | Description | Trainable Parameters | Data |
|-------|-------------|---------------------|------|
| A | DINOv2 classification head | MLP head (~66K) | FaceForensics++ |
| 1 | DINOv2 adapter pre-training | DINOv2 adapter (~2M) | LCS-558K |
| 2 | Adapter + LoRA fine-tuning | Adapter + MLP connector + LLM LoRA | DD-VQA |
| 3 | Localization fine-tuning | Adapter + MLP connector + LLM LoRA | DD-VQA + `[y1,x1,y2,x2]` |

## Project Structure

```
FaceGroundVLM/
├── configs/
│   ├── ablation/                              # Ablation study configs (G1-G6)
│   │   ├── g1_baseline.yaml                   # SigLIP only, end-to-end
│   │   ├── g2_imof.yaml                       # + DINOv2 I-MoF
│   │   ├── g4_classifier.yaml                 # + DINOv2 classifier
│   │   ├── g5_classifier_aug.yaml             # + augmentation
│   │   └── g6_full_stage3.yaml                # + localization
│   ├── archived/                              # Legacy PaliGemma configs
│   ├── dino_classifier.yaml                   # DINOv2 classifier (Stage A)
│   ├── tinyllava_stage1_adapter.yaml          # DINOv2 adapter pre-training
│   ├── tinyllava_stage2_finetune.yaml         # Stage 2 (Run E/F)
│   ├── tinyllava_stage2_cls.yaml              # Stage 2 with classifier (Run G)
│   ├── tinyllava_stage3_loc.yaml              # Stage 3 localization
│   └── tinyllava_stage3_cls_loc.yaml          # Stage 3 with classifier (Run G)
├── data/
│   ├── ddvqa_dataset.py                       # DD-VQA dataset (Stage 2+)
│   ├── lcs558k_dataset.py                     # LCS-558K dataset (Stage 1)
│   ├── prompt_formats.py                      # Prompt templates
│   └── extractors/                            # MediaPipe landmark extraction
├── models/
│   ├── __init__.py                            # build_model() factory
│   ├── tinyllava_ground_vlm.py                # TinyLLaVA backbone
│   ├── dino_classifier.py                     # DINOv2 CLS -> MLP binary classifier
│   ├── dino_adapter.py                        # DINOv2 -> LLM projection
│   ├── mixture_of_features.py                 # I-MoF / C-MoF
│   ├── lora_moe.py                            # LoRA-MoE router
│   └── _loss_utils.py                         # Custom loss functions
├── training/
│   ├── factory.py                             # Processor + augmentation factory
│   ├── train_classifier.py                    # DINOv2 classifier training (Stage A)
│   ├── train_stage1.py                        # Adapter pre-training
│   ├── train_stage2.py                        # Stage 2/3 fine-tuning
│   ├── train_stage4.py                        # LoRA-MoE (experimental)
│   └── sample_utils.py                        # Visual sample saving
├── evaluation/
│   ├── evaluate.py                            # Full evaluation pipeline
│   └── metrics.py                             # BLEU, ROUGE, CIDEr, IoU
├── scripts/
│   ├── run_ablation.py                        # Ablation study orchestration
│   ├── prepare_ddvqa.py                       # DD-VQA data preparation
│   ├── prepare_ff_classification.py           # FF++ classification data preparation
│   ├── extract_landmarks.py                   # MediaPipe landmark extraction
│   ├── create_loc_annotations.py              # [y1,x1,y2,x2] annotation generation
│   └── extract_tinyllava_weights.py           # TinyLLaVA weight extraction
├── docs/
│   └── apresentacao_progresso.md              # Progress presentation
├── analyze_evaluation.ipynb                   # Results analysis notebook
└── requirements.txt
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### One-time setup: extract TinyLLaVA weights

```bash
python scripts/extract_tinyllava_weights.py \
    --model-id bczhou/TinyLLaVA-1.5B \
    --output outputs/tinyllava_weights.pt
```

### Training pipeline

```bash
# Stage A: DINOv2 binary classifier on FF++
python scripts/prepare_ff_classification.py \
    --ff-root /datasets/deepfake/faceforensics \
    --output-dir /datasets/deepfake/ff_classification \
    --compression c23

python training/train_classifier.py --config configs/dino_classifier.yaml

# Stage 1: DINOv2 adapter pre-training on LCS-558K
python training/train_stage1.py --config configs/tinyllava_stage1_adapter.yaml

# Stage 2: Adapter + LoRA fine-tuning on DD-VQA
python training/train_stage2.py --config configs/tinyllava_stage2_cls.yaml

# Stage 3: Localization fine-tuning
python training/train_stage2.py --config configs/tinyllava_stage3_cls_loc.yaml
```

### Evaluation

```bash
# End-to-end verdict (LLM decides Real/Fake)
python evaluation/evaluate.py \
    --config configs/tinyllava_stage2_finetune.yaml \
    --checkpoint outputs/.../checkpoint-final.pt \
    --split test

# Decoupled verdict (DINOv2 classifier decides Real/Fake)
python evaluation/evaluate.py \
    --config configs/tinyllava_stage2_cls.yaml \
    --checkpoint outputs/.../checkpoint-final.pt \
    --classifier-checkpoint outputs/dino_classifier/best.pt \
    --split test
```

## Ablation Study

The ablation study systematically evaluates each architectural component.

| Run | Classifier | I-MoF | Aug | Loc | Description |
|:---:|:----------:|:-----:|:---:|:---:|-------------|
| G1 | end-to-end | no | no | no | Baseline (SigLIP only) |
| G2 | end-to-end | yes | no | no | + DINOv2 visual features |
| G4 | DINOv2 head | yes | no | no | + Decoupled classification |
| G5 | DINOv2 head | yes | yes | no | + Data augmentation |
| G6 | DINOv2 head | yes | yes | yes | Full pipeline |

### Running the full grid

```bash
# Dry-run first (validates the entire pipeline with minimal data)
python scripts/run_ablation.py --dry-run

# Full ablation study
python scripts/run_ablation.py

# Specific experiments only
python scripts/run_ablation.py --experiments G4 G5 G6
```

The script automatically:
1. Trains the DINOv2 classifier (once, shared by G4/G5/G6)
2. Trains Stage 2 for each experiment
3. Trains Stage 3 for G6
4. Evaluates **best** and **last** checkpoints on **val** and **test**
5. Waits 2 minutes between experiments for GPU cool-down
6. Saves a summary JSON to `outputs/ablation/summary.json`

## Data Augmentation

Augmentation is controlled by the `augmentation: true` config flag:

- **LLM training** (Stages 2/3): Color-only transforms (ColorJitter, GaussianBlur, RandomGrayscale) — no geometric transforms to preserve bounding box validity.
- **Classifier training** (Stage A): Full augmentation including RandomResizedCrop, HorizontalFlip, and color transforms.

## Metrics

Following the TruthLens evaluation protocol:

- **Detection**: Accuracy, Precision, Recall, F1
- **Text generation**: BLEU-3, BLEU-4, ROUGE-L, CIDEr
- **Localization** (Stage 3+): Mean IoU, Hit Rate @ 0.3

## Hardware Requirements

Tested on NVIDIA RTX A6000 (48 GB VRAM). Estimated VRAM usage:

| Stage | VRAM |
|-------|------|
| A (classifier) | ~8 GB |
| 1 (adapter only) | ~18 GB |
| 2 (adapter + LoRA, no DINOv2) | ~18 GB |
| 2 (adapter + LoRA, with DINOv2) | ~25 GB |
| 3 (localization) | ~25 GB |

## References

- Kundu et al. (2025). *TruthLens: Explainable DeepFake Detection via Large Foundation Vision-Language Models*. AAAI.
- Zhou et al. (2024). *TinyLLaVA: A Framework of Small-scale Large Multimodal Models*.
- Oquab et al. (2024). *DINOv2: Learning Robust Visual Features without Supervision*. Meta AI.
- Shiohara et al. (2024). *DD-VQA: Deepfake Detection through Visual Question Answering*.
