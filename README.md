# FaceGroundVLM

**Explainable and Localizing Deepfake Detection through Grounded Vision-Language Modeling**

FaceGroundVLM is a multimodal deepfake detection model that combines TinyLLaVA-1.5B (SigLIP-So400m + TinyLlama-1.1B) with DINOv2-Large through an Interleave Mixture-of-Features (I-MoF) strategy. The model generates natural-language explanations of its predictions and grounds them to specific facial regions using textual bounding boxes `[y1,x1,y2,x2]` derived from MediaPipe facial landmarks.

DINOv2 is fine-tuned with **LoRA** (Low-Rank Adaptation) to produce deepfake-aware visual features. A dual classification head on the CLS token (binary Real/Fake + forgery-type) provides the verdict independently, decoupling detection accuracy from explanation quality.

## Architecture

```
                          ┌─── CLS token ──> Binary Head ──> "Fake"
                          │                  Forgery Head ──> "FaceSwap"
Image ──> DINOv2-Large ───┤
          (LoRA)          │
                          └─── Patch tokens ──> Adapter ──> [B, 729, 2048]
                                                                  │
Image ──> SigLIP-So400m/14 ──> MLP Connector ──────────> [B, 729, 2048]
          (frozen)              (trainable)                       │
                                                            I-MoF interleave
                                                                  │
                                                      visual_embeds [B, 1458, 2048]
                                                                  │
"USER: Is this real? ASSISTANT: Fake." ──> embed_tokens ──> text_embeds
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
| A | DINOv2 LoRA + dual classifier | LoRA adapters + heads (~1.6M) | FaceForensics++ |
| 1 | DINOv2 adapter pre-training | DINOv2 adapter (~2M) | LCS-558K |
| 2 | Adapter + LLM LoRA fine-tuning | Adapter + MLP connector + LLM LoRA | DD-VQA |
| 3 | Localization fine-tuning | Adapter + MLP connector + LLM LoRA | DD-VQA + `[y1,x1,y2,x2]` |


Stage A trains DINOv2 with LoRA for deepfake-aware features and classification. Stage 1 pre-trains the adapter to project DINOv2 features into the LLM space. In Stage 2, the adapter adapts to LoRA-enhanced features naturally during VLM fine-tuning. Stage 3 adds localization.

## Project Structure

```
FaceGroundVLM/
├── configs/
│   ├── ablation/                              # Ablation study configs (G1-G6)
│   │   ├── g1_baseline.yaml                   # G1: SigLIP only, end-to-end
│   │   ├── g2_imof.yaml                       # G2: + DINOv2 frozen (I-MoF)
│   │   ├── g3_lora.yaml                       # G3: + DINOv2 LoRA (end-to-end)
│   │   └── g4_lora_loc.yaml                   # G4: + Localization (from G3)
│   │                                          # G5/G6: eval-only (reuse G3/G4 with classifier)
│   ├── archived/                              # Legacy / archived configs
│   ├── dino_classifier.yaml                   # DINOv2 frozen classifier (legacy)
│   ├── dino_lora_classifier.yaml              # DINOv2 LoRA + dual heads
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
│   ├── tinyllava_ground_vlm.py                # TinyLLaVA backbone (supports dino_lora_checkpoint)
│   ├── dino_classifier.py                     # DINOv2 frozen CLS -> MLP classifier (legacy)
│   ├── dino_lora_classifier.py                # DINOv2 LoRA + dual heads
│   ├── dino_adapter.py                        # DINOv2 -> LLM projection
│   ├── mixture_of_features.py                 # I-MoF / C-MoF
│   ├── lora_moe.py                            # LoRA-MoE router (Stage 4 legacy)
│   └── _loss_utils.py                         # Custom loss functions
├── training/
│   ├── factory.py                             # Processor + augmentation factory
│   ├── train_classifier.py                    # DINOv2 frozen classifier (legacy Stage A)
│   ├── train_dino_lora.py                     # DINOv2 LoRA + dual supervision (Stage A)
│   ├── train_stage1.py                        # Adapter pre-training (supports --init-adapter)
│   ├── train_stage2.py                        # Stage 2/3 fine-tuning
│   ├── train_stage4.py                        # LoRA-MoE on LLM (experimental, legacy)
│   └── sample_utils.py                        # Visual sample saving
├── evaluation/
│   ├── evaluate.py                            # Full evaluation pipeline (auto-detects classifier type)
│   └── metrics.py                             # BLEU, ROUGE, CIDEr, IoU
├── scripts/
│   ├── run_ablation.py                        # Ablation study orchestration (G1-G6)
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
# 0. Prepare data
python scripts/prepare_ff_classification.py \
    --ff-root /datasets/deepfake/faceforensics \
    --output-dir /datasets/deepfake/ff_classification \
    --compression c23

# Stage A: DINOv2 LoRA + dual classifier on FF++
python training/train_dino_lora.py --config configs/dino_lora_classifier.yaml

# Stage 1: DINOv2 adapter pre-training on LCS-558K (frozen DINOv2)
python training/train_stage1.py --config configs/tinyllava_stage1_adapter.yaml

# Stage 2: Adapter + LLM LoRA fine-tuning on DD-VQA
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

# Decoupled verdict (DINOv2 LoRA classifier decides Real/Fake)
python evaluation/evaluate.py \
    --config configs/tinyllava_stage2_cls.yaml \
    --checkpoint outputs/.../checkpoint-final.pt \
    --classifier-checkpoint outputs/dino_lora_classifier/best.pt \
    --split test
```

The evaluation script auto-detects whether the classifier checkpoint is from the legacy frozen-DINOv2 model or the new LoRA model.

## Ablation Study

The ablation study adds exactly **one component per step**, enabling isolated measurement of each contribution.


| Run | DINOv2 | LoRA | Verdict | Loc | Training | What it measures |
|-----|--------|------|---------|-----|----------|------------------|
| G1 | -- | -- | LLM | no | Stage 2 | Baseline (SigLIP only) |
| G2 | frozen | -- | LLM | no | Stage 2 | + generic visual features |
| G3 | LoRA | single | LLM | no | Stage 2 | + deepfake-aware features |
| G4 | LoRA | single | LLM | yes | Stage 3 (from G3) | + spatial localization |
| G5 | LoRA | single | classifier | no | eval-only (= G3) | + decoupled verdict |
| G6 | LoRA | single | classifier | yes | eval-only (= G4) | + localization + classifier |


Each step isolates one variable:

- **G1 -> G2**: value of additional visual features (generic DINOv2)
- **G2 -> G3**: value of fine-tuning the visual backbone for deepfakes
- **G3 -> G4**: value of spatial localization with end-to-end verdict
- **G3 -> G5**: value of decoupling the verdict from the LLM
- **G5 -> G6**: value of spatial localization with classifier verdict
- **G4 vs G6**: end-to-end vs classifier verdict (both with localization)

### Running the full grid

```bash
# Dry-run first (validates the entire pipeline with minimal data)
python scripts/run_ablation.py --dry-run

# Full ablation study
python scripts/run_ablation.py

# Specific experiments only
python scripts/run_ablation.py --experiments G3 G4 G5

# Skip Stage A (reuse existing LoRA checkpoint)
python scripts/run_ablation.py --skip-stage-a
```

The script automatically:

1. Trains DINOv2 LoRA classifier (Stage A, shared across G3-G6)
2. Trains G1/G2/G3 (Stage 2) and G4 (Stage 3, from G3 checkpoint)
3. Evaluates G5/G6 by reusing G3/G4 models with classifier verdict (no extra training)
4. Evaluates **best** and **last** checkpoints on **val** and **test**
5. Waits 2 minutes between experiments for GPU cool-down
6. Saves a summary JSON to `outputs/ablation/summary.json`

Each experiment writes to its own output directory — nothing is overwritten.

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
| A (DINOv2 LoRA classifier) | ~10 GB |
| 1 (adapter only) | ~18 GB |
| 2 (adapter + LLM LoRA, no DINOv2) | ~18 GB |
| 2 (adapter + LLM LoRA, with DINOv2+LoRA) | ~28 GB |
| 3 (localization) | ~28 GB |


## References

- Kundu et al. (2025). *TruthLens: Explainable DeepFake Detection via Large Foundation Vision-Language Models*. AAAI.
- Zhou et al. (2024). *TinyLLaVA: A Framework of Small-scale Large Multimodal Models*.
- Oquab et al. (2024). *DINOv2: Learning Robust Visual Features without Supervision*. Meta AI.
- Shiohara et al. (2024). *DD-VQA: Deepfake Detection through Visual Question Answering*.

