# FaceGroundVLM

**Explainable and Localizing Deepfake Detection through Grounded Vision-Language Modeling**

FaceGroundVLM is a multimodal deepfake detection model that combines PaliGemma2-3B with DINOv2-Large through an Interleave Mixture-of-Features (I-MoF) strategy. The model generates natural-language explanations of its predictions and grounds them to specific facial regions using PaliGemma2's native `<loc>` tokens derived from MediaPipe facial landmarks. Optionally, a LoRA Mixture-of-Experts (LoRA-MoE) module specialises detection per manipulation technique.

## Architecture

```
Image ──> SigLIP-So400m/14 ──> PaliGemma Projector (frozen) ──> [B, 1024, 2304]
                                                                        │
Image ──> DINOv2-Large (frozen) ──> DINOv2 Adapter (trainable) ──> [B, 1024, 2304]
                                                                        │
                                                                I-MoF interleave
                                                                        │
                                                            visual_embeds [B, 2048, 2304]
                                                                        │
Question ──> tokenizer ──> embed_tokens ─────────────────────> text_embeds
                                                                        │
                                                          Gemma2-2B (LoRA / LoRA-MoE)
                                                                        │
                                                                    LM Head
                                                                        │
                                                    "This is fake. The mouth <loc0340>..."
```

## Training Stages

| Stage | Description | Trainable Parameters | Data |
|-------|-------------|---------------------|------|
| 1 | DINOv2 Adapter pre-training | DINOv2 Adapter (~2.4M) | LCS-558K |
| 2 | Adapter + LoRA fine-tuning | Adapter + Gemma2 LoRA | DD-VQA |
| 3 | Localization token fine-tuning | Adapter + Gemma2 LoRA | DD-VQA + `<loc>` |
| 4 | LoRA-MoE specialisation | Router + Expert LoRAs | DD-VQA + `<loc>` |

## Project Structure

```
FaceGroundVLM/
├── configs/
│   ├── stage1_adapter.yaml      # Stage 1 configuration
│   ├── stage2_finetune.yaml     # Stage 2 configuration
│   ├── stage3_loc.yaml          # Stage 3 (loc tokens) configuration
│   └── stage4_moe.yaml          # Stage 4 (LoRA-MoE) configuration
├── data/
│   ├── lcs558k_dataset.py       # LCS-558K dataset (Stage 1)
│   ├── ddvqa_dataset.py         # DD-VQA dataset (Stage 2+)
│   └── extractors/
│       └── landmark_extractor.py
├── models/
│   ├── face_ground_vlm.py       # Main model
│   ├── dino_adapter.py          # DINOv2 → Gemma2 projection
│   ├── mixture_of_features.py   # I-MoF / C-MoF
│   └── lora_moe.py              # LoRA-MoE router
├── training/
│   ├── train_stage1.py          # Adapter pre-training
│   ├── train_stage2.py          # Adapter + LoRA fine-tuning
│   └── train_stage4.py          # LoRA-MoE training
├── evaluation/
│   ├── evaluate.py              # Full evaluation pipeline
│   └── metrics.py               # BLEU, ROUGE, CIDEr, accuracy, IoU
├── scripts/
│   ├── prepare_ddvqa.py         # DD-VQA data preparation
│   ├── extract_landmarks.py     # MediaPipe landmark extraction
│   └── create_loc_annotations.py # Generate <loc>-enriched annotations
└── requirements.txt
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

> **Note**: PaliGemma2 requires accepting Google's license on HuggingFace. Run `huggingface-cli login` and accept the terms at https://huggingface.co/google/paligemma2-3b-pt-448.

### Stage 1: DINOv2 Adapter Pre-training

```bash
python training/train_stage1.py --config configs/stage1_adapter.yaml
```

### Stage 2: Adapter + LoRA Fine-tuning

```bash
python training/train_stage2.py --config configs/stage2_finetune.yaml
```

### Stage 3: Localization Tokens

```bash
# Extract landmarks from DD-VQA frames
python scripts/extract_landmarks.py \
    --frames-dir /datasets/deepfake/ddvqa_prepared/frames \
    --output /datasets/deepfake/ddvqa_prepared/landmarks.jsonl

# Generate <loc>-enriched annotations
python scripts/create_loc_annotations.py \
    --ddvqa-jsonl /datasets/deepfake/ddvqa_prepared/train.jsonl \
    --landmarks-jsonl /datasets/deepfake/ddvqa_prepared/landmarks.jsonl \
    --output /datasets/deepfake/ddvqa_prepared/train_loc.jsonl

# Fine-tune with localization tokens
python training/train_stage2.py --config configs/stage3_loc.yaml
```

### Stage 4: LoRA-MoE

```bash
python training/train_stage4.py --config configs/stage4_moe.yaml
```

### Evaluation

```bash
python evaluation/evaluate.py \
    --config configs/stage2_finetune.yaml \
    --checkpoint outputs/stage2_finetune/checkpoint-final.pt
```

## Metrics

Following the TruthLens evaluation protocol:

- **Detection**: Accuracy, Precision, Recall, F1
- **Text generation**: BLEU-3, BLEU-4, ROUGE-L, CIDEr
- **Localization** (Stage 3+): Mean IoU, Hit Rate @ 0.3

## Hardware Requirements

Tested on NVIDIA RTX A6000 (48 GB VRAM). Estimated VRAM usage:

- Stage 1 (adapter only): ~18 GB
- Stage 2 (adapter + LoRA): ~22-25 GB
- Stage 4 (LoRA-MoE): ~28-30 GB

## References

- Kundu et al. (2025). *TruthLens: Explainable DeepFake Detection via Large Foundation Vision-Language Models*. AAAI.
- Beyer et al. (2024). *PaliGemma 2: A Family of Versatile VLMs for Transfer*. Google DeepMind.
- Oquab et al. (2024). *DINOv2: Learning Robust Visual Features without Supervision*. Meta AI.
