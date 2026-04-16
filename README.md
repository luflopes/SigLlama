# FaceGroundVLM

**Explainable and Localizing Deepfake Detection through Grounded Vision-Language Modeling**

FaceGroundVLM is a multimodal deepfake detection model that combines a vision-language backbone (PaliGemma2-3B **or** TinyLLaVA-1.5B) with DINOv2-Large through an Interleave Mixture-of-Features (I-MoF) strategy. The model generates natural-language explanations of its predictions and grounds them to specific facial regions using textual bounding boxes `[y1,x1,y2,x2]` derived from MediaPipe facial landmarks. Optionally, a LoRA Mixture-of-Experts (LoRA-MoE) module specialises detection per manipulation technique.

## Supported backbones

| Backbone | Vision | LLM | Total | Img size | Visual tokens |
|----------|--------|-----|-------|----------|---------------|
| `paligemma` | SigLIP-So400m/14 | Gemma2-2B | 3.0B | 448 | 1024 (+ 1024 DINOv2 via I-MoF) |
| `tinyllava` | SigLIP-So400m/14 (fine-tuned) | TinyLlama-1.1B (fine-tuned) | 1.5B | 384 | 729 (+ 729 DINOv2 via I-MoF) |

Switch backbones by setting `backbone: "paligemma"` or `backbone: "tinyllava"` in any YAML config. Both expose an identical training/eval API.

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
| 1 | DINOv2 adapter pre-training | DINOv2 adapter (~2.4M) | LCS-558K |
| 2 | Adapter + LoRA fine-tuning | Adapter + LLM LoRA | DD-VQA |
| 3 | Localization fine-tuning | Adapter + LLM LoRA | DD-VQA + `[y1,x1,y2,x2]` |
| 4 | LoRA-MoE specialisation | Router + Expert LoRAs | DD-VQA + `[y1,x1,y2,x2]` |

## Project Structure

```
FaceGroundVLM/
├── configs/
│   ├── stage{1,2,3,4}_*.yaml               # PaliGemma2 configs
│   └── tinyllava_stage{1,2,3,4}_*.yaml     # TinyLLaVA configs
├── data/
│   ├── lcs558k_dataset.py                  # Stage 1 dataset (backbone-aware)
│   ├── ddvqa_dataset.py                    # Stage 2+ dataset (backbone-aware)
│   └── prompt_formats.py                   # Per-backbone prompt templates
├── models/
│   ├── __init__.py                         # build_model() factory
│   ├── face_ground_vlm.py                  # PaliGemma2 backbone
│   ├── tinyllava_ground_vlm.py             # TinyLLaVA backbone
│   ├── dino_adapter.py                     # DINOv2 → LLM projection
│   ├── mixture_of_features.py              # I-MoF / C-MoF
│   └── lora_moe.py                         # LoRA-MoE router
├── training/
│   ├── factory.py                          # build_processor_and_transforms()
│   ├── train_stage1.py                     # Adapter pre-training
│   ├── train_stage2.py                     # Adapter + LoRA fine-tuning
│   └── train_stage4.py                     # LoRA-MoE training
├── evaluation/
│   ├── evaluate.py                         # Full evaluation pipeline
│   └── metrics.py                          # BLEU, ROUGE, CIDEr, IoU, ...
├── scripts/
│   ├── prepare_ddvqa.py                    # DD-VQA data preparation
│   ├── extract_landmarks.py                # MediaPipe landmark extraction
│   ├── create_loc_annotations.py           # Generate [y1,x1,y2,x2] annotations
│   └── extract_tinyllava_weights.py        # Extract fine-tuned TinyLLaVA weights
└── requirements.txt
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

> **Note**: PaliGemma2 requires accepting Google's license on HuggingFace. Run `huggingface-cli login` and accept the terms at https://huggingface.co/google/paligemma2-3b-pt-448. TinyLLaVA (`bczhou/TinyLLaVA-1.5B`) is ungated.

### Running with PaliGemma2 (default)

```bash
# Stage 1: DINOv2 adapter pre-training on LCS-558K
python training/train_stage1.py --config configs/stage1_adapter.yaml

# Stage 2: adapter + LoRA fine-tuning on DD-VQA
python training/train_stage2.py --config configs/stage2_finetune.yaml

# Stage 3: localization (textual [y1,x1,y2,x2] bounding boxes)
python scripts/extract_landmarks.py \
    --frames-dir /datasets/deepfake/ddvqa_prepared/frames \
    --output /datasets/deepfake/ddvqa_prepared/landmarks.jsonl
python scripts/create_loc_annotations.py \
    --ddvqa-jsonl /datasets/deepfake/ddvqa_prepared/train.jsonl \
    --landmarks-jsonl /datasets/deepfake/ddvqa_prepared/landmarks.jsonl \
    --output /datasets/deepfake/ddvqa_prepared/train_loc.jsonl
python training/train_stage2.py --config configs/stage3_loc.yaml

# Stage 4: LoRA-MoE specialisation
python training/train_stage4.py --config configs/stage4_moe.yaml
```

### Running with TinyLLaVA-1.5B

TinyLLaVA ships three fine-tuned components — SigLIP, the MLP connector, and TinyLlama — jointly trained via the paper's *share recipe*. We reuse all of them; the only new trainable component is the optional DINOv2 adapter.

**Step 1 — extract the fine-tuned weights (one-time setup)**

```bash
python scripts/extract_tinyllava_weights.py \
    --model-id bczhou/TinyLLaVA-1.5B \
    --output outputs/tinyllava_weights.pt
```

This produces `outputs/tinyllava_weights.pt` with three sub-state_dicts (`siglip`, `connector`, `llm`). Every TinyLLaVA config references this file via `tinyllava_weights`.

**Step 2 — run the usual training pipeline**

```bash
# Stage 1 can be SKIPPED when use_dino=false. When use_dino=true we still
# only train the lightweight DINOv2 adapter (connector stays frozen).
python training/train_stage1.py --config configs/tinyllava_stage1_adapter.yaml

python training/train_stage2.py --config configs/tinyllava_stage2_finetune.yaml
python training/train_stage2.py --config configs/tinyllava_stage3_loc.yaml
python training/train_stage4.py --config configs/tinyllava_stage4_moe.yaml
```

To ablate the DINOv2 branch, set `use_dino: false` in the TinyLLaVA configs — the MLP connector alone bridges SigLIP and TinyLlama (already aligned by the share recipe), so you can start directly at Stage 2.

### Evaluation

```bash
# Works identically for both backbones; the config picks the right one.
python evaluation/evaluate.py \
    --config configs/stage2_finetune.yaml \
    --checkpoint outputs/stage2_finetune/checkpoint-final.pt

python evaluation/evaluate.py \
    --config configs/tinyllava_stage2_finetune.yaml \
    --checkpoint outputs/tinyllava_stage2_finetune/checkpoint-final.pt
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
