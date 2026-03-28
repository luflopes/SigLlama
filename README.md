# SigLlama

**Explainable Deepfake Detection via Multimodal Vision-Language Model with Mixture of Experts**

> Master's thesis research project — Programa de Pós-Graduação em Informática, Universidade Federal do Paraná (UFPR).

---

## Abstract

SigLlama is a multimodal architecture for deepfake detection that goes beyond binary classification by generating natural-language explanations for its predictions. The model integrates three complementary input streams — visual patch embeddings (SigLIP), structured soft tokens (object detections and facial landmarks), and text prompts — into a unified autoregressive decoder (TinyLlama 1.1B). A domain-specialised Mixture of Experts (MoE) layer routes hidden representations through experts trained on distinct deepfake generation families (GAN, VAE, Diffusion, Transformer-based), enabling fine-grained forensic analysis.

The design draws inspiration from PaLiGemma 2, adapting its soft-token paradigm to the deepfake detection domain with the addition of facial landmark segmentation and domain-aware expert routing.

## Architecture

```
Image ──┬── SigLIP ──── Adapter (MLP) ──── Visual Embeddings ─────┐
        │                                                         │
        ├── YOLOv8 ─────────┐                                     │
        │                   ├── Soft Token Embedder ──────────────┤
        └── MediaPipe FM ───┘                                     │
                                                                  ▼
Text ───── Llama Tokenizer ──── Text Embeddings ─────────► [Concatenation]
                                                                  │
                                                              TinyLlama
                                                              (Decoder)
                                                                  │
                                                                Router
                                                            ┌───┬───┬───┐
                                                            GAN VAE DIF TF
                                                            └───┴─┬─┴───┘
                                                                  ▼
                                                            Output (text)
```

**Key components:**

| Component | Description |
|---|---|
| **SigLIP Encoder** | Frozen vision transformer that produces patch-level visual features. |
| **Visual Adapter** | Two-layer MLP (Linear-GELU-Linear) that projects SigLIP features into the TinyLlama embedding space. Trained during Stage 1. |
| **Soft Token Embedder** | Projects structured signals (bounding boxes, class IDs, 468 facial landmarks) into LLM-space embeddings, providing explicit geometric priors to the decoder. |
| **TinyLlama 1.1B** | Autoregressive language decoder that receives the concatenated multimodal sequence and generates textual explanations. |
| **MoE Layer** | Domain-specialised Mixture of Experts with top-k routing. Each expert (MLP or LoRA) specialises in a deepfake generation method (GAN, VAE, Diffusion, Transformers). |

## Training Pipeline

Training proceeds in two stages:

### Stage 1 — Adapter Pre-training (LCS-558K)

Objective: align the SigLIP visual representation with TinyLlama's embedding space.

- **Data**: LCS-558K (BLIP-captioned image-text pairs)
- **Trainable**: Adapter only (SigLIP and TinyLlama frozen)
- **Task**: Image captioning (next-token prediction on captions)

### Stage 2 — Deepfake Fine-tuning with MoE

Objective: train the full pipeline for explainable deepfake detection.

- **Data**: Deepfake detection datasets with textual explanations
- **Trainable**: Adapter + TinyLlama (via LoRA) + MoE layer
- **Task**: Given a face image and prompt, classify real/fake and generate a forensic explanation
- **Loss**: Language modelling loss + MoE auxiliary load-balancing loss

## Project Structure

```
SigLlama/
├── configs/                        YAML configurations per training stage
│   ├── base.yaml                   Base hyperparameters
│   ├── pretraining.yaml            Stage 1 configuration
│   ├── finetuning.yaml             Stage 2 configuration
│   └── moe.yaml                    MoE-specific parameters
├── data/
│   ├── extractors/                 Soft-token extraction pipeline
│   │   ├── yolo_detector.py        YOLOv8 object detection
│   │   ├── landmark_extractor.py   MediaPipe Face Mesh (468 landmarks)
│   │   └── soft_token_extractor.py Orchestrator
│   ├── writers/
│   │   └── ndjson_writer.py        NDJSON writer with resume support
│   ├── visualization/
│   │   └── plot_utils.py           Detection and landmark visualisation
│   ├── dataset.py                  PyTorch Dataset
│   └── collator.py                 Batch collation for mixed modalities
├── models/
│   ├── siglip_encoder.py           SigLIP vision encoder wrapper
│   ├── adapter.py                  Visual-to-LLM MLP adapter
│   ├── soft_token_embedder.py      Detection/landmark embedding
│   ├── moe/
│   │   ├── router.py               Top-k gating with load balancing
│   │   ├── experts.py              MLP and LoRA expert modules
│   │   └── moe_layer.py            MoE composition layer
│   └── sigllama.py                 Full model assembly
├── training/
│   ├── pretrain.py                 Stage 1 training script
│   ├── finetune.py                 Stage 2 training script
│   └── trainer.py                  Custom trainer with MoE loss
├── evaluation/
│   ├── metrics.py                  BLEU, ROUGE, accuracy, expert utilisation
│   └── evaluate.py                 Evaluation script
├── scripts/
│   ├── extract_soft_tokens.py      CLI for soft-token extraction
│   └── run_training.sh             Training launcher
├── sample/                         Sample data for local testing
├── requirements.txt
└── .gitignore
```

## Installation

```bash
git clone <repository-url> SigLlama
cd SigLlama
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Requirements**: Python 3.10+, CUDA-capable GPU (tested on NVIDIA A100/V100).

## Usage

### Soft-Token Extraction

Extract object detections and facial landmarks from a dataset in the LCS-558K metadata format:

```bash
# Full LCS-558K extraction (GPU server)
python scripts/extract_soft_tokens.py \
    --metadata /data/lcs-558k/blip_laion_cc_sbu_558k_meta.json \
    --image-root /data/lcs-558k \
    --output /data/lcs-558k/lcs_softtokens.ndjson \
    --yolo-model yolov8l.pt

# Local test with sample data
python scripts/extract_soft_tokens.py \
    --metadata sample/metadata.json \
    --image-root sample \
    --output sample_softtokens.ndjson \
    --visualize --vis-dir annotated
```

The script supports automatic **resume**: if interrupted, it skips already-processed images on re-run.

### Training

```bash
# Stage 1: Adapter pre-training
bash scripts/run_training.sh pretrain

# Stage 2: Deepfake fine-tuning with MoE
bash scripts/run_training.sh finetune
```

## Soft-Token Format

Each record in the output NDJSON file has the following schema:

```json
{
  "image_id": "000600956",
  "image_path": "00060/000600956.jpg",
  "caption": "a patent drawing of a battery system",
  "detection_tokens": [
    {
      "bbox": [0.51, 0.52, 0.89, 0.94],
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.97
    }
  ],
  "landmark_tokens": [
    {
      "landmarks_normalized": [[0.48, 0.58], ...],
      "landmarks_absolute": [[123.3, 114.7], ...],
      "confidence": 1.0
    }
  ]
}
```

- **`detection_tokens`**: YOLOv8 detections with normalised centre-format bounding boxes `[cx, cy, w, h]`.
- **`landmark_tokens`**: MediaPipe Face Mesh with 468 landmarks (normalised and absolute coordinates), extracted only when a face is detected.

## Dependencies

| Package | Purpose |
|---|---|
| `torch`, `torchvision` | Deep learning framework |
| `transformers` | TinyLlama and SigLIP model loading |
| `peft` | LoRA adapters for parameter-efficient fine-tuning |
| `accelerate` | Distributed training utilities |
| `ultralytics` | YOLOv8 object detection |
| `mediapipe` | Face Mesh landmark extraction |
| `Pillow`, `numpy`, `tqdm`, `pyyaml` | Data processing and utilities |

## Roadmap

- [x] Project structure and soft-token extraction pipeline
- [ ] Stage 1 — Adapter pre-training on LCS-558K
- [ ] Object detection and facial landmark integration as soft tokens
- [ ] TinyLlama adaptation for multimodal soft-token input
- [ ] Stage 2 — Full fine-tuning (SigLIP + Adapter + TinyLlama)
- [ ] MoE Router and domain-specific experts
- [ ] Ablation study
- [ ] Reliability evaluation and manuscript

## References

- Beyer, L. et al. *PaLiGemma 2: A Family of Versatile VLMs for Transfer*. arXiv:2412.03555, 2024.
- Zhai, X. et al. *Sigmoid Loss for Language Image Pre-Training (SigLIP)*. ICCV 2023.
- Zhang, P. et al. *TinyLlama: An Open-Source Small Language Model*. arXiv:2401.02385, 2024.
- Shazeer, N. et al. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. ICLR 2017.
- Hu, E. J. et al. *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.

## License

This project is part of ongoing academic research. License to be defined upon publication.

---

*Universidade Federal do Paraná — Programa de Pós-Graduação em Informática*
