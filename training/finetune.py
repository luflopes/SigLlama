"""Stage 2: Fine-tune SigLIP + Adapter + TinyLlama + MoE on deepfake data.

In this stage the full pipeline is trained (or partially via LoRA) on
deepfake detection datasets. The MoE layer is activated, and the model
learns to route inputs to domain-specific experts.
"""
from __future__ import annotations

import argparse

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2: Deepfake fine-tuning with MoE")
    p.add_argument("--config", required=True, help="Path to finetuning YAML config")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # TODO: implement fine-tuning loop
    #   1. Load pre-trained adapter checkpoint from Stage 1
    #   2. Initialise MoE layer
    #   3. Train on deepfake detection/explanation dataset
    #   4. Apply LoRA to TinyLlama if configured
    #   5. Monitor routing distribution across experts
    raise NotImplementedError("Fine-tuning loop not yet implemented.")


if __name__ == "__main__":
    main()
