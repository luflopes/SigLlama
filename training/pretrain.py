"""Stage 1: Pre-train the Visual Adapter on LCS-558K.

In this stage only the Adapter (MLP projection from SigLIP space to
TinyLlama space) is trained. Both SigLIP and TinyLlama are frozen.

The objective is image-caption alignment: given SigLIP features of an image,
the LLM should generate the corresponding BLIP caption.
"""
from __future__ import annotations

import argparse

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1: Adapter pre-training on LCS-558K")
    p.add_argument("--config", required=True, help="Path to pretraining YAML config")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # TODO: implement pre-training loop
    #   1. Load SigLlamaConfig with freeze_siglip=True, freeze_llm=True
    #   2. Build SigLlamaDataset from LCS-558K NDJSON
    #   3. Train only adapter parameters
    #   4. Validate with caption generation metrics
    raise NotImplementedError("Pre-training loop not yet implemented.")


if __name__ == "__main__":
    main()
