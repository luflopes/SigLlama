"""Evaluation script for SigLlama.

Loads a trained checkpoint and evaluates on a test dataset, reporting
generation quality (BLEU, ROUGE) and detection accuracy.
"""
from __future__ import annotations

import argparse

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained SigLlama model")
    p.add_argument("--config", required=True, help="Path to evaluation YAML config")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    p.add_argument("--split", default="test", help="Dataset split to evaluate")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # TODO: implement evaluation
    #   1. Load model from checkpoint
    #   2. Load test dataset
    #   3. Generate predictions
    #   4. Compute metrics (BLEU, ROUGE, accuracy)
    #   5. Print / save results
    raise NotImplementedError("Evaluation not yet implemented.")


if __name__ == "__main__":
    main()
