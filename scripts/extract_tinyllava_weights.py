"""Extract the three fine-tuned sub state-dicts from ``bczhou/TinyLLaVA-1.5B``.

The TinyLLaVA-1.5B checkpoint (TinyLLaVA-share-Sig-TL) bundles:

- A fine-tuned SigLIP-So400m/14 vision tower (layers >=12 updated by SFT).
- A 2-layer MLP (``mlp2x_gelu``) connector aligning SigLIP -> TinyLlama.
- A fine-tuned TinyLlama-1.1B (entire LLM updated by SFT).

All three components carry multimodal alignment that should NOT be discarded
when re-using the backbone. This script produces a single ``.pt`` file that
``TinyLLaVAGroundVLM`` loads into base ``transformers`` classes.

Key prefix mapping (TinyLLaVA checkpoint -> base ``transformers`` modules):

- ``model.vision_tower.vision_tower.vision_model.*``
    -> ``vision_model.*`` on ``SiglipVisionModel``
- ``model.mm_projector.<i>.{weight,bias}``
    -> ``<i>.{weight,bias}`` on ``nn.Sequential(Linear, GELU, Linear)``
- ``model.<...>`` (NOT under ``vision_tower`` / ``mm_projector``) and
  ``lm_head.weight``  -> unchanged (matches ``LlamaForCausalLM`` directly)

Usage::

    python scripts/extract_tinyllava_weights.py \
        --model-id bczhou/TinyLLaVA-1.5B \
        --output outputs/tinyllava_weights.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO,
)
logger = logging.getLogger(__name__)


VISION_PREFIX = "model.vision_tower.vision_tower."
CONNECTOR_PREFIX = "model.mm_projector."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-id", default="bczhou/TinyLLaVA-1.5B",
        help="HuggingFace model id or local path to the TinyLLaVA checkpoint.",
    )
    p.add_argument(
        "--output", default="outputs/tinyllava_weights.pt",
        help="Destination .pt file with the three sub state_dicts.",
    )
    p.add_argument(
        "--cache-dir", default=None,
        help="Optional HF cache directory (else default HF cache).",
    )
    return p.parse_args()


def _download_shards(model_id: str, cache_dir: str | None) -> list[Path]:
    """Return local paths to all safetensors shards of the given repo.

    Supports both local directories and HF hub repos. If ``model_id`` is a
    directory, we list its contents; otherwise we call ``snapshot_download``
    to fetch the relevant files only.
    """
    local = Path(model_id)
    if local.is_dir():
        shards = sorted(local.glob("model-*.safetensors"))
        if not shards:
            shards = sorted(local.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No safetensors found in {model_id}")
        return shards

    from huggingface_hub import snapshot_download
    repo_dir = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        allow_patterns=[
            "*.safetensors",
            "model.safetensors.index.json",
            "config.json",
        ],
    )
    repo_path = Path(repo_dir)
    shards = sorted(repo_path.glob("model-*.safetensors"))
    if not shards:
        shards = sorted(repo_path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensors shard found under {repo_dir}")
    return shards


def _read_all_weights(shards: list[Path]) -> dict[str, torch.Tensor]:
    """Load every tensor from every shard into a single flat dict."""
    from safetensors import safe_open

    combined: dict[str, torch.Tensor] = {}
    for shard in shards:
        logger.info("Reading shard %s", shard.name)
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                combined[key] = f.get_tensor(key)
    return combined


def _partition(
    raw: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Split the raw state_dict into (siglip, connector, llm)."""
    siglip: dict[str, torch.Tensor] = {}
    connector: dict[str, torch.Tensor] = {}
    llm: dict[str, torch.Tensor] = {}

    for key, tensor in raw.items():
        if key.startswith(VISION_PREFIX):
            new_key = key[len(VISION_PREFIX):]  # -> "vision_model.*"
            siglip[new_key] = tensor.clone()
        elif key.startswith(CONNECTOR_PREFIX):
            new_key = key[len(CONNECTOR_PREFIX):]  # -> "0.weight", "2.bias", ...
            connector[new_key] = tensor.clone()
        else:
            # Everything else (model.embed_tokens, model.layers.*, lm_head.*)
            # matches LlamaForCausalLM state_dict as-is.
            llm[key] = tensor.clone()

    return siglip, connector, llm


def _summarize(tag: str, sd: dict[str, torch.Tensor]) -> None:
    total = sum(t.numel() for t in sd.values())
    logger.info(
        "  %s: %d tensors, %.2fM parameters", tag, len(sd), total / 1e6,
    )


def main() -> None:
    args = parse_args()

    logger.info("Resolving TinyLLaVA checkpoint: %s", args.model_id)
    shards = _download_shards(args.model_id, args.cache_dir)
    raw = _read_all_weights(shards)
    logger.info("Loaded %d parameters from %d shards", len(raw), len(shards))

    siglip_sd, connector_sd, llm_sd = _partition(raw)

    _summarize("siglip    ", siglip_sd)
    _summarize("connector ", connector_sd)
    _summarize("llm       ", llm_sd)

    if not connector_sd:
        raise RuntimeError(
            "No connector weights found. Check that the checkpoint has the "
            f"expected prefix '{CONNECTOR_PREFIX}'."
        )
    if not siglip_sd:
        raise RuntimeError(
            "No vision tower weights found. Expected prefix "
            f"'{VISION_PREFIX}'."
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "source_model_id": args.model_id,
        "vision_prefix": VISION_PREFIX,
        "connector_prefix": CONNECTOR_PREFIX,
        "connector_keys": sorted(connector_sd.keys()),
    }
    torch.save(
        {
            "siglip": siglip_sd,
            "connector": connector_sd,
            "llm": llm_sd,
            "metadata": meta,
        },
        str(output),
    )
    logger.info("Saved extracted weights -> %s", output)
    logger.info("Metadata: %s", json.dumps(meta, indent=2)[:400])


if __name__ == "__main__":
    main()
