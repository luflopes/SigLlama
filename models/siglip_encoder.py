"""SigLIP visual encoder wrapper.

Loads the SigLIP vision tower from HuggingFace and returns patch-level
visual features that will be projected into the TinyLlama embedding space
by the Adapter.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import SiglipVisionModel, SiglipImageProcessor


class SigLIPEncoder(nn.Module):
    """Frozen (or fine-tunable) SigLIP vision encoder.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g. ``"google/siglip-base-patch16-224"``.
    freeze : bool
        If True the encoder weights are frozen during training.
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        freeze: bool = True,
    ):
        super().__init__()
        self.vision_model = SiglipVisionModel.from_pretrained(model_name)
        self.processor = SiglipImageProcessor.from_pretrained(model_name)

        if freeze:
            for p in self.vision_model.parameters():
                p.requires_grad = False

    @property
    def hidden_size(self) -> int:
        return self.vision_model.config.hidden_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pixel_values : Tensor [B, C, H, W]
            Pre-processed image tensors.

        Returns
        -------
        Tensor [B, num_patches, hidden_size]
            Patch-level visual features (last hidden state).
        """
        outputs = self.vision_model(pixel_values=pixel_values)
        return outputs.last_hidden_state
