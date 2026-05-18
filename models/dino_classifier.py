"""DINOv2-based binary deepfake classifier.

Uses the frozen DINOv2-Large CLS token (1x1024) as input to a small
trainable MLP head that predicts Real (0) vs Fake (1).  This is the
same CLS token that ``TinyLLaVAGroundVLM._encode_dino`` discards
(``dino_out[:, 1:, :]``), so the classifier and the VLM can share
the same DINOv2 backbone without redundancy at inference.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Dinov2Model


class DINOv2Classifier(nn.Module):
    """Frozen DINOv2 + trainable MLP head for binary classification."""

    def __init__(
        self,
        dino_model: str = "facebook/dinov2-large",
        num_classes: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dinov2 = Dinov2Model.from_pretrained(dino_model)
        for p in self.dinov2.parameters():
            p.requires_grad = False
        self.dinov2.eval()

        dino_dim = self.dinov2.config.hidden_size  # 1024 for dinov2-large

        self.head = nn.Sequential(
            nn.Linear(dino_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return logits [B, num_classes] from pixel values.

        Parameters
        ----------
        pixel_values : Tensor [B, 3, H, W]
            DINOv2-preprocessed images (ImageNet-normalised, 224x224 or 384x384).

        Returns
        -------
        Tensor [B, num_classes]
        """
        with torch.no_grad():
            dino_out = self.dinov2(pixel_values=pixel_values)
            cls_token = dino_out.last_hidden_state[:, 0, :]
        return self.head(cls_token.to(self.head[0].weight.dtype))

    def predict(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices [B]."""
        logits = self.forward(pixel_values)
        return logits.argmax(dim=-1)

    def predict_verdict(self, pixel_values: torch.Tensor) -> list[str]:
        """Return verdict strings ("Real" or "Fake") for a batch."""
        preds = self.predict(pixel_values)
        return ["Real" if p == 0 else "Fake" for p in preds.tolist()]
