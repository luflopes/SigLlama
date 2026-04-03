"""Soft Token Embedder.

Projects structured detection and landmark tokens into the LLM embedding
space so they can be concatenated with visual and text embeddings before
being fed to TinyLlama.

Soft tokens include:
  - Object detections: [cx, cy, w, h, class_id, confidence] -> embedding
  - Face landmarks: 478 normalised (x, y) points -> embedding sequence
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SoftTokenEmbedder(nn.Module):
    """Embeds detection bboxes and face-mesh landmarks into LLM-space vectors.

    Parameters
    ----------
    llm_dim : int
        Target embedding dimension (TinyLlama hidden size).
    max_detections : int
        Maximum number of detection tokens per image.
    num_landmarks : int
        Number of face-mesh landmarks (478 for MediaPipe Tasks API).
    landmark_dim : int
        Per-landmark input features (default 2: normalised x, y).
    """

    def __init__(
        self,
        llm_dim: int = 2048,
        max_detections: int = 20,
        num_landmarks: int = 478,
        landmark_dim: int = 2,
    ):
        super().__init__()
        self.llm_dim = llm_dim
        self.max_detections = max_detections
        self.num_landmarks = num_landmarks

        # Detection bbox embedding: 6 features -> llm_dim
        # [cx, cy, w, h, class_id (embedded), confidence]
        self.det_proj = nn.Sequential(
            nn.Linear(6, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

        # Landmark embedding: flatten all points or use 1D conv
        # 468 * 2 = 936 -> compressed into a fixed number of tokens
        self.landmark_proj = nn.Sequential(
            nn.Linear(num_landmarks * landmark_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def embed_detections(self, det_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        det_features : Tensor [B, num_det, 6]

        Returns
        -------
        Tensor [B, num_det, llm_dim]
        """
        return self.det_proj(det_features)

    def embed_landmarks(self, landmark_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        landmark_features : Tensor [B, num_landmarks * 2]
            Flattened normalised landmark coordinates.

        Returns
        -------
        Tensor [B, 1, llm_dim]
            Single landmark summary token.
        """
        out = self.landmark_proj(landmark_features)
        return out.unsqueeze(1)

    def forward(
        self,
        det_features: torch.Tensor | None = None,
        landmark_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed and concatenate all available soft tokens.

        Returns
        -------
        Tensor [B, num_soft_tokens, llm_dim]
        """
        parts: list[torch.Tensor] = []
        if det_features is not None:
            parts.append(self.embed_detections(det_features))
        if landmark_features is not None:
            parts.append(self.embed_landmarks(landmark_features))

        if not parts:
            raise ValueError("At least one of det_features or landmark_features required.")

        return torch.cat(parts, dim=1)
