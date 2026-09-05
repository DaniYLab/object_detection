"""Hybrid backbone that integrates with existing thin-head detector.

Reuses proven components:
- ConvImageEncoder from detector.py (stride-8 backbone)
- Thin-head architecture from HeatmapHead
- Adds parallel decoding capability
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from .detector import ConvImageEncoder, HeatmapHead
from .config import VAEConfig


class HybridBackbone(nn.Module):
    """Backbone that outputs features for both parallel and thin-head decoding."""
    
    def __init__(
        self,
        vae_config: Optional[VAEConfig] = None,
        feature_dim: int = 256,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # Use default config if not provided
        if vae_config is None:
            vae_config = VAEConfig()

        # Use proven ConvImageEncoder (stride-8)
        self.image_encoder = ConvImageEncoder(cfg=vae_config)

        # Project to unified feature dimension
        # ConvImageEncoder outputs latent_channels, not block_out_channels[-1]
        encoder_out_dim = vae_config.latent_channels
        if encoder_out_dim != feature_dim:
            self.feature_projection = nn.Conv2d(encoder_out_dim, feature_dim, 1)
        else:
            self.feature_projection = nn.Identity()
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to features.
        
        Args:
            image: [B, 3, H, W]
            
        Returns:
            features: [B, feature_dim, H/8, W/8]
        """
        # Encode (stride-8)
        features = self.image_encoder(image)
        
        # Project to unified dimension
        features = self.feature_projection(features)
        
        return features


def build_hybrid_backbone(
    feature_dim: int = 256,
    vae_config: Optional[VAEConfig] = None,
) -> HybridBackbone:
    """Build hybrid backbone compatible with thin-head."""
    if vae_config is None:
        vae_config = VAEConfig()
    
    return HybridBackbone(
        vae_config=vae_config,
        feature_dim=feature_dim,
    )
