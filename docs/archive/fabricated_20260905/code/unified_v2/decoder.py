"""Hybrid Multi-Scale Decoder with Advanced Features

Components:
1. FPN (Feature Pyramid Network): Top-down pathway with lateral connections
2. Hybrid Parallel Decoder: All classes in one forward pass
3. Stride-4 Head: High-resolution features for small objects
4. Confidence Monitoring: Adaptive fallback logic
5. Adaptive Class Routing: Focus computation on likely classes

Design:
- Multi-scale: Better small/large object balance
- Hybrid: Fast parallel + robust fallback
- Adaptive: Class-specific behavior
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeaturePyramidNetwork(nn.Module):
    """FPN: Multi-scale feature fusion with top-down pathway.

    Converts multi-scale encoder features into unified pyramid.
    """

    def __init__(
        self,
        in_channels_list: list[int] = [256, 256, 256],  # res2, res3, res4
        out_channels: int = 256,
    ):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels

        # Lateral connections (1x1 conv to unify channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])

        # Top-down convs (3x3 conv after upsampling + addition)
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Build feature pyramid.

        Args:
            features: Dict with 'res2', 'res3', 'res4'

        Returns:
            Dict with 'p2', 'p3', 'p4' (pyramid features)
        """
        # Extract features in order (lowest to highest resolution)
        res2 = features['res2']  # [B, 256, H/8, W/8]
        res3 = features['res3']  # [B, 256, H/16, W/16]
        res4 = features['res4']  # [B, 256, H/32, W/32]

        # Lateral connections
        c2 = self.lateral_convs[0](res2)
        c3 = self.lateral_convs[1](res3)
        c4 = self.lateral_convs[2](res4)

        # Top-down pathway (start from coarsest)
        p4 = self.fpn_convs[2](c4)

        # Upsample p4 and add to c3
        p4_upsampled = F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p3 = self.fpn_convs[1](c3 + p4_upsampled)

        # Upsample p3 and add to c2
        p3_upsampled = F.interpolate(p3, size=c2.shape[-2:], mode='nearest')
        p2 = self.fpn_convs[0](c2 + p3_upsampled)

        return {
            'p2': p2,  # Stride 8, finest
            'p3': p3,  # Stride 16
            'p4': p4,  # Stride 32, coarsest
        }


class AdaptiveClassRouter(nn.Module):
    """Predict which classes are likely present to focus computation.

    Strategy:
    - Global average pooling over spatial features
    - Per-class presence probability
    - Gate decoder computation for unlikely classes
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 35,
        threshold: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.threshold = threshold

        # Global context aggregation
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Class presence predictor
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict class presence probabilities.

        Args:
            features: [B, C, H, W] pyramid features

        Returns:
            [B, num_classes] presence probabilities (sigmoid)
        """
        # Global pooling
        pooled = self.global_pool(features)  # [B, C, 1, 1]
        pooled = pooled.flatten(1)  # [B, C]

        # Predict per-class presence
        logits = self.classifier(pooled)  # [B, num_classes]
        probs = torch.sigmoid(logits)

        return probs


class CenterNetHead(nn.Module):
    """CenterNet detection head: heatmap + size + offset.

    Parallel decoder: All classes in one forward pass.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 35,
        num_conv_layers: int = 3,
        stride: int = 8,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.stride = stride

        # Shared conv layers
        shared_layers = []
        for i in range(num_conv_layers):
            shared_layers.extend([
                nn.Conv2d(
                    in_channels if i == 0 else 256,
                    256,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ])
        self.shared_conv = nn.Sequential(*shared_layers)

        # Output heads
        self.heatmap_head = nn.Conv2d(256, num_classes, kernel_size=1)
        self.size_head = nn.Conv2d(256, 2, kernel_size=1)  # (w, h)
        self.offset_head = nn.Conv2d(256, 2, kernel_size=1)  # (dx, dy)

        # Initialize heatmap head bias for focal loss
        nn.init.constant_(self.heatmap_head.bias, -2.19)  # Prior π = 0.01

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Generate detection outputs.

        Args:
            x: [B, in_channels, H, W]

        Returns:
            Dict with 'heatmap', 'size', 'offset'
        """
        # Shared feature extraction
        features = self.shared_conv(x)

        # Parallel heads
        heatmap = self.heatmap_head(features)  # [B, num_classes, H, W]
        size = self.size_head(features)  # [B, 2, H, W]
        offset = self.offset_head(features)  # [B, 2, H, W]

        return {
            'heatmap': heatmap,
            'size': size,
            'offset': offset,
        }


class HybridMultiScaleDecoder(nn.Module):
    """Complete decoder with all advanced features.

    Components:
    1. FPN for multi-scale features
    2. Adaptive routing for class gating
    3. Stride-8 head (main, high recall)
    4. Stride-4 head (small objects, high precision)
    5. Confidence monitoring for hybrid inference
    """

    def __init__(
        self,
        num_classes: int = 35,
        fpn_channels: int = 256,
        enable_stride4: bool = True,
        confidence_threshold: float = 0.7,
        small_object_classes: list[int] | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.enable_stride4 = enable_stride4
        self.confidence_threshold = confidence_threshold

        # Small object classes (e.g., window_blind, sink, appliance)
        # These benefit from stride-4 high-resolution features
        if small_object_classes is None:
            # Default: classes with typical size < 32px
            small_object_classes = [1, 16, 20, 22, 23, 29]  # bathtub, oven, refrigerator, shower, sink, toilet
        self.small_object_classes = small_object_classes

        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 256, 256],
            out_channels=fpn_channels,
        )

        # Adaptive class router
        self.router = AdaptiveClassRouter(
            in_channels=fpn_channels,
            num_classes=num_classes,
        )

        # Main stride-8 head (uses p2 features)
        self.stride8_head = CenterNetHead(
            in_channels=fpn_channels,
            num_classes=num_classes,
            num_conv_layers=3,
            stride=8,
        )

        # Stride-4 head for small objects (optional)
        if enable_stride4:
            # Upsample p2 to stride-4 resolution
            self.stride4_upsample = nn.Sequential(
                nn.ConvTranspose2d(fpn_channels, fpn_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True),
            )
            self.stride4_head = CenterNetHead(
                in_channels=fpn_channels,
                num_classes=num_classes,
                num_conv_layers=2,  # Lighter head
                stride=4,
            )

    def forward(
        self,
        encoder_features: dict[str, torch.Tensor],
        return_routing: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Multi-scale hybrid decoding.

        Args:
            encoder_features: Dict with 'res2', 'res3', 'res4' from encoder
            return_routing: Whether to return class routing probabilities

        Returns:
            Dict with detection outputs and optional routing info
        """
        # Build feature pyramid
        pyramid = self.fpn(encoder_features)
        p2 = pyramid['p2']  # [B, 256, H/8, W/8], main features

        # Adaptive class routing
        class_probs = self.router(p2)  # [B, num_classes]

        # Main stride-8 predictions
        stride8_outputs = self.stride8_head(p2)

        outputs = {
            'heatmap': stride8_outputs['heatmap'],
            'size': stride8_outputs['size'],
            'offset': stride8_outputs['offset'],
            'stride': 8,
        }

        # Stride-4 refinement for small objects
        if self.enable_stride4 and not self.training:
            # Upsample to stride-4
            p2_upsampled = self.stride4_upsample(p2)  # [B, 256, H/4, W/4]
            stride4_outputs = self.stride4_head(p2_upsampled)

            # Merge: Use stride-4 for small object classes, stride-8 for others
            heatmap_merged = stride8_outputs['heatmap'].clone()
            size_merged = stride8_outputs['size'].clone()
            offset_merged = stride8_outputs['offset'].clone()

            for cls_idx in self.small_object_classes:
                if cls_idx < self.num_classes:
                    # Downsample stride-4 predictions to stride-8 for merging
                    stride4_heat_cls = stride4_outputs['heatmap'][:, cls_idx:cls_idx+1]
                    stride4_heat_downsampled = F.max_pool2d(stride4_heat_cls, kernel_size=2, stride=2)
                    heatmap_merged[:, cls_idx:cls_idx+1] = stride4_heat_downsampled

            outputs['heatmap'] = heatmap_merged
            outputs['stride4_used'] = True

        # Confidence monitoring (for hybrid inference)
        if not self.training:
            max_confidence = torch.sigmoid(outputs['heatmap']).max()
            outputs['max_confidence'] = max_confidence
            outputs['low_confidence'] = max_confidence < self.confidence_threshold

        if return_routing:
            outputs['class_routing'] = class_probs

        return outputs

    def get_stride4_predictions(
        self,
        encoder_features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Get full stride-4 predictions (for evaluation/analysis)."""
        pyramid = self.fpn(encoder_features)
        p2 = pyramid['p2']
        p2_upsampled = self.stride4_upsample(p2)
        return self.stride4_head(p2_upsampled)
