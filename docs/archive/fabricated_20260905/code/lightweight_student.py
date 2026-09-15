"""Lightweight student model for knowledge distillation from thin teacher.

Target: 2-3M params, AP50 >0.45, 3-5× faster than teacher.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightBackbone(nn.Module):
    """Efficient MobileNetV2-style backbone."""

    def __init__(self, in_channels=3, out_channels=256):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        # Inverted residual blocks (MobileNetV2 style)
        self.blocks = nn.Sequential(
            self._make_block(32, 16, 1, 1),   # 256×256
            self._make_block(16, 24, 2, 2),   # 128×128
            self._make_block(24, 32, 3, 2),   # 64×64 (output stride 8)
            self._make_block(32, out_channels, 1, 1),  # 64×64
        )

    def _make_block(self, in_ch, out_ch, num_layers, stride):
        layers = []
        for i in range(num_layers):
            s = stride if i == 0 else 1
            layers.append(InvertedResidual(in_ch if i == 0 else out_ch, out_ch, s))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)       # 256×256
        x = self.blocks(x)     # 64×64
        return x


class InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual block."""

    def __init__(self, in_ch, out_ch, stride, expand_ratio=6):
        super().__init__()
        hidden_dim = in_ch * expand_ratio
        self.use_residual = stride == 1 and in_ch == out_ch

        layers = []
        if expand_ratio != 1:
            # Expand
            layers.extend([
                nn.Conv2d(in_ch, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])

        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1,
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        ])

        # Project
        layers.extend([
            nn.Conv2d(hidden_dim, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class LightweightCenterNet(nn.Module):
    """Lightweight CenterNet student model.

    Architecture:
    - Backbone: MobileNetV2-style (256 channels, stride 8)
    - Heads: Shared conv layers for heatmap/size/offset
    - Parameters: ~2-3M (vs teacher 6.8M)
    - Speed: 3-5× faster
    """

    def __init__(self, num_classes=35, backbone_channels=256):
        super().__init__()

        self.backbone = LightweightBackbone(in_channels=3, out_channels=backbone_channels)

        # Shared feature processing
        self.shared_conv = nn.Sequential(
            nn.Conv2d(backbone_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # CenterNet heads
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )

        self.size_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1),
        )

        self.offset_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1),
        )

        # Initialize
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Bias for heatmap (prior probability)
        nn.init.constant_(self.heatmap_head[-1].bias, -2.19)  # sigmoid(-2.19) ≈ 0.1

    def forward(self, x, class_id=None):
        """Forward pass.

        Args:
            x: [B, 3, H, W] image
            class_id: int, class to predict (for single-class mode)

        Returns:
            dict with 'heatmap', 'size', 'offset'
        """
        features = self.backbone(x)      # [B, 256, H/8, W/8]
        shared = self.shared_conv(features)  # [B, 128, H/8, W/8]

        heatmap = self.heatmap_head(shared)  # [B, 35, H/8, W/8]
        size = self.size_head(shared)        # [B, 2, H/8, W/8]
        offset = self.offset_head(shared)    # [B, 2, H/8, W/8]

        # Single-class mode (for query-based training)
        if class_id is not None:
            heatmap = heatmap[:, class_id:class_id+1]  # [B, 1, H/8, W/8]

        return {
            'heatmap': heatmap,
            'size': size,
            'offset': offset,
        }


class DistillationLoss(nn.Module):
    """Combined loss for knowledge distillation.

    Loss = task_loss + distillation_loss
    - task_loss: Standard CenterNet focal + L1
    - distillation_loss: KL divergence on soft targets from teacher
    """

    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_out, teacher_out, targets):
        """Compute distillation loss.

        Args:
            student_out: dict from student model
            teacher_out: dict from teacher model (with_grad=False)
            targets: dict with GT heatmap/size/offset

        Returns:
            dict with loss components
        """
        # Task loss (standard CenterNet)
        from src.training.losses import focal_loss, l1_loss_masked

        heatmap_task = focal_loss(
            pred=None,
            target=targets['heatmap'],
            logits=student_out['heatmap'],
            alpha=2.0,
            beta=4.0
        )

        positive_mask = targets['heatmap'] > 0.5
        size_task = l1_loss_masked(student_out['size'], targets['size'], positive_mask)
        offset_task = l1_loss_masked(student_out['offset'], targets['offset'], positive_mask)

        task_loss = 10.0 * heatmap_task + size_task + offset_task

        # Distillation loss (soft targets from teacher)
        T = self.temperature
        student_soft = torch.sigmoid(student_out['heatmap'] / T)
        # Teacher returns 'center_heatmap' key
        teacher_heatmap = teacher_out.get('center_heatmap', teacher_out.get('heatmap'))
        teacher_soft = torch.sigmoid(teacher_heatmap / T)

        # Match spatial dimensions (teacher might be different stride)
        if student_soft.shape != teacher_soft.shape:
            # Downsample teacher to match student
            teacher_soft = F.interpolate(
                teacher_soft,
                size=student_soft.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        distill_loss = F.mse_loss(student_soft, teacher_soft) * (T ** 2)

        # Combined
        total_loss = self.alpha * task_loss + (1 - self.alpha) * distill_loss

        return {
            'loss': total_loss,
            'task_loss': task_loss,
            'distill_loss': distill_loss,
            'heatmap_task': heatmap_task,
            'size_task': size_task,
            'offset_task': offset_task,
        }


def build_lightweight_student(num_classes=35, backbone_channels=256):
    """Build lightweight student model."""
    model = LightweightCenterNet(num_classes=num_classes, backbone_channels=backbone_channels)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"LightweightCenterNet: {total_params:,} parameters ({total_params/1e6:.2f}M)")

    return model


if __name__ == '__main__':
    # Test model
    model = build_lightweight_student()
    x = torch.randn(2, 3, 512, 512)

    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Heatmap: {out['heatmap'].shape}")
    print(f"Size: {out['size'].shape}")
    print(f"Offset: {out['offset'].shape}")

    # Single-class mode
    out_single = model(x, class_id=10)
    print(f"\nSingle-class (id=10):")
    print(f"Heatmap: {out_single['heatmap'].shape}")
