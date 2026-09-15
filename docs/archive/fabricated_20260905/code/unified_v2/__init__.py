"""FloorPlanNet v2.1 - Unified Super Model

Integrates all best ideas:
- Dual-pathway encoder (image + vector)
- Cross-attention fusion
- Hybrid parallel decoder (with confidence monitoring)
- Multi-scale FPN features
- Stride-4 head for small objects
- Knowledge distillation training
- Adaptive class routing

Expected performance:
- AP50: 0.52-0.56
- AP50:95: 0.32-0.36
- Speed: 2-3× faster than thin-head
- Timeline: 70 epochs, ~3-4 days training
"""

from .encoder import UnifiedEncoder
from .decoder import HybridMultiScaleDecoder
from .model import FloorPlanNetV2

__all__ = [
    "UnifiedEncoder",
    "HybridMultiScaleDecoder",
    "FloorPlanNetV2",
]
