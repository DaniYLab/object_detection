import torch
from src.models.hybrid_parallel_detector import build_hybrid_parallel_detector

print("Building model...")
model = build_hybrid_parallel_detector(num_classes=35, use_reasoning=False)
model.eval()

print("Creating dummy input...")
x = torch.randn(1, 3, 512, 512)

print("Forward pass...")
with torch.no_grad():
    output = model(x)

print("Output shapes:")
for k, v in output.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape}")

print("\n✅ Model works!")
