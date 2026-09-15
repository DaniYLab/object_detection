# Training Issues & Solutions

## Issue: NaN Loss After ~30 Batches

**Observation**:
```
Batch 0:  Loss 1.4249, IoU 1.0000, BCE 0.8498
Batch 10: Loss 1.2711, IoU 0.9968, BCE 0.5486
Batch 20: Loss 1.1106, IoU 0.9990, BCE 0.2231
Batch 30: Loss nan, IoU nan, BCE nan
```

**Root Causes**:

### 1. Gradient Explosion
- Model từ random initialization → predictions không meaningful
- IoU = 1.0 nghĩa là predictions hoàn toàn không overlap với GT
- Loss lớn → gradients explode → weights diverge → NaN

### 2. BCE Loss Instability
- Confidence predictions từ random init có thể rất extreme
- `torch.sigmoid(very_large_logit)` → numerical overflow
- Need to use `F.binary_cross_entropy_with_logits` correctly

### 3. Learning Rate Too High
- 1e-4 có thể quá cao cho model lớn (18M params)
- Cần warmup hoặc reduce LR

---

## Solutions

### Immediate Fixes

**1. Lower Learning Rate**
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # Was 1e-4
```

**2. Add Gradient Clipping**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**3. Check IoU Computation**
```python
# Add epsilon to avoid division by zero
iou = inter / (union + 1e-6)  # Already in code
```

**4. Warmup Schedule**
```python
# Start with very small LR, gradually increase
for epoch in range(warmup_epochs):
    lr = base_lr * (epoch + 1) / warmup_epochs
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

### Medium-term Solutions

**5. Better Initialization**
- Use pretrained ViT weights (e.g., from ImageNet)
- Initialize decoder heads with small weights
- Xavier/Kaiming initialization for linear layers

**6. Loss Weighting**
```python
# Reduce BCE weight further
Loss = 1.0 * IoU_loss + 0.1 * BCE_loss  # Was 0.5
```

**7. Mixed Precision Training**
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(...)
    loss = criterion(...)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## Quick Fix Script

Create `train_floorplan_llm_fixed.py` with:

```python
# Key changes:
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # Lower LR
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clip
criterion = ParallelBoxLoss(iou_weight=1.0, bce_weight=0.1)  # Lower BCE weight

# In training loop, after loss.backward():
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

## Alternative Approach: Staged Training

**Stage 1: Train decoder only** (freeze encoder)
```python
# Freeze encoder
for param in model.encoder.parameters():
    param.requires_grad = False

# Train only decoder
optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=1e-4)
```

**Stage 2: Fine-tune full model** (unfreeze encoder)
```python
# Unfreeze encoder
for param in model.encoder.parameters():
    param.requires_grad = True

# Lower LR for full model
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
```

---

## Testing Strategy

**Test 1: Overfit on 1 batch**
```python
# Use single batch, train until loss < 0.1
batch = next(iter(dataloader))
for i in range(1000):
    loss = train_step(batch)
    if loss < 0.1:
        print(f"✓ Converged in {i} steps")
        break
```

**Test 2: Check gradients**
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if not math.isfinite(grad_norm):
            print(f"✗ NaN gradient in {name}")
```

---

## Next Actions

1. ✅ Implement quick fixes (LR + gradient clip)
2. ✅ Test overfit on 1 batch
3. ✅ If still NaN: try staged training (decoder first)
4. ⚠️ Consider using thin-head weights as initialization

---

**Priority**: Fix NaN before continuing full training.
