# Training Optimized: Batch Size 8 + Workers 4

**Goal**: Hoàn thành nghiên cứu  
**Date**: 2026-09-06 00:15  
**Status**: Optimized training running (PID 3755)

---

## Configuration

**Previous issues**:
- Batch size 4: Worked but GPU utilization only 40% (data loading bottleneck)
- Batch size 16: Started but OOMed mid-epoch (memory instability)

**Current optimized config**:
- Batch size: 8 (sweet spot for memory + throughput)
- Num workers: 4 for training, 2 for validation (parallel data loading)
- Resume from: epoch 10 checkpoint
- Remaining: epochs 11-100

---

## Expected Improvements

### GPU Utilization
- Previous (bs=4, workers=0): ~40% utilization
- Expected (bs=8, workers=4): ~70-90% utilization
- Reason: Parallel data loading eliminates CPU/IO bottleneck

### Training Speed
- Batch size 4: ~950ms/batch (3,800 batches/epoch = ~1h/epoch)
- Batch size 8 expected: ~500-600ms/batch (1,900 batches/epoch = ~20min/epoch)
- **Speedup**: ~3x faster per epoch!

### Timeline
- Previous estimate: 90 epochs × 1h = 90 hours (~3.75 days)
- New estimate: 90 epochs × 20min = 1,800 min = 30 hours (~1.25 days)
- **ETA completion**: Sunday Sept 7, ~06:00 AM (instead of Tuesday)

---

## Monitor

```bash
# Check training progress
tail -f training_bs8_workers4.log

# Check GPU utilization
nvidia-smi

# Check process
ps aux | grep "3755"

# Check checkpoints
ls -lh checkpoints/dual_stream_100ep/
```

---

## Technical Rationale

### Why batch_size=8?
- Memory safe: ~12GB GPU usage (16GB available)
- Computational efficient: GPU has work to do
- Stable: Won't OOM mid-training

### Why num_workers=4?
- Parallel data loading: 4 threads prefetch batches
- Reduces GPU idle time: Next batch ready when GPU finishes
- Optimal for 8-core CPU: Doesn't saturate CPU

### Why not batch_size=16?
- Memory unstable: Gets close to 16GB limit
- OOM risk: Process killed mid-epoch, lose progress
- No checkpoint between epochs 10-20: Can't resume easily

---

## Progress Tracking

**Current**:
- Epoch: 11/100 (resuming)
- Checkpoint available: Epoch 10

**Next milestones**:
- Epoch 15: First validation with new config
- Epoch 20: Save checkpoint, diagnostic check
- Epoch 50: Mid-training diagnostic
- Epoch 100: Complete!

---

## Expected Results

After 100 epochs:
- Loss: ~10-15 (from current ~17)
- Heatmap loss: ~1.0-1.5 (from current ~1.5)
- Val AP50: 0.15-0.18 (baseline for ablations)

---

**Status**: Optimized training running. Expected 3x speedup. ETA Sunday morning instead of Tuesday.
