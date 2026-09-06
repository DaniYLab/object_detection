# FloorPlanLLM - Next Steps Quick Reference

**Current Status**: Week 1 Foundation Complete (Architecture validated)

---

## Immediate Priority Queue

### 1. Dataset Wrapper (1 day)

**File**: `src/data/floorplan_llm_dataset.py`

```python
class FloorPlanLLMDataset(FloorPlanImageDataset):
    """Wrapper that extracts normalized boxes for parallel supervision."""
    
    def __getitem__(self, idx):
        # Load base sample
        sample = super().__getitem__(idx)
        
        # Extract from metadata
        # boxes: [N, 4] normalized (cx, cy, w, h) in [0, 1]
        # classes: [N] class indices
        
        return {
            'image': sample['image'],
            'stroke_tokens': sample['stroke_tokens'],
            'stroke_mask': sample['stroke_mask'],
            'boxes': boxes,  # torch.Tensor [N, 4]
            'classes': classes,  # torch.Tensor [N] dtype=long
        }
```

**Test script**: `test_dataset_wrapper.py`
```python
dataset = FloorPlanLLMDataset(root='data/FloorPlanCAD_original', 
                              split='train_new', 
                              manifest_path='splits_v2.json')
sample = dataset[0]
print(f"boxes shape: {sample['boxes'].shape}")
print(f"classes shape: {sample['classes'].shape}")
print(f"boxes range: [{sample['boxes'].min():.3f}, {sample['boxes'].max():.3f}]")
```

---

### 2. Baseline Training (2 days)

**Command**:
```bash
python train_floorplan_llm_simple.py \
    --preset small \
    --batch_size 8 \
    --epochs 10 \
    --lr 1e-4 \
    --output_dir checkpoints/floorplan_llm_baseline
```

**Expected results** (10 epochs):
- AP50:95 ≥ 0.25 (acceptable baseline)
- Speed: ~150-200 img/s
- Loss convergence: < 0.5 by epoch 10

**Checkpoints**:
- `best_val_ap.pt` (save when val_select AP improves)
- `last.pt` (last epoch)
- `config.json` (hyperparameters)

---

### 3. Evaluation (1 day)

**File**: `evaluate_floorplan_llm.py`

**Convert parallel predictions → CenterNet format**:
```python
def parallel_to_centernet_for_eval(parallel_output, threshold=0.3):
    """Convert parallel boxes to list of detections."""
    boxes = parallel_output['boxes']  # [B, 35, 4]
    conf = torch.sigmoid(parallel_output['confidence'])  # [B, 35]
    
    detections = []
    for b in range(B):
        for c in range(35):
            if conf[b, c] > threshold:
                # Add to detections: (class, confidence, x, y, w, h)
                detections.append({
                    'class': c,
                    'confidence': conf[b, c].item(),
                    'box': boxes[b, c].cpu().numpy(),  # (cx, cy, w, h)
                })
    return detections
```

**Metrics**:
- Use existing `evaluate.py` logic
- AP50, AP50:95, per-class AP
- Compare against thin-head (AP50:95 = 0.3063)

---

### 4. Teacher Integration (3 days)

**Step 4.1**: Parse teacher reasoning → extract relationships

**File**: `scripts/data/parse_teacher_to_relationships.py`

```python
def extract_spatial_relations(reasoning: str) -> list[dict]:
    """
    Input: "The bedroom contains two beds. Door connects bedroom and hallway."
    
    Output: [
        {'source': 'bedroom', 'relation': 'contains', 'target': 'bed'},
        {'source': 'door', 'relation': 'connects', 'target': 'bedroom'},
        {'source': 'door', 'relation': 'connects', 'target': 'hallway'},
    ]
    """
    # Use rule-based extraction or small LLM
    pass
```

**Relation types to extract**:
- `contains`: room contains furniture
- `connects`: door/window connects two spaces
- `adjacent`: objects next to each other
- `aligned`: horizontal/vertical alignment
- `symmetry`: mirror symmetry

**Step 4.2**: Add relationship head to model

```python
class RelationshipHead(nn.Module):
    """Predict pairwise relationships between detected objects."""
    
    def forward(self, node_features):
        # node_features: [B, num_objects, D]
        # Output: [B, num_objects, num_objects, num_relations]
        pass
```

**Step 4.3**: Multi-task training

```python
loss = λ1 * detection_loss + λ2 * relationship_loss
# λ1=1.0, λ2=0.3
```

---

## Timeline (Week 1-2)

```
Day 1-2:  Dataset wrapper + testing
Day 3-5:  Baseline training (10 epochs)
Day 6:    Evaluation + comparison
Day 7-9:  Teacher integration (parse + relationship head)
Day 10:   Multi-task training
```

---

## Success Metrics

### Baseline (End of Week 2)
- ✅ AP50:95 ≥ 0.25 (detection baseline)
- ✅ Speed ≥ 150 img/s
- ✅ Memory ≤ 1 GB (base preset, batch=1)

### Extended (End of Week 3)
- ✅ Relationship F1 ≥ 0.60
- ✅ Multi-task AP50:95 ≥ 0.28 (detection + relationships)

---

## Troubleshooting

### If AP50:95 < 0.20 after 10 epochs
1. Check ground truth boxes normalized correctly
2. Visualize predictions (overlay on images)
3. Try larger model (base instead of small)
4. Increase training epochs (30 instead of 10)

### If training too slow
1. Use tiny preset for prototyping
2. Reduce batch size
3. Freeze image encoder (train decoder only)

### If OOM
1. Reduce batch size
2. Use gradient checkpointing
3. Use tiny/small preset

---

## Commands Reference

```bash
# Test architecture
python test_floorplan_llm.py

# Test dataset wrapper (after implementation)
python test_dataset_wrapper.py

# Train baseline
python train_floorplan_llm_simple.py --preset small --batch_size 8 --epochs 10

# Evaluate
python evaluate_floorplan_llm.py --checkpoint checkpoints/floorplan_llm_baseline/best_val_ap.pt

# Parse teacher data
python scripts/data/parse_teacher_to_relationships.py --input data/Teacher/teacher_distill.jsonl --output data/Teacher/relationships.json
```

---

**Status**: Ready to proceed with Priority 1 (Dataset Wrapper)

**Next decision point**: After baseline training (Day 6) - if AP50:95 < 0.25, debug before proceeding to teacher integration.
