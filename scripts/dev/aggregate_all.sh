#!/bin/bash
# Aggregate all ablation results (A0/A1/B across seeds 1337/2024/4242)

echo "=== Validation AP50 summary ==="
.venv/Scripts/python.exe scripts/dev/aggregate_results.py \
  --reports \
    outputs/evaluation_val_centernet_baseline_seed1337.json \
    outputs/evaluation_val_shared_no_condition_seed1337.json \
    outputs/evaluation_val_floorplan_base_seed1337.json \
    outputs/evaluation_val_shared_no_condition_seed2024.json \
    outputs/evaluation_val_floorplan_base_seed2024.json \
    outputs/evaluation_val_shared_no_condition_seed4242.json \
    outputs/evaluation_val_floorplan_base_seed4242.json \
  --output-json outputs/ablation_summary_val.json

echo ""
echo "=== Test AP50 summary ==="
.venv/Scripts/python.exe scripts/dev/aggregate_results.py \
  --reports \
    outputs/evaluation_test_centernet_baseline_seed1337.json \
    outputs/evaluation_test_shared_no_condition_seed1337.json \
    outputs/evaluation_test_floorplan_base_seed1337.json \
    outputs/evaluation_test_shared_no_condition_seed2024.json \
    outputs/evaluation_test_floorplan_base_seed2024.json \
    outputs/evaluation_test_shared_no_condition_seed4242.json \
    outputs/evaluation_test_floorplan_base_seed4242.json \
  --output-json outputs/ablation_summary_test.json
