"""
Shared constants for FloorPlanCAD dataset.

Single source of truth for class mappings used by both:
  - scripts/data/build_dataset.py
  - scripts/data/generate_metadata.py
  - src/data/dataset.py
"""

# Semantic ID → class name (FloorPlanCAD 35-class panoptic)
# Based on: "FloorPlanCAD: A Large-Scale CAD Floor Plan Dataset"
SEMANTIC_ID_TO_NAME = {
    1:  "wall",
    2:  "door_single",
    3:  "door_double",
    4:  "door_sliding",
    5:  "window",
    6:  "door_revolving",
    7:  "window_bay",
    8:  "window_blind",
    9:  "stair",
    10: "ramp",
    11: "elevator",
    12: "escalator",
    13: "column",
    14: "toilet",
    15: "sink",
    16: "bathtub",
    17: "shower",
    18: "washing_machine",
    19: "refrigerator",
    20: "oven",
    21: "bed",
    22: "sofa",
    23: "table",
    24: "chair",
    25: "room_label",
    26: "floor_plan_area",
    27: "parking",
    28: "plant",
    29: "counter",
    30: "cabinet",
    31: "tv",
    32: "escalator_stair",
    33: "dimension_line",
    34: "symbol_misc",
    35: "annotation_text",
}

# Sorted alphabetically — index = class_id used by model
CLASS_NAMES = sorted(set(SEMANTIC_ID_TO_NAME.values()))
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)  # 35

TEXT_TEMPLATE = "Find {cls} in this floor plan drawing"
