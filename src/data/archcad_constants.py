"""ArchCAD-400K semantic taxonomy and configuration constants.

Defines the 27/29 canonical classes, countable instance mapping,
and standard image space attributes for ArchCAD floor plans.
"""

from __future__ import annotations

# Full ArchCAD semantic mapping (raw semantic ID -> human-readable name)
ARCHCAD_RAW_SEMANTIC_NAMES: dict[int, str] = {
    0: "axis_and_grid",
    1: "single_door",
    2: "double_door",
    3: "parent_child_door",
    4: "other_door",
    5: "elevator",
    6: "staircase",
    7: "sink",
    8: "urinal",
    9: "toilet",
    10: "bathtub",
    11: "squat_toilet",
    12: "other_fixtures",
    13: "drain",
    14: "table",
    15: "chair",
    16: "bed",
    17: "sofa",
    18: "hole",
    19: "glass",
    20: "wall",
    21: "concrete_column",
    22: "steel_column",
    23: "concrete_beam",
    24: "steel_beam",
    25: "parking_space",
    26: "foundation",
    27: "pile",
    28: "rebar",
    29: "fire_hydrant",
    100: "others",
}

# Countable object classes (entities with bounding boxes / instances)
# Sorted in canonical benchmark order (0-indexed for detector heads)
ARCHCAD_COUNTABLE_CLASSES: list[str] = [
    "single_door",  # ID 1
    "double_door",  # ID 2
    "parent_child_door",  # ID 3
    "other_door",  # ID 4
    "elevator",  # ID 5
    "staircase",  # ID 6
    "sink",  # ID 7
    "urinal",  # ID 8
    "toilet",  # ID 9
    "bathtub",  # ID 10
    "squat_toilet",  # ID 11
    "table",  # ID 14
    "chair",  # ID 15
    "bed",  # ID 16
    "sofa",  # ID 17
    "hole",  # ID 18
    "concrete_column",  # ID 21
    "steel_column",  # ID 22
    "parking_space",  # ID 25
    "pile",  # ID 27
    "fire_hydrant",  # ID 29
]

# Mapping from class name to 0-indexed detector class index
ARCHCAD_CLASS_TO_IDX: dict[str, int] = {
    name: idx for idx, name in enumerate(ARCHCAD_COUNTABLE_CLASSES)
}
ARCHCAD_IDX_TO_CLASS: dict[int, str] = {
    idx: name for idx, name in enumerate(ARCHCAD_COUNTABLE_CLASSES)
}

# Mapping from raw semantic ID to detector class index (or None if stuff/uncountable)
ARCHCAD_RAW_ID_TO_DETECTOR_IDX: dict[int, int] = {}
for raw_id, raw_name in ARCHCAD_RAW_SEMANTIC_NAMES.items():
    if raw_name in ARCHCAD_CLASS_TO_IDX:
        ARCHCAD_RAW_ID_TO_DETECTOR_IDX[raw_id] = ARCHCAD_CLASS_TO_IDX[raw_name]

NUM_ARCHCAD_CLASSES = len(ARCHCAD_COUNTABLE_CLASSES)  # 21 classes

# Native ArchCAD slice properties
ARCHCAD_IMAGE_SIZE = (980, 980)  # Native slice resolution (W, H)
ARCHCAD_PHYSICAL_SIZE_M = 14.0  # 14m x 14m physical chunk size
ARCHCAD_BACKGROUND_IS_WHITE = True
