"""
Rename class_XX_NNN.png files to their proper class names.
Only renames files that still have placeholder names.
"""

from pathlib import Path

DATASET_ROOT = Path("./data/FloorPlanCAD_dataset")

RENAME_MAP = {
    "class_06": "door_revolving",
    "class_08": "window_blind",
    "class_10": "ramp",
    "class_25": "room_label",
    "class_26": "floor_plan_area",
    "class_32": "escalator_stair",
    "class_35": "annotation_text",
}


def main() -> None:
    print(f"Scanning: {DATASET_ROOT.resolve()}")
    print(f"Rename rules: {len(RENAME_MAP)} patterns\n")

    total = 0
    for old_prefix, new_prefix in RENAME_MAP.items():
        count = 0
        for f in DATASET_ROOT.rglob(f"{old_prefix}_*.png"):
            new_name = f.name.replace(old_prefix, new_prefix, 1)
            f.rename(f.parent / new_name)
            count += 1
        total += count
        print(f"  {old_prefix:12s} -> {new_prefix:20s} : {count:6d} files renamed")

    print(f"\nDone! Total renamed: {total} files")


if __name__ == "__main__":
    main()
