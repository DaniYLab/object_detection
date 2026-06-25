"""
Scan FloorPlanCAD_dataset, collect all unique class names from filenames,
save to classes.txt (comma-separated).

Class names are extracted from crop filenames like:
  door_double_001.png  ->  door_double
  wall_001.png         ->  wall
  original.png         ->  skipped
"""

import re
from pathlib import Path

DATASET_ROOT = Path("./data/FloorPlanCAD_dataset")
OUTPUT_FILE  = Path("./data/classes.txt")

# Strip trailing _NNN index to get class name
INDEX_RE = re.compile(r"_\d+$")


def extract_class(filename: str) -> str | None:
    stem = Path(filename).stem          # remove .png
    if stem == "original":
        return None
    return INDEX_RE.sub("", stem)       # remove _001, _023, etc.


def collect_classes(root: Path) -> set[str]:
    classes: set[str] = set()
    for f in root.rglob("*.png"):
        cls = extract_class(f.name)
        if cls:
            classes.add(cls)
    return classes


def main() -> None:
    print(f"Scanning: {DATASET_ROOT.resolve()}")
    classes = collect_classes(DATASET_ROOT)
    sorted_classes = sorted(classes)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(",".join(sorted_classes), encoding="utf-8")

    print(f"Found {len(sorted_classes)} classes:")
    print(" ", ", ".join(sorted_classes))
    print(f"\nSaved -> {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
