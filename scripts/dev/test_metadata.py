import json
from pathlib import Path
from generate_metadata import parse_svg

for svg in sorted(Path("./data/FloorPlanCAD_original/train_set_1").glob("*.svg"))[:3]:
    meta = parse_svg(svg)
    n = meta["num_instances"]
    sz = meta["image_size"]
    print(f"{svg.stem}: {n} instances, image={sz}")
    for inst in meta["instances"][:4]:
        cls = inst["class"]
        iid = inst["instance_id"]
        bbox = inst["bbox_px"]
        print(f"  {cls:22s} iid={iid:3d}  bbox={bbox}")
    print()
