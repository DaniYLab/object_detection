"""
Download FloorPlanCAD dataset dùng streaming mode — ảnh được lưu ngay lập tức.
Windows-compatible, không cần symlink, không cần chờ download toàn bộ.

Dataset: https://huggingface.co/datasets/Voxel51/FloorPlanCAD
- 5308 samples, floor plan CAD images + object detection labels
- License: CC-BY-SA 4.0
"""

import json
import os
from pathlib import Path

# Tắt cảnh báo symlink của HuggingFace trên Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

TOTAL_SAMPLES = 5308


def download_floorplan_cad(output_dir: str = "./data/FloorPlanCAD") -> None:
    """
    Download và extract FloorPlanCAD dùng streaming=True.
    Ảnh được lưu ngay khi từng sample về — không cần chờ toàn bộ dataset.

    Args:
        output_dir: Thư mục lưu ảnh và metadata.
    """
    from datasets import load_dataset

    out = Path(output_dir)
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  FloorPlanCAD - Streaming Download (Windows OK)")
    print("  Images will appear in ./data/FloorPlanCAD/images/ immediately")
    print("=" * 60)

    # streaming=True: không download toàn bộ trước, xử lý từng sample
    print("\n[INFO] Starting streaming from HuggingFace...")
    ds = load_dataset(
        "Voxel51/FloorPlanCAD",
        split="train",
        streaming=True,
    )

    metadata_records = []
    saved = 0

    for sample in ds:
        img = sample.get("image")
        if img is None:
            continue

        # Convert to RGB (JPEG does not support RGBA/alpha channel)
        if img.mode in ("RGBA", "P", "LA"):
            from PIL import Image as PILImage
            if img.mode == "P":
                img = img.convert("RGBA")
            background = PILImage.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # Save image immediately
        img_path = images_dir / f"sample_{saved:05d}.jpg"
        img.save(str(img_path), format="JPEG", quality=95)

        # Thu thập metadata
        record = {"idx": saved, "image_path": str(img_path)}
        for key, val in sample.items():
            if key == "image":
                continue
            record[key] = val if not hasattr(val, "__dict__") else str(val)
        metadata_records.append(record)
        saved += 1

        # Print progress every 50 images
        if saved % 50 == 0:
            pct = saved / TOTAL_SAMPLES * 100
            print(f"  [{saved:>4}/{TOTAL_SAMPLES}] {pct:5.1f}% - saved: {img_path.name}")

        # Save metadata checkpoint every 200 images (crash recovery)
        if saved % 200 == 0:
            _save_metadata(metadata_records, out)

    # Lưu metadata lần cuối
    _save_metadata(metadata_records, out)

    print(f"\n{'=' * 60}")
    print(f"  DONE!")
    print(f"  Images : {images_dir}  ({saved} files)")
    print(f"  Meta   : {out / 'metadata.json'}")
    print(f"{'=' * 60}")


def _save_metadata(records: list, out: Path) -> None:
    meta = {
        "dataset": "Voxel51/FloorPlanCAD",
        "source": "https://huggingface.co/datasets/Voxel51/FloorPlanCAD",
        "license": "CC-BY-SA 4.0",
        "num_samples": len(records),
        "samples": records,
    }
    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    try:
        import datasets  # noqa: F401
    except ImportError:
        print("[ERROR] Run: pip install datasets")
        raise

    download_floorplan_cad(output_dir="./data/FloorPlanCAD")

