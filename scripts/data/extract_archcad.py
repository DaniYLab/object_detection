"""Extract ArchCAD-400K raw zip files into structured directories."""

import os
import time
import zipfile
from pathlib import Path


def extract_zip(
    zip_path: Path, dest_dir: Path, expected_count: int | None = None
) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(dest_dir.glob("*.*")))
    if expected_count is not None and existing >= expected_count:
        print(
            f"[{zip_path.name}] Already extracted ({existing} files in {dest_dir}). Skipping."
        )
        return

    print(f"[{zip_path.name}] Extracting to {dest_dir}...")
    t0 = time.time()
    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.namelist()
        total = len(members)
        # Flatten directory if files are inside a subfolder like 'png/' or 'json/'
        for i, member in enumerate(members):
            filename = os.path.basename(member)
            if not filename:
                continue
            source = z.open(member)
            target = open(dest_dir / filename, "wb")
            with source, target:
                target.write(source.read())

            if (i + 1) % 10000 == 0 or i + 1 == total:
                elapsed = time.time() - t0
                rate = (i + 1) / max(1e-5, elapsed)
                print(f"  Extracted {i + 1}/{total} files ({rate:.1f} files/s)...")

    elapsed = time.time() - t0
    final_count = len(list(dest_dir.glob("*.*")))
    print(f"[{zip_path.name}] Done! Extracted {final_count} files in {elapsed:.1f}s.\n")


def main() -> None:
    raw_dir = Path("data/ArchCAD_raw/data")
    dest_root = Path("data/ArchCAD")

    dest_root.mkdir(parents=True, exist_ok=True)

    tasks = [
        (raw_dir / "json.zip", dest_root / "json", 41097),
        (raw_dir / "svg.zip", dest_root / "svg", 41097),
        (raw_dir / "png.zip", dest_root / "png", 41097),
    ]

    for zip_path, dest_dir, expected_count in tasks:
        extract_zip(zip_path, dest_dir, expected_count)

    print("All primary ArchCAD modalities extracted successfully!")


if __name__ == "__main__":
    main()
