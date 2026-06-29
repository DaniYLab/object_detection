"""
Download FloorPlanCAD dataset goc tu Google Drive.
Source: https://floorplancad.github.io/
- 15,663 CAD drawings voi day du annotations
- License: CC-BY-SA 4.0

Usage:
    python scripts/data/download_gdrive.py
    python scripts/data/download_gdrive.py --output_dir /content/FloorPlanCAD_orig
    OUTPUT_DIR=/content/FloorPlanCAD_orig python scripts/data/download_gdrive.py
"""

import os
import zipfile
from pathlib import Path

# Google Drive file IDs tu floorplancad.github.io
GDRIVE_FILES = [
    {
        "id": "1HcyKt6qWeXog-tRfvEjdO3O3TN91PXGL",
        "name": "train_set_1.zip",
        "desc": "Train set 1",
    },
    {
        "id": "1kSS7OB_EEu7VJzb0W8DK9_nu1EvshioV",
        "name": "train_set_2.zip",
        "desc": "Train set 2",
    },
    {
        "id": "1jxpYgxnLUbXEzMOsjaMPQFSuvmvHimiZ",
        "name": "test_set.zip",
        "desc": "Test set",
    },
]


def download_from_gdrive(file_id: str, output_path: Path) -> bool:
    """Download a file from Google Drive using gdown."""
    import gdown

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"  Downloading: {output_path.name}")
    try:
        gdown.download(url, str(output_path), quiet=False)
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def detect_format(path: Path) -> str:
    """Detect file format from magic bytes."""
    with open(path, "rb") as f:
        magic = f.read(6)
    if magic[:4] == b"PK\x03\x04":
        return "zip"
    if magic[:6] == b"\xfd7zXZ\x00" or magic[:5] == b"y7zXZ":
        return "xz"
    if magic[:2] == b"\x1f\x8b":
        return "gz"
    return "unknown"


def extract_archive(archive_path: Path, extract_to: Path) -> None:
    """Extract zip, tar.gz, or tar.xz archive."""
    import tarfile

    fmt = detect_format(archive_path)
    print(f"  Format detected: {fmt}")
    os.makedirs(extract_to, exist_ok=True)

    if fmt == "zip":
        import zipfile
        with zipfile.ZipFile(archive_path, "r") as zf:
            total = len(zf.namelist())
            zf.extractall(extract_to)
            print(f"  Extracted {total} files.")
    elif fmt in ("xz", "gz"):
        mode = "r:xz" if fmt == "xz" else "r:gz"
        with tarfile.open(archive_path, mode) as tf:
            members = tf.getmembers()
            tf.extractall(extract_to)
            print(f"  Extracted {len(members)} files.")
    else:
        # Try tarfile auto-detection as fallback
        try:
            with tarfile.open(archive_path, "r:*") as tf:
                members = tf.getmembers()
                tf.extractall(extract_to)
                print(f"  Extracted {len(members)} files.")
        except Exception as e:
            print(f"  [ERROR] Cannot extract: {e}")
            raise


def main(output_dir: Path) -> None:
    # Use os.makedirs — safe with symlinks (unlike Path.mkdir on Python 3.12+)
    os.makedirs(output_dir, exist_ok=True)
    zip_dir = output_dir / "zips"
    os.makedirs(zip_dir, exist_ok=True)

    print("=" * 60)
    print("  FloorPlanCAD - Google Drive Download")
    print("  Source: https://floorplancad.github.io/")
    print(f"  Output : {output_dir.resolve()}")
    print("  Total: ~15,663 CAD drawings with annotations")
    print("=" * 60)

    for i, file_info in enumerate(GDRIVE_FILES, 1):
        print(f"\n[{i}/{len(GDRIVE_FILES)}] {file_info['desc']}")
        zip_path = zip_dir / file_info["name"]

        # Skip if already downloaded
        if zip_path.exists() and zip_path.stat().st_size > 10_000:
            print(f"  Already downloaded: {zip_path.name} "
                  f"({zip_path.stat().st_size / 1e6:.1f} MB) - skipping")
        else:
            success = download_from_gdrive(file_info["id"], zip_path)
            if not success:
                print(f"  [SKIP] Failed to download {file_info['name']}")
                continue

        # Extract
        extract_to = output_dir / file_info["name"].replace(".zip", "")
        if extract_to.exists() and any(extract_to.iterdir()):
            print(f"  Already extracted to: {extract_to} - skipping")
        else:
            print(f"  Extracting: {zip_path.name} -> {extract_to}")
            extract_archive(zip_path, extract_to)

    # Summary
    print(f"\n{'=' * 60}")
    print("  DONE! Dataset structure:")
    for d in sorted(output_dir.iterdir()):
        if d.is_dir() and d.name != "zips":
            n_files = sum(1 for _ in d.rglob("*") if _.is_file())
            print(f"  {d.name}/  ({n_files} files)")
    print(f"{'=' * 60}")
    print(f"\n  Full path: {output_dir.resolve()}")


if __name__ == "__main__":
    import argparse

    try:
        import gdown  # noqa: F401
    except ImportError:
        print("[ERROR] Run: pip install gdown")
        raise

    parser = argparse.ArgumentParser(description="Download FloorPlanCAD from Google Drive")
    parser.add_argument(
        "--output_dir",
        default=os.environ.get("OUTPUT_DIR", "./data/FloorPlanCAD_original"),
        help="Directory to save downloaded data (default: ./data/FloorPlanCAD_original)",
    )
    args = parser.parse_args()

    main(output_dir=Path(args.output_dir))

