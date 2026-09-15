import json

import pytest
from PIL import Image

from src.data.archcad_dataset import ArchCADImageDataset
from src.data.transforms import PairedTrainTransform


def test_archcad_vector_strokes_follow_raster_flip(tmp_path):
    root = tmp_path / "ArchCAD"
    (root / "png").mkdir(parents=True)
    (root / "metadata").mkdir()
    Image.new("RGB", (100, 100), "white").save(root / "png" / "sample.png")
    metadata = {
        "image_size": [100, 100],
        "instances": [],
        "strokes": [
            [0.10, 0.20, 0.30, 0.40, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ],
    }
    (root / "metadata" / "sample_meta.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    manifest = {"splits": {"train": {"images": ["sample"]}}}
    (root / "splits.json").write_text(json.dumps(manifest), encoding="utf-8")
    transform = PairedTrainTransform(
        image_size=100,
        horizontal_flip_probability=1.0,
        vertical_flip_probability=0.0,
        rotation_probability=0.0,
        photometric_probability=0.0,
        scale_probability=0.0,
    )
    dataset = ArchCADImageDataset(
        root,
        split="train",
        vector_mode=True,
        n_max_strokes=4,
        transform=transform,
    )

    sample = dataset[0]

    assert sample["strokes"][0, 0].item() == pytest.approx(0.90)
    assert sample["strokes"][0, 2].item() == pytest.approx(0.70)
    assert sample["stroke_mask"].tolist() == [True, False, False, False]
