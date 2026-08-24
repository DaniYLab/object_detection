from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

import train
from src.data import NUM_CLASSES
from src.training.checkpoint import CHECKPOINT_SCHEMA_VERSION, CheckpointError
from src.training.losses import (
    centernet_loss as canonical_centernet_loss,
    focal_loss as canonical_focal_loss,
    l1_loss_masked as canonical_l1_loss_masked,
)
from src.training.reproducibility import make_generator


class _WeightedDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> int:
        return index

    def get_sample_weights(self, balance_power: float = 0.5) -> torch.Tensor:
        return torch.full((len(self),), balance_power, dtype=torch.double)


def test_parser_defaults_to_floorplan_base_and_reexports_losses() -> None:
    args = train.build_argument_parser().parse_args([])
    config = train.resolve_model_config_from_args(args)

    assert args.preset == "floorplan_base"
    assert config.image_size == 512
    assert config.model_dim == 256
    assert config.num_classes == NUM_CLASSES
    assert config.output_stride == 8
    assert args.val_ap_interval == 5
    assert args.val_ap_chunk_size == 4
    assert args.limit_val_ap_images == 0
    assert args.validate_sources is False
    assert args.precision == "fp32"
    assert train.focal_loss is canonical_focal_loss
    assert train.l1_loss_masked is canonical_l1_loss_masked
    assert train.centernet_loss is canonical_centernet_loss


def test_legacy_model_overrides_resolve_into_validated_config() -> None:
    parser = train.build_argument_parser()
    args = parser.parse_args(
        [
            "--model_dim",
            "128",
            "--depth_per_class",
            "3",
            "--fusion_mode",
            "film_cross_attn",
            "--output_stride",
            "8",
        ]
    )
    config = train.resolve_model_config_from_args(args)

    assert config.model_dim == 128
    assert config.depth_per_class == 3
    assert config.fusion_mode == "film_cross_attn"
    assert config.output_stride == 8

    invalid = parser.parse_args(["--output_stride", "16"])
    with pytest.raises(ValueError, match="stride.*8|stride=8"):
        train.resolve_model_config_from_args(invalid)


def test_loaders_receive_the_seeded_generator_and_worker_initializer() -> None:
    args = train.build_argument_parser().parse_args(
        ["--num_workers", "0", "--sampler", "balanced"]
    )
    dataset = _WeightedDataset()
    generator = make_generator(123)

    loader = train._make_train_loader(
        dataset,
        dataset,  # type: ignore[arg-type]
        args,
        pin_memory=False,
        data_generator=generator,
    )

    assert loader.generator is generator
    assert loader.worker_init_fn is train.seed_worker
    assert isinstance(loader.sampler, WeightedRandomSampler)
    assert loader.sampler.generator is generator


def test_training_checkpoint_wrapper_builds_complete_schema_v2_payload() -> None:
    args = train.build_argument_parser().parse_args([])
    config = train.resolve_model_config_from_args(args)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    generator = make_generator(77)
    runtime_config = {"scheduler": {"total_steps": 10}}

    payload = train.build_training_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=2,
        global_step=20,
        best_metric=0.5,
        metrics={"val": {"val_loss": 0.5}},
        model_config=config,
        preset=args.preset,
        runtime_config=runtime_config,
        split_fingerprint="split-fingerprint",
        metadata_fingerprint="metadata-fingerprint",
        data_generator=generator,
    )

    assert payload["schema"]["version"] == CHECKPOINT_SCHEMA_VERSION
    assert payload["optimizer_state"] is not None
    assert payload["scheduler_state"] is not None
    assert payload["rng_state"]["data_generator"] is not None
    assert payload["preset"] == "floorplan_base"
    assert type(config).from_dict(payload["model_config"]).to_dict() == config.to_dict()
    assert payload["class_names"]
    assert payload["output_stride"] == 8
    assert payload["split_manifest_fingerprint"] == "split-fingerprint"
    assert payload["metadata_fingerprint"] == "metadata-fingerprint"
    assert payload["global_step"] == 20
    assert payload["metrics"]["val"]["val_loss"] == 0.5

    train.validate_exact_resume_compatibility(
        payload,
        runtime_config=runtime_config,
        split_fingerprint="split-fingerprint",
        metadata_fingerprint="metadata-fingerprint",
    )
    with pytest.raises(CheckpointError, match="Runtime configuration mismatch"):
        train.validate_exact_resume_compatibility(
            payload,
            runtime_config={"scheduler": {"total_steps": 11}},
            split_fingerprint="split-fingerprint",
            metadata_fingerprint="metadata-fingerprint",
        )


def test_weights_only_requires_resume() -> None:
    args = train.build_argument_parser().parse_args(["--weights-only"])
    with pytest.raises(ValueError, match="requires --resume"):
        train._validate_args(args)


def test_validation_ap_arguments_are_checked() -> None:
    parser = train.build_argument_parser()
    for option, value, message in (
        ("--val-ap-interval", "-1", "val_ap_interval"),
        ("--val-ap-chunk-size", "0", "val_ap_chunk_size"),
        ("--limit-val-ap-images", "-1", "limit_val_ap_images"),
    ):
        args = parser.parse_args([option, value])
        with pytest.raises(ValueError, match=message):
            train._validate_args(args)


def test_metadata_source_path_prefers_schema_v2_source_entry(tmp_path) -> None:
    metadata_path = tmp_path / "sample_meta.json"
    resolved = train._metadata_source_path(
        {"source": {"svg": {"path": "sample.svg"}}},
        metadata_path,
        "svg",
        tmp_path / "fallback.svg",
    )
    assert resolved == tmp_path / "sample.svg"


def test_precision_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    assert train._resolve_precision("auto", torch.device("cpu")) == "fp32"
    with pytest.raises(ValueError, match="FP16.*CUDA"):
        train._resolve_precision("fp16", torch.device("cpu"))

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert train._resolve_precision("auto", torch.device("cuda")) == "bf16"
    assert train._build_grad_scaler(torch.device("cuda"), "bf16") is None
