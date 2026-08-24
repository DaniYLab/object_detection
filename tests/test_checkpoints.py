from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
import torch

from src.models.conditioning import (
    LazyHFTextConditioner,
    materialize_pretrained_conditioners,
)
from src.training.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointError,
    build_checkpoint,
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
)
from src.training.reproducibility import make_generator, seed_everything


def _objects():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    return model, optimizer, scheduler


class _FakeHFEncoder(torch.nn.Module):
    def __init__(self, hidden_size: int = 5) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = torch.nn.Embedding(13, hidden_size)


def _hf_holder(*, freeze: bool) -> torch.nn.Module:
    holder = torch.nn.Module()
    holder.conditioner = LazyHFTextConditioner(
        "fake/checkpoint-model",
        model_dim=7,
        freeze=freeze,
        backend_loader=lambda _name, _kwargs: (object(), _FakeHFEncoder()),
    )
    return holder


def test_complete_checkpoint_round_trip_without_training(tmp_path) -> None:
    seed_everything(1337)
    generator = make_generator(99)
    model, optimizer, scheduler = _objects()
    expected_weight = model.weight.detach().clone()

    payload = build_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=4,
        global_step=123,
        best_metric=0.25,
        metrics={"AP50": 0.25},
        model_config={"model_dim": 16, "output_stride": 8},
        preset="tiny",
        class_names=["chair", "table"],
        output_stride=8,
        data_generator=generator,
    )
    path = save_checkpoint(tmp_path / "checkpoint.pt", payload)

    with torch.no_grad():
        model.weight.zero_()
    loaded = load_checkpoint(path)
    resume = restore_training_state(
        loaded,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        data_generator=generator,
        expected_model_config={"model_dim": 16, "output_stride": 8},
        expected_class_names=["chair", "table"],
        expected_output_stride=8,
    )

    assert loaded["schema"]["version"] == CHECKPOINT_SCHEMA_VERSION
    assert torch.equal(model.weight, expected_weight)
    assert resume == {
        "start_epoch": 5,
        "global_step": 123,
        "best_metric": 0.25,
        "legacy": False,
    }


def test_weights_only_materializes_hf_keys_before_strict_load() -> None:
    source = _hf_holder(freeze=True)
    materialize_pretrained_conditioners(source)
    payload = build_checkpoint(
        model=source,
        epoch=0,
        global_step=0,
        model_config={"conditioner": "fake-frozen"},
        preset="fake",
        class_names=["chair"],
        output_stride=8,
    )
    assert any(key.startswith("conditioner.hf_model.") for key in payload["model_state"])

    target = _hf_holder(freeze=True)
    assert target.conditioner.hf_model is None
    result = restore_training_state(
        payload,
        model=target,
        expected_model_config={"conditioner": "fake-frozen"},
        expected_class_names=["chair"],
        expected_output_stride=8,
        weights_only=True,
        strict=True,
    )

    assert result["legacy"] is False
    assert target.conditioner.hf_model is not None
    for key, expected in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], expected)


def test_exact_resume_restores_materialized_trainable_hf_without_step() -> None:
    source = _hf_holder(freeze=False)
    materialize_pretrained_conditioners(source)
    source_optimizer = torch.optim.AdamW(
        (parameter for parameter in source.parameters() if parameter.requires_grad),
        lr=1e-3,
    )
    payload = build_checkpoint(
        model=source,
        optimizer=source_optimizer,
        epoch=3,
        global_step=17,
        best_metric=0.4,
        model_config={"conditioner": "fake-trainable"},
        preset="fake",
        class_names=["chair"],
        output_stride=8,
    )

    target = _hf_holder(freeze=False)
    materialize_pretrained_conditioners(target)
    target_optimizer = torch.optim.AdamW(
        (parameter for parameter in target.parameters() if parameter.requires_grad),
        lr=1e-3,
    )
    result = restore_training_state(
        payload,
        model=target,
        optimizer=target_optimizer,
        expected_model_config={"conditioner": "fake-trainable"},
        expected_class_names=["chair"],
        expected_output_stride=8,
        strict=True,
    )

    assert result == {
        "start_epoch": 4,
        "global_step": 17,
        "best_metric": 0.4,
        "legacy": False,
    }
    source_hf_parameter_ids = {
        id(parameter) for parameter in source.conditioner.hf_model.parameters()
    }
    optimizer_parameter_ids = {
        id(parameter)
        for group in source_optimizer.param_groups
        for parameter in group["params"]
    }
    assert source_hf_parameter_ids <= optimizer_parameter_ids
    for key, expected in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], expected)


def test_resume_rejects_incompatible_config(tmp_path) -> None:
    model, optimizer, scheduler = _objects()
    payload = build_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=0,
        global_step=0,
        model_config={"model_dim": 16},
        preset="tiny",
        class_names=["chair"],
        output_stride=8,
    )
    loaded = load_checkpoint(save_checkpoint(tmp_path / "checkpoint.pt", payload))

    with pytest.raises(CheckpointError, match="configuration mismatch"):
        restore_training_state(
            loaded,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_model_config={"model_dim": 32},
            expected_class_names=["chair"],
            expected_output_stride=8,
        )


def test_pre_schema_checkpoint_is_rejected_before_state_loading(tmp_path) -> None:
    model, _, _ = _objects()
    path = tmp_path / "pre_schema.pt"
    torch.save({"model_state": model.state_dict(), "epoch": 2}, path)

    with pytest.raises(
        CheckpointError,
        match="pre-schema.*cannot be migrated automatically",
    ):
        load_checkpoint(path)

    with pytest.raises(
        CheckpointError,
        match="pre-schema.*cannot be migrated automatically",
    ):
        restore_training_state(
            {"model_state": {"unexpected.old_key": torch.ones(1)}, "_legacy": True},
            model=model,
            weights_only=True,
        )


def test_checkpoint_restores_python_rng(tmp_path) -> None:
    seed_everything(5)
    model, _, _ = _objects()
    payload = build_checkpoint(
        model=model,
        epoch=0,
        global_step=0,
        model_config={},
        preset="tiny",
        class_names=["chair"],
        output_stride=8,
    )
    expected = random.random()
    loaded = load_checkpoint(save_checkpoint(tmp_path / "rng.pt", payload))

    restore_training_state(
        loaded,
        model=model,
        expected_model_config={},
        expected_class_names=["chair"],
        expected_output_stride=8,
    )
    assert random.random() == expected
