from __future__ import annotations

from types import SimpleNamespace

import torch

from src.models.conditioning import (
    ByteTextConditioner,
    ClassEmbeddingConditioner,
    LazyHFTextConditioner,
    NoConditioner,
    masked_mean,
    materialize_conditioners_for_state_dict,
    materialize_pretrained_conditioners,
)


def test_masked_mean_ignores_padding_and_handles_empty_rows() -> None:
    tokens = torch.tensor(
        [[[1.0, 3.0], [3.0, 5.0], [99.0, 99.0]], [[7.0, 9.0], [8.0, 8.0], [9.0, 9.0]]]
    )
    mask = torch.tensor([[True, True, False], [False, False, False]])
    pooled = masked_mean(tokens, mask)
    torch.testing.assert_close(pooled[0], torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(pooled[1], torch.zeros(2))


def test_no_and_class_embedding_conditioners_expose_masks() -> None:
    no_conditioner = NoConditioner(model_dim=8)
    empty = no_conditioner(batch_size=2)
    assert empty.tokens.shape == (2, 1, 8)
    assert not empty.attention_mask.any()
    assert torch.count_nonzero(empty.pooled) == 0

    class_conditioner = ClassEmbeddingConditioner(num_classes=3, model_dim=8).eval()
    with torch.no_grad():
        conditioned = class_conditioner(class_ids=torch.tensor([2, 0]))
    assert conditioned.tokens.shape == (2, 1, 8)
    assert conditioned.attention_mask.all()
    torch.testing.assert_close(conditioned.tokens[:, 0], conditioned.pooled)


def test_byte_conditioner_uses_utf8_masks_and_masked_pooling() -> None:
    conditioner = ByteTextConditioner(model_dim=12, max_length=6, embedding_dim=8).eval()
    with torch.no_grad():
        output = conditioner(["A", "é", ""])
    assert output.attention_mask.sum(dim=1).tolist() == [1, 2, 0]
    assert torch.count_nonzero(output.tokens[~output.attention_mask]) == 0
    torch.testing.assert_close(output.pooled, masked_mean(output.tokens, output.attention_mask))
    torch.testing.assert_close(output.pooled[2], torch.zeros(12))


def test_hf_conditioner_is_lazy_and_does_not_download_on_construction() -> None:
    conditioner = LazyHFTextConditioner(
        "unused/test-model",
        model_dim=8,
        max_length=4,
        local_files_only=True,
    )
    assert not conditioner.is_loaded
    assert conditioner.hf_model is None
    assert conditioner.tokenizer is None


class _FakeHFEncoder(torch.nn.Module):
    def __init__(self, hidden_size: int = 6) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = torch.nn.Embedding(11, hidden_size)


def test_hf_materialize_uses_injected_backend_without_transformers(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def loader(model_name: str, kwargs) -> tuple[object, torch.nn.Module]:
        calls.append((model_name, dict(kwargs)))
        return object(), _FakeHFEncoder()

    def fail_import(_name: str):
        raise AssertionError("transformers import is forbidden")

    monkeypatch.setattr("src.models.conditioning.import_module", fail_import)
    conditioner = LazyHFTextConditioner(
        "fake/local-model",
        model_dim=8,
        freeze=True,
        local_files_only=True,
        revision="test-revision",
        backend_loader=loader,
    )

    returned = conditioner.materialize()
    conditioner.materialize()

    assert returned is conditioner
    assert conditioner.is_loaded
    assert conditioner.projection.weight.shape == (8, 6)
    assert calls == [
        (
            "fake/local-model",
            {
                "local_files_only": True,
                "trust_remote_code": False,
                "revision": "test-revision",
            },
        )
    ]
    assert conditioner.hf_model is not None
    assert not any(parameter.requires_grad for parameter in conditioner.hf_model.parameters())
    conditioner.train()
    assert not conditioner.hf_model.training


def test_materialize_pretrained_conditioners_registers_trainable_hf_parameters() -> None:
    holder = torch.nn.Module()
    holder.conditioner = LazyHFTextConditioner(
        "fake/trainable-model",
        model_dim=8,
        freeze=False,
        backend_loader=lambda _name, _kwargs: (object(), _FakeHFEncoder()),
    )

    assert holder.conditioner.hf_model is None
    assert materialize_conditioners_for_state_dict(
        holder,
        {"unrelated.weight": torch.ones(1)},
    ) == ()
    assert holder.conditioner.hf_model is None
    assert materialize_pretrained_conditioners(holder) == ("conditioner",)
    assert holder.conditioner.hf_model is not None

    optimizer = torch.optim.AdamW(
        (parameter for parameter in holder.parameters() if parameter.requires_grad),
        lr=1e-3,
    )
    optimizer_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert all(
        id(parameter) in optimizer_parameters
        for parameter in holder.conditioner.hf_model.parameters()
    )
