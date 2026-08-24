"""Conditioning modules for class IDs, lightweight text, and pretrained text."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import torch
import torch.nn as nn

from .config import ConditionerConfig


@dataclass
class ConditioningOutput:
    """Token-level and pooled conditioning with a valid-token mask."""

    tokens: torch.Tensor
    attention_mask: torch.Tensor
    pooled: torch.Tensor

    @property
    def token_embeddings(self) -> torch.Tensor:
        return self.tokens

    @property
    def pooled_embedding(self) -> torch.Tensor:
        return self.pooled

    @property
    def mask(self) -> torch.Tensor:
        return self.attention_mask

    def __iter__(self):
        yield self.tokens
        yield self.attention_mask
        yield self.pooled


def masked_mean(tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool ``[B, L, D]`` tokens over valid positions without NaNs."""
    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape [B,L,D], got {tuple(tokens.shape)}")
    if attention_mask.shape != tokens.shape[:2]:
        raise ValueError(
            "attention_mask must match token batch/length dimensions: "
            f"tokens={tuple(tokens.shape)}, mask={tuple(attention_mask.shape)}"
        )
    mask = attention_mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
    numerator = (tokens * mask).sum(dim=1)
    denominator = mask.sum(dim=1).clamp_min(1.0)
    return numerator / denominator


def _normalise_texts(texts: str | Sequence[str] | None) -> list[str] | None:
    if texts is None:
        return None
    if isinstance(texts, str):
        return [texts]
    values = list(texts)
    if not all(isinstance(text, str) for text in values):
        raise TypeError("texts must be a string or a sequence of strings")
    return values


def _infer_batch_size(
    texts: list[str] | None,
    class_ids: torch.Tensor | None,
    batch_size: int | None,
) -> int:
    sizes = []
    if texts is not None:
        sizes.append(len(texts))
    if class_ids is not None:
        if class_ids.ndim != 1:
            raise ValueError("class_ids must have shape [B]")
        sizes.append(class_ids.shape[0])
    if batch_size is not None:
        sizes.append(batch_size)
    if not sizes:
        raise ValueError("batch_size, texts, or class_ids is required")
    if any(size != sizes[0] for size in sizes[1:]):
        raise ValueError(f"inconsistent conditioning batch sizes: {sizes}")
    if sizes[0] <= 0:
        raise ValueError("conditioning batch size must be positive")
    return sizes[0]


class NoConditioner(nn.Module):
    """Return an explicit empty conditioning signal with no learned parameters."""

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.model_dim = model_dim

    def forward(
        self,
        texts: str | Sequence[str] | None = None,
        class_ids: torch.Tensor | None = None,
        *,
        batch_size: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ConditioningOutput:
        normalised = _normalise_texts(texts)
        batch = _infer_batch_size(normalised, class_ids, batch_size)
        if device is None and class_ids is not None:
            device = class_ids.device
        dtype = dtype or torch.get_default_dtype()
        tokens = torch.zeros(batch, 1, self.model_dim, device=device, dtype=dtype)
        mask = torch.zeros(batch, 1, device=device, dtype=torch.bool)
        pooled = torch.zeros(batch, self.model_dim, device=device, dtype=dtype)
        return ConditioningOutput(tokens=tokens, attention_mask=mask, pooled=pooled)


class ClassEmbeddingConditioner(nn.Module):
    """Condition on a learned class embedding represented as one valid token."""

    def __init__(self, num_classes: int, model_dim: int) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.num_classes = num_classes
        self.model_dim = model_dim
        self.embedding = nn.Embedding(num_classes, model_dim)

    def forward(
        self,
        texts: str | Sequence[str] | None = None,
        class_ids: torch.Tensor | None = None,
        *,
        batch_size: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ConditioningOutput:
        del texts, device, dtype
        if class_ids is None:
            raise ValueError("ClassEmbeddingConditioner requires class_ids")
        _infer_batch_size(None, class_ids, batch_size)
        class_ids = class_ids.to(device=self.embedding.weight.device, dtype=torch.long)
        if torch.any(class_ids < 0) or torch.any(class_ids >= self.num_classes):
            raise ValueError(f"class_ids must be in [0, {self.num_classes})")
        pooled = self.embedding(class_ids)
        tokens = pooled.unsqueeze(1)
        mask = torch.ones(tokens.shape[:2], device=tokens.device, dtype=torch.bool)
        return ConditioningOutput(tokens=tokens, attention_mask=mask, pooled=pooled)


class ByteTextConditioner(nn.Module):
    """Small UTF-8 byte-token text encoder with deterministic local tokenization."""

    def __init__(
        self,
        model_dim: int,
        max_length: int = 64,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        if max_length <= 0 or embedding_dim <= 0:
            raise ValueError("max_length and embedding_dim must be positive")
        self.model_dim = model_dim
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        # Token 0 is padding; UTF-8 bytes 0..255 map to IDs 1..256.
        self.token_embedding = nn.Embedding(257, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.norm = nn.LayerNorm(model_dim)

    def tokenize(
        self,
        texts: str | Sequence[str],
        *,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = _normalise_texts(texts)
        if values is None or not values:
            raise ValueError("ByteTextConditioner requires at least one text")
        input_ids = torch.zeros(len(values), self.max_length, dtype=torch.long, device=device)
        attention_mask = torch.zeros(len(values), self.max_length, dtype=torch.bool, device=device)
        for row, text in enumerate(values):
            encoded = text.encode("utf-8")[: self.max_length]
            if encoded:
                ids = torch.tensor([byte + 1 for byte in encoded], dtype=torch.long, device=device)
                input_ids[row, : len(encoded)] = ids
                attention_mask[row, : len(encoded)] = True
        return input_ids, attention_mask

    def encode_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> ConditioningOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B,L]")
        batch, length = input_ids.shape
        if length > self.max_length:
            raise ValueError(f"token length {length} exceeds max_length={self.max_length}")
        if attention_mask is None:
            attention_mask = input_ids.ne(0)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        tokens = self.norm(self.projection(hidden))
        tokens = tokens * attention_mask.unsqueeze(-1).to(tokens.dtype)
        pooled = masked_mean(tokens, attention_mask)
        return ConditioningOutput(tokens=tokens, attention_mask=attention_mask, pooled=pooled)

    def forward(
        self,
        texts: str | Sequence[str] | None = None,
        class_ids: torch.Tensor | None = None,
        *,
        batch_size: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ConditioningOutput:
        del class_ids, dtype
        values = _normalise_texts(texts)
        if values is None:
            raise ValueError("ByteTextConditioner requires texts")
        _infer_batch_size(values, None, batch_size)
        if device is None:
            device = self.token_embedding.weight.device
        input_ids, attention_mask = self.tokenize(values, device=device)
        return self.encode_ids(input_ids, attention_mask)


class LazyHFTextConditioner(nn.Module):
    """Lazily load an optional Hugging Face encoder when it is required.

    Construction does not import ``transformers``, access the network, or
    allocate pretrained weights. ``materialize()`` is the explicit boundary for
    checkpoint restoration and optimizer setup; ordinary frozen inference can
    remain lazy until the first forward call.
    """

    def __init__(
        self,
        model_name: str,
        model_dim: int,
        max_length: int = 64,
        *,
        freeze: bool = True,
        local_files_only: bool = False,
        revision: str | None = None,
        backend_loader: Callable[
            [str, Mapping[str, object]], tuple[Any, nn.Module]
        ]
        | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.model_dim = model_dim
        self.max_length = max_length
        self.freeze = freeze
        self.local_files_only = local_files_only
        self.revision = revision
        self.projection = nn.LazyLinear(model_dim, bias=False)
        self.norm = nn.LayerNorm(model_dim)
        self.hf_model: nn.Module | None = None
        self.tokenizer = None
        self._backend_loader = backend_loader

    @property
    def is_loaded(self) -> bool:
        return self.hf_model is not None

    @staticmethod
    def _load_transformers_backend(
        model_name: str,
        kwargs: Mapping[str, object],
    ) -> tuple[Any, nn.Module]:
        try:
            transformers = import_module("transformers")
        except ImportError as exc:
            raise ImportError(
                "LazyHFTextConditioner requires optional dependencies. "
                "Install requirements-pretrained.txt."
            ) from exc
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, **dict(kwargs))
        model = transformers.AutoModel.from_pretrained(model_name, **dict(kwargs))
        return tokenizer, model

    @staticmethod
    def _infer_hidden_size(model: nn.Module) -> int:
        config = getattr(model, "config", None)
        for name in ("hidden_size", "d_model", "dim", "n_embd"):
            value = getattr(config, name, None)
            if isinstance(value, int) and value > 0:
                return value

        get_embeddings = getattr(model, "get_input_embeddings", None)
        if callable(get_embeddings):
            embeddings = get_embeddings()
            embedding_dim = getattr(embeddings, "embedding_dim", None)
            if isinstance(embedding_dim, int) and embedding_dim > 0:
                return embedding_dim
            weight = getattr(embeddings, "weight", None)
            if isinstance(weight, torch.Tensor) and weight.ndim == 2:
                return int(weight.shape[1])

        raise RuntimeError(
            "Could not infer the pretrained encoder hidden size needed to "
            "materialize the conditioner projection"
        )

    def _materialize_projection(self, input_dim: int) -> None:
        if input_dim <= 0:
            raise ValueError("projection input_dim must be positive")
        projection = self.projection
        has_uninitialized = getattr(projection, "has_uninitialized_params", None)
        if callable(has_uninitialized) and has_uninitialized():
            reference = projection.weight
            sample = torch.empty(
                1,
                input_dim,
                device=reference.device,
                dtype=reference.dtype,
            )
            projection.initialize_parameters(sample)
            return
        if projection.weight.shape[1] != input_dim:
            raise RuntimeError(
                "Pretrained encoder hidden size does not match the materialized "
                f"projection: encoder={input_dim}, projection={projection.weight.shape[1]}"
            )

    def materialize(
        self,
        *,
        projection_input_dim: int | None = None,
    ) -> LazyHFTextConditioner:
        """Load/register the tokenizer and encoder and initialize the projection.

        ``projection_input_dim`` is normally inferred from the encoder config.
        Checkpoint restoration may pass the saved projection width so strict
        loading does not depend on provider-specific config attribute names.
        The operation is idempotent.
        """
        if self.hf_model is not None:
            if self.tokenizer is None:
                raise RuntimeError("hf_model is registered but tokenizer is missing")
            input_dim = projection_input_dim or self._infer_hidden_size(self.hf_model)
            self._materialize_projection(input_dim)
            return self

        kwargs: dict[str, object] = {
            "local_files_only": self.local_files_only,
            "trust_remote_code": False,
        }
        if self.revision is not None:
            kwargs["revision"] = self.revision
        loader = self._backend_loader or self._load_transformers_backend
        tokenizer, model = loader(self.model_name, kwargs)
        if tokenizer is None or not isinstance(model, nn.Module):
            raise TypeError("backend_loader must return a tokenizer and torch.nn.Module")

        input_dim = projection_input_dim or self._infer_hidden_size(model)
        self._materialize_projection(input_dim)
        if self.freeze:
            model.requires_grad_(False)
            model.eval()
        model = model.to(self.projection.weight.device)
        self.tokenizer = tokenizer
        self.hf_model = model
        return self

    def _load_pretrained(self) -> None:
        self.materialize()

    def train(self, mode: bool = True) -> LazyHFTextConditioner:
        super().train(mode)
        if self.freeze and self.hf_model is not None:
            self.hf_model.eval()
        return self

    def forward(
        self,
        texts: str | Sequence[str] | None = None,
        class_ids: torch.Tensor | None = None,
        *,
        batch_size: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ConditioningOutput:
        del class_ids, dtype
        values = _normalise_texts(texts)
        if values is None:
            raise ValueError("LazyHFTextConditioner requires texts")
        _infer_batch_size(values, None, batch_size)
        self._load_pretrained()
        assert self.tokenizer is not None and self.hf_model is not None

        model_device = next(self.hf_model.parameters()).device
        encoded = self.tokenizer(
            values,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {name: value.to(model_device) for name, value in encoded.items()}
        if self.freeze:
            with torch.no_grad():
                hidden = self.hf_model(**encoded).last_hidden_state
        else:
            hidden = self.hf_model(**encoded).last_hidden_state
        tokens = self.norm(self.projection(hidden))
        attention_mask = encoded.get("attention_mask", torch.ones(tokens.shape[:2], device=model_device))
        attention_mask = attention_mask.to(dtype=torch.bool)
        tokens = tokens * attention_mask.unsqueeze(-1).to(tokens.dtype)
        pooled = masked_mean(tokens, attention_mask)
        if device is not None and torch.device(device) != tokens.device:
            tokens = tokens.to(device)
            attention_mask = attention_mask.to(device)
            pooled = pooled.to(device)
        return ConditioningOutput(tokens=tokens, attention_mask=attention_mask, pooled=pooled)


def materialize_pretrained_conditioners(module: nn.Module) -> tuple[str, ...]:
    """Materialize every lazy pretrained conditioner before optimizer setup."""
    materialized: list[str] = []
    for name, child in list(module.named_modules()):
        if isinstance(child, LazyHFTextConditioner):
            child.materialize()
            materialized.append(name)
    return tuple(materialized)


def materialize_conditioners_for_state_dict(
    module: nn.Module,
    state_dict: Mapping[str, Any],
) -> tuple[str, ...]:
    """Materialize only conditioners represented by HF keys in ``state_dict``."""
    materialized: list[str] = []
    state_keys = tuple(state_dict)
    for name, child in list(module.named_modules()):
        if not isinstance(child, LazyHFTextConditioner):
            continue
        prefix = f"{name}." if name else ""
        hf_prefix = f"{prefix}hf_model."
        if not any(key.startswith(hf_prefix) for key in state_keys):
            continue

        projection_input_dim = None
        projection_weight = state_dict.get(f"{prefix}projection.weight")
        if isinstance(projection_weight, torch.Tensor):
            try:
                if projection_weight.ndim == 2:
                    projection_input_dim = int(projection_weight.shape[1])
            except RuntimeError:
                # An uninitialized lazy checkpoint tensor has no usable shape;
                # fall back to the encoder config during materialization.
                pass
        child.materialize(projection_input_dim=projection_input_dim)
        materialized.append(name)
    return tuple(materialized)


# Concise aliases for callers that prefer the provider-oriented name.
HFPretrainedConditioner = LazyHFTextConditioner
LightweightTextConditioner = ByteTextConditioner


def build_conditioner(
    config: ConditionerConfig | str,
    *,
    model_dim: int,
    num_classes: int,
) -> nn.Module:
    """Build a conditioner from a serializable config or kind string."""
    if isinstance(config, str):
        config = ConditionerConfig(kind=config)
    if config.kind == "none":
        return NoConditioner(model_dim)
    if config.kind == "class_embedding":
        return ClassEmbeddingConditioner(num_classes, model_dim)
    if config.kind == "lightweight_text":
        return ByteTextConditioner(
            model_dim=model_dim,
            max_length=config.max_length,
            embedding_dim=config.embedding_dim,
        )
    if config.kind == "pretrained_text":
        return LazyHFTextConditioner(
            model_name=config.model_name,
            model_dim=model_dim,
            max_length=config.max_length,
            freeze=config.freeze_pretrained,
            local_files_only=config.local_files_only,
            revision=config.revision,
        )
    raise ValueError(f"unsupported conditioner kind: {config.kind}")
