"""tinther — local drop-in for the `tinker` SDK.

Covers the API surface used (directly and transitively) by
``tinker_cookbook/distillation/train_on_policy.py`` and
``tinker_cookbook/distillation/train_off_policy.py``.

Backed by transformers + peft + torch + accelerate + vllm, with DDP.
Multi-GPU launch via ``accelerate launch`` or ``torchrun``.

Register as ``tinker`` before importing cookbook modules::

    import tinther
    tinther.install_as_tinker()
    from tinker_cookbook.distillation.train_on_policy import Config, main
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import time
import types as _types_module
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("tinther")

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Exceptions and simple type aliases
# ---------------------------------------------------------------------------


class TinkerError(Exception):
    """Raised for tinther/tinker-level failures."""


StopReason = str
LossFnType = Literal["cross_entropy", "importance_sampling", "ppo"]


# ---------------------------------------------------------------------------
# Data types (mirror tinker's public surface)
# ---------------------------------------------------------------------------


@dataclass
class EncodedTextChunk:
    """A chunk of already-encoded text tokens."""

    tokens: list[int]

    @property
    def length(self) -> int:
        return len(self.tokens)


@dataclass
class ImageChunk:
    """JPEG-encoded image chunk (stub; distillation scripts don't use images)."""

    data: bytes = b""
    width: int = 0
    height: int = 0

    @property
    def length(self) -> int:
        return 0


@dataclass
class ImageAssetPointerChunk:
    """Pointer to an uploaded image asset (stub)."""

    url: str = ""

    @property
    def length(self) -> int:
        return 0


ModelInputChunk = EncodedTextChunk  # runtime alias; type is broader at type-check time


@dataclass
class ModelInput:
    """Tokenized prompt built from one or more chunks."""

    chunks: list[ModelInputChunk] = field(default_factory=list)

    @classmethod
    def from_ints(cls, ints: list[int]) -> "ModelInput":
        return cls(chunks=[EncodedTextChunk(tokens=list(ints))])

    @classmethod
    def empty(cls) -> "ModelInput":
        return cls(chunks=[])

    def append_int(self, token_id: int) -> "ModelInput":
        new_chunks = [EncodedTextChunk(tokens=list(c.tokens)) for c in self.chunks]
        if new_chunks and isinstance(new_chunks[-1], EncodedTextChunk):
            new_chunks[-1].tokens.append(int(token_id))
        else:
            new_chunks.append(EncodedTextChunk(tokens=[int(token_id)]))
        return ModelInput(chunks=new_chunks)

    def to_ints(self) -> list[int]:
        out: list[int] = []
        for c in self.chunks:
            if isinstance(c, EncodedTextChunk):
                out.extend(c.tokens)
            else:
                raise TinkerError(f"Cannot flatten non-text chunk: {type(c).__name__}")
        return out

    @property
    def length(self) -> int:
        return sum(c.length for c in self.chunks)


@dataclass
class TensorData:
    """Numpy-backed tensor with metadata, round-trippable to torch."""

    data: np.ndarray
    dtype: str
    shape: tuple[int, ...]

    @classmethod
    def from_torch(cls, t: torch.Tensor) -> "TensorData":
        t = t.detach().to("cpu").contiguous()
        # numpy does not support bf16; promote to float32 for round-tripping.
        if t.dtype == torch.bfloat16:
            t = t.to(torch.float32)
        arr = t.numpy()
        return cls(data=arr, dtype=str(arr.dtype), shape=tuple(arr.shape))

    def to_torch(self) -> torch.Tensor:
        arr = np.asarray(self.data)
        # Honor the declared dtype (cookbook builds TensorData from Python lists
        # with dtype="float32" / "int64"; numpy would otherwise infer float64
        # from Python floats and break dtype-strict ops like Tensor.dot).
        if self.dtype:
            try:
                target = np.dtype(self.dtype)
            except TypeError:
                target = None
            if target is not None and arr.dtype != target:
                arr = arr.astype(target)
        return torch.from_numpy(arr)

    def tolist(self) -> list:
        """Mirror tinker.TensorData.tolist() — flatten to a nested Python list."""
        if isinstance(self.data, list):
            return list(self.data)
        return np.asarray(self.data).tolist()


@dataclass
class Datum:
    """A single training example."""

    model_input: ModelInput
    loss_fn_inputs: dict[str, TensorData]


@dataclass
class SamplingParams:
    """Sampling knobs."""

    max_tokens: int
    temperature: float = 1.0
    stop: Any | None = None  # list[str] | list[int] | None


@dataclass
class AdamParams:
    """Adam optimizer knobs."""

    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8


@dataclass
class _Sequence:
    tokens: list[int]
    logprobs: list[float] | None
    stop_reason: StopReason


@dataclass
class SampleResponse:
    sequences: list[_Sequence]
    topk_prompt_logprobs: list[list[tuple[int, float]] | None] | None = None


@dataclass
class ForwardBackwardOutput:
    loss_fn_outputs: list[dict[str, TensorData]]
    metrics: dict[str, float] | None = None


@dataclass
class OptimStepResponse:
    metrics: dict[str, float] | None = None


@dataclass
class _SavePath:
    """Return value for save_state / save_weights_for_sampler futures."""

    path: str


class APIFuture(Generic[T]):
    """Already-resolved future; preserves the tinker async contract."""

    def __init__(self, value: T):
        self._value = value

    async def result_async(self) -> T:
        return self._value

    @property
    def result(self) -> T:
        return self._value


# ---------------------------------------------------------------------------
# Environment and distribution helpers
# ---------------------------------------------------------------------------


class _DistEnv:
    """Detects rank/world_size and lazily initializes NCCL."""

    def __init__(self) -> None:
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self._initialized = False
        self._gloo_group: torch.distributed.ProcessGroup | None = None

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def ensure_initialized(self) -> None:
        if self._initialized or self.world_size <= 1:
            self._initialized = True
            return
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        if self._gloo_group is None:
            self._gloo_group = torch.distributed.new_group(backend="gloo")
        self._initialized = True

    def barrier(self) -> None:
        if self.world_size > 1 and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def broadcast_object(self, obj: T | None) -> T | None:
        if self.world_size <= 1:
            return obj
        self.ensure_initialized()
        payload = [obj]
        torch.distributed.broadcast_object_list(payload, src=0, group=self._gloo_group)
        return payload[0]

    def all_gather_object(self, obj: T) -> list[T]:
        if self.world_size <= 1:
            return [obj]
        self.ensure_initialized()
        payload: list[T | None] = [None for _ in range(self.world_size)]
        torch.distributed.all_gather_object(payload, obj, group=self._gloo_group)
        return payload  # type: ignore[return-value]

    def all_reduce_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.world_size <= 1:
            return tensor
        self.ensure_initialized()
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        return tensor

    def all_reduce_min_int(self, value: int) -> int:
        if self.world_size <= 1:
            return int(value)
        self.ensure_initialized()
        t = torch.tensor([int(value)], dtype=torch.int64)
        if torch.distributed.get_backend() == "nccl" and torch.cuda.is_available():
            t = t.to(f"cuda:{self.local_rank}")
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)
        return int(t.item())


_DIST = _DistEnv()


_DATA_SHARDING_FALSY = {"", "none", "off", "0", "false", "no"}
_DATA_SHARDING_VALID = {"shard", "replica"}


def _resolve_data_sharding() -> Literal["shard", "replica", "none"]:
    """TINTHER_DATA_SHARDING: 'shard' (off-policy: slice the same batch across
    ranks) or 'replica' (on-policy: each rank already has a unique batch; sum
    sizes across ranks). Falsy/unset, or single-process runs, return 'none'."""
    if _DIST.world_size <= 1:
        return "none"
    raw = (os.environ.get("TINTHER_DATA_SHARDING") or "").strip().lower()
    if raw in _DATA_SHARDING_FALSY:
        return "none"
    if raw not in _DATA_SHARDING_VALID:
        raise TinkerError(
            f"TINTHER_DATA_SHARDING={raw!r} not supported; "
            "expected one of: shard, replica, none."
        )
    return raw  # type: ignore[return-value]


_HTTP_TEACHER_SHARD_SENTINEL_TOKEN_ID = -1


class _HTTPTeacherShardState:
    """Tracks one off-policy teacher-inference batch per rank."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sample_calls = 0

    def record_sample_call(self) -> None:
        with self._lock:
            self._sample_calls += 1

    def pending_sample_calls(self) -> int:
        with self._lock:
            return self._sample_calls

    def reset_sample_calls(self) -> int:
        with self._lock:
            count = self._sample_calls
            self._sample_calls = 0
            return count


_HTTP_TEACHER_SHARD_STATE = _HTTPTeacherShardState()


def _stable_token_hash(tokens: list[int]) -> int:
    arr = np.asarray(tokens, dtype=np.int64)
    h = hashlib.blake2b(digest_size=8)
    h.update(len(tokens).to_bytes(8, "little", signed=False))
    h.update(arr.tobytes())
    return int.from_bytes(h.digest(), "little", signed=False)


def _http_teacher_owner_rank_from_prompt_tokens(tokens: list[int]) -> int:
    if _DIST.world_size <= 1:
        return 0
    # Off-policy teacher forcing passes model_input + last target token. The
    # teacher distribution for every supervised target position is determined by
    # model_input, so drop the appended last target for a stable per-datum owner.
    owner_key_tokens = tokens[:-1] if tokens else tokens
    return _stable_token_hash(owner_key_tokens) % _DIST.world_size


def _http_teacher_prompt_logprobs_sharding_enabled(
    include_prompt_logprobs: bool,
) -> bool:
    return (
        include_prompt_logprobs
        and _DIST.world_size > 1
        and _resolve_data_sharding() == "shard"
    )


def _sharded_teacher_prompt_logprobs(
    prompt_len: int,
) -> list[list[tuple[int, float]] | None]:
    return [[(_HTTP_TEACHER_SHARD_SENTINEL_TOKEN_ID, 0.0)] for _ in range(prompt_len)]


def _datum_has_sharded_teacher_sentinel(datum: Datum) -> bool:
    target_tokens = datum.loss_fn_inputs.get("target_tokens")
    if target_tokens is None:
        return False
    arr = np.asarray(target_tokens.data)
    return bool(arr.size and np.any(arr == _HTTP_TEACHER_SHARD_SENTINEL_TOKEN_ID))


def _tensor_data_equal(a: TensorData, b: TensorData) -> bool:
    return (
        a.dtype == b.dtype
        and tuple(a.shape) == tuple(b.shape)
        and np.array_equal(np.asarray(a.data), np.asarray(b.data))
    )


def _loss_fn_inputs_equal(
    a: dict[str, TensorData],
    b: dict[str, TensorData],
) -> bool:
    if set(a) != set(b):
        return False
    return all(_tensor_data_equal(a[k], b[k]) for k in a)


def _repair_sharded_teacher_datums(data_D: list[Datum]) -> list[Datum]:
    """Fill non-owner sentinel datums with the owner rank's teacher outputs.

    Teacher ownership is independent of batch order, while student training still
    uses ``data_D[rank::world_size]``. Gathering per-index owner payloads restores
    a full, ordered batch on every rank before student sharding happens.
    """
    if _DIST.world_size <= 1 or not data_D:
        return data_D

    local_payloads: list[dict[str, TensorData] | None] = [
        None if _datum_has_sharded_teacher_sentinel(datum) else datum.loss_fn_inputs
        for datum in data_D
    ]
    gathered_payloads = _DIST.all_gather_object(local_payloads)
    if len(gathered_payloads) != _DIST.world_size:
        raise TinkerError(
            "TINTHER_DATA_SHARDING=shard: failed to gather HTTP teacher "
            f"payloads from all ranks; got {len(gathered_payloads)} of "
            f"{_DIST.world_size} ranks."
        )

    missing: list[int] = []
    conflicts: list[int] = []
    repaired: list[Datum] = []
    expected_n = len(data_D)

    for datum_idx, datum in enumerate(data_D):
        payloads: list[dict[str, TensorData]] = []
        for rank, rank_payloads in enumerate(gathered_payloads):
            if len(rank_payloads) != expected_n:
                raise TinkerError(
                    "TINTHER_DATA_SHARDING=shard: ranks disagree on teacher "
                    f"batch size; local={expected_n}, rank{rank}={len(rank_payloads)}."
                )
            payload = rank_payloads[datum_idx]
            if payload is not None:
                payloads.append(payload)

        if not payloads:
            missing.append(datum_idx)
            repaired.append(datum)
            continue

        first_payload = payloads[0]
        if any(not _loss_fn_inputs_equal(first_payload, payload) for payload in payloads[1:]):
            conflicts.append(datum_idx)
        repaired.append(Datum(model_input=datum.model_input, loss_fn_inputs=first_payload))

    if missing or conflicts:
        details: list[str] = []
        if missing:
            details.append(f"missing owner payloads at batch indices {missing[:10]}")
        if conflicts:
            details.append(f"conflicting owner payloads at batch indices {conflicts[:10]}")
        raise TinkerError(
            "TINTHER_DATA_SHARDING=shard: could not reconstruct sharded HTTP "
            "teacher outputs before student training (" + "; ".join(details) + ")."
        )

    return repaired


def _cache_root() -> Path:
    root = Path(os.environ.get("TINTHER_CACHE_DIR", "/tmp/tinther"))
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# Checkpoint store (tinther:// path <-> on-disk directory)
# ---------------------------------------------------------------------------


class _CheckpointStore:
    """Maps tinther://{state|sampler}/<uuid> strings to local directories."""

    @staticmethod
    def new_path(kind: Literal["state", "sampler"], name: str | None = None) -> tuple[str, Path]:
        uid = name or uuid.uuid4().hex
        d = _cache_root() / kind / uid
        d.mkdir(parents=True, exist_ok=True)
        return f"tinther://{kind}/{uid}", d

    @staticmethod
    def resolve(path: str) -> Path:
        if not path.startswith("tinther://"):
            # Treat as a direct filesystem path (useful for pre-existing HF snapshots).
            return Path(path)
        rest = path[len("tinther://") :]
        parts = rest.split("/", 1)
        if len(parts) != 2:
            raise TinkerError(f"Malformed tinther path: {path}")
        kind, uid = parts
        d = _cache_root() / kind / uid
        if not d.exists():
            raise TinkerError(f"Checkpoint path does not exist: {path}")
        return d

    @staticmethod
    def write_meta(path: str, meta: dict[str, Any]) -> None:
        d = _CheckpointStore.resolve(path)
        with (d / "meta.json").open("w") as f:
            json.dump(meta, f)

    @staticmethod
    def read_meta(path: str) -> dict[str, Any]:
        d = _CheckpointStore.resolve(path)
        p = d / "meta.json"
        if not p.exists():
            return {}
        try:
            with p.open("r") as f:
                return json.load(f)
        except Exception:
            return {}

    @staticmethod
    def delete(path: str) -> None:
        try:
            d = _CheckpointStore.resolve(path)
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass


def _is_peft_adapter_dir(path: str | Path) -> bool:
    p = Path(path)
    return p.exists() and (p / "adapter_config.json").exists()


def _has_tokenizer_files(path: str | Path) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    return any(
        (p / name).exists()
        for name in (
            "tokenizer.json",
            "tokenizer.model",
            "vocab.json",
            "spiece.model",
        )
    )


def _adapter_base_model_from_config(adapter_dir: str | Path) -> str | None:
    config_path = Path(adapter_dir) / "adapter_config.json"
    if not config_path.exists():
        return None
    try:
        with config_path.open("r") as f:
            adapter_config = json.load(f)
    except Exception:
        return None
    base_model = adapter_config.get("base_model_name_or_path")
    if base_model is None:
        return None
    base_model = str(base_model)
    return base_model if base_model else None


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


class _LossFns:
    """Loss function implementations sharing a single signature."""

    @staticmethod
    def _gather_target_logits(
        logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Return logp at target positions. Handles hard (N,) and soft (N,K) targets."""
        logp = F.log_softmax(logits, dim=-1)
        if targets.ndim == 1:
            return logp.gather(-1, targets.long().unsqueeze(-1)).squeeze(-1)
        return logp.gather(-1, targets.long())

    @staticmethod
    def cross_entropy(
        logits: torch.Tensor,
        inputs: dict[str, torch.Tensor],
        loss_fn_config: dict[str, Any] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        del loss_fn_config
        targets = inputs["target_tokens"]
        weights = inputs.get("weights")
        if weights is None:
            weights = torch.ones_like(targets, dtype=logits.dtype)
        tgt_lp = _LossFns._gather_target_logits(logits, targets)  # (N,) or (N,K)
        if targets.ndim == 1:
            per_tok = -weights.to(tgt_lp.dtype) * tgt_lp
            out_lp = tgt_lp
        else:
            per_tok = -(weights.to(tgt_lp.dtype) * tgt_lp).sum(-1)  # (N,)
            out_lp = tgt_lp[..., 0]
        denom = weights.sum().clamp_min(1.0)
        loss = per_tok.sum() / denom
        metrics = {
            "total_loss": float(loss.detach().item()),
            "n_tokens": float(weights.sum().detach().item()),
        }
        return loss, out_lp.detach(), metrics

    @staticmethod
    def importance_sampling(
        logits: torch.Tensor,
        inputs: dict[str, torch.Tensor],
        loss_fn_config: dict[str, Any] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        del loss_fn_config
        targets = inputs["target_tokens"].long()
        old_lp = inputs["logprobs"].to(logits.dtype)
        adv = inputs["advantages"].to(logits.dtype)
        mask = inputs.get("mask")
        if mask is None:
            mask = torch.ones_like(adv)
        else:
            mask = mask.to(logits.dtype)
        new_lp = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        ratio = torch.exp(new_lp - old_lp)
        pg = -(ratio * adv) * mask
        denom = mask.sum().clamp_min(1.0)
        loss = pg.sum() / denom
        metrics = {
            "total_loss": float(loss.detach().item()),
            "mean_ratio": float(ratio.mean().detach().item()),
            "kl_sample_train": float((old_lp - new_lp).mean().detach().item()),
        }
        return loss, new_lp.detach(), metrics

    @staticmethod
    def ppo(
        logits: torch.Tensor,
        inputs: dict[str, torch.Tensor],
        loss_fn_config: dict[str, Any] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        clip_eps = float((loss_fn_config or {}).get("clip_eps", 0.2))
        targets = inputs["target_tokens"].long()
        old_lp = inputs["logprobs"].to(logits.dtype)
        adv = inputs["advantages"].to(logits.dtype)
        mask = inputs.get("mask")
        if mask is None:
            mask = torch.ones_like(adv)
        else:
            mask = mask.to(logits.dtype)
        new_lp = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        ratio = torch.exp(new_lp - old_lp)
        unclipped = ratio * adv
        clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv
        pg = -torch.minimum(unclipped, clipped) * mask
        denom = mask.sum().clamp_min(1.0)
        loss = pg.sum() / denom
        metrics = {
            "total_loss": float(loss.detach().item()),
            "mean_ratio": float(ratio.mean().detach().item()),
            "clip_frac": float(((unclipped > clipped).float() * mask).sum().item()
                               / float(denom.item())),
        }
        return loss, new_lp.detach(), metrics

    DISPATCH: dict[str, Any] = {}


_LossFns.DISPATCH = {
    "cross_entropy": _LossFns.cross_entropy,
    "importance_sampling": _LossFns.importance_sampling,
    "ppo": _LossFns.ppo,
}


# ---------------------------------------------------------------------------
# Trainer backend: HF + PEFT + accelerate
# ---------------------------------------------------------------------------


class _TrainerBackend:
    """Owns the training model, optimizer, and tokenizer."""

    _SUPPORTED_LR_SCHEDULERS = ("cosine", "linear")

    @staticmethod
    def _lr_scheduler_kind() -> str | None:
        raw = (os.environ.get("TINTHER_LR_SCHEDULER") or "").lower()
        if raw == "":
            return None
        if raw not in _TrainerBackend._SUPPORTED_LR_SCHEDULERS:
            raise TinkerError(
                f"TINTHER_LR_SCHEDULER={raw!r} not supported; "
                f"expected one of {_TrainerBackend._SUPPORTED_LR_SCHEDULERS}."
            )
        return raw

    @staticmethod
    def _resolve_initial_lr() -> float:
        kind = _TrainerBackend._lr_scheduler_kind()
        if kind is None:
            return 1e-5
        peak = os.environ.get("TINTHER_LR_PEAK")
        if peak is None:
            raise TinkerError(
                f"TINTHER_LR_SCHEDULER={kind} requires TINTHER_LR_PEAK (peak learning rate)."
            )
        try:
            return float(peak)
        except ValueError as e:
            raise TinkerError(f"TINTHER_LR_PEAK must be a float, got {peak!r}") from e

    @staticmethod
    def _resolve_optim_kwargs() -> dict[str, Any]:
        """Read optional AdamW knobs from the environment.

        TINTHER_WEIGHT_DECAY: AdamW weight_decay (default 0.01 — PyTorch default).
        TINTHER_OPTIM_FUSED: enable fused AdamW (CUDA-only). Truthy strings: 1/true/yes.
        """
        kwargs: dict[str, Any] = {}
        wd_raw = os.environ.get("TINTHER_WEIGHT_DECAY")
        if wd_raw is not None and wd_raw != "":
            try:
                kwargs["weight_decay"] = float(wd_raw)
            except ValueError as e:
                raise TinkerError(
                    f"TINTHER_WEIGHT_DECAY must be a float, got {wd_raw!r}"
                ) from e
        if (os.environ.get("TINTHER_OPTIM_FUSED") or "").lower() in ("1", "true", "yes"):
            if torch.cuda.is_available():
                kwargs["fused"] = True
            else:
                logger.warning(
                    "TINTHER_OPTIM_FUSED requested but CUDA is unavailable; ignoring."
                )
        return kwargs

    @staticmethod
    def _resolve_grad_accum_steps() -> int:
        """TINTHER_GRAD_ACCUM_STEPS: number of micro-batches per
        forward_backward call. 1 (or unset) = current behavior."""
        raw = os.environ.get("TINTHER_GRAD_ACCUM_STEPS")
        if raw is None or raw == "":
            return 1
        try:
            value = int(raw)
        except ValueError as e:
            raise TinkerError(
                f"TINTHER_GRAD_ACCUM_STEPS must be an int, got {raw!r}"
            ) from e
        if value < 1:
            raise TinkerError("TINTHER_GRAD_ACCUM_STEPS must be >= 1")
        return value

    @staticmethod
    def _resolve_max_grad_norm() -> float | None:
        """TINTHER_MAX_GRAD_NORM: clip gradients to this L2 norm before optim step."""
        raw = os.environ.get("TINTHER_MAX_GRAD_NORM")
        if raw is None or raw == "":
            return None
        try:
            value = float(raw)
        except ValueError as e:
            raise TinkerError(
                f"TINTHER_MAX_GRAD_NORM must be a float, got {raw!r}"
            ) from e
        if value <= 0:
            raise TinkerError("TINTHER_MAX_GRAD_NORM must be > 0")
        return value

    @staticmethod
    def _build_lr_scheduler(optimizer: torch.optim.Optimizer):
        kind = _TrainerBackend._lr_scheduler_kind()
        if kind is None:
            return None
        total_raw = os.environ.get("TINTHER_LR_TOTAL_STEPS")
        if total_raw is None:
            raise TinkerError(
                f"TINTHER_LR_SCHEDULER={kind} requires TINTHER_LR_TOTAL_STEPS "
                "(total number of optim steps)."
            )
        try:
            total_steps = int(total_raw)
        except ValueError as e:
            raise TinkerError(
                f"TINTHER_LR_TOTAL_STEPS must be an int, got {total_raw!r}"
            ) from e
        if total_steps <= 0:
            raise TinkerError("TINTHER_LR_TOTAL_STEPS must be > 0")
        warmup_steps = int(os.environ.get("TINTHER_LR_WARMUP_STEPS", "0"))
        if warmup_steps < 0:
            raise TinkerError("TINTHER_LR_WARMUP_STEPS must be >= 0")
        min_lr_ratio = float(os.environ.get("TINTHER_LR_MIN_RATIO", "0.0"))
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise TinkerError("TINTHER_LR_MIN_RATIO must be in [0.0, 1.0]")

        if kind == "cosine":
            try:
                from transformers import get_cosine_with_min_lr_schedule_with_warmup
            except ImportError:
                # transformers 4.56.0 doesn't re-export this at the top level.
                from transformers.optimization import (
                    get_cosine_with_min_lr_schedule_with_warmup,
                )

            logger.info(
                "Enabling transformers cosine LR scheduler: peak=%s total=%d warmup=%d min_ratio=%.4f",
                optimizer.param_groups[0]["lr"],
                total_steps,
                warmup_steps,
                min_lr_ratio,
            )
            return get_cosine_with_min_lr_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                min_lr_rate=min_lr_ratio,
            )

        # kind == "linear": warmup linearly to peak, then linearly decay to
        # peak * min_lr_ratio over the remaining steps.
        peak_lr = optimizer.param_groups[0]["lr"]

        def linear_lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            decay_steps = max(1, total_steps - warmup_steps)
            progress = (current_step - warmup_steps) / decay_steps
            progress = min(max(progress, 0.0), 1.0)
            return max(min_lr_ratio, 1.0 - (1.0 - min_lr_ratio) * progress)

        logger.info(
            "Enabling linear LR scheduler: peak=%s total=%d warmup=%d min_ratio=%.4f",
            peak_lr,
            total_steps,
            warmup_steps,
            min_lr_ratio,
        )
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_lr_lambda)

    @staticmethod
    def _resolve_trainer_attn_implementation() -> tuple[str | None, bool]:
        requested = os.environ.get("TINTHER_TRAINER_ATTN_IMPLEMENTATION")
        if requested == "fa2":
            return "flash_attention_2", False
        if requested is None or requested == "auto":
            if not torch.cuda.is_available():
                return None, True
            try:
                from transformers.utils import is_flash_attn_2_available
            except Exception:
                return None, True
            if is_flash_attn_2_available():
                return "flash_attention_2", True
            return None, True
        raise TinkerError(
            "TINTHER_TRAINER_ATTN_IMPLEMENTATION only supports 'fa2' or 'auto'. "
            f"Got {requested!r}."
        )

    def __init__(
        self,
        model_name: str,
        lora_rank: int | None,
        from_state: str | None = None,
        load_optimizer: bool = False,
    ) -> None:
        from accelerate import Accelerator
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.lora_rank = lora_rank
        self._lock = asyncio.Lock()
        self._last_backward_metrics: dict[str, float] | None = None

        mixed_precision = os.environ.get("TINTHER_MIXED_PRECISION", "bf16")
        self.accelerator: Accelerator = Accelerator(mixed_precision=mixed_precision)
        _DIST.ensure_initialized()

        state_dir = _CheckpointStore.resolve(from_state) if from_state else None
        is_peft_state = state_dir is not None and _is_peft_adapter_dir(state_dir)
        peft_base_source = (
            _adapter_base_model_from_config(state_dir)
            if is_peft_state and state_dir is not None
            else None
        )
        if is_peft_state and peft_base_source is not None and model_name == from_state:
            self.model_name = peft_base_source

        # Tokenizer. Tinther checkpoints save tokenizer files next to the state,
        # but external PEFT adapter dirs may not; fall back to the base model.
        tok_source: str | Path = (
            state_dir
            if state_dir is not None and _has_tokenizer_files(state_dir)
            else (peft_base_source or model_name)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tok_source) if isinstance(tok_source, Path) else tok_source,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Base model
        dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
        base_source = (
            (peft_base_source or model_name)
            if is_peft_state
            else (str(state_dir) if state_dir is not None else model_name)
        )
        trainer_attn_implementation, auto_attn = self._resolve_trainer_attn_implementation()
        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }
        if trainer_attn_implementation is not None:
            model_kwargs["attn_implementation"] = trainer_attn_implementation
        try:
            base = AutoModelForCausalLM.from_pretrained(base_source, **model_kwargs)
        except Exception as e:
            error_message = str(e).lower()
            should_retry_with_default_attn = (
                trainer_attn_implementation is not None
                and auto_attn
                and any(
                    needle in error_message
                    for needle in (
                        "flash attention",
                        "flash_attn",
                        "attn_implementation",
                        "attention implementation",
                    )
                )
            )
            if (
                should_retry_with_default_attn
            ):
                logger.warning(
                    "Auto-selected attn_implementation=%s for training model, "
                    "but loading failed (%s). Retrying with the Transformers default.",
                    trainer_attn_implementation,
                    e,
                )
                model_kwargs.pop("attn_implementation", None)
                base = AutoModelForCausalLM.from_pretrained(base_source, **model_kwargs)
            else:
                raise
        if trainer_attn_implementation is not None:
            actual_attn = getattr(base.config, "_attn_implementation", None)
            logger.info(
                "Training model attention implementation: requested=%s actual=%s",
                trainer_attn_implementation,
                actual_attn or "default",
            )
        if getattr(base.config, "use_cache", None):
            base.config.use_cache = False
        base.gradient_checkpointing_enable()

        # LoRA adapter. State checkpoints produced by PeftModel.save_pretrained()
        # are adapter-only dirs, so load the base model first and then attach the
        # saved adapter instead of treating the checkpoint as a full HF model.
        if is_peft_state:
            from peft import PeftModel

            assert state_dir is not None
            self.model = PeftModel.from_pretrained(base, str(state_dir), is_trainable=True)
        elif lora_rank and lora_rank > 0:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=2 * lora_rank,
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(base, lora_config)
        else:
            self.model = base

        # Optimizer (params on fresh model; DDP-aware via accelerator.prepare).
        # When TINTHER_LR_SCHEDULER=cosine, initial lr becomes the scheduler's
        # base lr (param_group["lr"] at scheduler creation time), so the peak
        # must be set here — not later in _optim_step_sync.
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        initial_lr = self._resolve_initial_lr()
        optim_kwargs = self._resolve_optim_kwargs()
        self.optimizer = torch.optim.AdamW(
            trainable, lr=initial_lr, betas=(0.9, 0.95), eps=1e-8, **optim_kwargs
        )
        self._max_grad_norm = self._resolve_max_grad_norm()

        self._lr_scheduler = self._build_lr_scheduler(self.optimizer)
        if self._lr_scheduler is not None:
            self.model, self.optimizer, self._lr_scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self._lr_scheduler
            )
        else:
            self.model, self.optimizer = self.accelerator.prepare(
                self.model, self.optimizer
            )

        # Optionally restore optimizer state
        if from_state and load_optimizer:
            try:
                opt_path = _CheckpointStore.resolve(from_state) / "optim.pt"
                if opt_path.exists():
                    self.optimizer.load_state_dict(
                        torch.load(opt_path, map_location="cpu")
                    )
            except Exception as e:
                logger.warning(f"Could not restore optimizer state: {e}")
            if self._lr_scheduler is not None:
                try:
                    sched_path = _CheckpointStore.resolve(from_state) / "lr_scheduler.pt"
                    if sched_path.exists():
                        self._lr_scheduler.load_state_dict(
                            torch.load(sched_path, map_location="cpu")
                        )
                except Exception as e:
                    logger.warning(f"Could not restore LR scheduler state: {e}")

        self._user_metadata: dict[str, str] = {}

    # ---- Public-ish methods called by TrainingClient -------------------

    def set_user_metadata(self, md: dict[str, str] | None) -> None:
        self._user_metadata = dict(md or {})

    def get_tokenizer(self):
        return self.tokenizer

    async def forward_backward(
        self,
        data_D: list[Datum],
        loss_fn: str,
        loss_fn_config: dict[str, Any] | None,
    ) -> ForwardBackwardOutput:
        async with self._lock:
            return await asyncio.to_thread(
                self._forward_backward_sync, data_D, loss_fn, loss_fn_config
            )

    def _forward_backward_sync(
        self,
        data_D: list[Datum],
        loss_fn: str,
        loss_fn_config: dict[str, Any] | None,
    ) -> ForwardBackwardOutput:
        if loss_fn not in _LossFns.DISPATCH:
            raise TinkerError(f"Unknown loss_fn: {loss_fn}")
        loss_impl = _LossFns.DISPATCH[loss_fn]

        device = self.accelerator.device
        self.model.train()

        # DDP data sharding. See _resolve_data_sharding().
        #   shard: cookbook handed every rank the same batch (off-policy SFT);
        #          slice to data_D[rank::world_size] and use the pre-slice size
        #          as the global denominator so DDP gradient averaging yields
        #          the global-mean gradient.
        #   replica: each rank already has unique data (on-policy rollouts);
        #          do not slice; sum local sizes via all-reduce to get the
        #          global denominator.
        sharding = _resolve_data_sharding()
        teacher_shard_sample_count = _HTTP_TEACHER_SHARD_STATE.pending_sample_calls()
        if teacher_shard_sample_count > 0 and sharding != "shard":
            observed = _HTTP_TEACHER_SHARD_STATE.reset_sample_calls()
            raise TinkerError(
                "HTTP teacher sharding state is active, but "
                f"TINTHER_DATA_SHARDING={sharding!r}; observed {observed} "
                "teacher prompt-logprob calls before forward_backward."
            )

        derived_global_bs: int | None = None
        if sharding == "shard":
            original_n = len(data_D)
            http_teacher_sharding_active = teacher_shard_sample_count > 0
            if http_teacher_sharding_active:
                observed = _HTTP_TEACHER_SHARD_STATE.reset_sample_calls()
                count_records = _DIST.all_gather_object((observed, original_n))
                bad_count_records = [
                    (rank, observed_n, batch_n)
                    for rank, (observed_n, batch_n) in enumerate(count_records)
                    if observed_n != batch_n
                ]
                if bad_count_records:
                    raise TinkerError(
                        "TINTHER_DATA_SHARDING=shard: HTTP teacher prompt-logprob "
                        "call count does not match training batch size on all "
                        f"ranks: {bad_count_records}. Teacher inference sharding requires "
                        "exactly one prompt-logprob sample call per Datum before "
                        "forward_backward."
                    )
                data_D = _repair_sharded_teacher_datums(data_D)

            if 0 < original_n < _DIST.world_size:
                # Too few examples to split. Fallback: every rank computes the
                # full batch; DDP averages identical gradients (no speedup, but
                # no deadlock and math is identical to single-GPU). DDP scales
                # the averaged gradient by 1/world_size, so to recover the
                # per-example mean we set global_batch_size = original_n *
                # world_size (cancels with the existing world_size factor).
                if _DIST.is_main:
                    warnings.warn(
                        f"TINTHER_DATA_SHARDING=shard: batch size {original_n}"
                        f" < world_size {_DIST.world_size}; falling back to"
                        " replicated compute. Increase batch_size to benefit"
                        " from DDP."
                    )
                derived_global_bs = original_n * _DIST.world_size
            elif original_n >= _DIST.world_size:
                data_D = data_D[_DIST.rank :: _DIST.world_size]
                derived_global_bs = original_n
                if http_teacher_sharding_active and any(
                    _datum_has_sharded_teacher_sentinel(datum) for datum in data_D
                ):
                    raise TinkerError(
                        "TINTHER_DATA_SHARDING=shard: a non-owner HTTP teacher "
                        "sentinel reached this rank's student shard after "
                        "cross-rank repair; refusing to train on corrupt soft "
                        "targets."
                    )
            # else original_n == 0: every rank has no data — symmetric early
            # return below is safe.
        elif sharding == "replica":
            sizes = torch.tensor([len(data_D)], device=device, dtype=torch.int64)
            _DIST.all_reduce_tensor(sizes)
            global_n = int(sizes.item())
            if global_n > 0 and len(data_D) == 0:
                raise TinkerError(
                    "TINTHER_DATA_SHARDING=replica: this rank has no data"
                    " while other ranks do. DDP collectives would deadlock."
                    " Ensure each rank produces at least one Datum per"
                    " forward_backward call."
                )
            if global_n > 0:
                derived_global_bs = global_n

        if any(_datum_has_sharded_teacher_sentinel(datum) for datum in data_D):
            raise TinkerError(
                "Sharded HTTP teacher sentinel reached training data. This means "
                "teacher inference sharding and student data sharding are out of "
                "sync; refusing to train on corrupt soft targets."
            )

        local_metric_sums: dict[str, float] = {}
        n_data = len(data_D)
        if n_data == 0:
            self._last_backward_metrics = {}
            return ForwardBackwardOutput(loss_fn_outputs=[], metrics={})

        cfg_gbs = (loss_fn_config or {}).get("global_batch_size")
        if cfg_gbs is not None:
            global_batch_size = int(cfg_gbs)
            if global_batch_size <= 0:
                raise TinkerError("loss_fn_config.global_batch_size must be positive")
            use_global_batch_scaling = True
        elif derived_global_bs is not None:
            global_batch_size = derived_global_bs
            use_global_batch_scaling = True
        else:
            global_batch_size = n_data
            use_global_batch_scaling = False

        # Gradient accumulation: split data_D into K micro-batches. Each
        # micro-batch's loss is scaled by the same fb-level denominator
        # (global_batch_size or n_data), so the sum of micro-batch losses
        # equals the original single-shot total_loss — gradients accumulate
        # to the same value. DDP all-reduce is suppressed via no_sync() on
        # all but the last micro-batch so collectives fire once per fb call.
        k_requested = self._resolve_grad_accum_steps()
        k_eff = max(1, min(k_requested, _DIST.all_reduce_min_int(n_data)))

        idx_groups: list[list[int]] = [
            list(g) for g in np.array_split(np.arange(n_data), k_eff)
        ]
        idx_groups = [g for g in idx_groups if len(g) > 0]
        loss_fn_outputs: list[dict[str, TensorData]] = [None] * n_data  # type: ignore[list-item]
        token_lists_full = [datum.model_input.to_ints() for datum in data_D]

        for mb_i, idxs in enumerate(idx_groups):
            is_last = mb_i == len(idx_groups) - 1
            sync_ctx = (
                contextlib.nullcontext()
                if is_last
                else self.accelerator.no_sync(self.model)
            )
            with sync_ctx:
                # SKELETON: MAKE DATA (per micro-batch)
                mb_token_lists = [token_lists_full[i] for i in idxs]
                mb_seq_lens = [len(t) for t in mb_token_lists]
                mb_max_seq_len = max(mb_seq_lens)
                input_ids = torch.full(
                    (len(idxs), mb_max_seq_len),
                    self.tokenizer.pad_token_id,
                    dtype=torch.long,
                    device=device,
                )
                attention_mask = torch.zeros(
                    (len(idxs), mb_max_seq_len), dtype=torch.long, device=device
                )
                for j, tokens in enumerate(mb_token_lists):
                    seq_len = len(tokens)
                    input_ids[j, :seq_len] = torch.tensor(
                        tokens, dtype=torch.long, device=device
                    )
                    attention_mask[j, :seq_len] = 1

                # SKELETON: FORWARD
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                batch_logits = outputs.logits

                mb_loss_sum = torch.zeros((), device=device)
                for j, datum_idx in enumerate(idxs):
                    datum = data_D[datum_idx]
                    logits = batch_logits[j, : mb_seq_lens[j]]
                    targets_t = logits.size(0)
                    inputs = {
                        k: v.to_torch().to(device)
                        for k, v in datum.loss_fn_inputs.items()
                    }

                    # -------------------------------- DEBUG
                    print(f"len(data_D): {len(data_D)}")
                    print({k: tuple(v.shape) for k, v in inputs.items()})
                    # soft-target distillation 디버그 프린트: (N, K) 형태의 target_tokens/weights에서
                    # 각 response 위치별 상위 teacher 분포를 확인한다.
                    _tt, _ww = inputs.get("target_tokens"), inputs.get("weights")
                    if _tt is not None and _ww is not None and _tt.ndim == 2:
                        _N, _K = _tt.shape  # N=시퀀스 길이, K=teacher top-K
                        # weight 합이 0보다 큰 행만 실제 response 위치. 전부 0이면 앞 20개로 fallback.
                        _rows = (_ww.sum(-1) > 0).nonzero(as_tuple=True)[0].tolist() or list(range(min(20, _N)))
                        # 20개 초과면 앞 10 + 뒤 10만 샘플링해서 출력량 제한.
                        _positions = _rows if len(_rows) <= 20 else _rows[:10] + _rows[-10:]
                        print(f"  [N={_N}, K={_K}] showing {len(_positions)} response positions:")
                        for _i, _pos in enumerate(_positions):
                            # 해당 위치의 K개 (토큰ID, weight) 쌍을 weight 내림차순 상위 5개만.
                            _pairs = sorted(zip(_tt[_pos].tolist(), _ww[_pos].tolist()), key=lambda x: -x[1])[:5]
                            print(f"  pos={_pos} (top-5 of {_K}):")
                            for _tid, _w in _pairs:
                                print(f"    tok={_tid:>6}  w={_w:.4f}  {self.tokenizer.decode([int(_tid)])!r}")
                            # 앞 10개 출력 직후 생략 표시.
                            if _i == 9 and len(_rows) > 20:
                                print("  .....")

                    # Align any length mismatch by truncating to common length.
                    target_len = inputs["target_tokens"].shape[0]
                    common = min(targets_t, target_len)
                    logits = logits[:common]
                    for k, v in list(inputs.items()):
                        if v.ndim >= 1 and v.shape[0] >= common:
                            inputs[k] = v[:common]

                    # SKELETON: LOSS
                    loss, per_pos_lp, metrics = loss_impl(logits, inputs, loss_fn_config)
                    mb_loss_sum = mb_loss_sum + loss
                    loss_fn_outputs[datum_idx] = {
                        "logprobs": TensorData.from_torch(per_pos_lp)
                    }
                    for k, v in metrics.items():
                        local_metric_sums[k] = local_metric_sums.get(k, 0.0) + float(v)

                if use_global_batch_scaling:
                    mb_total_loss = mb_loss_sum * (_DIST.world_size / global_batch_size)
                else:
                    mb_total_loss = mb_loss_sum / n_data

                # SKELETON: BACKWARD (per micro-batch; no_sync until last)
                self.accelerator.backward(mb_total_loss)

        if use_global_batch_scaling and local_metric_sums:
            metric_names = sorted(local_metric_sums)
            metric_tensor = torch.tensor(
                [local_metric_sums[name] for name in metric_names],
                device=device,
                dtype=torch.float64,
            )
            _DIST.all_reduce_tensor(metric_tensor)
            aggregated = {
                name: float(metric_tensor[i].item() / global_batch_size)
                for i, name in enumerate(metric_names)
            }
        else:
            aggregated = {k: v / n_data for k, v in local_metric_sums.items()}

        self._last_backward_metrics = aggregated
        return ForwardBackwardOutput(loss_fn_outputs=loss_fn_outputs, metrics=aggregated)

    async def optim_step(self, adam: AdamParams) -> OptimStepResponse:
        async with self._lock:
            return await asyncio.to_thread(self._optim_step_sync, adam)

    def _optim_step_sync(self, adam: AdamParams) -> OptimStepResponse:
        # Update betas/eps on the fly (tinker exposes per-step Adam params).
        # When the cosine scheduler owns LR, ignore adam.learning_rate — the
        # scheduler sets param_group["lr"] at construction and after each step.
        scheduler_owns_lr = self._lr_scheduler is not None
        for g in self.optimizer.param_groups:
            if not scheduler_owns_lr:
                g["lr"] = float(adam.learning_rate)
            g["betas"] = (float(adam.beta1), float(adam.beta2))
            g["eps"] = float(adam.eps)

        # Gradient norm before step (for logging).
        grad_norm = 0.0
        n = 0
        for p in self.model.parameters():
            if p.grad is not None:
                grad_norm += float(p.grad.detach().norm(2).item()) ** 2
                n += 1
        grad_norm = float(grad_norm**0.5) if n else 0.0

        if self._max_grad_norm is not None:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(), self._max_grad_norm
            )

        self.optimizer.step()
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        effective_lr = float(self.optimizer.param_groups[0]["lr"])
        metrics: dict[str, float] = {
            "optim/lr": effective_lr,
            "optim/grad_norm": grad_norm,
        }
        if self._last_backward_metrics:
            for k, v in self._last_backward_metrics.items():
                metrics[f"train/{k}"] = float(v)
        return OptimStepResponse(metrics=metrics)

    async def save_state(self, name: str) -> _SavePath:
        return await asyncio.to_thread(self._save_state_sync, name)

    def _new_distributed_checkpoint_path(
        self, kind: Literal["state", "sampler"], name: str
    ) -> tuple[str, Path]:
        shared_name = _DIST.broadcast_object(
            f"{name}-{uuid.uuid4().hex[:6]}" if _DIST.is_main else None
        )
        if not isinstance(shared_name, str):
            raise TinkerError(f"Could not broadcast checkpoint name for {kind}")
        return _CheckpointStore.new_path(kind, name=shared_name)

    def _save_state_sync(self, name: str) -> _SavePath:
        path, d = self._new_distributed_checkpoint_path("state", name)
        unwrapped = self.accelerator.unwrap_model(self.model)
        if _DIST.is_main:
            unwrapped.save_pretrained(str(d))
            self.tokenizer.save_pretrained(str(d))
            try:
                torch.save(self.optimizer.state_dict(), str(d / "optim.pt"))
            except Exception as e:
                logger.warning(f"Could not save optimizer state: {e}")
            if self._lr_scheduler is not None:
                try:
                    torch.save(
                        self._lr_scheduler.state_dict(), str(d / "lr_scheduler.pt")
                    )
                except Exception as e:
                    logger.warning(f"Could not save LR scheduler state: {e}")
            _CheckpointStore.write_meta(
                path,
                {
                    "user_metadata": self._user_metadata,
                    "model_name": self.model_name,
                    "lora_rank": self.lora_rank,
                    "is_peft_adapter": _is_peft_adapter_dir(d),
                    "ts": time.time(),
                },
            )
        _DIST.barrier()
        return _SavePath(path=path)

    async def save_weights_for_sampler(self, name: str) -> _SavePath:
        return await asyncio.to_thread(self._save_weights_for_sampler_sync, name)

    def _save_weights_for_sampler_sync(self, name: str) -> _SavePath:
        path, d = self._new_distributed_checkpoint_path("sampler", name)
        unwrapped = self.accelerator.unwrap_model(self.model)
        if _DIST.is_main:
            is_peft_adapter = False
            try:
                from peft import PeftModel

                if isinstance(unwrapped, PeftModel):
                    # Do not call merge_and_unload() on the live training model:
                    # it mutates the PeftModel in place and desynchronizes DDP
                    # ranks. Save the adapter-only snapshot and let the sampler
                    # load/merge it on its own inference copy.
                    unwrapped.save_pretrained(str(d))
                    is_peft_adapter = True
                else:
                    unwrapped.save_pretrained(str(d))
            except Exception:
                unwrapped.save_pretrained(str(d))
                is_peft_adapter = _is_peft_adapter_dir(d)
            self.tokenizer.save_pretrained(str(d))
            _CheckpointStore.write_meta(
                path,
                {
                    "user_metadata": self._user_metadata,
                    "model_name": self.model_name,
                    "lora_rank": self.lora_rank,
                    "is_peft_adapter": is_peft_adapter,
                    "ts": time.time(),
                },
            )
        _DIST.barrier()
        return _SavePath(path=path)


# ---------------------------------------------------------------------------
# Sampler backend: local vLLM/HF, used **only** for fresh student checkpoints
# saved at ``tinther://sampler/...``. External teacher models are served by a
# separate OS process and accessed through ``_HTTPSamplerBackend`` below.
# ---------------------------------------------------------------------------


class _SamplerBackend:
    """In-process sampler for student checkpoints.

    Do not use this for teacher models — teachers run in a separate OS process
    serving vLLM's OpenAI-compatible HTTP API, and tinther's ``ServiceClient``
    routes them through :class:`_HTTPSamplerBackend`.
    """

    def __init__(self, model_ref: str):
        self.model_ref = model_ref
        self._llm = None
        self._tokenizer = None
        self._hf_model = None  # fallback
        self._sem = asyncio.Semaphore(int(os.environ.get("TINTHER_SAMPLER_CONCURRENCY", "16")))

    def _resolve_model_path(self) -> str:
        if self.model_ref.startswith("tinther://"):
            return str(_CheckpointStore.resolve(self.model_ref))
        return self.model_ref

    def _checkpoint_meta(self) -> dict[str, Any]:
        if not self.model_ref.startswith("tinther://"):
            return {}
        return _CheckpointStore.read_meta(self.model_ref)

    def _is_peft_adapter_checkpoint(self) -> bool:
        return _is_peft_adapter_dir(self._resolve_model_path())

    def _adapter_base_model(self) -> str:
        path = self._resolve_model_path()
        meta_model = self._checkpoint_meta().get("model_name")
        config_model = _adapter_base_model_from_config(path)
        if meta_model is not None and str(meta_model) in {self.model_ref, path}:
            meta_model = None
        base_model = meta_model or config_model
        if not base_model:
            raise TinkerError(
                f"PEFT adapter checkpoint at {path} does not record a base model. "
                "Expected meta.json:model_name or adapter_config.json:base_model_name_or_path."
            )
        return str(base_model)

    def _ensure_llm(self) -> None:
        if self._llm is not None:
            return
        if self._is_peft_adapter_checkpoint():
            # vLLM cannot consume this adapter-only tinther checkpoint through
            # the plain `model=` path. Use the HF sampler path, which loads the
            # base model plus adapter into an isolated inference model.
            self._ensure_hf()
            return
        if os.environ.get("TINTHER_SAMPLER_DISABLE_VLLM", "0") == "1":
            self._ensure_hf()
            return
        try:
            from vllm import LLM

            tp = int(os.environ.get("TINTHER_SAMPLER_TP", "1"))
            gpu_mem = float(os.environ.get("TINTHER_SAMPLER_GPU_MEM", "0.45"))
            dtype = os.environ.get("TINTHER_SAMPLER_DTYPE", "bfloat16")
            kwargs: dict[str, Any] = {
                "model": self._resolve_model_path(),
                "tensor_parallel_size": tp,
                "dtype": dtype,
                "gpu_memory_utilization": gpu_mem,
                "enforce_eager": os.environ.get("TINTHER_SAMPLER_EAGER", "0") == "1",
                "trust_remote_code": True,
            }
            max_model_len = os.environ.get("TINTHER_SAMPLER_MAX_MODEL_LEN")
            if max_model_len:
                kwargs["max_model_len"] = int(max_model_len)
            self._llm = LLM(**kwargs)
        except Exception as e:
            logger.warning(
                f"vLLM unavailable ({e}); falling back to HF generate for sampling."
            )
            self._ensure_hf()

    def _ensure_hf(self) -> None:
        if self._hf_model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = self._resolve_model_path()
        is_adapter = self._is_peft_adapter_checkpoint()
        base_model = self._adapter_base_model() if is_adapter else path
        tokenizer_source = path if is_adapter and _has_tokenizer_files(path) else base_model
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        cuda_available = torch.cuda.is_available()
        device = f"cuda:{_DIST.local_rank}" if cuda_available else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if cuda_available else torch.float32,
            trust_remote_code=True,
        )
        if is_adapter:
            from peft import PeftModel

            peft_model = PeftModel.from_pretrained(model, path)
            try:
                model = peft_model.merge_and_unload()
            except Exception:
                model = peft_model
        self._hf_model = model.to(device)
        self._hf_model.eval()

    def close(self) -> None:
        if self._llm is not None:
            try:
                del self._llm
            except Exception:
                pass
            self._llm = None
        if self._hf_model is not None:
            try:
                del self._hf_model
            except Exception:
                pass
            self._hf_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- sample ----------------------------------------------------------

    async def sample(
        self,
        prompt: ModelInput,
        num_samples: int,
        params: SamplingParams,
        include_prompt_logprobs: bool,
        topk_prompt_logprobs: int | None,
    ) -> SampleResponse:
        async with self._sem:
            return await asyncio.to_thread(
                self._sample_sync, prompt, num_samples, params, include_prompt_logprobs,
                topk_prompt_logprobs,
            )

    def _sample_sync(
        self,
        prompt: ModelInput,
        num_samples: int,
        params: SamplingParams,
        include_prompt_logprobs: bool,
        topk_prompt_logprobs: int | None,
    ) -> SampleResponse:
        self._ensure_llm()
        tokens = prompt.to_ints()

        if self._llm is not None:
            return self._sample_vllm(
                tokens, num_samples, params, include_prompt_logprobs, topk_prompt_logprobs
            )
        return self._sample_hf(
            tokens, num_samples, params, include_prompt_logprobs, topk_prompt_logprobs
        )

    def _sample_vllm(
        self,
        tokens: list[int],
        num_samples: int,
        params: SamplingParams,
        include_prompt_logprobs: bool,
        topk_prompt_logprobs: int | None,
    ) -> SampleResponse:
        from vllm import SamplingParams as VLLMSamplingParams

        stop_strs: list[str] | None = None
        stop_ids: list[int] | None = None
        if params.stop is not None:
            stop_seq = list(params.stop)
            if stop_seq and isinstance(stop_seq[0], int):
                stop_ids = [int(x) for x in stop_seq]
            elif stop_seq:
                stop_strs = [str(x) for x in stop_seq]

        vllm_kwargs: dict[str, Any] = {
            "n": num_samples,
            "max_tokens": int(params.max_tokens),
            "temperature": float(params.temperature),
            "logprobs": 1,
        }
        if stop_strs:
            vllm_kwargs["stop"] = stop_strs
        if stop_ids:
            vllm_kwargs["stop_token_ids"] = stop_ids
        if include_prompt_logprobs:
            vllm_kwargs["prompt_logprobs"] = int(topk_prompt_logprobs or 1)

        sp = VLLMSamplingParams(**vllm_kwargs)
        from vllm import TokensPrompt

        outputs = self._llm.generate(
            [TokensPrompt(prompt_token_ids=tokens)], sampling_params=sp, use_tqdm=False
        )
        req = outputs[0]

        seqs: list[_Sequence] = []
        for comp in req.outputs:
            out_tokens = list(comp.token_ids)
            out_lps: list[float] | None = None
            if comp.logprobs is not None:
                lps: list[float] = []
                for pos_dict, tok in zip(comp.logprobs, out_tokens):
                    if pos_dict is None:
                        lps.append(0.0)
                        continue
                    entry = pos_dict.get(tok)
                    if entry is None:
                        lps.append(0.0)
                    else:
                        lps.append(float(getattr(entry, "logprob", entry)))
                out_lps = lps
            stop_reason = getattr(comp, "finish_reason", None) or "stop"
            seqs.append(
                _Sequence(tokens=out_tokens, logprobs=out_lps, stop_reason=str(stop_reason))
            )

        topk: list[list[tuple[int, float]] | None] | None = None
        if include_prompt_logprobs:
            raw = getattr(req, "prompt_logprobs", None)
            if raw is not None:
                topk = []
                for pos_dict in raw:
                    if pos_dict is None:
                        topk.append(None)
                        continue
                    items: list[tuple[int, float]] = []
                    for tok_id, lp_obj in pos_dict.items():
                        lp_val = float(getattr(lp_obj, "logprob", lp_obj))
                        items.append((int(tok_id), lp_val))
                    items.sort(key=lambda x: x[1], reverse=True)
                    topk.append(items)

        return SampleResponse(sequences=seqs, topk_prompt_logprobs=topk)

    def _sample_hf(
        self,
        tokens: list[int],
        num_samples: int,
        params: SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int | None = None,
    ) -> SampleResponse:
        self._ensure_hf()
        assert self._hf_model is not None and self._tokenizer is not None
        input_ids = torch.tensor([tokens] * num_samples, dtype=torch.long).to(
            self._hf_model.device
        )
        do_sample = float(params.temperature) > 0.0
        with torch.no_grad():
            gen = self._hf_model.generate(
                input_ids=input_ids,
                max_new_tokens=int(params.max_tokens),
                do_sample=do_sample,
                temperature=max(float(params.temperature), 1e-6),
                pad_token_id=self._tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        seqs: list[_Sequence] = []
        for i in range(num_samples):
            out_tokens = gen.sequences[i].tolist()[len(tokens):]
            lps: list[float] = []
            for step_scores, tok in zip(gen.scores or [], out_tokens):
                logp = F.log_softmax(step_scores[i], dim=-1)
                lps.append(float(logp[int(tok)].item()))
            seqs.append(_Sequence(tokens=out_tokens, logprobs=lps, stop_reason="stop"))

        topk: list[list[tuple[int, float]] | None] | None = None
        if include_prompt_logprobs:
            # Compute top-K log-probabilities on the prompt positions.
            prompt_ids = torch.tensor([tokens], dtype=torch.long, device=self._hf_model.device)
            with torch.no_grad():
                out = self._hf_model(input_ids=prompt_ids)
            prompt_logits = out.logits[0]  # (T, V) — logit at pos t predicts tok t+1
            k = int(topk_prompt_logprobs or 1)
            # Follow the tinker-style convention: one entry per input position,
            # position 0 has no logprob, positions 1..T-1 carry (tok, lp) lists
            # describing the distribution that produced the token at that slot.
            topk = [None]
            for t in range(len(tokens) - 1):
                logp = F.log_softmax(prompt_logits[t], dim=-1)
                top_lp, top_ids = torch.topk(logp, k=min(k, logp.shape[-1]))
                items = [(int(i.item()), float(v.item())) for v, i in zip(top_lp, top_ids)]
                topk.append(items)

        return SampleResponse(sequences=seqs, topk_prompt_logprobs=topk)

    # -- compute_logprobs ------------------------------------------------

    async def compute_logprobs(self, sequence: ModelInput) -> list[float]:
        async with self._sem:
            return await asyncio.to_thread(self._compute_logprobs_sync, sequence)

    def _compute_logprobs_sync(self, sequence: ModelInput) -> list[float]:
        tokens = sequence.to_ints()
        self._ensure_llm()
        if self._llm is not None:
            return self._compute_logprobs_vllm(tokens)
        return self._compute_logprobs_hf(tokens)

    def _compute_logprobs_vllm(self, tokens: list[int]) -> list[float]:
        from vllm import SamplingParams as VLLMSamplingParams, TokensPrompt

        sp = VLLMSamplingParams(n=1, max_tokens=1, temperature=0.0, prompt_logprobs=0)
        try:
            outputs = self._llm.generate(
                [TokensPrompt(prompt_token_ids=tokens)], sampling_params=sp, use_tqdm=False
            )
        except Exception:
            # Some vllm versions require prompt_logprobs >= 1.
            sp = VLLMSamplingParams(n=1, max_tokens=1, temperature=0.0, prompt_logprobs=1)
            outputs = self._llm.generate(
                [TokensPrompt(prompt_token_ids=tokens)], sampling_params=sp, use_tqdm=False
            )
        req = outputs[0]
        raw = getattr(req, "prompt_logprobs", None) or []
        lps: list[float] = []
        for pos_dict, tok in zip(raw, tokens):
            if pos_dict is None:
                continue
            entry = pos_dict.get(tok)
            if entry is None:
                # Fall back to any top entry that matches; otherwise 0.
                lps.append(0.0)
            else:
                lps.append(float(getattr(entry, "logprob", entry)))
        # tinker returns N-1 logprobs for an N-token sequence; drop the leading
        # 0-position if vllm included it.
        if len(lps) == len(tokens):
            lps = lps[1:]
        return lps

    def _compute_logprobs_hf(self, tokens: list[int]) -> list[float]:
        self._ensure_hf()
        assert self._hf_model is not None
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self._hf_model.device)
        with torch.no_grad():
            out = self._hf_model(input_ids=input_ids)
        logits = out.logits[0, :-1, :]  # predict tokens[1:]
        logp = F.log_softmax(logits, dim=-1)
        tgt = torch.tensor(tokens[1:], device=logits.device, dtype=torch.long)
        return logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).detach().cpu().tolist()


# ---------------------------------------------------------------------------
# Sampler backend: external vLLM server via HTTP (OpenAI-compatible)
# ---------------------------------------------------------------------------


class _HTTPSamplerBackend:
    """Talks to an out-of-process vLLM server via its OpenAI-compatible API.

    vLLM's ``POST /v1/completions`` accepts the standard OpenAI request shape
    plus two extensions we rely on:

    * ``prompt_logprobs: int`` — return top-K logprobs for each prompt position,
      including the teacher distribution needed for off-policy distillation.
    * ``return_token_ids: bool`` — populate ``choice.token_ids`` and
      ``choice.prompt_token_ids`` (raw IDs, no re-tokenization roundtrip).

    The response payload serializes ``list[dict[int, Logprob] | None]``;
    Pydantic writes the dict keys as strings (``"12345"``), and each ``Logprob``
    is ``{"logprob": float, "rank": int | None, "decoded_token": str | None}``.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str | None = None,
        timeout_s: float = 600.0,
        concurrency: int = 64,
        base_urls: list[str] | None = None,
        initial_url_index: int = 0,
    ) -> None:
        normalized_urls = [url.rstrip("/") for url in (base_urls or [base_url]) if url]
        if not normalized_urls:
            raise TinkerError("Teacher HTTP backend requires at least one base URL")
        self._base_urls = normalized_urls
        self.base_url = self._base_urls[0]
        self.model_name = model_name
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._sem = asyncio.Semaphore(concurrency)
        self._sessions: dict[str, Any] = {}
        self._session_lock = asyncio.Lock()
        self._url_index = initial_url_index % len(self._base_urls)
        self._url_lock = asyncio.Lock()

    # ---- construction --------------------------------------------------

    @classmethod
    def from_env(cls, model_ref: str) -> "_HTTPSamplerBackend":
        """Build a backend from ``TINTHER_TEACHER_*`` env vars.

        ``TINTHER_TEACHER_URLS`` (JSON dict) lets multi-teacher configs pick a
        URL per model ref using substring match on the key; this is the MOPD
        pattern in ``train_off_policy.py``. Falls back to
        ``TINTHER_TEACHER_URL``.
        """
        base_url = os.environ.get("TINTHER_TEACHER_URL")
        selected_urls: list[str] | None = None
        urls_json = os.environ.get("TINTHER_TEACHER_URLS")
        rank = int(os.environ.get("RANK", "0"))
        balance_mode = os.environ.get("TINTHER_TEACHER_BALANCE", "rank")

        def _normalize_urls(urls: list[Any]) -> list[str]:
            str_urls = [str(url) for url in urls if str(url)]
            if not str_urls:
                raise TinkerError("TINTHER_TEACHER_URLS list must not be empty")
            return str_urls

        if urls_json:
            try:
                urls_map = json.loads(urls_json)
            except Exception as e:
                raise TinkerError(
                    f"Invalid TINTHER_TEACHER_URLS (must be JSON dict or list): {e}"
                ) from e
            if isinstance(urls_map, list):
                selected_urls = _normalize_urls(urls_map)
            elif isinstance(urls_map, dict):
                for key, url in urls_map.items():
                    if key in str(model_ref):
                        if isinstance(url, list):
                            selected_urls = _normalize_urls(url)
                        else:
                            base_url = str(url)
                        break
            else:
                raise TinkerError("TINTHER_TEACHER_URLS must be a JSON dict or list")
        initial_url_index = 0
        base_urls = None
        if selected_urls:
            if balance_mode == "request_rr":
                base_urls = selected_urls
                initial_url_index = rank % len(selected_urls)
                base_url = selected_urls[initial_url_index]
            else:
                base_url = selected_urls[rank % len(selected_urls)]
        if not base_url:
            raise TinkerError(
                "Teacher sampling requested but TINTHER_TEACHER_URL is not set. "
                "Start an external vLLM server (e.g. "
                "`python -m vllm.entrypoints.openai.api_server --model ... --port 8765`) "
                "and export TINTHER_TEACHER_URL=http://127.0.0.1:8765"
            )
        model_name = os.environ.get("TINTHER_TEACHER_MODEL_NAME") or model_ref
        api_key = os.environ.get("TINTHER_TEACHER_API_KEY")
        timeout_s = float(os.environ.get("TINTHER_TEACHER_TIMEOUT", "600"))
        concurrency = int(os.environ.get("TINTHER_TEACHER_CONCURRENCY", "64"))
        return cls(
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            timeout_s=timeout_s,
            concurrency=concurrency,
            base_urls=base_urls,
            initial_url_index=initial_url_index,
        )

    # ---- session management -------------------------------------------

    async def _pick_base_url(self) -> str:
        if len(self._base_urls) == 1:
            return self._base_urls[0]
        async with self._url_lock:
            url = self._base_urls[self._url_index]
            self._url_index = (self._url_index + 1) % len(self._base_urls)
            return url

    async def _ensure_session(self, base_url: str):
        import aiohttp

        session = self._sessions.get(base_url)
        if session is not None and not session.closed:
            return session
        async with self._session_lock:
            session = self._sessions.get(base_url)
            if session is not None and not session.closed:
                return session
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            timeout = aiohttp.ClientTimeout(total=self._timeout_s)
            session = aiohttp.ClientSession(headers=headers, timeout=timeout)
            self._sessions[base_url] = session
            return session

    async def _reset_session(self, base_url: str | None = None) -> None:
        async with self._session_lock:
            if base_url is None:
                old_sessions = list(self._sessions.values())
                self._sessions = {}
            else:
                old_session = self._sessions.pop(base_url, None)
                old_sessions = [old_session] if old_session is not None else []
            for old_session in old_sessions:
                if old_session is not None and not old_session.closed:
                    await old_session.close()

    def close(self) -> None:
        # aiohttp sessions should be closed from an event loop; tinther uses
        # clients for the full training run, so best-effort drop-on-shutdown
        # is sufficient here.
        self._sessions = {}

    # ---- request helpers ----------------------------------------------

    @staticmethod
    def _parse_prompt_logprobs(
        raw: list | None,
    ) -> list[list[tuple[int, float]] | None] | None:
        """Convert vLLM's response-side prompt_logprobs to tinker's format.

        Input: ``[None | {"tok_id_str": {"logprob": float, ...}}]``
        Output: ``[None | [(int, float), ...]]`` (sorted by logprob desc).
        """
        if raw is None:
            return None
        out: list[list[tuple[int, float]] | None] = []
        for pos in raw:
            if pos is None:
                out.append(None)
                continue
            items: list[tuple[int, float]] = []
            for tok_key, lp_obj in pos.items():
                # JSON serialization coerces dict[int, ...] keys to strings.
                try:
                    tok_id = int(tok_key)
                except (TypeError, ValueError):
                    continue
                if isinstance(lp_obj, dict):
                    lp_val = float(lp_obj.get("logprob", 0.0))
                else:
                    lp_val = float(lp_obj)
                items.append((tok_id, lp_val))
            items.sort(key=lambda x: x[1], reverse=True)
            out.append(items)
        return out

    async def _post_completions(self, body: dict[str, Any]) -> dict[str, Any]:
        import aiohttp

        max_attempts = max(1, int(os.environ.get("TINTHER_TEACHER_RETRIES", "3")))
        backoff_s = float(os.environ.get("TINTHER_TEACHER_RETRY_BACKOFF_S", "1.0"))
        retryable_statuses = {408, 409, 425, 429, 500, 502, 503, 504}
        last_exc: Exception | None = None

        def _is_retryable_exception(exc: Exception) -> bool:
            if isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError)):
                return True
            if isinstance(exc, RuntimeError):
                msg = str(exc).lower()
                return "connection closed" in msg or "session is closed" in msg
            return False

        for attempt in range(1, max_attempts + 1):
            base_url = await self._pick_base_url()
            url = f"{base_url}/v1/completions"
            session = await self._ensure_session(base_url)
            try:
                async with session.post(url, json=body) as resp:
                    if resp.status == 200:
                        try:
                            return await resp.json()
                        except Exception as exc:
                            last_exc = exc
                            if not _is_retryable_exception(exc) or attempt >= max_attempts:
                                raise
                            logger.warning(
                                "Teacher response body read from %s failed on attempt %s/%s: %s",
                                url,
                                attempt,
                                max_attempts,
                                exc,
                            )
                            await self._reset_session(base_url)
                            await asyncio.sleep(backoff_s * attempt)
                            continue

                    text = await resp.text()
                    if resp.status in retryable_statuses and attempt < max_attempts:
                        logger.warning(
                            "Teacher server at %s returned retryable HTTP %s on attempt %s/%s",
                            url,
                            resp.status,
                            attempt,
                            max_attempts,
                        )
                        await self._reset_session(base_url)
                        await asyncio.sleep(backoff_s * attempt)
                        continue

                    raise TinkerError(
                        f"Teacher server at {url} returned HTTP {resp.status}: {text[:500]}"
                    )
            except Exception as exc:
                last_exc = exc
                if not _is_retryable_exception(exc) or attempt >= max_attempts:
                    raise
                logger.warning(
                    "Teacher request to %s failed on attempt %s/%s: %s",
                    url,
                    attempt,
                    max_attempts,
                    exc,
                )
                await self._reset_session(base_url)
                await asyncio.sleep(backoff_s * attempt)

        if last_exc is not None:
            raise last_exc
        raise TinkerError(f"Teacher request to {url} failed without a response")

    # ---- public surface (mirrors _SamplerBackend) ---------------------

    async def sample(
        self,
        prompt: ModelInput,
        num_samples: int,
        params: SamplingParams,
        include_prompt_logprobs: bool,
        topk_prompt_logprobs: int | None,
    ) -> SampleResponse:
        tokens = prompt.to_ints()
        if _http_teacher_prompt_logprobs_sharding_enabled(include_prompt_logprobs):
            _HTTP_TEACHER_SHARD_STATE.record_sample_call()
            owner_rank = _http_teacher_owner_rank_from_prompt_tokens(tokens)
            if owner_rank != _DIST.rank:
                return SampleResponse(
                    sequences=[
                        _Sequence(
                            tokens=[],
                            logprobs=[],
                            stop_reason="tinther_sharded_non_owner",
                        )
                    ],
                    topk_prompt_logprobs=_sharded_teacher_prompt_logprobs(len(tokens)),
                )

        body: dict[str, Any] = {
            "model": self.model_name,
            "prompt": tokens,
            "max_tokens": int(params.max_tokens),
            "temperature": float(params.temperature),
            "n": int(num_samples),
            "logprobs": 1,
            "return_token_ids": True,
        }
        if include_prompt_logprobs:
            body["prompt_logprobs"] = int(topk_prompt_logprobs or 1)
        if params.stop is not None:
            stop_seq = list(params.stop)
            if stop_seq and isinstance(stop_seq[0], int):
                body["stop_token_ids"] = [int(x) for x in stop_seq]
            elif stop_seq:
                body["stop"] = [str(x) for x in stop_seq]

        async with self._sem:
            payload = await self._post_completions(body)

        choices = payload.get("choices") or []
        seqs: list[_Sequence] = []
        for choice in choices:
            out_tokens = choice.get("token_ids") or []
            lp_info = choice.get("logprobs") or {}
            out_lps = lp_info.get("token_logprobs")
            if out_lps is not None:
                out_lps = [float(x) if x is not None else 0.0 for x in out_lps]
            stop_reason = choice.get("finish_reason") or "stop"
            seqs.append(
                _Sequence(
                    tokens=[int(t) for t in out_tokens],
                    logprobs=out_lps,
                    stop_reason=str(stop_reason),
                )
            )

        topk_prompt = None
        if include_prompt_logprobs and choices:
            topk_prompt = self._parse_prompt_logprobs(choices[0].get("prompt_logprobs"))

        return SampleResponse(sequences=seqs, topk_prompt_logprobs=topk_prompt)

    async def compute_logprobs(self, sequence: ModelInput) -> list[float]:
        """Return per-position logprob of each token in ``sequence`` (len-1 aligned).

        Implemented via ``prompt_logprobs=1`` (top-1) on an empty generation;
        we pick the entry matching the actual prompt token at each position.
        """
        tokens = sequence.to_ints()
        body = {
            "model": self.model_name,
            "prompt": tokens,
            "max_tokens": 1,
            "temperature": 0.0,
            "n": 1,
            "logprobs": 1,
            "prompt_logprobs": 1,
            "return_token_ids": True,
        }
        async with self._sem:
            payload = await self._post_completions(body)
        choices = payload.get("choices") or []
        if not choices:
            return []
        raw = self._parse_prompt_logprobs(choices[0].get("prompt_logprobs")) or []
        # First position has no logprob; subsequent positions have the logprob
        # of the actual prompt token at that position. With top-1, the server
        # may return a different token than the ground-truth prompt token when
        # the prompt token is not the argmax — we must scan for the exact ID.
        out: list[float] = []
        for i, pos in enumerate(raw):
            if i == 0 or pos is None:
                continue
            tok = tokens[i]
            match = next((lp for (tid, lp) in pos if tid == tok), None)
            # If top-1 missed, fall back to the top entry's logprob. In practice
            # callers of compute_logprobs feed sequences that were sampled from
            # this same model, so the argmax is usually the true token.
            out.append(match if match is not None else (pos[0][1] if pos else 0.0))
        return out


# ---------------------------------------------------------------------------
# Rest client stub (checkpoint metadata)
# ---------------------------------------------------------------------------


class _TrainingRunInfo:
    def __init__(self, user_metadata: dict[str, str]):
        self.user_metadata = dict(user_metadata)


class _TrainingRunFuture:
    def __init__(self, info: _TrainingRunInfo):
        self._info = info

    def result(self) -> _TrainingRunInfo:
        return self._info


class _RestClientStub:
    """Minimal fake covering checkpoint_utils metadata calls."""

    def get_training_run_by_tinker_path(self, path: str) -> _TrainingRunFuture:
        meta = _CheckpointStore.read_meta(path)
        return _TrainingRunFuture(_TrainingRunInfo(meta.get("user_metadata", {})))

    async def get_training_run_by_tinker_path_async(self, path: str) -> _TrainingRunInfo:
        meta = _CheckpointStore.read_meta(path)
        return _TrainingRunInfo(meta.get("user_metadata", {}))

    async def delete_checkpoint_from_tinker_path_async(self, path: str) -> None:
        _CheckpointStore.delete(path)


# ---------------------------------------------------------------------------
# Public clients
# ---------------------------------------------------------------------------


class SamplingClient:
    """Mirrors tinker.SamplingClient.

    The backend is either an in-process ``_SamplerBackend`` (student
    checkpoint) or an ``_HTTPSamplerBackend`` (external teacher server).
    """

    def __init__(self, backend: "_SamplerBackend | _HTTPSamplerBackend"):
        self._backend = backend

    async def sample_async(
        self,
        prompt: ModelInput,
        num_samples: int,
        sampling_params: SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int | None = None,
    ) -> SampleResponse:
        return await self._backend.sample(
            prompt, num_samples, sampling_params, include_prompt_logprobs, topk_prompt_logprobs
        )

    async def compute_logprobs_async(self, sequence: ModelInput) -> list[float]:
        return await self._backend.compute_logprobs(sequence)


class TrainingClient:
    """Mirrors tinker.TrainingClient."""

    def __init__(self, backend: _TrainerBackend):
        self._backend = backend
        self._student_sampler: _SamplerBackend | None = None

    def get_tokenizer(self):
        return self._backend.get_tokenizer()

    async def forward_backward_async(
        self,
        data_D: list[Datum],
        loss_fn: LossFnType,
        loss_fn_config: dict[str, Any] | None = None,
    ) -> APIFuture[ForwardBackwardOutput]:
        out = await self._backend.forward_backward(data_D, loss_fn, loss_fn_config)
        return APIFuture(out)

    async def optim_step_async(self, adam_params: AdamParams) -> APIFuture[OptimStepResponse]:
        out = await self._backend.optim_step(adam_params)
        return APIFuture(out)

    async def save_state_async(
        self, name: str, ttl_seconds: int | None = None
    ) -> APIFuture[_SavePath]:
        del ttl_seconds
        out = await self._backend.save_state(name)
        return APIFuture(out)

    async def save_weights_for_sampler_async(
        self, name: str, ttl_seconds: int | None = None
    ) -> APIFuture[_SavePath]:
        del ttl_seconds
        out = await self._backend.save_weights_for_sampler(name)
        return APIFuture(out)

    async def save_weights_and_get_sampling_client_async(self) -> SamplingClient:
        saved = await self._backend.save_weights_for_sampler(f"adhoc-{int(time.time())}")
        return self._swap_student_sampler(saved.path)

    def create_sampling_client(self, sampler_path: str) -> SamplingClient:
        return self._swap_student_sampler(sampler_path)

    def _swap_student_sampler(self, path: str) -> SamplingClient:
        if self._student_sampler is not None:
            self._student_sampler.close()
        backend = _SamplerBackend(path)
        self._student_sampler = backend
        return SamplingClient(backend)


class ServiceClient:
    """Mirrors tinker.ServiceClient."""

    def __init__(self, base_url: str | None = None) -> None:
        # base_url is accepted for API compatibility; it has no effect locally.
        self.base_url = base_url

    async def create_lora_training_client_async(
        self,
        model_name: str | None = None,
        rank: int | None = None,
        user_metadata: dict[str, str] | None = None,
        *,
        base_model: str | None = None,
    ) -> TrainingClient:
        # Real tinker uses ``base_model=`` as the kwarg name; the cookbook's
        # ``supervised/train.py`` calls us with that name. Accept either.
        resolved_model = base_model if base_model is not None else model_name
        if resolved_model is None:
            raise TinkerError(
                "create_lora_training_client_async requires `model_name` or `base_model`"
            )
        if rank is None:
            raise TinkerError("create_lora_training_client_async requires `rank`")
        backend = await asyncio.to_thread(_TrainerBackend, resolved_model, rank, None, False)
        backend.set_user_metadata(user_metadata)
        return TrainingClient(backend)

    async def create_training_client_from_state_async(
        self,
        state_path: str,
        user_metadata: dict[str, str] | None = None,
    ) -> TrainingClient:
        meta = _CheckpointStore.read_meta(state_path)
        model_name = meta.get("model_name") or state_path
        lora_rank = meta.get("lora_rank")
        backend = await asyncio.to_thread(
            _TrainerBackend, model_name, lora_rank, state_path, False
        )
        backend.set_user_metadata(user_metadata)
        return TrainingClient(backend)

    async def create_training_client_from_state_with_optimizer_async(
        self,
        state_path: str,
        user_metadata: dict[str, str] | None = None,
    ) -> TrainingClient:
        meta = _CheckpointStore.read_meta(state_path)
        model_name = meta.get("model_name") or state_path
        lora_rank = meta.get("lora_rank")
        backend = await asyncio.to_thread(
            _TrainerBackend, model_name, lora_rank, state_path, True
        )
        backend.set_user_metadata(user_metadata)
        return TrainingClient(backend)

    def create_sampling_client(
        self, base_model: str, model_path: str | None = None
    ) -> SamplingClient:
        """Create a sampler for an external (teacher) model.

        Routes through an external vLLM HTTP server configured via
        ``TINTHER_TEACHER_URL`` / ``TINTHER_TEACHER_URLS``. In-process student
        checkpoints (``tinther://sampler/...``) must go through
        :meth:`TrainingClient.create_sampling_client` instead.
        """
        model_ref = model_path or base_model
        if str(model_ref).startswith("tinther://"):
            raise TinkerError(
                "ServiceClient.create_sampling_client is for external teacher models. "
                "Use TrainingClient.create_sampling_client for in-process student "
                "checkpoints at tinther:// paths."
            )
        return SamplingClient(_HTTPSamplerBackend.from_env(model_ref))

    def create_rest_client(self) -> _RestClientStub:
        return _RestClientStub()


# ---------------------------------------------------------------------------
# `tinker.types` submodule facade + sys.modules aliasing
# ---------------------------------------------------------------------------


def _build_types_module() -> _types_module.ModuleType:
    mod = _types_module.ModuleType("tinker.types")
    mod.LossFnType = LossFnType  # type: ignore[attr-defined]
    mod.SampleResponse = SampleResponse
    mod.ModelInput = ModelInput
    mod.TensorData = TensorData
    mod.Datum = Datum
    mod.SamplingParams = SamplingParams
    mod.AdamParams = AdamParams
    mod.ForwardBackwardOutput = ForwardBackwardOutput
    mod.OptimStepResponse = OptimStepResponse
    mod.StopReason = StopReason  # type: ignore[attr-defined]
    mod.EncodedTextChunk = EncodedTextChunk
    mod.ImageChunk = ImageChunk
    mod.ImageAssetPointerChunk = ImageAssetPointerChunk
    mod.ModelInputChunk = ModelInputChunk
    # Nested submodules referenced as `tinker.types.tensor_data`, etc.
    tensor_data_mod = _types_module.ModuleType("tinker.types.tensor_data")
    tensor_data_mod.TensorData = TensorData
    mod.tensor_data = tensor_data_mod
    image_chunk_mod = _types_module.ModuleType("tinker.types.image_chunk")
    image_chunk_mod.ImageChunk = ImageChunk
    image_chunk_mod.ImageAssetPointerChunk = ImageAssetPointerChunk
    mod.image_chunk = image_chunk_mod
    return mod


types = _build_types_module()


def _build_lib_module() -> _types_module.ModuleType:
    """Build a stub for ``tinker.lib`` and ``tinker.lib.public_interfaces``.

    The cookbook's ``supervised/train.py`` imports ``APIFuture`` from
    ``tinker.lib.public_interfaces``. We expose the same symbol our flat
    tinther module already defines.
    """
    lib_mod = _types_module.ModuleType("tinker.lib")
    public_interfaces_mod = _types_module.ModuleType("tinker.lib.public_interfaces")
    public_interfaces_mod.APIFuture = APIFuture
    lib_mod.public_interfaces = public_interfaces_mod
    return lib_mod


lib = _build_lib_module()


def install_as_tinker() -> None:
    """Register this module as ``tinker`` / ``tinker.types`` / ``tinker.lib``.

    Call this once, before importing any ``tinker_cookbook`` module, so
    every downstream ``import tinker`` resolves to ``tinther``.
    """
    this_mod = sys.modules[__name__]
    sys.modules["tinker"] = this_mod
    sys.modules["tinker.types"] = types
    sys.modules["tinker.types.tensor_data"] = types.tensor_data  # type: ignore[attr-defined]
    sys.modules["tinker.types.image_chunk"] = types.image_chunk  # type: ignore[attr-defined]
    sys.modules["tinker.lib"] = lib
    sys.modules["tinker.lib.public_interfaces"] = lib.public_interfaces  # type: ignore[attr-defined]


__all__ = [
    "APIFuture",
    "AdamParams",
    "Datum",
    "EncodedTextChunk",
    "ForwardBackwardOutput",
    "ImageAssetPointerChunk",
    "ImageChunk",
    "LossFnType",
    "ModelInput",
    "ModelInputChunk",
    "OptimStepResponse",
    "SampleResponse",
    "SamplingClient",
    "SamplingParams",
    "ServiceClient",
    "StopReason",
    "TensorData",
    "TinkerError",
    "TrainingClient",
    "_HTTPSamplerBackend",
    "install_as_tinker",
    "types",
]
