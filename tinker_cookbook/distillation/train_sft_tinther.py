"""SFT fine-tune Qwen3-8B-Base on OpenThoughts3-1.2M-shuffle-1k via tinther.

This is the tinther counterpart of ``trl/train_sft.py``. We register
``tinther`` as ``sys.modules["tinker"]`` (so every transitive
``import tinker`` inside ``tinker_cookbook`` resolves to tinther) and
then reuse ``tinker_cookbook.supervised.train.main`` with a custom
dataset builder that loads the OpenThoughts3 parquet.

Hyperparameters mirror ``trl/train_sft.py``:
  - full fine-tuning (no LoRA)
  - bf16, FlashAttention-2, fused AdamW
  - lr 5e-5 with cosine schedule and warmup_ratio=0.03
  - weight_decay=0, max_grad_norm=1.0
  - 3 epochs, batch_size=1, max_length=8192
  - assistant-only loss (TrainOnWhat.ALL_ASSISTANT_MESSAGES)

Launch from the repo root (so ``tinker_cookbook`` is importable)::

    cd /workspace/tinker-cookbook-arisohn
    export PYTHONPATH=.

Multi-GPU::

    accelerate launch --num_processes=$NGPUS \
        --config_file configs/accelerate_fsdp.yaml \
        tinker_cookbook/distillation/train_sft_tinther.py

Single GPU::

    TINTHER_BACKEND=ddp python tinker_cookbook/distillation/train_sft_tinther.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from pathlib import Path

# When launched as a script, Python prepends this file's directory to
# ``sys.path``. That directory contains ``datasets.py``, which shadows the
# HuggingFace ``datasets`` package and creates a circular import during
# cookbook startup. Remove it before anything inside ``tinker_cookbook``
# tries ``import datasets``. (Same trick as ``train_off_policy_tinther.py``.)
_THIS_DIR = Path(__file__).resolve().parent
for _entry in (str(_THIS_DIR), ""):
    while _entry in sys.path:
        sys.path.remove(_entry)

# Load the sibling ``tinther.py`` directly and register it as ``tinker``
# BEFORE any tinker_cookbook module is imported.
_tinther_spec = importlib.util.spec_from_file_location("tinther", _THIS_DIR / "tinther.py")
assert _tinther_spec is not None and _tinther_spec.loader is not None
tinther = importlib.util.module_from_spec(_tinther_spec)
sys.modules["tinther"] = tinther
_tinther_spec.loader.exec_module(tinther)
tinther.install_as_tinker()


# ---------------------------------------------------------------------------
# Constants — mirror ``trl/train_sft.py``.
# ---------------------------------------------------------------------------

MODEL_PATH = (
    "/workspace/models--Qwen--Qwen3-8B-Base/"
    "snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
)
DATA_PATH = (
    "/workspace/datasets--open-thoughts--OpenThoughts3-1.2M-shuffle-1k/"
    "data/train-00000-of-00001.parquet"
)
OUTPUT_DIR = (
    "/workspace/tinker-cookbook-arisohn/tinker_cookbook/distillation/"
    "outputs/qwen3-8b-openthoughts-sft-tinther"
)

NUM_EPOCHS = 3
BATCH_SIZE = 1
LEARNING_RATE = 5e-5
WARMUP_RATIO = 0.03
MAX_LENGTH = 8192
RENDERER_NAME = "qwen3"


# ---------------------------------------------------------------------------
# Configure tinther via env vars BEFORE the cookbook builds the trainer.
# ---------------------------------------------------------------------------

import pandas as pd  # noqa: E402  (after path purge)

_N_EXAMPLES = len(pd.read_parquet(DATA_PATH))
_N_BATCHES = _N_EXAMPLES // BATCH_SIZE
_TOTAL_STEPS = _N_BATCHES * NUM_EPOCHS
_WARMUP_STEPS = max(1, round(WARMUP_RATIO * _TOTAL_STEPS))

# Checkpoints (state + sampler weights) live under TINTHER_CACHE_DIR. Default is
# /tmp/tinther which is too small for an 8B optimizer state — pin it under
# /workspace so saves go to the large network volume next to the run logs.
os.environ.setdefault(
    "TINTHER_CACHE_DIR",
    "/workspace/tinker-cookbook-arisohn/tinker_cookbook/distillation/outputs/"
    "qwen3-8b-openthoughts-sft-tinther/_tinther_cache",
)

os.environ.setdefault("TINTHER_MIXED_PRECISION", "bf16")
os.environ.setdefault("TINTHER_TRAINER_ATTN_IMPLEMENTATION", "fa2")
os.environ.setdefault("TINTHER_LR_SCHEDULER", "cosine")
os.environ.setdefault("TINTHER_LR_PEAK", str(LEARNING_RATE))
os.environ.setdefault("TINTHER_LR_TOTAL_STEPS", str(_TOTAL_STEPS))
os.environ.setdefault("TINTHER_LR_WARMUP_STEPS", str(_WARMUP_STEPS))
os.environ.setdefault("TINTHER_LR_MIN_RATIO", "0.0")
os.environ.setdefault("TINTHER_WEIGHT_DECAY", "0.0")
os.environ.setdefault("TINTHER_OPTIM_FUSED", "1")
os.environ.setdefault("TINTHER_MAX_GRAD_NORM", "1.0")


# ---------------------------------------------------------------------------
# Imports that depend on the tinker → tinther alias being installed.
# ---------------------------------------------------------------------------

import chz  # noqa: E402
import datasets as hf_datasets  # noqa: E402
import tinker  # noqa: E402  (resolves to tinther via install_as_tinker)

from tinker_cookbook.renderers import TrainOnWhat  # noqa: E402
from tinker_cookbook.supervised import train  # noqa: E402
from tinker_cookbook.supervised.data import (  # noqa: E402
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import (  # noqa: E402
    ChatDatasetBuilder,
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
)


# ---------------------------------------------------------------------------
# Dataset builder for OpenThoughts3 (parquet of {from, value} conversations).
# ---------------------------------------------------------------------------


@chz.chz
class OpenThoughts3Builder(ChatDatasetBuilder):
    """Loads the OpenThoughts3 parquet and maps {human, gpt} → {user, assistant}."""

    file_path: str = DATA_PATH

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        df = pd.read_parquet(self.file_path)
        role_map = {"human": "user", "gpt": "assistant"}
        msgs = [
            [{"role": role_map[t["from"]], "content": t["value"]} for t in conv]
            for conv in df["conversations"]
        ]
        ds = hf_datasets.Dataset.from_dict({"messages": msgs})

        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )
        renderer = self.renderer
        max_length = self.common_config.max_length

        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(
                row["messages"], renderer, max_length, train_on_what
            )

        train_ds = SupervisedDatasetFromHFDataset(
            ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        )
        return train_ds, None


# ---------------------------------------------------------------------------
# Build the cookbook training config and run.
# ---------------------------------------------------------------------------


def build_config() -> train.Config:
    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=MODEL_PATH,
        renderer_name=RENDERER_NAME,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    builder = OpenThoughts3Builder(common_config=common, file_path=DATA_PATH)

    # learning_rate / lr_schedule are set for completeness but tinther's own
    # cosine scheduler (TINTHER_LR_*) owns the per-step LR when enabled.
    return train.Config(
        log_path=OUTPUT_DIR,
        model_name=MODEL_PATH,
        renderer_name=RENDERER_NAME,
        dataset_builder=builder,
        learning_rate=LEARNING_RATE,
        lr_schedule="cosine",
        num_epochs=NUM_EPOCHS,
        lora_rank=0,  # full fine-tuning — tinther skips PEFT when rank == 0
        adam_beta1=0.9,
        adam_beta2=0.999,  # match HF AdamW default used by TRL
        adam_eps=1e-8,
        save_every=_N_BATCHES,  # one checkpoint per epoch
        eval_every=0,
        infrequent_eval_every=0,
        wandb_project=None,
        submit_ahead=0,  # local trainer, no API pipelining
    )


if __name__ == "__main__":
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    asyncio.run(train.main(build_config()))
