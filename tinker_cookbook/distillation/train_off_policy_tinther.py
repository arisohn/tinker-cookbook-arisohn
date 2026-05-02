"""Off-policy distillation using `tinther` in place of `tinker`.

This script is effectively ``tinker_cookbook/distillation/train_off_policy.py``
with ``tinker`` renamed to ``tinther``: we register ``tinther`` as
``sys.modules["tinker"]`` before importing the cookbook, so every transitive
``import tinker`` inside the cookbook resolves to this library.

Launch (multi-GPU)::

    accelerate launch --num_processes=$NGPUS \
        --config_file configs/accelerate_fsdp.yaml \
        tinker_cookbook/distillation/train_off_policy_tinther.py <chz args...>

Single GPU::

    TINTHER_BACKEND=ddp python tinker_cookbook/distillation/train_off_policy_tinther.py <chz args...>
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

# When launched as a script, Python prepends this file's directory to
# ``sys.path``. That directory contains ``datasets.py``, which shadows the
# HuggingFace ``datasets`` package and creates a circular import during
# cookbook startup. Remove it before anything inside ``tinker_cookbook``
# tries ``import datasets``.
_THIS_DIR = Path(__file__).resolve().parent
for _entry in (str(_THIS_DIR), ""):
    while _entry in sys.path:
        sys.path.remove(_entry)

# Load the sibling ``tinther.py`` file directly, BEFORE any tinker_cookbook
# import runs. See train_on_policy_tinther.py for the rationale.
_tinther_spec = importlib.util.spec_from_file_location("tinther", _THIS_DIR / "tinther.py")
assert _tinther_spec is not None and _tinther_spec.loader is not None
tinther = importlib.util.module_from_spec(_tinther_spec)
sys.modules["tinther"] = tinther
_tinther_spec.loader.exec_module(tinther)
tinther.install_as_tinker()

import chz  # noqa: E402

from tinker_cookbook.distillation.train_off_policy import Config, main  # noqa: E402


if __name__ == "__main__":
    cfg = chz.entrypoint(Config)
    asyncio.run(main(cfg))
