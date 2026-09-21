"""Glue code for the Laya Multilingual playground.

Import order matters: ``USE_TF=0`` must be set before ``transformers`` (and therefore
``laya``) is imported, otherwise ``transformers`` probes for TensorFlow at import time and
an installed TensorFlow can deadlock model construction. See docs/troubleshooting.md.
"""

from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")

import time
from pathlib import Path
from typing import Any

MODEL_ID = "convaiinnovations/laya-multilingual"

MODELSCOPE_URL = f"https://www.modelscope.cn/models/{MODEL_ID}"
HUGGINGFACE_URL = f"https://huggingface.co/{MODEL_ID}"

REQUIRED_FILES = (
    "README.md",
    "model.safetensors",
    "rl_agent_config.json",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def model_dir() -> Path:
    return repo_root() / "models" / "laya-multilingual"


def missing_files(directory: Path | None = None) -> list[str]:
    """Return required checkpoint files that are absent (per file, not just the directory)."""
    directory = Path(directory) if directory is not None else model_dir()
    return [name for name in REQUIRED_FILES if not (directory / name).is_file()]


def dir_size_mib(directory: Path) -> float:
    return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file()) / 1024**2


def load_agent(directory: Path | None = None, device: str | None = None):
    """Load the ModelScope-downloaded checkpoint from disk with the Laya SDK.

    ``laya.load()`` accepts a local directory: it only reaches for a hub download when the
    path does not exist (laya/agent.py, ``Agent.__init__``). Nothing is fetched from
    Hugging Face here.
    """
    import laya

    path = Path(directory) if directory is not None else model_dir()
    missing = missing_files(path)
    if missing:
        raise FileNotFoundError(
            f"{path} is not a complete Laya checkpoint, missing: {', '.join(missing)}. "
            "Run `make download` (or `uv run python scripts/download_model.py`) first."
        )
    return laya.load(str(path.resolve()), device=device)


def describe_runtime(agent) -> str:
    """Device Laya actually picked, plus whether torch sees a CUDA device."""
    device = str(getattr(getattr(agent, "device", None), "type", "cpu")).upper()
    return f"{device} (CUDA available: {'yes' if cuda_available() else 'no'})"


def cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def predict_timed(agent, state: Any, questions: dict) -> tuple[dict, float]:
    """One forward pass, measured with time.perf_counter(); returns (result, milliseconds)."""
    start = time.perf_counter()
    result = agent.predict(state, questions)
    return result, (time.perf_counter() - start) * 1000.0


def banner(title: str) -> None:
    """Header shared by the examples: what model, from where."""
    print("=" * 40)
    print(title)
    print("=" * 40)
    print("\nModel:")
    print(MODEL_ID)
    print("\nSource:")
    print("ModelScope")
    print()


def answer_lines(name: str, answer: dict[str, Any]) -> list[str]:
    """Human-readable rendering of one typed answer, whatever primitive it is."""
    kind = answer.get("type")
    out = [f"{name} [{kind}]"]
    probabilities = answer.get("probabilities") or {}
    if kind == "choice":
        out.append(f"  choice     = {answer['choice']}")
    elif kind == "score":
        levels = len(probabilities)
        best = max(probabilities, key=probabilities.get)
        out.append(f"  score      = {answer['score']}  (expected level, 0-based)")
        out.append(f"  level      = {int(best) + 1}/{levels}  {answer['legend'][best]}")
    elif kind == "noul":
        out.append(f"  noul       = {answer['noul']}  (P(true))")
    out.append(f"  confidence = {answer.get('confidence')}")
    legend = answer.get("legend", {})
    for rank, (key, value) in enumerate(probabilities.items()):
        out.append(f"    {'level ' + str(rank + 1) if kind == 'score' else key:<10}"
                   f"{value:>7.4f}  {legend.get(key, '')}")
    out.append(f"  action     = {answer.get('action')}")
    return out
