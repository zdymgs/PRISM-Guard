"""Deterministic seeds and immutable configuration snapshots."""

from __future__ import annotations
import hashlib
import json
import random
from pathlib import Path
from collections.abc import Mapping
import numpy as np


def seed_everything(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonical_config_hash(config: Mapping) -> str:
    raw = json.dumps(
        dict(config), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_config_snapshot(config: Mapping, output_root: Path) -> str:
    raw = json.dumps(
        dict(config), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    digest = canonical_config_hash(config)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    files = {output_root / "config.json": raw, output_root / "config_hash.txt": digest + "\n"}
    for path, value in files.items():
        if path.exists() and path.read_text(encoding="utf-8") != value:
            raise FileExistsError("frozen configuration differs")
    for path, value in files.items():
        if not path.exists():
            with path.open("x", encoding="utf-8") as handle:
                handle.write(value)
    return digest
