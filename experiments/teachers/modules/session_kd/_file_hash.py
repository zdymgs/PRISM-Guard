from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import hashlib
from pathlib import Path

def sha256_file(path: Path, block_size: int=4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while (block := handle.read(block_size)):
            digest.update(block)
    return digest.hexdigest()
