from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import hashlib
from pathlib import Path
import numpy as np

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def native3streams_from_raw(raw: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reorder raw slots (phone acc, watch acc, phone gyro) explicitly."""
    values, present = (np.asarray(raw), np.asarray(mask, dtype=bool))
    if values.ndim not in (4, 5) or values.shape[-4:] != (2, 16, 4, 9) or present.shape != values.shape:
        raise ValueError('raw streams must have shape [2,16,4,9] or [B,2,16,4,9]')
    return ((values[..., 0:3].astype(np.float32, copy=True), values[..., 6:9].astype(np.float32, copy=True), values[..., 3:6].astype(np.float32, copy=True)), (present[..., 0:3].copy(), present[..., 6:9].copy(), present[..., 3:6].copy()))

def assemblemodules_part_grid_from_raw(raw: np.ndarray, mask: np.ndarray):
    """Adapt a cache sample/batch to the mature native part-grid encoder."""
    from session_kd.mmact_imu import assemblemodules_part_grid
    values = np.asarray(raw)
    streams, masks = native3streams_from_raw(values, mask)
    if values.ndim == 4:
        streams = tuple((value[None] for value in streams))
        masks = tuple((value[None] for value in masks))
    return assemblemodules_part_grid(*streams, *masks)
