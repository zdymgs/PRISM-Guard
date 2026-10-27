from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
from functools import lru_cache
import cv2
import numpy as np

@lru_cache(maxsize=1)
def jet_palette_rgb() -> np.ndarray:
    """Return OpenCV COLORMAP_JET as 256 unique RGB triplets."""
    values = np.arange(256, dtype=np.uint8)[:, None]
    return cv2.applyColorMap(values, cv2.COLORMAP_JET)[:, 0, ::-1].copy()

@lru_cache(maxsize=1)
def jet_lookup_table() -> np.ndarray:
    """Map packed 24-bit RGB values to their JET index; -1 means no exact match."""
    palette = jet_palette_rgb().astype(np.int64)
    keys = palette[:, 0] << 16 | palette[:, 1] << 8 | palette[:, 2]
    lookup = np.full(1 << 24, -1, dtype=np.int16)
    lookup[keys] = np.arange(256, dtype=np.int16)
    return lookup

def decode_jet_rgb(rgb: np.ndarray, *, repair_unmatched: bool=True) -> tuple[np.ndarray, np.ndarray, int]:
    """Decode an RGB JET image into uint8 indices and a separate validity mask.

    Pure black is treated as invalid because black is not present in OpenCV's JET
    palette. Non-black off-palette pixels can be repaired with nearest-palette
    matching, which is useful for rare damaged or recompressed inputs.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f'Depth RGB shape must be HxWx3, got {rgb.shape}')
    rgb = np.asarray(rgb, dtype=np.uint8)
    valid = np.any(rgb != 0, axis=2)
    packed = rgb[..., 0].astype(np.int64) << 16 | rgb[..., 1].astype(np.int64) << 8 | rgb[..., 2].astype(np.int64)
    decoded = jet_lookup_table()[packed]
    unmatched = valid & (decoded < 0)
    unmatched_count = int(unmatched.sum())
    if unmatched_count:
        if not repair_unmatched:
            raise ValueError(f'Found {unmatched_count} non-black pixels outside JET palette')
        palette = jet_palette_rgb().astype(np.int32)
        colors, inverse = np.unique(rgb[unmatched].reshape(-1, 3), axis=0, return_inverse=True)
        distances = ((colors[:, None].astype(np.int32) - palette[None]) ** 2).sum(axis=2)
        repaired = distances.argmin(axis=1).astype(np.int16)
        decoded[unmatched] = repaired[inverse]
    decoded = np.where(valid, decoded, 0).astype(np.uint8)
    return (decoded, valid.astype(np.uint8), unmatched_count)

def resize_decoded_depth(depth_index: np.ndarray, valid: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Resize decoded depth without mixing invalid pixels into valid depth values."""
    depth = np.asarray(depth_index, dtype=np.float32)
    mask = np.asarray(valid, dtype=np.float32)
    size = (int(width), int(height))
    numerator = cv2.resize(depth * mask, size, interpolation=cv2.INTER_LINEAR)
    denominator = cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)
    resized_valid = denominator >= 0.5
    resized_depth = np.zeros((height, width), dtype=np.float32)
    np.divide(numerator, np.maximum(denominator, 1e-06), out=resized_depth, where=denominator > 1e-06)
    resized_depth = np.clip(np.rint(resized_depth), 0, 255).astype(np.uint8)
    resized_depth[~resized_valid] = 0
    return (resized_depth, resized_valid.astype(np.uint8))
