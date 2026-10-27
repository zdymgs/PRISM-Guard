from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
from typing import Any
import numpy as np
from PIL import Image

def bounded_indices(low: int, high: int, count: int) -> np.ndarray:
    if high <= low:
        return np.full(count, max(low, 0), dtype=np.int64)
    return np.rint(np.linspace(low, high - 1, count)).astype(np.int64)

def one_trial(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError('trial loader must use batch_size=1')
    return batch[0]

def model_ready_video(video: list[np.ndarray], frames_per_clip: int, crop_size: int) -> np.ndarray:
    """Match V-JEPA resize+center-crop before the processor batches frames."""
    frames = []
    for frame in video:
        image = Image.fromarray(np.asarray(frame))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        width, height = image.size
        resize_short = int(round(crop_size * 292.0 / 256.0))
        scale = resize_short / max(min(width, height), 1)
        resized = image.resize((max(crop_size, int(round(width * scale))), max(crop_size, int(round(height * scale)))), Image.Resampling.BILINEAR)
        left = max((resized.width - crop_size) // 2, 0)
        top = max((resized.height - crop_size) // 2, 0)
        frames.append(np.asarray(resized.crop((left, top, left + crop_size, top + crop_size))))
    value = np.stack(frames)
    if value.shape != (frames_per_clip, crop_size, crop_size, 3):
        raise ValueError(f'unexpected prepared video shape: {value.shape}')
    return value
