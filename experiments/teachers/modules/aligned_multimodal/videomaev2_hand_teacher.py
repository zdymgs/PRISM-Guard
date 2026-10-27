from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import numpy as np

def centers(boxes: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.full((len(boxes), 2), np.nan, dtype=np.float32)
    output[valid] = (boxes[valid, :2] + boxes[valid, 2:]) * 0.5
    for coordinate in range(2):
        values = output[:, coordinate]
        known = np.flatnonzero(np.isfinite(values))
        if len(known):
            output[:, coordinate] = np.interp(np.arange(len(values)), known, values[known])
        else:
            output[:, coordinate] = 0
    return output

def fallback_box(boxes: np.ndarray, valid: np.ndarray, frame: int, primary: int, workspace: int, person: int) -> np.ndarray:
    for index in (primary, workspace, person):
        if valid[frame, index]:
            return boxes[frame, index]
    return np.full(4, np.nan, dtype=np.float32)

def interaction_box(boxes: np.ndarray, valid: np.ndarray, frame: int, left: int, right: int, workspace: int, person: int) -> np.ndarray:
    hands = [boxes[frame, index] for index in (left, right) if valid[frame, index]]
    if hands:
        values = np.stack(hands)
        return np.asarray([values[:, 0].min(), values[:, 1].min(), values[:, 2].max(), values[:, 3].max()], dtype=np.float32)
    return fallback_box(boxes, valid, frame, workspace, workspace, person)
