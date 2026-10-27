from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import cv2
import numpy as np
from ultralytics import YOLO

def letterbox_rgb(image_bgr: np.ndarray, size: int=160) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = image_bgr.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = (max(1, round(w * scale)), max(1, round(h * scale)))
    rgb = cv2.cvtColor(cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    canvas = np.zeros((size, size, 3), np.uint8)
    x = (size - nw) // 2
    y = (size - nh) // 2
    canvas[y:y + nh, x:x + nw] = rgb
    return (canvas.transpose(2, 0, 1), scale, (x, y))

def square_crop(image: np.ndarray, box: np.ndarray, scale: float) -> np.ndarray:
    h, w = image.shape[:2]
    if box.shape != (4,) or not np.isfinite(box).all():
        return image
    x1, y1, x2, y2 = map(float, box)
    side = max(x2 - x1, y2 - y1) * scale
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    left = max(0, round(cx - side / 2))
    right = min(w, round(cx + side / 2))
    top = max(0, round(cy - side / 2))
    bottom = min(h, round(cy + side / 2))
    return image[top:bottom, left:right] if right > left + 2 and bottom > top + 2 else image

def workspace_box(keypoints: np.ndarray, person: np.ndarray, conf: float=0.25) -> tuple[np.ndarray, bool]:
    boxes = []
    for shoulder, elbow, wrist in ((5, 7, 9), (6, 8, 10)):
        valid = [i for i in (shoulder, elbow, wrist) if np.isfinite(keypoints[i]).all() and keypoints[i, 2] >= conf]
        if len(valid) >= 2:
            xy = keypoints[valid, :2]
            span = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 0.12 * float(person[3] - person[1]))
            boxes.append(np.array([xy[:, 0].min() - 1.6 * span / 2, xy[:, 1].min() - 1.6 * span / 2, xy[:, 0].max() + 1.6 * span / 2, xy[:, 1].max() + 1.6 * span / 2], np.float32))
    if boxes:
        b = np.array([min((x[0] for x in boxes)), min((x[1] for x in boxes)), max((x[2] for x in boxes)), max((x[3] for x in boxes))], np.float32)
        return (b, True)
    h = float(person[3] - person[1])
    return (np.array([person[0], person[1], person[2], person[1] + 0.68 * h], np.float32), False)

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = (max(a[0], b[0]), max(a[1], b[1]))
    x2, y2 = (min(a[2], b[2]), min(a[3], b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = max(1e-09, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return float(inter / union)

def select_person(boxes: np.ndarray, previous: np.ndarray | None) -> int:
    if len(boxes) == 0:
        return -1
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    if previous is None:
        return int(np.argmax(areas))
    score = np.asarray([_iou(b, previous) for b in boxes]) * 2.0 + areas / max(float(areas.max()), 1.0)
    return int(np.argmax(score))
