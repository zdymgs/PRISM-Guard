from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
from dataclasses import asdict, dataclass
from pathlib import Path
import cv2
import numpy as np

@dataclass(frozen=True)
class VideoMetadata:
    path: str
    frame_count: int
    fps: float
    width: int
    height: int
    duration_seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

def inspect_video_metadata(path: Path) -> VideoMetadata:
    """Read container metadata without materializing a frame cache."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f'Could not open MM-Fit RGB video: {path}')
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()
    if frame_count <= 0 or fps <= 0 or width <= 0 or (height <= 0):
        raise ValueError(f'Invalid video metadata for {path}')
    return VideoMetadata(path=str(path.resolve()), frame_count=frame_count, fps=fps, width=width, height=height, duration_seconds=frame_count / fps)

def validate_reference_mapping(reference_frames: np.ndarray, metadata: VideoMetadata, *, decoder_offset: int=0) -> dict[str, object]:
    """Check one explicit reference-frame to decoder-index hypothesis."""
    if not isinstance(decoder_offset, int):
        raise TypeError('decoder_offset must be an explicit integer')
    frames = np.asarray(reference_frames, dtype=np.float64)
    if frames.ndim != 1 or frames.size == 0 or (not np.isfinite(frames).all()):
        raise ValueError('Reference frames must be a non-empty finite vector')
    rounded = np.rint(frames)
    if not np.array_equal(frames, rounded):
        raise ValueError('MM-Fit reference frames must be integral')
    if np.any(np.diff(frames) < 0):
        raise ValueError('MM-Fit reference frames must be monotonic')
    minimum, maximum = (int(rounded[0]), int(rounded[-1]))
    decoder_minimum = minimum + decoder_offset
    decoder_maximum = maximum + decoder_offset
    if decoder_minimum < 0 or decoder_maximum >= metadata.frame_count:
        raise ValueError(f'Reference mapping is out of video bounds: {decoder_minimum}..{decoder_maximum} versus 0..{metadata.frame_count - 1}')
    return {'mapping': 'decoder_index = official_left_camera_reference_frame_id + decoder_offset', 'decoder_offset': decoder_offset, 'status': 'bounds_valid_visual_overlay_still_required', 'reference_frame_min': minimum, 'reference_frame_max': maximum, 'video_index_min': 0, 'video_index_max': metadata.frame_count - 1}

def draw_pose_overlay(rgb: np.ndarray, pose_2d: np.ndarray, frame_id: int) -> np.ndarray:
    """Draw same-id 2D joints for a human-verifiable zero-offset audit."""
    pose = np.asarray(pose_2d)
    clock = pose[0, :, 0]
    index = int(np.searchsorted(clock, frame_id, side='left'))
    if index >= pose.shape[1] or clock[index] != frame_id:
        raise ValueError(f'Pose frame {frame_id} unavailable')
    output = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR).copy()
    points = np.stack([pose[0, index, 1:], pose[1, index, 1:]], axis=1)
    for x, y in points:
        if np.isfinite(x) and np.isfinite(y) and (x >= 0) and (y >= 0):
            cv2.circle(output, (int(round(x)), int(round(y))), 5, (0, 255, 0), -1)
    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
