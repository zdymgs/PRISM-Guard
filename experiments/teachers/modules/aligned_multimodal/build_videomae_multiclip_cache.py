from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
from pathlib import Path
import numpy as np
from aligned_multimodal.audit_yolo11_pose_skeleton import frame_map
from aligned_multimodal.build_shared_dir_roi_feature_cache import read_ir
from aligned_multimodal.build_videomae_cache import VIEW_NAMES, atomic_npz, encode, restore_attention_biases, safe_relative, square_crop
WINDOW_NAMES = ('early', 'late')
WINDOW_BOUNDS = ((0.0, 0.7), (0.3, 1.0))
FRAME_COUNT = 16
HIDDEN_SIZE = 1024
CLASSIFIER_CLASSES = 400

def window_indices(frame_count: int, low: float, high: float) -> np.ndarray:
    if frame_count < 1 or not 0.0 <= low < high <= 1.0:
        raise ValueError('Invalid temporal window')
    end = frame_count - 1
    return np.rint(np.linspace(low * end, high * end, FRAME_COUNT)).astype(np.int64)

def prepare_trial(row: dict[str, str], roi_run: Path) -> tuple[list[list[np.ndarray]], dict[str, np.ndarray]]:
    roi_path = roi_run / 'trial_roi_cache' / safe_relative(row['source_id']).with_suffix('.npz')
    with np.load(roi_path, allow_pickle=False) as data:
        frame_ids = np.asarray(data['frame_ids']).astype(str)
        region_names = tuple(np.asarray(data['region_names']).astype(str))
        boxes = np.asarray(data['roi_boxes_xyxy'], dtype=np.float32)
        valid = np.asarray(data['roi_valid'], dtype=bool)
        quality = np.asarray(data['roi_quality'], dtype=np.float32)
    person_index = region_names.index('full_body')
    workspace_index = region_names.index('hand_workspace')
    ir_paths = frame_map(Path(row['ir_dir']), 'ir')
    clips: list[list[np.ndarray]] = []
    all_indices: list[np.ndarray] = []
    person_valid: list[np.ndarray] = []
    workspace_valid: list[np.ndarray] = []
    person_quality: list[np.ndarray] = []
    workspace_quality: list[np.ndarray] = []
    for low, high in WINDOW_BOUNDS:
        chosen = window_indices(len(frame_ids), low, high)
        all_indices.append(chosen)
        scene: list[np.ndarray] = []
        person: list[np.ndarray] = []
        workspace: list[np.ndarray] = []
        for index in chosen:
            frame_id = frame_ids[index]
            if frame_id not in ir_paths:
                raise RuntimeError(f'Missing IR frame {frame_id}: {row['source_id']}')
            image = read_ir(ir_paths[frame_id])
            person_box = boxes[index, person_index] if valid[index, person_index] else np.full(4, np.nan)
            workspace_box = boxes[index, workspace_index] if valid[index, workspace_index] else person_box
            scene.append(image)
            person.append(square_crop(image, person_box, scale=1.15))
            workspace.append(square_crop(image, workspace_box, scale=1.4))
        clips.extend((scene, person, workspace))
        person_valid.append(valid[chosen, person_index].astype(np.uint8))
        workspace_valid.append(valid[chosen, workspace_index].astype(np.uint8))
        person_quality.append(quality[chosen, person_index].astype(np.float32))
        workspace_quality.append(quality[chosen, workspace_index].astype(np.float32))
    return (clips, {'frame_indices': np.stack(all_indices).astype(np.int16), 'person_valid': np.stack(person_valid), 'workspace_valid': np.stack(workspace_valid), 'person_quality': np.stack(person_quality), 'workspace_quality': np.stack(workspace_quality)})
