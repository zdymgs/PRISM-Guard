from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
from pathlib import Path
import cv2
import numpy as np
from aligned_multimodal.audit_yolo11_pose_skeleton import frame_map
from aligned_multimodal.build_shared_dir_roi_feature_cache import read_depth
from aligned_multimodal.build_videomae_cache import DEFAULT_ROI, MODEL_NAME, PROJECT_DIR, VIEW_NAMES, atomic_npz, encode, read_rows, restorecore_attention_biases, safe_relative, square_crop, uniform_indices

def thermal_dir(row: dict[str, str]) -> Path:
    source = str(Path(row['ir_dir']).resolve())
    marker = f'{Path().anchor}IR{Path().anchor}' if Path().anchor else ''
    candidates = ('\\IR\\', '/IR/')
    for candidate in candidates:
        if candidate in source:
            return Path(source.replace(candidate, candidate[0] + 'Thermal' + candidate[-1], 1))
    source_path = Path(source)
    if source_path.name.casefold() == 'ir':
        return source_path.parent / 'Thermal'
    raise RuntimeError(f'could not derive Thermal path from {row['ir_dir']}')

def read_thermal(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f'could not read Thermal image: {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def prepare_trial(row: dict[str, str], roi_run: Path, modality: str) -> tuple[list[list[np.ndarray]], dict[str, np.ndarray]]:
    roi_path = roi_run / 'trial_roi_cache' / safe_relative(row['source_id']).with_suffix('.npz')
    with np.load(roi_path, allow_pickle=False) as data:
        ir_frame_ids = np.asarray(data['frame_ids']).astype(str)
        region_names = tuple(np.asarray(data['region_names']).astype(str))
        boxes = np.asarray(data['roi_boxes_xyxy'], dtype=np.float32)
        valid = np.asarray(data['roi_valid'], dtype=bool)
    person_index = region_names.index('full_body')
    workspace_index = region_names.index('hand_workspace')
    if modality == 'depth':
        paths = frame_map(Path(row['depth_dir']), 'depth')
        chosen_roi = uniform_indices(len(ir_frame_ids))
        selected_paths = [paths[ir_frame_ids[index]] for index in chosen_roi]
        reader = read_depth
    else:
        files = sorted(thermal_dir(row).glob('*.jpg'))
        if not files:
            raise RuntimeError(f'missing Thermal frames: {row['source_id']}')
        chosen_thermal = uniform_indices(len(files))
        selected_paths = [files[index] for index in chosen_thermal]
        if len(files) == 1:
            chosen_roi = np.zeros(16, dtype=np.int64)
        else:
            chosen_roi = np.rint(chosen_thermal / (len(files) - 1) * (len(ir_frame_ids) - 1)).astype(np.int64)
        reader = read_thermal
    scene: list[np.ndarray] = []
    person: list[np.ndarray] = []
    workspace: list[np.ndarray] = []
    for path, roi_index in zip(selected_paths, chosen_roi):
        image = reader(path)
        height, width = image.shape[:2]
        scale = np.asarray([width / 640.0, height / 480.0] * 2, dtype=np.float32)
        person_box = boxes[roi_index, person_index] * scale if valid[roi_index, person_index] else np.asarray([np.nan] * 4)
        workspace_box = boxes[roi_index, workspace_index] * scale if valid[roi_index, workspace_index] else person_box
        scene.append(image)
        person.append(square_crop(image, person_box, scale=1.15))
        workspace.append(square_crop(image, workspace_box, scale=1.4))
    return ([scene, person, workspace], {'roi_frame_indices': chosen_roi.astype(np.int16), 'person_valid': valid[chosen_roi, person_index].astype(np.uint8), 'workspace_valid': valid[chosen_roi, workspace_index].astype(np.uint8)})
