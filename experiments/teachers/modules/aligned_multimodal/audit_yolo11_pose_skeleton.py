from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
from pathlib import Path

def canonical_frame_id(path: Path, modality: str) -> str:
    stem = path.stem
    if modality == 'depth':
        if not stem.startswith('Depth_') or not stem.endswith('_Color'):
            raise ValueError(f'Unexpected Depth filename: {path.name}')
        return stem[len('Depth_'):-len('_Color')]
    if modality == 'ir':
        if not stem.startswith('IR_'):
            raise ValueError(f'Unexpected IR filename: {path.name}')
        return stem[len('IR_'):]
    if modality == 'skeleton':
        if not stem.startswith('Color_'):
            raise ValueError(f'Unexpected Skeleton filename: {path.name}')
        value = stem[len('Color_'):]
        suffix = value.rsplit('_', 1)[-1]
        if len(suffix) == 8 and suffix.isdigit():
            return suffix
        return value
    raise ValueError(modality)

def frame_map(trial_dir: Path, modality: str) -> dict[str, Path]:
    if modality == 'skeleton':
        files = (trial_dir / 'predictions').glob('*.json')
    else:
        files = (path for path in trial_dir.iterdir() if path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    return {canonical_frame_id(path, modality): path for path in files if path.is_file()}
