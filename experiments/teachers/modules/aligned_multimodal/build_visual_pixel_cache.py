from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import argparse
import csv
import json
import time
from pathlib import Path
import cv2
import numpy as np
from aligned_multimodal.audit_yolo11_pose_skeleton import frame_map
from aligned_multimodal.build_shared_dir_roi_feature_cache import read_ir
from aligned_multimodal.build_videomae_cache import safe_relative, square_crop
PROJECT_DIR = WORK_ROOT / 'aligned_multimodal'
DEFAULT_MANIFEST = PROJECT_DIR / 'data/single_split.csv'
DEFAULT_ROI = PROJECT_DIR / 'runs/dir_multiscale_roi_full'
DEFAULT_OUTPUT = PROJECT_DIR / 'runs/visual_pixel_cache'
WINDOW_NAMES = ('early', 'late')
WINDOW_BOUNDS = ((0.0, 0.7), (0.3, 1.0))
VIEW_NAMES = ('scene', 'person', 'workspace')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build compact uint8 IR clips for student V2.')
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--roi-run', type=Path, default=DEFAULT_ROI)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--frames', type=int, default=8)
    parser.add_argument('--resolution', type=int, default=112)
    parser.add_argument('--max-trials', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()

def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: row['sample_id'])
    if len(rows) != 2914:
        raise RuntimeError(f'student full40 universe changed: {len(rows)}')
    return rows

def window_indices(frame_count: int, low: float, high: float, frames: int) -> np.ndarray:
    if frame_count < 1 or frames < 2:
        raise ValueError('invalid frame count')
    end = frame_count - 1
    return np.rint(np.linspace(low * end, high * end, frames)).astype(np.int64)

def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + '.building')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)

def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=('sample_id', 'source_id', 'user_id', 'class_id'))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

def create_or_open(output: Path, rows: list[dict[str, str]], frames: int, resolution: int, overwrite: bool) -> tuple[np.memmap, np.memmap, np.memmap, np.memmap, np.memmap]:
    images_path = output / 'images.npy'
    completed_path = output / 'completed.npy'
    valid_path = output / 'view_valid.npy'
    quality_path = output / 'view_quality.npy'
    source_index_path = output / 'source_frame_indices.npy'
    shape = (len(rows), 2, frames, 3, resolution, resolution)
    if overwrite or not images_path.exists():
        images = np.lib.format.open_memmap(images_path, mode='w+', dtype=np.uint8, shape=shape)
        completed = np.lib.format.open_memmap(completed_path, mode='w+', dtype=np.uint8, shape=(len(rows),))
        valid = np.lib.format.open_memmap(valid_path, mode='w+', dtype=np.uint8, shape=(len(rows), 2, frames, 3))
        quality = np.lib.format.open_memmap(quality_path, mode='w+', dtype=np.float16, shape=(len(rows), 2, frames, 3))
        source_index = np.lib.format.open_memmap(source_index_path, mode='w+', dtype=np.int32, shape=(len(rows), 2, frames))
        images[:] = 0
        completed[:] = 0
        valid[:] = 0
        quality[:] = 0
        source_index[:] = -1
        images.flush()
        completed.flush()
        valid.flush()
        quality.flush()
        source_index.flush()
        write_rows(output / 'rows.csv', rows)
        return (images, completed, valid, quality, source_index)
    images = np.lib.format.open_memmap(images_path, mode='r+')
    completed = np.lib.format.open_memmap(completed_path, mode='r+')
    valid = np.lib.format.open_memmap(valid_path, mode='r+')
    quality = np.lib.format.open_memmap(quality_path, mode='r+')
    if source_index_path.exists():
        source_index = np.lib.format.open_memmap(source_index_path, mode='r+')
    else:
        source_index = np.lib.format.open_memmap(source_index_path, mode='w+', dtype=np.int32, shape=(len(rows), 2, frames))
        source_index[:] = -1
        source_index.flush()
    if images.shape != shape or completed.shape != (len(rows),):
        raise RuntimeError('existing student pixel cache shape differs; use --overwrite')
    with (output / 'rows.csv').open('r', encoding='utf-8-sig', newline='') as handle:
        existing = [row['sample_id'] for row in csv.DictReader(handle)]
    if existing != [row['sample_id'] for row in rows]:
        raise RuntimeError('existing student pixel cache row order differs; use --overwrite')
    if source_index.shape != (len(rows), 2, frames):
        raise RuntimeError('existing student source-index cache shape differs; use --overwrite')
    return (images, completed, valid, quality, source_index)

def main() -> None:
    args = parse_args()
    if args.frames < 2 or args.resolution < 64:
        raise ValueError('frames must be >=2 and resolution must be >=64')
    rows = read_rows(args.manifest.resolve())
    if args.max_trials > 0:
        rows = rows[:args.max_trials]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    images, completed, valid_out, quality_out, source_index_out = create_or_open(output, rows, args.frames, args.resolution, args.overwrite)
    started = time.perf_counter()
    built = skipped = 0
    for row_index, row in enumerate(rows):
        pixels_complete = bool(completed[row_index])
        indices_complete = bool(np.all(source_index_out[row_index] >= 0))
        if pixels_complete and indices_complete:
            skipped += 1
            continue
        roi_path = args.roi_run.resolve() / 'trial_roi_cache' / safe_relative(row['source_id']).with_suffix('.npz')
        with np.load(roi_path, allow_pickle=False) as data:
            frame_ids = np.asarray(data['frame_ids']).astype(str)
            region_names = tuple(np.asarray(data['region_names']).astype(str))
            boxes = np.asarray(data['roi_boxes_xyxy'], dtype=np.float32)
            valid = np.asarray(data['roi_valid'], dtype=bool)
            quality = np.asarray(data['roi_quality'], dtype=np.float32)
        person_index = region_names.index('full_body')
        workspace_index = region_names.index('hand_workspace')
        chosen_windows = [window_indices(len(frame_ids), low, high, args.frames) for low, high in WINDOW_BOUNDS]
        source_index_out[row_index] = np.stack(chosen_windows).astype(np.int32)
        if pixels_complete:
            built += 1
            if built % 25 == 0 or row_index + 1 == len(rows):
                source_index_out.flush()
                print(json.dumps({'processed': row_index + 1, 'total': len(rows), 'source_indices_backfilled': built, 'skipped': skipped, 'elapsed_seconds': round(time.perf_counter() - started, 1)}), flush=True)
            continue
        ir_paths = frame_map(Path(row['ir_dir']), 'ir')
        unique_indices = sorted(set(np.concatenate(chosen_windows).tolist()))
        frame_images: dict[int, np.ndarray] = {}
        for frame_index in unique_indices:
            frame_id = frame_ids[frame_index]
            if frame_id not in ir_paths:
                raise RuntimeError(f'missing IR frame {frame_id}: {row['source_id']}')
            frame_images[frame_index] = read_ir(ir_paths[frame_id])[:, :, 0]
        for window_index, chosen in enumerate(chosen_windows):
            for time_index, frame_index in enumerate(chosen):
                image = frame_images[int(frame_index)]
                person_valid = bool(valid[frame_index, person_index])
                workspace_valid = bool(valid[frame_index, workspace_index])
                person_box = boxes[frame_index, person_index] if person_valid else np.full(4, np.nan, dtype=np.float32)
                workspace_box = boxes[frame_index, workspace_index] if workspace_valid else person_box
                crops = (image, square_crop(image, person_box, scale=1.15), square_crop(image, workspace_box, scale=1.4))
                for view_index, crop in enumerate(crops):
                    images[row_index, window_index, time_index, view_index] = cv2.resize(crop, (args.resolution, args.resolution), interpolation=cv2.INTER_AREA)
                valid_out[row_index, window_index, time_index] = (1, int(person_valid), int(workspace_valid or person_valid))
                quality_out[row_index, window_index, time_index] = (1.0, quality[frame_index, person_index] if person_valid else 0.25, quality[frame_index, workspace_index] if workspace_valid else quality[frame_index, person_index] * 0.5 if person_valid else 0.2)
        completed[row_index] = 1
        built += 1
        if built % 25 == 0 or row_index + 1 == len(rows):
            images.flush()
            completed.flush()
            valid_out.flush()
            quality_out.flush()
            source_index_out.flush()
            print(json.dumps({'processed': row_index + 1, 'total': len(rows), 'built': built, 'skipped': skipped, 'elapsed_seconds': round(time.perf_counter() - started, 1)}), flush=True)
    summary = {'protocol': 'student V2 trainable-small-visual pixel cache', 'trials': len(rows), 'windows': list(WINDOW_NAMES), 'window_bounds': WINDOW_BOUNDS, 'frames_per_window': args.frames, 'views': list(VIEW_NAMES), 'resolution': args.resolution, 'shape': list(images.shape), 'dtype': 'uint8', 'cache_bytes': int(sum(((output / name).stat().st_size for name in ('images.npy', 'completed.npy', 'view_valid.npy', 'view_quality.npy', 'source_frame_indices.npy')))), 'source_frame_indices': {'shape': [len(rows), 2, args.frames], 'meaning': 'zero-based indices into the exact roi/motion_source shared trial frame axis'}, 'large_videomae_input': False, 'elapsed_seconds': time.perf_counter() - started}
    atomic_json(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
