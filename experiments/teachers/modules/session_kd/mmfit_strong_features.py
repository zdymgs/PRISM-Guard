from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
import cv2
import numpy as np
import pandas as pd
import torch
INPUT_COLUMNS = ('segment_id', 'split', 'workout', 'person', 'start_pose_index', 'start_frame', 'center_frame')
FRAME_OFFSETS = np.rint(np.linspace(0, 149, 16)).astype(np.int64)
EXPECTED_ROWS = 40475
MODEL_ORDER = ('videomaev2_large', 'vjepa2_vitl_ssv2')
MODEL_FILES = {'videomaev2_large': 'videomaev2_large_16x1024_fp16.npy', 'vjepa2_vitl_ssv2': 'vjepa2_vitl_ssv2_16x1024_fp16.npy'}
MODEL_SNAPSHOTS = {'videomaev2_large': Path(str(PRETRAINED_ROOT / 'huggingface/hub/models--OpenGVLab--VideoMAEv2-Large/snapshots/9981a9c8f77118c421e5228e1b219468a4b0238d')), 'vjepa2_vitl_ssv2': Path(str(PRETRAINED_ROOT / 'huggingface/hub/models--facebook--vjepa2-vitl-fpc16-256-ssv2/snapshots/4aa02df83918538fc21cfaf576382fa20e489a80'))}

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def ordered_id_sha256(ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in ids:
        digest.update(str(value).encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()

def load_inputs(path: Path, expected_rows: int=EXPECTED_ROWS) -> pd.DataFrame:
    """Load an explicit non-label column allow-list from the frozen V1 index."""
    rows = pd.read_csv(path, usecols=list(INPUT_COLUMNS))
    if tuple(rows.columns) != INPUT_COLUMNS:
        rows = rows.loc[:, list(INPUT_COLUMNS)]
    if expected_rows and len(rows) != expected_rows:
        raise ValueError(f'expected {expected_rows} V1 rows, found {len(rows)}')
    if rows.segment_id.isna().any() or rows.segment_id.duplicated().any():
        raise ValueError('segment_id must be complete and unique')
    if not rows.segment_id.astype(str).str.startswith('mmfit:').all():
        raise ValueError('unexpected segment_id namespace')
    starts = rows.start_frame.to_numpy()
    if not np.issubdtype(starts.dtype, np.integer):
        if not np.equal(starts, np.rint(starts)).all():
            raise ValueError('start_frame must be integral')
        rows['start_frame'] = starts.astype(np.int64)
    return rows

def center_resize(frame: np.ndarray, size: int) -> np.ndarray:
    """Center-square crop followed by resize, returned as contiguous RGB uint8."""
    height, width = frame.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    crop = frame[top:top + side, left:left + side]
    return np.ascontiguousarray(cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA))

def vjepa_resize_crop(frame: np.ndarray, crop_size: int=256) -> np.ndarray:
    """Match the established V-JEPA short-side 292/256 resize and center crop."""
    height, width = frame.shape[:2]
    resize_short = int(round(crop_size * 292.0 / 256.0))
    scale = resize_short / max(min(height, width), 1)
    new_width = max(crop_size, int(round(width * scale)))
    new_height = max(crop_size, int(round(height * scale)))
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    left = max((new_width - crop_size) // 2, 0)
    top = max((new_height - crop_size) // 2, 0)
    return np.ascontiguousarray(resized[top:top + crop_size, left:left + crop_size])

@dataclass
class Encoder:
    name: str
    feature_dim: int
    input_size: int
    parameters: int
    checkpoint: Path
    preprocess: str
    kernel: str
    encode: Callable[[list[np.ndarray]], np.ndarray]

def load_encoder(name: str, device: torch.device) -> Encoder:
    snapshot = MODEL_SNAPSHOTS[name]
    if name == 'videomaev2_large':
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, VideoMAEImageProcessor
        config = AutoConfig.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True)
        model = AutoModel.from_config(config, trust_remote_code=True)
        checkpoint = snapshot / 'model.safetensors'
        model.load_state_dict(load_file(str(checkpoint), device='cpu'), strict=True)
        model = model.eval().to(device=device, dtype=torch.float16)
        from session_kd.mmfit_videomae_sdpa import enable_sdpa
        patch = enable_sdpa(model)
        if patch.class_count != 1:
            raise ValueError(f'expected one VideoMAEv2 Attention class, found {patch.class_count}')
        processor = VideoMAEImageProcessor.from_pretrained(snapshot, local_files_only=True)

        @torch.inference_mode()
        def encode(clips: list[np.ndarray]) -> np.ndarray:
            values = processor(clips, return_tensors='pt', do_resize=False, do_center_crop=False).pixel_values.permute(0, 2, 1, 3, 4)
            values = values.to(device=device, dtype=torch.float16, non_blocking=True)
            result = model(pixel_values=values)
            return result.float().cpu().numpy()
        dim = 1024
        preprocess = 'RGB center-square crop -> 224x224; processor normalize; 16 uniform frames'
        kernel = 'torch_sdpa_scale1_after_explicit_q_scale'
    elif name == 'vjepa2_vitl_ssv2':
        from transformers import AutoVideoProcessor, VJEPA2ForVideoClassification
        checkpoint = snapshot / 'model.safetensors'
        processor = AutoVideoProcessor.from_pretrained(snapshot, local_files_only=True)
        model = VJEPA2ForVideoClassification.from_pretrained(snapshot, local_files_only=True, dtype=torch.float16).eval().to(device)

        @torch.inference_mode()
        def encode(clips: list[np.ndarray]) -> np.ndarray:
            values = processor(clips, return_tensors='pt', do_resize=False, do_center_crop=False).pixel_values_videos.to(device=device, dtype=torch.float16, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                backbone = model.vjepa2(pixel_values_videos=values, skip_predictor=True)
                result = model.pooler(backbone.last_hidden_state)
            return result.float().cpu().numpy()
        dim = int(model.config.hidden_size)
        preprocess = 'RGB short-side 292 then center crop 256x256; processor normalize; 16 uniform frames'
        kernel = 'transformers_vjepa2_default'
    else:
        raise ValueError(f'unknown model: {name}')
    parameters = int(sum((parameter.numel() for parameter in model.parameters())))
    return Encoder(name, dim, 224 if name == 'videomaev2_large' else 256, parameters, checkpoint, preprocess, kernel, encode)

def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.building')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)

def _workout_plan(rows: pd.DataFrame) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[np.ndarray | None]]]:
    requests: dict[int, list[tuple[int, int]]] = {}
    pending: dict[int, list[np.ndarray | None]] = {}
    for index, row in rows.iterrows():
        row_index = int(index)
        pending[row_index] = [None] * len(FRAME_OFFSETS)
        start = int(row.start_frame)
        for slot, frame_id in enumerate(start + FRAME_OFFSETS):
            requests.setdefault(int(frame_id), []).append((row_index, slot))
    return (requests, pending)

def extract_workout(video_path: Path, rows: pd.DataFrame, encoder: Encoder, destination: np.ndarray | None, batch_size: int) -> tuple[np.ndarray | None, dict]:
    """Sequentially decode one video and encode requested overlapping clips."""
    if rows.empty:
        raise ValueError('workout has no rows')
    requests, pending = _workout_plan(rows)
    maximum = max(requests)
    cv2.setNumThreads(2)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f'could not open {video_path}')
    completed_clips: list[np.ndarray] = []
    completed_rows: list[int] = []
    collected: list[np.ndarray] = []
    decoded = 0
    encoded = 0
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def flush() -> None:
        nonlocal encoded
        if not completed_clips:
            return
        values = encoder.encode(completed_clips)
        if values.shape != (len(completed_clips), encoder.feature_dim):
            raise ValueError(f'unexpected {encoder.name} feature shape {values.shape}')
        if not np.isfinite(values).all():
            raise ValueError(f'non-finite {encoder.name} feature')
        if destination is None:
            collected.extend(values.astype(np.float32, copy=False))
        else:
            destination[np.asarray(completed_rows)] = values.astype(np.float16)
        encoded += len(completed_rows)
        completed_clips.clear()
        completed_rows.clear()
    try:
        frame_index = 0
        while frame_index <= maximum:
            ok = capture.grab()
            if not ok:
                raise RuntimeError(f'decode ended at frame {frame_index}, required through {maximum}: {video_path}')
            decoded += 1
            slots = requests.get(frame_index)
            if slots:
                ok, bgr = capture.retrieve()
                if not ok or bgr is None:
                    raise RuntimeError(f'could not retrieve requested frame {frame_index}: {video_path}')
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                prepared = center_resize(rgb, 224) if encoder.name == 'videomaev2_large' else vjepa_resize_crop(rgb, 256)
                for row_index, slot in slots:
                    clip = pending[row_index]
                    clip[slot] = prepared
                    if all((value is not None for value in clip)):
                        completed_clips.append(np.stack(clip))
                        completed_rows.append(row_index)
                        del pending[row_index]
                        if len(completed_clips) >= batch_size:
                            flush()
            frame_index += 1
        flush()
    finally:
        capture.release()
    if pending or encoded != len(rows):
        raise RuntimeError(f'incomplete workout: pending={len(pending)}, encoded={encoded}/{len(rows)}')
    elapsed = time.perf_counter() - started
    report = {'rows': int(len(rows)), 'decoded_frames': decoded, 'encoded_clips': encoded, 'retrieved_frames': int(len(requests)), 'decode_mode': 'opencv_grab_all_retrieve_requested_threads2', 'elapsed_seconds': elapsed, 'clips_per_second': encoded / elapsed, 'peak_cuda_gib': torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0}
    array = np.stack(collected) if destination is None else None
    return (array, report)

def validate_features(values: np.ndarray, rows: int, dim: int, chunk: int=1024) -> dict:
    if values.shape != (rows, dim) or values.dtype != np.float16:
        raise ValueError(f'feature cache contract mismatch: {values.shape}/{values.dtype}')
    norms = []
    for start in range(0, rows, chunk):
        block = np.asarray(values[start:start + chunk], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError(f'non-finite features in rows {start}:{start + len(block)}')
        norms.append(np.linalg.norm(block, axis=1))
    all_norms = np.concatenate(norms)
    return {'norm_min': float(all_norms.min()), 'norm_max': float(all_norms.max()), 'norm_mean': float(all_norms.mean()), 'finite': True}

def pilot_model(name: str, inputs: Path, rgb_root: Path, batch_size: int, count: int=16, workout: str='w01') -> dict:
    rows = load_inputs(inputs)
    chosen = rows[rows.workout == workout].iloc[:count].copy()
    if len(chosen) < 16:
        raise ValueError('pilot requires at least 16 source clips')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('strong encoder pilot requires CUDA')
    started = time.perf_counter()
    encoder = load_encoder(name, device)
    load_seconds = time.perf_counter() - started
    values, report = extract_workout(rgb_root / f'{workout}_rgb.mp4', chosen, encoder, None, batch_size)
    assert values is not None
    pairwise_delta = np.linalg.norm(values[1:] - values[:-1], axis=1)
    if not np.isfinite(values).all() or not np.any(pairwise_delta > 1e-05):
        raise ValueError('pilot features are non-finite or identical across clips')
    report.update({'status': 'pilot_complete', 'model': name, 'workout': workout, 'feature_shape': list(values.shape), 'feature_dim': encoder.feature_dim, 'load_seconds': load_seconds, 'parameters': encoder.parameters, 'checkpoint': str(encoder.checkpoint), 'checkpoint_sha256': sha256_file(encoder.checkpoint), 'preprocess': encoder.preprocess, 'kernel': encoder.kernel, 'frame_offsets': FRAME_OFFSETS.tolist(), 'minimum_adjacent_feature_l2': float(pairwise_delta.min()), 'maximum_adjacent_feature_l2': float(pairwise_delta.max()), 'estimated_full_hours_at_pilot_rate': EXPECTED_ROWS / report['clips_per_second'] / 3600, 'scientific_scope': 'source-w01 execution/throughput check only; no accuracy claim'})
    return report

def build_cache(name: str, inputs: Path, rgb_root: Path, output: Path, batch_size: int) -> dict:
    rows = load_inputs(inputs)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'cache_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    else:
        manifest = {'status': 'building', 'protocol': 'MMFIT_STRONG_FEATURES_V2', 'rows': len(rows), 'input_columns': list(INPUT_COLUMNS), 'test_label_columns_read': [], 'inputs_csv': str(inputs.resolve()), 'inputs_csv_sha256': sha256_file(inputs), 'ordered_segment_id_sha256': ordered_id_sha256(rows.segment_id.astype(str)), 'frame_offsets': FRAME_OFFSETS.tolist(), 'window_frames': 150, 'sampling': '16 fixed rounded linspace offsets over [start_frame,start_frame+149]', 'models': {}}
    if manifest.get('inputs_csv_sha256') != sha256_file(inputs):
        raise ValueError('resume manifest belongs to a different V1 inputs.csv')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('full strong feature cache requires CUDA')
    encoder = load_encoder(name, device)
    feature_path = output / MODEL_FILES[name]
    state = manifest['models'].get(name, {})
    completed = list(state.get('completed_workouts', []))
    if feature_path.exists():
        features = np.lib.format.open_memmap(feature_path, mode='r+')
        if features.shape != (len(rows), encoder.feature_dim) or features.dtype != np.float16:
            raise ValueError('existing feature file has wrong contract')
    else:
        features = np.lib.format.open_memmap(feature_path, mode='w+', dtype=np.float16, shape=(len(rows), encoder.feature_dim))
        features[:] = np.nan
        features.flush()
        completed = []
    state.update({'status': 'building', 'feature_file': str(feature_path.resolve()), 'shape': [len(rows), encoder.feature_dim], 'dtype': 'float16', 'parameters': encoder.parameters, 'checkpoint': str(encoder.checkpoint), 'checkpoint_bytes': encoder.checkpoint.stat().st_size, 'checkpoint_sha256': sha256_file(encoder.checkpoint), 'preprocess': encoder.preprocess, 'kernel': encoder.kernel, 'batch_size': batch_size, 'completed_workouts': completed, 'workout_stats': state.get('workout_stats', {}), 'kernel_by_workout': state.get('kernel_by_workout', {workout: 'upstream_eager_attention' if name == 'videomaev2_large' else encoder.kernel for workout in completed}), 'decoder_by_workout': state.get('decoder_by_workout', {workout: 'opencv_read_all' for workout in completed}), 'execution_source_sha256': {str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()), **({str(Path(__file__).with_name('mmfit_videomae_sdpa.py').resolve()): sha256_file(Path(__file__).with_name('mmfit_videomae_sdpa.py'))} if name == 'videomaev2_large' else {})}})
    manifest['models'][name] = state
    _atomic_json(manifest_path, manifest)
    started = time.perf_counter()
    for workout in rows.workout.drop_duplicates().astype(str):
        if workout in completed:
            continue
        selected = rows[rows.workout == workout]
        _, stats = extract_workout(rgb_root / f'{workout}_rgb.mp4', selected, encoder, features, batch_size)
        features.flush()
        block = np.asarray(features[selected.index.to_numpy()], dtype=np.float32)
        if not np.isfinite(block).all() or np.allclose(block[0], block[-1]):
            raise ValueError(f'invalid or constant completed workout features: {workout}')
        completed.append(workout)
        state['completed_workouts'] = completed
        state['workout_stats'][workout] = stats
        state['kernel_by_workout'][workout] = encoder.kernel
        state['decoder_by_workout'][workout] = stats['decode_mode']
        state['elapsed_seconds_this_run'] = time.perf_counter() - started
        _atomic_json(manifest_path, manifest)
        print(json.dumps({'model': name, 'workout': workout, **stats}), flush=True)
    state['validation'] = validate_features(features, len(rows), encoder.feature_dim)
    state['feature_sha256'] = sha256_file(feature_path)
    state['feature_bytes'] = feature_path.stat().st_size
    state['status'] = 'complete'
    state['elapsed_seconds_this_run'] = time.perf_counter() - started
    manifest['status'] = 'complete' if all((manifest['models'].get(model, {}).get('status') == 'complete' for model in MODEL_ORDER)) else 'partial_complete'
    _atomic_json(manifest_path, manifest)
    return state

def _timing_summary(state: dict) -> dict:
    stats = state['workout_stats']
    kernels = state['kernel_by_workout']
    decoders = state['decoder_by_workout']
    return {'sum_of_workout_seconds_not_parallel_wall': float(sum((value['elapsed_seconds'] for value in stats.values()))), 'kernel_seconds': {key: float(sum((stats[w]['elapsed_seconds'] for w, value in kernels.items() if value == key))) for key in sorted(set(kernels.values()))}, 'decoder_seconds': {key: float(sum((stats[w]['elapsed_seconds'] for w, value in decoders.items() if value == key))) for key in sorted(set(decoders.values()))}}

def merge_model_cache(canonical: Path, source: Path, name: str) -> dict:
    """Verify and copy one completed isolated model cache into the canonical root."""
    canonical_manifest_path = canonical / 'cache_manifest.json'
    source_manifest_path = source / 'cache_manifest.json'
    canonical_manifest = json.loads(canonical_manifest_path.read_text(encoding='utf-8'))
    source_manifest = json.loads(source_manifest_path.read_text(encoding='utf-8'))
    for key in ('rows', 'inputs_csv_sha256', 'ordered_segment_id_sha256', 'frame_offsets'):
        if canonical_manifest.get(key) != source_manifest.get(key):
            raise ValueError(f'manifest merge mismatch for {key}')
    state = source_manifest.get('models', {}).get(name)
    if not state or state.get('status') != 'complete':
        raise ValueError(f'source model is not complete: {name}')
    source_feature = Path(state['feature_file'])
    if sha256_file(source_feature) != state['feature_sha256']:
        raise ValueError('source feature hash mismatch')
    destination = canonical / MODEL_FILES[name]
    temporary = destination.with_suffix(destination.suffix + '.copying')
    shutil.copyfile(source_feature, temporary)
    if sha256_file(temporary) != state['feature_sha256']:
        temporary.unlink(missing_ok=True)
        raise ValueError('copied feature hash mismatch')
    temporary.replace(destination)
    merged = json.loads(json.dumps(state))
    merged['feature_file'] = str(destination.resolve())
    if name == 'vjepa2_vitl_ssv2':
        merged['kernel_by_workout'] = {workout: 'transformers_vjepa2_default' for workout in merged['completed_workouts']}
    merged['timing_summary'] = _timing_summary(merged)
    merged['isolated_manifest'] = str(source_manifest_path.resolve())
    merged['isolated_manifest_sha256'] = sha256_file(source_manifest_path)
    canonical_manifest['models'][name] = merged
    for model_name, model_state in canonical_manifest['models'].items():
        if model_state.get('status') == 'complete':
            model_state['timing_summary'] = _timing_summary(model_state)
    canonical_manifest['status'] = 'complete' if all((canonical_manifest['models'].get(model, {}).get('status') == 'complete' for model in MODEL_ORDER)) else 'partial_complete'
    canonical_manifest['ready'] = canonical_manifest['status'] == 'complete'
    _atomic_json(canonical_manifest_path, canonical_manifest)
    provenance_copy = canonical / f'isolated_{name}_cache_manifest.json'
    shutil.copyfile(source_manifest_path, provenance_copy)
    return canonical_manifest
