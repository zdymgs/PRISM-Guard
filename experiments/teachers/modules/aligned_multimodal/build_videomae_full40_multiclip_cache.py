from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from aligned_multimodal.build_videomae_cache import atomic_npz, encode, restorecore_attention_biases, safe_relative
from aligned_multimodal.build_videomae_multiclip_cache import CLASSIFIER_CLASSES, HIDDEN_SIZE, VIEW_NAMES, WINDOW_NAMES, prepare_trial
PROJECT_DIR = WORK_ROOT / 'aligned_multimodal'
DEFAULT_MANIFEST = PROJECT_DIR / 'data/single_split.csv'
DEFAULT_ROI = PROJECT_DIR / 'runs/dir_multiscale_roi_full'
DEFAULT_REUSE = PROJECT_DIR / 'runs/videomae_large_multiclip'
DEFAULT_OUTPUT = PROJECT_DIR / 'runs/videomae_large_multiclip_full40'
MODEL_NAME = 'MCG-NJU/videomae-large-finetuned-kinetics'
EXPECTED_ROWS = 2914

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Extract Large VideoMAE early/late x three-view features for all 40 classes, reusing the audited Detail21 cache when possible.')
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--roi-run', type=Path, default=DEFAULT_ROI)
    parser.add_argument('--reuse-run', type=Path, default=DEFAULT_REUSE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--model', default=MODEL_NAME)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--trial-batch', type=int, default=16)
    parser.add_argument('--max-trials', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()

def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: row['sample_id'])
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f'Frozen aligned full-40 universe changed: {len(rows)}')
    labels = {int(row['class_id']) for row in rows}
    users = {row['user_id'] for row in rows}
    if labels != set(range(40)) or len(users) != 18:
        raise RuntimeError(f'Expected 40 classes and 18 users, got {len(labels)}, {len(users)}')
    return rows

def cache_path(run: Path, row: dict[str, str]) -> Path:
    return run / 'trial_feature_cache' / safe_relative(row['source_id']).with_suffix('.npz')

def valid_cache(path: Path, row: dict[str, str]) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return str(data['sample_id'].item()) == row['sample_id'] and int(data['label'].item()) == int(row['class_id']) and (tuple(np.asarray(data['window_names']).astype(str)) == WINDOW_NAMES) and (tuple(np.asarray(data['view_names']).astype(str)) == tuple(VIEW_NAMES)) and (data['features'].shape == (2, 3, HIDDEN_SIZE)) and (data['kinetics_logits'].shape == (2, 3, CLASSIFIER_CLASSES)) and np.isfinite(data['features']).all() and np.isfinite(data['kinetics_logits']).all()
    except (OSError, ValueError, KeyError, EOFError):
        return False

def resolve_cache(row: dict[str, str], output: Path, reuse: Path, allow_reuse: bool=True) -> tuple[Path | None, str]:
    own = cache_path(output, row)
    if valid_cache(own, row):
        return (own, 'output')
    legacy = cache_path(reuse, row)
    if allow_reuse and valid_cache(legacy, row):
        return (legacy, 'detail21_reuse')
    return (None, 'missing')

def aggregate(rows: list[dict[str, str]], output: Path, reuse: Path) -> Path:
    features: list[np.ndarray] = []
    kinetics: list[np.ndarray] = []
    sources: list[str] = []
    for row in rows:
        path, source = resolve_cache(row, output, reuse)
        if path is None:
            raise RuntimeError(f'Missing full-40 VideoMAE cache: {row['source_id']}')
        with np.load(path, allow_pickle=False) as data:
            features.append(np.asarray(data['features'], dtype=np.float16))
            kinetics.append(np.asarray(data['kinetics_logits'], dtype=np.float16))
        sources.append(source)
    complete = output / 'complete_features.npz'
    atomic_npz(complete, sample_ids=np.asarray([row['sample_id'] for row in rows]), source_ids=np.asarray([row['source_id'] for row in rows]), users=np.asarray([row['user_id'] for row in rows]), labels=np.asarray([int(row['class_id']) for row in rows], dtype=np.int64), cache_sources=np.asarray(sources), window_names=np.asarray(WINDOW_NAMES), view_names=np.asarray(VIEW_NAMES), features=np.stack(features), kinetics_logits=np.stack(kinetics))
    return complete

def main() -> None:
    args = parse_args()
    if args.trial_batch < 1:
        raise ValueError('--trial-batch must be positive')
    rows = read_rows(args.manifest.resolve())
    if args.max_trials > 0:
        rows = rows[:args.max_trials]
    output = args.output_dir.resolve()
    reuse = args.reuse_run.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reuse_count = int(sum((valid_cache(cache_path(reuse, row), row) for row in rows)))
    own_count = int(sum((valid_cache(cache_path(output, row), row) for row in rows)))
    missing_count = int(sum((resolve_cache(row, output, reuse)[0] is None for row in rows)))
    print(json.dumps({'rows': len(rows), 'existing_output': own_count, 'reusable_detail21': reuse_count, 'inference_required': missing_count}), flush=True)
    snapshot = Path(snapshot_download(args.model, allow_patterns=('*.json', '*.safetensors', '*.txt'), local_files_only=True))
    processor = VideoMAEImageProcessor.from_pretrained(snapshot, local_files_only=True)
    model = VideoMAEForVideoClassification.from_pretrained(snapshot, local_files_only=True)
    bias_report = restorecore_attention_biases(model, snapshot)
    if int(model.config.hidden_size) != HIDDEN_SIZE or int(model.config.num_labels) != CLASSIFIER_CLASSES:
        raise RuntimeError('Large VideoMAE architecture changed')
    device = torch.device(args.device)
    model = model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    started = time.perf_counter()
    built = skipped_output = reused = 0
    peak_cuda_gib = 0.0
    for batch_start in range(0, len(rows), args.trial_batch):
        batch_rows = rows[batch_start:batch_start + args.trial_batch]
        work_rows: list[dict[str, str]] = []
        videos: list[list[np.ndarray]] = []
        metadata: list[dict[str, np.ndarray]] = []
        for row in batch_rows:
            own = cache_path(output, row)
            legacy = cache_path(reuse, row)
            if not args.overwrite and valid_cache(own, row):
                skipped_output += 1
                continue
            if not args.overwrite and valid_cache(legacy, row):
                reused += 1
                continue
            row_videos, row_metadata = prepare_trial(row, args.roi_run.resolve())
            work_rows.append(row)
            videos.extend(row_videos)
            metadata.append(row_metadata)
        if work_rows:
            feature, logits, peak = encode(model, processor, videos, device)
            peak_cuda_gib = max(peak_cuda_gib, peak)
            feature = feature.reshape(len(work_rows), 2, 3, HIDDEN_SIZE)
            logits = logits.reshape(len(work_rows), 2, 3, CLASSIFIER_CLASSES)
            for index, row in enumerate(work_rows):
                atomic_npz(cache_path(output, row), sample_id=np.asarray(row['sample_id']), source_id=np.asarray(row['source_id']), user=np.asarray(row['user_id']), label=np.asarray(int(row['class_id']), dtype=np.int64), window_names=np.asarray(WINDOW_NAMES), view_names=np.asarray(VIEW_NAMES), features=feature[index].astype(np.float16), kinetics_logits=logits[index].astype(np.float16), **metadata[index])
                built += 1
        processed = min(batch_start + len(batch_rows), len(rows))
        if processed == len(rows) or processed % 80 == 0:
            print(json.dumps({'processed': processed, 'total': len(rows), 'built': built, 'reused': reused, 'skipped_output': skipped_output, 'elapsed_seconds': round(time.perf_counter() - started, 1), 'peak_cuda_gib': round(peak_cuda_gib, 3)}), flush=True)
    complete = aggregate(rows, output, reuse) if len(rows) == EXPECTED_ROWS else None
    summary = {'protocol': 'all 40 classes, frozen Large VideoMAE early/late x three synchronized IR views', 'model': args.model, 'snapshot': str(snapshot), 'attention_bias_compatibility': bias_report, 'trials': len(rows), 'classes': len({int(row['class_id']) for row in rows}), 'users': len({row['user_id'] for row in rows}), 'built': built, 'reused_detail21': reused, 'skipped_output': skipped_output, 'elapsed_seconds': time.perf_counter() - started, 'peak_cuda_gib': peak_cuda_gib, 'complete_features': str(complete) if complete is not None else None}
    (output / 'cache_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
