from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
ROOT = WORK_ROOT
from session_kd.mmfit_rgb_benchmark import CompactStudent, MMFitDataset, ModalityTeacher, fit_normalizer, make_loader as make_loader, predict as predict
from session_kd.mmfit_strong_benchmark import *

def load_and_verify_assets(feature_dir: Path, v1_dir: Path, rows: list[dict[str, str]]):
    manifest_path = feature_dir / 'cache_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete' or manifest.get('rows') != 40475:
        raise RuntimeError('both strong caches must be complete before formal training')
    if manifest.get('inputs_csv_sha256') != sha256_file(v1_dir / 'inputs.csv'):
        raise ValueError('strong cache input CSV hash differs from frozen V1')
    ids = [row['segment_id'] for row in rows]
    if manifest.get('ordered_segment_id_sha256') != ordered_id_hash(ids):
        raise ValueError('strong cache row ID order differs from frozen V1')
    if manifest.get('test_label_columns_read') not in ([], None, False):
        raise ValueError('strong extraction manifest reports reading TEST label columns')
    names = ('videomaev2_large_16x1024_fp16.npy', 'vjepa2_vitl_ssv2_16x1024_fp16.npy')
    arrays = []
    for name in names:
        path = feature_dir / name
        array = np.load(path, mmap_mode='r')
        validate_strong_cache(array)
        if array.dtype != np.float16:
            raise ValueError(f'{name} must be float16')
        model_records = manifest.get('models', {})
        match = [value for value in model_records.values() if Path(value.get('feature_file', '')).name == name]
        if len(match) != 1 or match[0].get('status') != 'complete':
            raise ValueError(f'manifest has no complete model record for {name}')
        if match[0].get('feature_sha256') != sha256_file(path):
            raise ValueError(f'strong feature hash mismatch: {name}')
        arrays.append(array)
    return (manifest, arrays)

def full_prob(model, arrays, indices, device):
    loader = make_loader(ArrayDataset(arrays, indices), False, SEED)
    return probabilities_from_logits(predict_logits(model, loader, device))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'runs/mmfit_strong_benchmark')
    parser.add_argument('--feature-dir', type=Path, default=ROOT / 'runs/mmfit_strong_features')
    parser.add_argument('--v1-dir', type=Path, default=ROOT / 'runs/mmfit_rgb_benchmark')
    parser.add_argument('--imu-cache', type=Path, default=ROOT / 'runs/mmfit_unseen_cache')
    args = parser.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError('formal V2 run directory must be absent or empty')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(6)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('formal V2 training requires the released GPU')
    started = time.perf_counter()
    rows = list(csv.DictReader((args.v1_dir / 'inputs.csv').open(encoding='utf-8')))
    v1_freeze = json.loads((args.v1_dir / 'pre_evaluation_freeze.json').read_text(encoding='utf-8'))
    if v1_freeze.get('inputs_csv_sha256') != sha256_file(args.v1_dir / 'inputs.csv'):
        raise ValueError('frozen V1 input index no longer matches its own manifest')
    for reused_name in ('teacher_rgb.pt', 'teacher_pose.pt', 'teacher_imu.pt'):
        if v1_freeze.get('artifact_sha256', {}).get(reused_name) != sha256_file(args.v1_dir / reused_name):
            raise ValueError(f'frozen V1 artifact no longer matches its manifest: {reused_name}')
    labels = np.load(args.v1_dir / 'source_labels_test_minus1.npy', mmap_mode='r')
    indices = validate_split_firewall(rows, labels)
    feature_manifest, strong_raw = load_and_verify_assets(args.feature_dir, args.v1_dir, rows)
    rgb = np.load(args.v1_dir / 'rgb_4x512_fp16.npy', mmap_mode='r')
    pose = np.load(args.v1_dir / 'pose_51x150_fp16.npy', mmap_mode='r')
    imu = np.load(args.imu_cache / 'teacher_sensor_24x250.npy', mmap_mode='r')
    presence = np.load(args.imu_cache / 'modality_presence_8.npy', mmap_mode='r')
    with np.load(args.v1_dir / 'normalizer_train_only.npz') as saved:
        normalizer = {name: saved[name] for name in saved.files}
    recalculated = fit_normalizer(rgb, pose, imu, presence, indices['train'])
    if any((not np.array_equal(normalizer[name], recalculated[name]) for name in normalizer)):
        raise ValueError('stored V1 normalizer does not equal TRAIN-only recomputation')
    strong_norm = []
    normalizer_artifact = {}
    for name, raw in zip(('videomaev2_large', 'vjepa2'), strong_raw):
        mean, std = fit_feature_normalizer(raw, indices['train'])
        normalizer_artifact[f'{name}_mean'] = mean
        normalizer_artifact[f'{name}_std'] = std
        strong_norm.append(standardize(raw, mean, std))
    np.savez(args.run_dir / 'strong_normalizer_train_only.npz', **normalizer_artifact)
    all_indices = np.arange(len(rows), dtype=np.int64)
    pose_stats, imu_stats, present_stats = normalized_pooled_statistics(pose, imu, presence, normalizer, all_indices)
    relation_features = np.concatenate((strong_norm[0], strong_norm[1], pose_stats), axis=1)
    counts = np.bincount(np.asarray(labels[indices['train']]), minlength=11).astype(np.float32)
    class_weights = 1.0 / np.sqrt(counts)
    class_weights /= class_weights.mean()
    summaries = {}
    new_models = {}
    specifications = (('videomaev2_large', [strong_norm[0]], 1024, 10), ('vjepa2', [strong_norm[1]], 1024, 11), ('joint', [strong_norm[0], strong_norm[1], pose_stats, imu_stats, present_stats], 2206, 12))
    for name, arrays, width, offset in specifications:
        torch.manual_seed(SEED + offset)
        np.random.seed(SEED + offset)
        model = StrongMLP(width)
        train_loader = make_loader(ArrayDataset(arrays, indices['train'], labels), True, SEED + offset)
        val_loader = make_loader(ArrayDataset(arrays, indices['validation']), False, SEED + offset)
        summaries[name] = train_strong_head(model, train_loader, val_loader, np.asarray(labels[indices['validation']]), class_weights, device, args.run_dir / f'teacher_{name}.pt', args.run_dir / f'teacher_{name}_train.jsonl')
        new_models[name] = (model, arrays)
    v1_datasets = {name: MMFitDataset(rgb, pose, imu, presence, idx, labels if name != 'test' else None, normalizer) for name, idx in indices.items()}
    v1_loaders = {name: make_loader(ds, 256, False, SEED) for name, ds in v1_datasets.items()}
    teacher_by_split = {name: [] for name in indices}
    for old_name in ('rgb', 'pose', 'imu'):
        model = ModalityTeacher(old_name).to(device)
        saved = torch.load(args.v1_dir / f'teacher_{old_name}.pt', map_location=device, weights_only=True)
        model.load_state_dict(saved['state_dict'])
        for split_name in indices:
            teacher_by_split[split_name].append(probabilities_from_logits(predict(model, v1_loaders[split_name], device)))
    for name in ('videomaev2_large', 'vjepa2', 'joint'):
        model, arrays = new_models[name]
        for split_name, idx in indices.items():
            teacher_by_split[split_name].append(full_prob(model, arrays, idx, device))
    teacher_by_split = {name: np.stack(value) for name, value in teacher_by_split.items()}
    np.savez(args.run_dir / 'teacher_bank_source_validation.npz', **{name: teacher_by_split['validation'][i] for i, name in enumerate(TEACHER_ORDER)})
    np.savez(args.run_dir / 'teacher_bank_unlabeled_test.npz', **{name: teacher_by_split['test'][i] for i, name in enumerate(TEACHER_ORDER)})
    artifacts = list(args.run_dir.glob('teacher_*.pt')) + list(args.run_dir.glob('teacher_*.npz'))
    freeze = {'inputs_csv_sha256': sha256_file(args.v1_dir / 'inputs.csv'), 'artifact_sha256': {p.name: sha256_file(p) for p in artifacts}, 'test_labels_read': False}
    (args.run_dir / 'pre_evaluation_freeze.json').write_text(json.dumps(freeze, indent=2), encoding='utf-8')
