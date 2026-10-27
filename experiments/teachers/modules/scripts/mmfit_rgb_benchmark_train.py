from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT = WORK_ROOT
from session_kd.mmfit_rgb_benchmark import *
from session_kd.mmfit_rgb_benchmark_cache import validate_derived_arrays, validatecore_imu_alignment

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'runs/mmfit_rgb_benchmark')
    parser.add_argument('--imu-cache', type=Path, default=ROOT / 'runs/mmfit_unseen_cache')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--adapt-epochs', type=int, default=5)
    args = parser.parse_args()
    if args.epochs != 30 or args.adapt_epochs != 5:
        raise ValueError('Formal V1 requires epochs=30 and adapt-epochs=5')
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(6)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = list(csv.DictReader((args.run_dir / 'inputs.csv').open(encoding='utf-8')))
    if len(rows) != 40475:
        raise ValueError(f'Formal cache must contain 40475 rows, got {len(rows)}')
    labels = np.load(args.run_dir / 'source_labels_test_minus1.npy', mmap_mode='r')
    split = np.array([r['split'] for r in rows])
    train_idx, val_idx, test_idx = (np.flatnonzero(split == x) for x in ('train', 'validation', 'test'))
    if (len(train_idx), len(val_idx), len(test_idx)) != (23612, 5195, 11668):
        raise ValueError('Frozen split counts changed')
    if not (np.all(labels[test_idx] == -1) and np.all(labels[np.r_[train_idx, val_idx]] >= 0)):
        raise ValueError('Label firewall violated')
    validatecore_imu_alignment(rows, args.imu_cache / 'segments.csv')
    rgb = np.load(args.run_dir / 'rgb_4x512_fp16.npy', mmap_mode='r')
    pose = np.load(args.run_dir / 'pose_51x150_fp16.npy', mmap_mode='r')
    imu = np.load(args.imu_cache / 'teacher_sensor_24x250.npy', mmap_mode='r')
    presence = np.load(args.imu_cache / 'modality_presence_8.npy', mmap_mode='r')
    validate_derived_arrays(rgb, pose)
    if imu.shape != (len(rows), 24, 250) or presence.shape != (len(rows), 8):
        raise ValueError('Legacy IMU cache shape mismatch')
    normalizer = fit_normalizer(rgb, pose, imu, presence, train_idx)
    np.savez(args.run_dir / 'normalizer_train_only.npz', **normalizer)
    counts = np.bincount(np.asarray(labels[train_idx]), minlength=CLASSES)
    class_weights = 1 / np.sqrt(counts)
    class_weights /= class_weights.mean()
    datasets = {name: MMFitDataset(rgb, pose, imu, presence, idx, labels if name != 'test' else None, normalizer) for name, idx in (('train', train_idx), ('validation', val_idx), ('test', test_idx))}
    train_loader = make_loader(datasets['train'], 256, True, SEED)
    val_loader = make_loader(datasets['validation'], 256, False, SEED)
    test_loader = make_loader(datasets['test'], 256, False, SEED)
    val_y = np.asarray(labels[val_idx])
    summaries = {}
    teacher_val = []
    teacher_test = []
    for offset, modality in enumerate(('rgb', 'pose', 'imu')):
        torch.manual_seed(SEED + offset)
        model = ModalityTeacher(modality)
        summaries[modality] = train_source(model, train_loader, val_loader, val_y, class_weights, device, args.run_dir / f'teacher_{modality}.pt', args.run_dir / f'teacher_{modality}_train.jsonl')
        teacher_val.append(F.softmax(torch.from_numpy(predict(model, val_loader, device)) / 2.0, 1).numpy())
        teacher_test.append(F.softmax(torch.from_numpy(predict(model, test_loader, device)) / 2.0, 1).numpy())
    tv, tt = (np.stack(teacher_val), np.stack(teacher_test))
    np.savez(args.run_dir / 'teacher_individual_unlabeled_test.npz', rgb=tt[0].astype(np.float32), pose=tt[1].astype(np.float32), imu=tt[2].astype(np.float32))
    artifacts = list(args.run_dir.glob('teacher_*.pt')) + list(args.run_dir.glob('teacher_*.npz'))
    freeze = {'inputs_csv_sha256': sha256_file(args.run_dir / 'inputs.csv'), 'artifact_sha256': {p.name: sha256_file(p) for p in artifacts}, 'test_labels_read': False}
    (args.run_dir / 'pre_evaluation_freeze.json').write_text(json.dumps(freeze, indent=2), encoding='utf-8')
