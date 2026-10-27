from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import argparse
import csv
import json
import sys
import time
import os
from pathlib import Path
import numpy as np
ROOT = WORK_ROOT
from session_kd.mmfit_fullroute import EARLY_LATE, skeleton_3d_features, imu_grid, DEVICE_MAPPING, REGIONS
from session_kd.mmfit_strong_benchmark import sha256_file, ordered_id_hash, validate_split_firewall

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT / 'runs/mmfit_fullroute_v3/native_motion_refclock')
    p.add_argument('--pilot', action='store_true')
    a = p.parse_args()
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    v1 = ROOT / 'runs/mmfit_rgb_benchmark'
    ic = ROOT / 'runs/mmfit_unseen_cache'
    rows = list(csv.DictReader((v1 / 'inputs.csv').open()))
    y = np.load(v1 / 'source_labels_test_minus1.npy')
    idx = validate_split_firewall(rows, y)
    indices = idx['train'][np.linspace(0, len(idx['train']) - 1, 32).astype(int)] if a.pilot else np.arange(len(rows))
    with np.load(v1 / 'normalizer_train_only.npz') as z:
        mean = z['imu_mean']
        std = z['imu_std']
    raw = np.load(ic / 'teacher_sensor_24x250.npy', mmap_mode='r')
    presence = np.load(ic / 'modality_presence_8.npy', mmap_mode='r')
    import pandas as pd
    legacy = pd.read_csv(ic / 'segments.csv', usecols=['segment_id'])['segment_id'].astype(str).to_numpy()
    if not np.array_equal(legacy, np.asarray([r['segment_id'] for r in rows])):
        raise ValueError('IMU IDs differ')
    contract = {'pilot': a.pilot, 'rows': len(indices), 'input_sha256': sha256_file(v1 / 'inputs.csv'), 'ordered_id_sha256': ordered_id_hash([rows[i]['segment_id'] for i in indices]), 'source_normalizer_sha256': sha256_file(v1 / 'normalizer_train_only.npz'), 'early_late_offsets': EARLY_LATE.tolist(), 'device_mapping': DEVICE_MAPPING, 'earbud_semantics': 'measured left-ear/head sensor in contextual torso slot; not torso measurement', 'missing_parts': ['left_leg'], 'sensor_layout': 'four real devices x acc/gyr xyz; all 24 native channels', 'skeleton_layout': '17 native H36M 3D; zero root is valid; full xyz feature/mask slots; exact reference-frame join and missing-frame masks', 'test_labels_read': False, 'regions': REGIONS}
    cp = out / 'contract.json'
    if cp.exists() and json.loads(cp.read_text()) != json.loads(json.dumps(contract)):
        raise ValueError('cache contract mismatch')
    cp.write_text(json.dumps(contract, indent=2) + '\n')
    np.savez(out / 'identity.npz', indices=indices, ids=np.array([rows[i]['segment_id'] for i in indices]), source_labels=y[indices])
    arrays = {}
    completed_path = out / 'completed.npy'
    complete = np.load(completed_path) if completed_path.exists() else np.zeros(len(indices), bool)
    began = time.perf_counter()
    current = None
    pose = None
    valid_skeleton = 0
    valid_imu = 0
    for j, i in enumerate(indices):
        if (ROOT / 'STOP').exists() or (out.parent / 'STOP').exists():
            raise RuntimeError('STOP requested')
        if complete[j]:
            continue
        row = rows[i]
        if current != row['workout']:
            current = row['workout']
            pose = np.load(ROOT / f'data/MM-Fit/mm-fit/{current}/{current}_pose_3d.npy', mmap_mode='r')
        clock = int(row['start_frame']) + EARLY_LATE
        sel = np.searchsorted(pose[0, :, 0], clock)
        clipped = np.minimum(sel, pose.shape[1] - 1)
        observed = (sel < pose.shape[1]) & (pose[0, clipped, 0] == clock)
        xyz = pose[:, clipped, 1:].transpose(1, 2, 3, 0).copy()
        xyz[~observed] = np.nan
        samples = [skeleton_3d_features(xyz[w], clock[w] / 30.0) for w in range(2)]
        values = {k: np.stack([r[k] for r in samples]) for k in samples[0]}
        values.update(imu_grid(raw[i], presence[i], mean, std))
        values['global_time_position'] = (EARLY_LATE / 149.0).astype(np.float32)
        for name, x in values.items():
            if name not in arrays:
                path = out / f'{name}.npy'
                arrays[name] = np.lib.format.open_memmap(path, mode='r+' if path.exists() else 'w+', dtype=x.dtype, shape=(len(indices), *x.shape))
            if not np.isfinite(x).all():
                raise ValueError(f'nonfinite {name}:{i}')
            arrays[name][j] = x
        complete[j] = True
        valid_skeleton += int(values['skeleton_joint_mask'].sum())
        valid_imu += int(values['imu_sequence_mask'].sum())
        if j % 512 == 0:
            for x in arrays.values():
                x.flush()
            np.save(completed_path, complete)
            print(json.dumps({'row': j, 'total': len(indices), 'seconds': time.perf_counter() - began}), flush=True)
    for x in arrays.values():
        x.flush()
    np.save(completed_path, complete)
    paths = sorted(out.glob('*.npy')) + [out / 'identity.npz', cp]
    status = {'status': 'pilot_complete' if a.pilot else 'complete', 'rows': len(indices), 'seconds': time.perf_counter() - began, 'complete': bool(complete.all()), 'test_labels_read': False, 'bytes': sum((p.stat().st_size for p in out.iterdir() if p.is_file())), 'artifact_sha256': {p.name: sha256_file(p) for p in paths}}
    (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status), flush=True)
