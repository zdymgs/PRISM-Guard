from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import json
import argparse
import time
from pathlib import Path
import numpy as np
ROOT = WORK_ROOT
from session_kd.mmact_cache_dataset import assemble_part_grid_from_raw, sha256_file
from session_kd.mmact_imu import fit_channel_normalizer

def main():
    parser = argparse.ArgumentParser(description="Fit the MMAct IMU normalizer on source folds only.")
    parser.add_argument('--cache', type=Path, default=ROOT / 'runs/mmact_input_cache')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/mmact_student_preparation')
    parser.add_argument('--input-manifest', type=Path, default=ROOT / 'runs/mmact_mc3_features/input_manifest.json')
    args = parser.parse_args()
    start = time.perf_counter()
    cache = args.cache
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    destination = out / 'source_imu_normalizer.npz'
    if destination.exists():
        raise FileExistsError('Do not overwrite the frozen source normalizer')
    audit = json.loads((cache / 'full_array_audit.json').read_text(encoding='utf-8'))
    if audit['status'] != 'PASS' or not audit['full_coverage']:
        raise ValueError('full cache audit required')
    manifest_path = args.input_manifest
    rows = json.loads(manifest_path.read_text(encoding='utf-8'))['rows_manifest']
    train = np.asarray([i for i, r in enumerate(rows) if (int(r['subject'][7:]) - 1) % 5 in (2, 3, 4)], np.int64)
    if len(train) != 3243:
        raise ValueError('source count mismatch')
    values = np.empty((len(train), 2, 16, 5, 4, 6), np.float32)
    masks = np.empty(values.shape, bool)
    ids = []
    for j, i in enumerate(train):
        path = cache / f'interval_{i:06d}.npz'
        with np.load(path, allow_pickle=False) as data:
            if int(data['global_interval_index'].item()) != int(i) or str(data['interval_id'].item()) != rows[i]['interval_id']:
                raise ValueError(f'identity mismatch: {path}')
            grid, mask, schema = assemble_part_grid_from_raw(data['imu_raw_phone_watch'], data['imu_channel_mask'])
        values[j] = grid[0]
        masks[j] = mask[0]
        ids.append(rows[i]['interval_id'])
    normalizer = fit_channel_normalizer(values, masks, ids, np.arange(len(train)), schema)
    np.savez(destination, mean=normalizer.mean, std=normalizer.std, training_indices=train, training_ids=np.asarray(ids), channel_names=np.asarray(normalizer.channel_names))
    report = {'status': 'COMPLETE', 'source_folds': ['F2', 'F3', 'F4'], 'source_rows': len(train), 'validation_or_target_used': False, 'labels_loaded': False, 'mean': normalizer.mean.tolist(), 'std': normalizer.std.tolist(), 'normalizer_sha256': sha256_file(destination), 'input_manifest_sha256': sha256_file(manifest_path), 'full_cache_audit_sha256': sha256_file(cache / 'full_array_audit.json'), 'elapsed_seconds': time.perf_counter() - start}
    (out / 'SOURCE_NORMALIZER_RECEIPT.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
