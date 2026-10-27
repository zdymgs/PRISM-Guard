from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import argparse
import csv
import json
from pathlib import Path
import numpy as np
ROOT = WORK_ROOT
from session_kd.mmfit_rgb_benchmark import sha256_file
from session_kd.mmfit_rgb_benchmark_cache import validate_derived_arrays, validatecore_imu_alignment

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--imu-cache', type=Path, required=True)
    parser.add_argument('--rgb-root', type=Path, required=True)
    parser.add_argument('--executed-core-sha256', required=True)
    parser.add_argument('--executed-script-sha256', required=True)
    args = parser.parse_args()
    manifest_path = args.run_dir / 'cache_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    rows = list(csv.DictReader((args.run_dir / 'inputs.csv').open(encoding='utf-8')))
    validatecore_imu_alignment(rows, args.imu_cache / 'segments.csv')
    rgb = np.load(args.run_dir / 'rgb_4x512_fp16.npy', mmap_mode='r')
    pose = np.load(args.run_dir / 'pose_51x150_fp16.npy', mmap_mode='r')
    validate_derived_arrays(rgb, pose)
    if len(rows) != 40475 or rgb.shape[0] != len(rows):
        raise ValueError('Formal V1 count mismatch')
    split_counts = {name: sum((r['split'] == name for r in rows)) for name in ('train', 'validation', 'test')}
    if split_counts != {'train': 23612, 'validation': 5195, 'test': 11668}:
        raise ValueError('Formal V1 split mismatch')
    provenance = {'status': 'cache_integrity_complete_ready_for_formal_training', 'row_order_matchescore_imu_without_reading_label_or_activity_columns': True, 'derived_arrays_chunkwise_finite': True, 'rows': len(rows), 'split_counts': split_counts, 'inputs_csv_sha256': sha256_file(args.run_dir / 'inputs.csv'), 'rgb_features_sha256': sha256_file(args.run_dir / 'rgb_4x512_fp16.npy'), 'pose_features_sha256': sha256_file(args.run_dir / 'pose_51x150_fp16.npy'), 'cache_execution_source_sha256': {'src/session_kd/mmfit_rgb_benchmark_cache.py': args.executed_core_sha256.lower(), 'scripts/mmfit_rgb_benchmark_cache.py': args.executed_script_sha256.lower()}, 'post_run_validation_source_sha256': {'src/session_kd/mmfit_rgb_benchmark_cache.py': sha256_file(MODULE_ROOT / 'session_kd/mmfit_rgb_benchmark_cache.py'), 'scripts/mmfit_rgb_benchmark_finalize_cache.py': sha256_file(Path(__file__))}, 'cache_manifest_before_finalize_sha256': sha256_file(manifest_path), 'cache_elapsed_seconds': manifest['elapsed_seconds'], 'cache_bytes': manifest['bytes']}
    (args.run_dir / 'cache_integrity.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(provenance, indent=2))
