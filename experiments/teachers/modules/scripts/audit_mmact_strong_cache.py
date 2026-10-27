from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
import numpy as np
from session_kd.mmfit_strong_features import sha256_file
MODELS = ('videomaev2_large', 'vjepa2_vitl_ssv2')
SENSORS = ('acc_phone_clip', 'acc_watch_clip', 'gyro_clip')

def digest_lines(values: list[str]) -> str:
    return hashlib.sha256(('\n'.join(values) + '\n').encode()).hexdigest()

def run(run_root: Path, sensor_csv: Path) -> dict:
    cache_root = run_root / 'cache'
    manifest = json.loads((cache_root / 'interval_manifest.json').read_text(encoding='utf-8'))
    intervals = manifest['intervals']
    by_session: dict[tuple[str, str, str], list[tuple[int, dict]]] = {}
    for index, row in enumerate(intervals):
        by_session.setdefault((row['subject'], row['scene'], row['session']), []).append((index, row))
    sensor_lookup = {}
    with sensor_csv.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            key = (row['subject'], row['scene'], row['session'], int(row['line_number']), row['sensor'])
            sensor_lookup[key] = str(row['usable']).lower() == 'true'
    inventory = json.loads((run_root / 'archive_inventory.json').read_text(encoding='utf-8'))
    source_lookup = {}
    for archive in inventory['archives']:
        for video in archive['videos']:
            source_lookup[str(Path(archive['path']).resolve()), video['name']] = video
    failures = []
    model_counts = Counter()
    global_covered = {model: np.zeros(len(intervals), dtype=np.int16) for model in MODELS}
    expected_markers = int(inventory['video_count']) * len(MODELS)
    markers = sorted(cache_root.glob('**/*.complete.json'))
    for number, marker_path in enumerate(markers, 1):
        try:
            marker = json.loads(marker_path.read_text(encoding='utf-8'))
            model = marker['model']
            video = marker['video']
            session_key = (video['subject'], video['scene'], video['session'])
            indexed = by_session[session_key]
            indices = [value[0] for value in indexed]
            rows = [value[1] for value in indexed]
            ids = [f'{row['annotation_file']}:{row['line_number']}' for row in rows]
            timings = [f'{row['start_ms']}:{row['end_ms']}' for row in rows]
            source = source_lookup[str(Path(marker['archive']).resolve()), marker['member']]
            cache_path = Path(marker['cache_file'])
            checks = {'status': marker.get('status') == 'complete', 'source_crc': marker['source_crc32'] == source['crc32'] == video['crc32'], 'source_size': video['size'] == source['bytes'], 'id_hash': marker['ordered_interval_id_sha256'] == digest_lines(ids), 'timing_hash': marker['ordered_interval_timing_sha256'] == digest_lines(timings), 'sensor_hash': marker['sensor_coverage_sha256'] == sha256_file(sensor_csv), 'cache_hash': marker['cache_sha256'] == sha256_file(cache_path), 'label_firewall': marker.get('test_label_columns_used_for_feature_choices') == []}
            with np.load(cache_path, allow_pickle=False) as data:
                features = data['features']
                covered = data['video_covered']
                sampled = data['sampled_timestamps_ms']
                cached_ids = data['interval_id'].tolist()
                sensor_mask = data['sensor_usable']
                expected_sensor = np.asarray([[sensor_lookup[row['subject'], row['scene'], row['session'], int(row['line_number']), sensor] for sensor in SENSORS] for row in rows], dtype=np.bool_)
                expected_sampled = np.asarray([np.linspace(row['start_ms'], row['end_ms'], 16) for row in rows])
                checks.update({'shape': features.shape == (len(rows), 1024) and features.dtype == np.float16, 'mask_shape': covered.shape == (len(rows),) and covered.dtype == np.bool_, 'ids': cached_ids == ids, 'timestamps': sampled.shape == (len(rows), 16) and np.allclose(sampled[covered], expected_sampled[covered], atol=0.01) and np.isnan(sampled[~covered]).all(), 'sensor_masks': sensor_mask.shape == expected_sensor.shape and np.array_equal(sensor_mask, expected_sensor), 'finite_contract': np.isfinite(features[covered].astype(np.float32)).all() and (not np.isfinite(features[~covered].astype(np.float32)).any()), 'nonzero_covered': int(covered.sum()) > 0})
                global_covered[model][np.asarray(indices)] += covered.astype(np.int16)
            failed = [key for key, value in checks.items() if not value]
            if failed:
                failures.append({'marker': str(marker_path), 'failed_checks': failed})
            model_counts[model] += 1
        except Exception as error:
            failures.append({'marker': str(marker_path), 'exception': repr(error)})
        if number % 200 == 0:
            print(f'audited {number}/{len(markers)} markers', flush=True)
    stderr = (run_root / 'full_queue.stderr.log').read_text(encoding='utf-8', errors='replace')
    h264_lines = [line for line in stderr.splitlines() if '[h264' in line.lower()]
    h264_kinds = Counter((re.sub('^.*?\\] ', '', line).split(', bytestream')[0] for line in h264_lines))
    coverage = {model: {'mapped_camera_interval_rows': int(values.sum()), 'all_5406_intervals_have_at_least_one_view': bool(np.all(values > 0)), 'intervals_without_any_view': int(np.count_nonzero(values == 0)), 'minimum_views_per_interval': int(values.min()), 'maximum_views_per_interval': int(values.max())} for model, values in global_covered.items()}
    result = {'status': 'pass_with_decoder_warning_limitation' if not failures else 'fail', 'expected_markers': expected_markers, 'observed_markers': len(markers), 'model_marker_counts': dict(model_counts), 'annotation_intervals': len(intervals), 'coverage': coverage, 'failure_count': len(failures), 'failures': failures[:100], 'zip_evidence': inventory['status'], 'h264_stderr': {'warning_lines': len(h264_lines), 'distinct_message_prefixes': len(h264_kinds), 'most_common': h264_kinds.most_common(10), 'attribution': 'unavailable: stderr contains decoder contexts but no video/member names or timestamps', 'effect_claim': 'cache contracts prove requested frames returned and features are finite; they do not prove concealed H.264 macroblocks had no visual effect', 'selective_followup': 'retain logs and archives; if visual-fidelity sensitivity becomes load-bearing, rerun only a separately identified affected member with timestamped decoder instrumentation and compare sampled-frame hashes'}, 'safe_for_developmental_teacher_training': not failures and len(markers) == expected_markers and all((value['all_5406_intervals_have_at_least_one_view'] for value in coverage.values()))}
    return result

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--sensor-coverage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.run_root, args.sensor_coverage)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('status', 'expected_markers', 'observed_markers', 'failure_count', 'safe_for_developmental_teacher_training')}, indent=2))
