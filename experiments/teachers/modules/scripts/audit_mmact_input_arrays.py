from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import json
from pathlib import Path
import numpy as np

def audit(cache_root: Path, input_manifest: Path, report_path: Path, allow_partial=False):
    rows = json.loads(input_manifest.read_text(encoding='utf-8'))['rows_manifest']
    seen, failures, bytes_total = (set(), [], 0)
    frame_sum = pose_sum = 0
    region_sum = np.zeros(3, dtype=np.int64)
    imu_sum = np.zeros(9, dtype=np.int64)
    paths = sorted(cache_root.rglob('*.npz'))
    shapes = {'video_uint8': (2, 16, 3, 3, 160, 160), 'pose_coco2d': (2, 16, 17, 3), 'pose_joint_mask': (2, 16, 17), 'frame_available': (2, 16), 'region_available': (2, 16, 3), 'imu_raw_phone_watch': (2, 16, 4, 9), 'imu_channel_mask': (2, 16, 4, 9), 'source_time_ms': (2, 16), 'source_frame_index': (2, 16), 'frame_width_height': (2, 16, 2)}
    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as data:
                arrays = {k: data[k] for k in shapes}
                for k, shape in shapes.items():
                    if arrays[k].shape != shape:
                        raise ValueError(f'{k} shape {arrays[k].shape}, expected {shape}')
                index = int(data['global_interval_index'].item())
                identity = str(data['interval_id'].item())
                if index < 0 or index >= len(rows) or index in seen:
                    raise ValueError(f'invalid/duplicate global ID {index}')
                row = rows[index]
                if identity != row['interval_id']:
                    raise ValueError('native interval_id mismatch')
                start, end = (int(row['start_ms']), int(row['end_ms']))
                expected = np.stack([np.linspace(start + lo * (end - start), start + hi * (end - start), 16) for lo, hi in ((0.0, 0.7), (0.3, 1.0))])
                if not np.array_equal(arrays['source_time_ms'], expected):
                    raise ValueError('sampling timestamps mismatch')
                fa, ra, pm, im = (arrays[k] for k in ('frame_available', 'region_available', 'pose_joint_mask', 'imu_channel_mask'))
                if any((x.dtype != np.bool_ for x in (fa, ra, pm, im))):
                    raise ValueError('mask dtype must be bool')
                rgb, pose, imu = (arrays['video_uint8'], arrays['pose_coco2d'], arrays['imu_raw_phone_watch'])
                if rgb.dtype != np.uint8 or pose.dtype != np.float32 or imu.dtype != np.float32:
                    raise ValueError('pixel/pose/IMU dtype mismatch')
                if not np.array_equal(ra[:, :, 0], fa) or np.any(ra & ~fa[:, :, None]):
                    raise ValueError('ROI/frame mask disagreement')
                if np.any(rgb[~ra] != 0):
                    raise ValueError('missing ROI pixels not zero')
                if np.any(pm & ~fa[:, :, None]) or not np.isfinite(pose[pm]).all():
                    raise ValueError('invalid observed pose')
                if np.any((pose[:, :, :, 2][pm] <= 0) | (pose[:, :, :, 2][pm] > 1.00001)):
                    raise ValueError('invalid observed confidence')
                if not np.isfinite(imu[im]).all() or not np.isnan(imu[~im]).all():
                    raise ValueError('IMU observed/missing values inconsistent')
                if np.any(arrays['source_frame_index'][fa] < 0) or np.any(arrays['frame_width_height'][fa] <= 0):
                    raise ValueError('observed frame has invalid provenance')
                frame_sum += int(fa.sum())
                pose_sum += int(pm.sum())
                region_sum += ra.sum((0, 1))
                imu_sum += im.sum((0, 1, 2))
                seen.add(index)
                bytes_total += path.stat().st_size
        except Exception as exc:
            failures.append({'file': str(path), 'error': repr(exc)})
    missing = sorted(set(range(len(rows))) - seen)
    complete = not missing and (not failures)
    passed = not failures and bool(seen) and (allow_partial or complete)
    n = len(seen)
    report = {'status': 'PASS' if passed else 'FAIL', 'full_coverage': complete, 'expected_intervals': len(rows), 'validated_intervals': n, 'npz_files': len(paths), 'bytes': bytes_total, 'missing_indices': missing, 'failures': failures, 'frame_available_fraction': frame_sum / max(1, n * 32), 'pose_joint_available_fraction': pose_sum / max(1, n * 32 * 17), 'region_available_fractions': (region_sum / max(1, n * 32)).tolist(), 'imu_channel_available_fractions': (imu_sum / max(1, n * 32 * 4)).tolist(), 'scope': 'Independent array semantics, timestamps and exactly-once native identity; does not certify pose accuracy or run classifier evaluation.'}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('missing_indices', 'failures')}, indent=2))
    return 0 if passed else 1
