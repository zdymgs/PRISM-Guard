from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import argparse
import csv
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path
import cv2
import numpy as np
ROOT = WORK_ROOT
from session_kd.mmfit_rgb_alignment import inspect_video_metadata, validate_reference_mapping, draw_pose_overlay
from session_kd.mmfit_strong_benchmark import sha256_file, ordered_id_hash, validate_split_firewall

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT / 'runs/mmfit_fullroute_v3/assets')
    p.add_argument('--rgb-root', type=Path, default=Path(str(DATA_ROOT / 'Downloads')))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    rows = list(csv.DictReader((ROOT / 'runs/mmfit_rgb_benchmark/inputs.csv').open()))
    labels = np.load(ROOT / 'runs/mmfit_rgb_benchmark/source_labels_test_minus1.npy')
    idx = validate_split_firewall(rows, labels)
    records = []
    missing = []
    total_unique = 0
    offsets = np.stack([np.linspace(0, 0.7 * 149, 16), np.linspace(0.3 * 149, 149, 16)])
    for workout in sorted(set((r['workout'] for r in rows))):
        if (a.output.parent / 'STOP').exists() or (ROOT / 'STOP').exists():
            raise RuntimeError('STOP requested')
        path = a.rgb_root / f'{workout}_rgb.mp4'
        if not path.is_file():
            missing.append(str(path))
            continue
        meta = inspect_video_metadata(path)
        pose = np.load(ROOT / f'data/MM-Fit/mm-fit/{workout}/{workout}_pose_3d.npy', mmap_mode='r')
        clock = np.asarray(pose[0, :, 0])
        starts = np.asarray([int(r['start_frame']) for r in rows if r['workout'] == workout])
        origin = int(clock[0])
        unique = np.unique(origin + np.rint((starts[:, None, None] + offsets - origin) / 5).astype(np.int64) * 5)
        mapping = validate_reference_mapping(np.asarray([int(unique.min()), int(unique.max())]), meta)
        mapping['raw_pose_overhang_frames'] = int(max(0, clock[-1] - meta.frame_count + 1))
        mapping['checked_scope'] = 'all requested shared-grid RGB frames; raw pose tail reported separately'
        total_unique += len(unique)
        probes = np.unique(np.rint(np.linspace(unique.min(), unique.max(), 5)).astype(int))
        cap = cv2.VideoCapture(str(path))
        samples = []
        started = time.perf_counter()
        pose2 = np.load(ROOT / f'data/MM-Fit/mm-fit/{workout}/{workout}_pose_2d.npy', mmap_mode='r')
        for frame in probes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            ok, bgr = cap.read()
            after = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            if not ok or bgr is None or after != int(frame) + 1:
                raise RuntimeError(f'bad decode {path}:{frame}')
            if frame == probes[len(probes) // 2]:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                overlay = draw_pose_overlay(rgb, pose2, int(frame))
                cv2.imwrite(str(a.output / f'{workout}_overlay.jpg'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            samples.append({'frame': int(frame), 'next_frame': after, 'shape': list(bgr.shape), 'std': float(bgr.std())})
        cap.release()
        rec = {'workout': workout, 'bytes': path.stat().st_size, 'metadata': meta.to_dict(), 'reference_mapping': mapping, 'pose_shape': list(pose.shape), 'pose_contiguous': bool(np.all(np.diff(clock) == 1)), 'pose_2d_shape': list(pose2.shape), 'requested_unique_frames_6hz': len(unique), 'decode_probes': samples, 'probe_seconds': time.perf_counter() - started}
        records.append(rec)
        print(json.dumps(rec), flush=True)
    archive = a.rgb_root / 'mm-fit.zip'
    archive_info = {}
    if archive.is_file():
        with zipfile.ZipFile(archive) as z:
            video = [n for n in z.namelist() if n.lower().endswith(('.mp4', '.avi', '.mov'))]
            archive_info = {'path': str(archive), 'entries': len(z.infolist()), 'video_entries': video}
    cache_files = []
    for name in ('mmfit_rgb_benchmark', 'mmfit_strong_features', 'mmfit_unseen_cache'):
        for path in (ROOT / 'runs' / name).glob('*.npy'):
            if path.name == 'labels.npy':
                continue
            x = np.load(path, mmap_mode='r')
            cache_files.append({'path': str(path), 'shape': list(x.shape), 'dtype': str(x.dtype), 'bytes': path.stat().st_size})
    disk = shutil.disk_usage(ROOT)
    report = {'status': 'ready' if not missing else 'missing_rgb', 'test_labels_read': False, 'missing': missing, 'videos': records, 'archive': archive_info, 'cache_files': cache_files, 'disk_free_bytes': disk.free, 'rgb_bytes': sum((r['bytes'] for r in records)), 'ordered_id_sha256': ordered_id_hash([r['segment_id'] for r in rows]), 'shared_frame_plan': {'rate_hz': 6, 'rounding_max_error_frames': 2, 'unique_frames': total_unique, 'uint8_scene_person_bodymotion_bytes': total_unique * 3 * 3 * 160 * 160, 'per_window_full_pixels_bytes': len(rows) * 2 * 16 * 3 * 3 * 160 * 160, 'backbone_sequence_fp16_bytes': len(rows) * 2 * 3 * 16 * 512 * 2}, 'split_counts': {k: len(v) for k, v in idx.items()}}
    (a.output / 'asset_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('videos', 'cache_files')}, indent=2))
