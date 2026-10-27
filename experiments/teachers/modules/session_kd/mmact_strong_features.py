from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import stat
import time
import zipfile
import zlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo
import cv2
import numpy as np
import torch
from session_kd.mmact import MMActInterval, discover_annotation_files, parse_annotation_file
from session_kd.mmfit_strong_features import Encoder, load_encoder, sha256_file
MODEL_NAMES = ('videomaev2_large', 'vjepa2_vitl_ssv2')
VIDEO_RE = re.compile('(?:^|/)video/trainval/(?P<camera>cam[1-4])/(?P<subject>subject\\d+)/(?P<scene>scene\\d+)/(?P<session>session\\d+)/(?P<end>\\d{8}-\\d{6})\\.mp4$', re.IGNORECASE)
SAMPLE_COUNT = 16

def letterbox_rgb(frame: np.ndarray, size: int) -> np.ndarray:
    """Aspect-preserving full-frame resize with symmetric black padding."""
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    output = np.zeros((size, size, 3), dtype=np.uint8)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    output[top:top + resized_height, left:left + resized_width] = resized
    return np.ascontiguousarray(output)

@dataclass(frozen=True)
class VideoMember:
    archive: str
    member: str
    camera: str
    subject: str
    scene: str
    session: str
    end_ms: int
    size: int
    crc32: int

    @property
    def session_key(self) -> tuple[str, str, str]:
        return (self.subject, self.scene, self.session)

    @property
    def video_id(self) -> str:
        return f'{self.camera}:{self.subject}:{self.scene}:{self.session}'

def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.building')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)

def _end_timestamp_ms(text: str) -> int:
    parsed = datetime.strptime(text, '%Y%m%d-%H%M%S').replace(tzinfo=ZoneInfo('Asia/Tokyo'))
    return int(parsed.timestamp() * 1000)

def inventory_archives(archive_dir: Path) -> list[VideoMember]:
    members: list[VideoMember] = []
    seen: set[tuple[str, str, str, str]] = set()
    for archive in sorted(archive_dir.glob('untrimmed_cam*.zip')):
        with zipfile.ZipFile(archive) as handle:
            for info in handle.infolist():
                match = VIDEO_RE.search(info.filename)
                if not match:
                    continue
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError(f'ZIP video member is a symlink: {archive}:{info.filename}')
                values = {key: value.lower() for key, value in match.groupdict().items() if key != 'end'}
                key = (values['camera'], values['subject'], values['scene'], values['session'])
                if key in seen:
                    raise ValueError(f'duplicate MMAct camera/session member: {key}')
                seen.add(key)
                members.append(VideoMember(str(archive.resolve()), info.filename, **values, end_ms=_end_timestamp_ms(match.group('end')), size=info.file_size, crc32=info.CRC))
    if not members:
        raise FileNotFoundError(f'no MMAct camera ZIP members under {archive_dir}')
    return sorted(members, key=lambda item: item.video_id)

def load_all_intervals(dataset_root: Path) -> tuple[list[MMActInterval], dict[tuple[str, str, str], list[int]]]:
    intervals = [item for path in discover_annotation_files(dataset_root) for item in parse_annotation_file(path, dataset_root)]
    intervals.sort(key=lambda x: (x.subject, x.scene, x.session, x.start_ms, x.end_ms, x.line_number))
    grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, interval in enumerate(intervals):
        grouped[interval.subject, interval.scene, interval.session].append(index)
    if not intervals:
        raise ValueError('no MMAct annotations')
    return (intervals, grouped)

def _safe_scratch_file(scratch_root: Path, member: VideoMember) -> Path:
    root = scratch_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    leaf = hashlib.sha256(f'{member.archive}\n{member.member}'.encode()).hexdigest() + '.mp4'
    target = (root / leaf).resolve()
    if target.parent != root or PurePosixPath(member.member).is_absolute() or '..' in PurePosixPath(member.member).parts:
        raise ValueError(f'unsafe ZIP member path: {member.member}')
    return target

def extract_member(member: VideoMember, scratch_root: Path) -> Path:
    target = _safe_scratch_file(scratch_root, member)
    if target.exists() and target.stat().st_size == member.size:
        crc = 0
        with target.open('rb') as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
                crc = zlib.crc32(block, crc)
        if crc & 4294967295 == member.crc32:
            return target
        target.unlink()
    temporary = target.with_suffix('.mp4.extracting')
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(member.archive) as archive, archive.open(member.member) as source, temporary.open('wb') as destination:
        shutil.copyfileobj(source, destination, length=4 * 1024 * 1024)
        destination.flush()
        os.fsync(destination.fileno())
    if temporary.stat().st_size != member.size:
        temporary.unlink(missing_ok=True)
        raise ValueError(f'extracted size mismatch for {member.video_id}')
    crc = 0
    with temporary.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            crc = zlib.crc32(block, crc)
    if crc & 4294967295 != member.crc32:
        temporary.unlink(missing_ok=True)
        raise ValueError(f'extracted CRC mismatch for {member.video_id}')
    temporary.replace(target)
    return target

def probe_video(path: Path, end_ms: int) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f'could not open {path}')
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    capture.release()
    if not np.isfinite(fps) or fps <= 0 or frames <= 0:
        raise ValueError(f'invalid video probe fps={fps}, frames={frames}: {path}')
    duration_ms = frames / fps * 1000.0
    return {'fps': fps, 'frames': frames, 'duration_ms': duration_ms, 'start_ms': float(end_ms - duration_ms), 'end_ms': int(end_ms)}

def interval_frame_plan(intervals: list[MMActInterval], indices: list[int], probe: dict) -> tuple[dict[int, list[tuple[int, int]]], np.ndarray, np.ndarray]:
    requests: dict[int, list[tuple[int, int]]] = defaultdict(list)
    covered = np.zeros(len(indices), dtype=np.bool_)
    sampled_ms = np.full((len(indices), SAMPLE_COUNT), np.nan, dtype=np.float64)
    tolerance = 1000.0 / float(probe['fps']) + 1.0
    for local, global_index in enumerate(indices):
        row = intervals[global_index]
        if row.start_ms < float(probe['start_ms']) - tolerance or row.end_ms > float(probe['end_ms']) + tolerance:
            continue
        times = np.linspace(row.start_ms, row.end_ms, SAMPLE_COUNT, dtype=np.float64)
        frame_ids = np.rint((times - float(probe['start_ms'])) * float(probe['fps']) / 1000.0).astype(np.int64)
        frame_ids = np.clip(frame_ids, 0, int(probe['frames']) - 1)
        covered[local] = True
        sampled_ms[local] = times
        for slot, frame_id in enumerate(frame_ids):
            requests[int(frame_id)].append((local, slot))
    return (dict(requests), covered, sampled_ms)

def encode_video(path: Path, intervals: list[MMActInterval], indices: list[int], encoder: Encoder, probe: dict, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    requests, covered, sampled_ms = interval_frame_plan(intervals, indices, probe)
    features = np.full((len(indices), encoder.feature_dim), np.nan, dtype=np.float16)
    pending = {row: [None] * SAMPLE_COUNT for row in np.flatnonzero(covered)}
    clips: list[np.ndarray] = []
    clip_rows: list[int] = []
    capture = cv2.VideoCapture(str(path))
    cv2.setNumThreads(2)
    started = time.perf_counter()

    def flush() -> None:
        if not clips:
            return
        values = encoder.encode(clips)
        if values.shape != (len(clips), encoder.feature_dim) or not np.isfinite(values).all():
            raise ValueError(f'invalid {encoder.name} encoder output {values.shape}')
        features[np.asarray(clip_rows)] = values.astype(np.float16)
        clips.clear()
        clip_rows.clear()
    try:
        for frame_id in range(max(requests, default=-1) + 1):
            if not capture.grab():
                raise RuntimeError(f'decode ended at frame {frame_id}: {path}')
            if frame_id not in requests:
                continue
            ok, bgr = capture.retrieve()
            if not ok or bgr is None:
                raise RuntimeError(f'failed requested frame {frame_id}: {path}')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            prepared = letterbox_rgb(rgb, encoder.input_size)
            for row, slot in requests[frame_id]:
                pending[row][slot] = prepared
                if all((value is not None for value in pending[row])):
                    clips.append(np.stack(pending.pop(row)))
                    clip_rows.append(row)
                    if len(clips) >= batch_size:
                        flush()
        flush()
    finally:
        capture.release()
    if pending or not np.isfinite(features[covered].astype(np.float32)).all():
        raise ValueError(f'incomplete/nonfinite covered features for {path}')
    return (features, covered, sampled_ms, {'elapsed_seconds': time.perf_counter() - started, 'covered_intervals': int(covered.sum()), 'total_session_intervals': len(indices), 'requested_unique_frames': len(requests), 'decode_mode': 'opencv_grab_all_retrieve_requested_threads2'})

def _cache_path(output: Path, model: str, member: VideoMember) -> Path:
    return output / model / member.camera / member.subject / member.scene / f'{member.session}.npz'

def validate_cache(path: Path, expected_rows: int, expected_dim: int, expected_ids: list[str] | None=None) -> dict:
    with np.load(path, allow_pickle=False) as cache:
        features = cache['features']
        covered = cache['video_covered']
        if features.shape != (expected_rows, expected_dim) or features.dtype != np.float16:
            raise ValueError(f'cache shape/dtype mismatch: {features.shape}/{features.dtype}')
        if covered.shape != (expected_rows,) or covered.dtype != np.bool_:
            raise ValueError('coverage mask contract mismatch')
        if not np.isfinite(features[covered].astype(np.float32)).all():
            raise ValueError('nonfinite covered cache features')
        if np.isfinite(features[~covered].astype(np.float32)).any():
            raise ValueError('uncovered feature rows must remain NaN')
        if expected_ids is not None and cache['interval_id'].tolist() != expected_ids:
            raise ValueError('cache interval identity/order mismatch')
        return {'rows': expected_rows, 'covered': int(covered.sum()), 'finite': True, 'shape': list(features.shape), 'dtype': str(features.dtype)}

def write_cache(path: Path, features: np.ndarray, covered: np.ndarray, sampled_ms: np.ndarray, sensor_usable: np.ndarray, indices: list[int], intervals: list[MMActInterval], metadata: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.npz.building')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, features=features, video_covered=covered, sampled_timestamps_ms=sampled_ms, sensor_usable=sensor_usable, global_interval_index=np.asarray(indices, dtype=np.int32), interval_id=np.asarray([f'{intervals[i].annotation_file}:{intervals[i].line_number}' for i in indices]))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    ids = [f'{intervals[i].annotation_file}:{intervals[i].line_number}' for i in indices]
    validation = validate_cache(path, len(indices), features.shape[1], ids)
    metadata.update({'status': 'complete', 'cache_file': str(path.resolve()), 'cache_sha256': sha256_file(path), 'validation': validation})
    _atomic_json(path.with_suffix('.complete.json'), metadata)
    return validation

def _sensor_masks(path: Path, intervals: list[MMActInterval]) -> tuple[np.ndarray, list[str]]:
    sensors = ['acc_phone_clip', 'acc_watch_clip', 'gyro_clip']
    lookup: dict[tuple[str, str, str, int, str], bool] = {}
    with path.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            key = (row['subject'], row['scene'], row['session'], int(row['line_number']), row['sensor'])
            lookup[key] = str(row['usable']).lower() == 'true'
    masks = np.zeros((len(intervals), len(sensors)), dtype=np.bool_)
    for index, interval in enumerate(intervals):
        for column, sensor in enumerate(sensors):
            key = (interval.subject, interval.scene, interval.session, interval.line_number, sensor)
            if key not in lookup:
                raise ValueError(f'sensor coverage row missing: {key}')
            masks[index, column] = lookup[key]
    return (masks, sensors)

def run(dataset_root: Path, archive_dir: Path, output: Path, scratch: Path, sensor_coverage: Path, models: list[str], batch_size: int, delete_extracted: bool, limit: int | None=None) -> dict:
    intervals, grouped = load_all_intervals(dataset_root)
    all_sensor_masks, sensor_names = _sensor_masks(sensor_coverage, intervals)
    members = inventory_archives(archive_dir)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / 'interval_manifest.json', {'protocol': 'MMACT_STRONG_FEATURES_V1', 'instances': len(intervals), 'label_use': 'provenance_only_not_used_for_feature_selection', 'sensor_coverage_csv': str(sensor_coverage.resolve()), 'sensor_coverage_sha256': sha256_file(sensor_coverage), 'sensor_mask_columns': sensor_names, 'intervals': [asdict(row) for row in intervals]})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('strong feature cache requires CUDA')
    summary = {'videos_seen': len(members), 'models': models, 'completed': 0, 'skipped': 0}
    encoders = {name: load_encoder(name, device) for name in models}
    for member in members[:limit]:
        indices = grouped.get(member.session_key, [])
        if not indices:
            raise ValueError(f'video has no annotation session: {member.video_id}')
        needed = []
        for name, encoder in encoders.items():
            path = _cache_path(output, name, member)
            marker = path.with_suffix('.complete.json')
            ids = [f'{intervals[i].annotation_file}:{intervals[i].line_number}' for i in indices]
            contract = {'archive': member.archive, 'member': member.member, 'source_crc32': member.crc32, 'model': name, 'checkpoint': str(encoder.checkpoint), 'preprocess': f'RGB full-frame aspect-preserving letterbox {encoder.input_size}x{encoder.input_size}; frozen local processor normalization; 16 uniform frames', 'sampling': '16 uniform annotation timestamps mapped by filename-end minus frames/fps', 'ordered_interval_id_sha256': hashlib.sha256(('\n'.join(ids) + '\n').encode()).hexdigest(), 'ordered_interval_timing_sha256': hashlib.sha256(('\n'.join((f'{intervals[i].start_ms}:{intervals[i].end_ms}' for i in indices)) + '\n').encode()).hexdigest(), 'sensor_coverage_sha256': sha256_file(sensor_coverage)}
            if path.exists() and marker.exists():
                state = json.loads(marker.read_text(encoding='utf-8'))
                if all((state.get(key) == value for key, value in contract.items())):
                    validate_cache(path, len(indices), encoder.feature_dim, ids)
                    if sha256_file(path) != state.get('cache_sha256'):
                        raise ValueError(f'resume cache hash mismatch for {member.video_id}/{name}')
                    summary['skipped'] += 1
                else:
                    raise ValueError(f'resume contract mismatch for {member.video_id}/{name}')
            else:
                needed.append((name, encoder, path, contract))
        if not needed:
            continue
        video = extract_member(member, scratch)
        probe = probe_video(video, member.end_ms)
        for name, encoder, path, contract in needed:
            features, covered, sampled_ms, stats = encode_video(video, intervals, indices, encoder, probe, batch_size)
            if int(covered.sum()) == 0:
                raise ValueError(f'zero annotation coverage for {member.video_id}; check end-time alignment')
            metadata = {'video': asdict(member), 'probe': probe, **contract, 'test_label_columns_used_for_feature_choices': [], **stats}
            write_cache(path, features, covered, sampled_ms, all_sensor_masks[indices], indices, intervals, metadata)
            summary['completed'] += 1
            _atomic_json(output / 'run_summary.json', {**summary, 'status': 'building', 'last_completed_video': member.video_id, 'last_completed_model': name})
            print(json.dumps({'video_id': member.video_id, 'model': name, **stats}), flush=True)
        if delete_extracted:
            expected = _safe_scratch_file(scratch, member)
            if video.resolve() != expected or video.parent.resolve() != scratch.resolve():
                raise ValueError('refusing to delete file outside resolved scratch root')
            video.unlink()
    _atomic_json(output / 'run_summary.json', {**summary, 'status': 'invocation_complete'})
    return summary

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--archive-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--sensor-coverage', type=Path, required=True, help='audited mmact_sensor_interval_coverage.csv; retained as masks, never a filter')
    parser.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--delete-extracted', action='store_true', help='delete only validated per-member scratch MP4s; ZIPs are always retained')
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.archive_dir, args.output, args.scratch, args.sensor_coverage, args.models, args.batch_size, args.delete_extracted, args.limit), indent=2))
