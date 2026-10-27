from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import zipfile
import zlib
from datetime import datetime
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from zoneinfo import ZoneInfo
import numpy as np
import psutil
from session_kd.cache_mmact import SENSORS, _sensor_directory, load_sensor_session
SCHEMA = 'MMACT_INPUT_CACHE_V1'
WINDOW_BOUNDS = ((0.0, 0.7), (0.3, 1.0))
DEFAULT_FLOOR = 10 * 2 ** 30
VIDEO_RE = re.compile('(?:^|/)video/trainval/(cam[1-4])/(subject\\d+)/(scene\\d+)/(session\\d+)/(\\d{8}-\\d{6})\\.mp4$', re.I)

class DiskPaused(RuntimeError):
    pass

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def normalized_times(start_ms, end_ms, frames=16):
    if frames < 2 or end_ms <= start_ms:
        raise ValueError('invalid interval/grid')
    return np.stack([np.linspace(start_ms + lo * (end_ms - start_ms), start_ms + hi * (end_ms - start_ms), frames) for lo, hi in WINDOW_BOUNDS]).astype(np.float64)

def binned_raw_imu(sensor_data, centers, points=4):
    out = np.full((2, len(centers[0]), points, 9), np.nan, np.float32)
    mask = np.zeros_like(out, dtype=bool)
    for w, c in enumerate(centers):
        bounds = np.concatenate(([c[0]], (c[:-1] + c[1:]) / 2, [c[-1]]))
        for slot, (_, times, values) in enumerate(sensor_data):
            for j in range(len(c)):
                chosen = values[(times >= bounds[j]) & (times <= bounds[j + 1])]
                if len(chosen):
                    ids = np.rint(np.linspace(0, len(chosen) - 1, min(points, len(chosen)))).astype(int)
                    out[w, j, :len(ids), slot * 3:(slot + 1) * 3] = chosen[ids]
                    mask[w, j, :len(ids), slot * 3:(slot + 1) * 3] = True
    return (out, mask)

def validate_input_manifest(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    rows = data.get('rows_manifest')
    required = {'global_interval_index', 'interval_id', 'subject', 'scene', 'session', 'selected_camera', 'start_ms', 'end_ms'}
    forbidden = {'label', 'action', 'class', 'label_id', 'operation'}
    if not isinstance(rows, list) or not rows:
        raise ValueError('input manifest has no rows_manifest')
    ids = []
    native = []
    for r in rows:
        if not required <= set(r):
            raise ValueError(f'manifest row missing {sorted(required - set(r))}')
        if forbidden & set(r):
            raise ValueError('labels in input manifest')
        ids.append(int(r['global_interval_index']))
        native.append(str(r['interval_id']))
        if int(r['end_ms']) <= int(r['start_ms']):
            raise ValueError('invalid interval bounds')
        if r['selected_camera'] not in {'cam1', 'cam2', 'cam3', 'cam4'}:
            raise ValueError('invalid selected_camera')
    if ids != list(range(len(rows))) or len(set(ids)) != len(ids):
        raise ValueError('global_interval_index must be unique and ordered from zero')
    if len(set(native)) != len(native):
        raise ValueError('interval_id must be unique')
    if data.get('rows') is not None and int(data['rows']) != len(rows):
        raise ValueError('manifest rows count mismatch')
    if data.get('label_columns_used_for_selection', []) != []:
        raise ValueError('input manifest is not label-free')
    return data

def _safe_member(member):
    p = PurePosixPath(member)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError(f'unsafe ZIP member: {member}')

def scratch_path(scratch, archive, member):
    _safe_member(member)
    root = Path(scratch).resolve()
    root.mkdir(parents=True, exist_ok=True)
    out = (root / (hashlib.sha256(f'{Path(archive).resolve()}\n{member}'.encode()).hexdigest() + '.mp4')).resolve()
    if out.parent != root:
        raise ValueError('scratch containment failure')
    return out

def assert_scratch_contained(path, scratch):
    p, root = (Path(path).resolve(), Path(scratch).resolve())
    if p.parent != root:
        raise ValueError('scratch deletion confinement failure')
    return p

def disk_guard(path, required, floor=DEFAULT_FLOOR):
    free = shutil.disk_usage(path).free
    if free < int(floor) + max(0, int(required)):
        raise DiskPaused(f'free disk {free} below floor+required')
    return free

class OutputLock:

    def __init__(self, path):
        self.path = Path(path)
        self.held = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps({'pid': os.getpid(), 'created': time.time()}).encode())
            os.close(fd)
            self.held = True
            return self
        except FileExistsError:
            try:
                old = json.loads(self.path.read_text())
            except Exception:
                old = {}
            pid = int(old.get('pid', -1))
            if pid > 0 and psutil.pid_exists(pid):
                raise RuntimeError(f'output lock held by PID {pid}')
            raise RuntimeError('stale output lock exists; verify the prior process before removing it')

    def __exit__(self, *_):
        if self.held:
            self.path.unlink(missing_ok=True)

def _member_inventory(inventory, archive_dir):
    old = json.loads(Path(inventory).read_text(encoding='utf-8'))
    expected = {v['name']: v for a in old.get('archives', []) for v in a.get('videos', [])}
    current = {}
    if not expected:
        raise ValueError('archive inventory has no videos')
    for zpath in sorted(Path(archive_dir).glob('untrimmed_cam*.zip')):
        with zipfile.ZipFile(zpath) as z:
            for info in z.infolist():
                if info.filename not in expected:
                    continue
                if info.filename in current:
                    raise ValueError(f'duplicate current archive member: {info.filename}')
                m = VIDEO_RE.search(info.filename)
                if not m:
                    raise ValueError(f'invalid selected video member: {info.filename}')
                hist = expected[info.filename]
                if int(hist.get('bytes', info.file_size)) != info.file_size or int(hist.get('crc32', info.CRC)) != info.CRC:
                    raise ValueError(f'archive member changed: {info.filename}')
                end = int(datetime.strptime(m.group(5), '%Y%m%d-%H%M%S').replace(tzinfo=ZoneInfo('Asia/Tokyo')).timestamp() * 1000)
                current[info.filename] = {'archive': zpath.resolve(), 'member': info.filename, 'size': info.file_size, 'crc32': info.CRC, 'compressed_size': info.compress_size, 'end_ms': end}
    if not current:
        raise FileNotFoundError('no historical archive members found')
    return current

def extract_member(meta, scratch):
    target = scratch_path(scratch, meta['archive'], meta['member'])
    if target.exists():
        crc = 0
        with target.open('rb') as f:
            for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
                crc = zlib.crc32(b, crc)
        if target.stat().st_size == meta['size'] and crc & 4294967295 == meta['crc32']:
            return target
        target.unlink()
    tmp = target.with_suffix('.extracting')
    tmp.unlink(missing_ok=True)
    with zipfile.ZipFile(meta['archive']) as z, z.open(meta['member']) as src, tmp.open('wb') as dst:
        shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    crc = 0
    with tmp.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            crc = zlib.crc32(b, crc)
    if tmp.stat().st_size != meta['size'] or crc & 4294967295 != meta['crc32']:
        tmp.unlink(missing_ok=True)
        raise ValueError('extracted member size/CRC mismatch')
    os.replace(tmp, target)
    return target

def _source_hashes():
    root = Path(__file__).resolve().parent
    return {n: sha256_file(root / n) for n in ('mmact_input_cache.py', '_roi_preprocessing.py', 'cache_mmact.py')}

def build_run_contract(manifest_path, manifest, pose_model, inventory, config):
    pose_model = Path(pose_model)
    if not pose_model.is_file():
        raise FileNotFoundError(f'pose model missing: {pose_model}')
    rows = manifest['rows_manifest']
    mapping = [{k: r[k] for k in ('global_interval_index', 'interval_id', 'selected_camera', 'subject', 'scene', 'session', 'start_ms', 'end_ms')} for r in rows]
    return {'schema': SCHEMA, 'input_manifest': str(Path(manifest_path).resolve()), 'input_manifest_sha256': sha256_file(manifest_path), 'rows': len(rows), 'camera_mapping_sha256': hashlib.sha256(json.dumps(mapping, sort_keys=True, separators=(',', ':')).encode()).hexdigest(), 'camera_counts': {c: sum((r['selected_camera'] == c for r in rows)) for c in ('cam1', 'cam2', 'cam3', 'cam4')}, 'pose_model': str(pose_model.resolve()), 'pose_model_sha256': sha256_file(pose_model), 'archive_inventory_sha256': sha256_file(inventory), 'source_helper_sha256': _source_hashes(), 'config': config, 'label_free': True, 'sensor_source_timestamps_cached': False, 'sensor_source_timestamps_note': 'values/masks use cache_mmact absolute-time convention; original observation timestamps omitted'}

def _video_contract(run_contract, meta, rows):
    source = {k: str(v) if isinstance(v, Path) else v for k, v in meta.items()}
    return {'run_contract_sha256': hashlib.sha256(json.dumps(run_contract, sort_keys=True).encode()).hexdigest(), 'source_member': source, 'native_rows': [{'global_interval_index': int(r['global_interval_index']), 'interval_id': str(r['interval_id'])} for r in rows]}

def _interval_path(output, row):
    return Path(output) / f'interval_{int(row['global_interval_index']):06d}.npz'

def _expected_shapes(frames, size):
    return {'video_uint8': (2, frames, 3, 3, size, size), 'pose_coco2d': (2, frames, 17, 3), 'pose_joint_mask': (2, frames, 17), 'frame_available': (2, frames), 'region_available': (2, frames, 3), 'roi_person_workspace_xyxy': (2, frames, 2, 4), 'workspace_broad_fallback': (2, frames), 'frame_width_height': (2, frames, 2), 'imu_raw_phone_watch': (2, frames, 4, 9), 'imu_channel_mask': (2, frames, 4, 9), 'source_time_ms': (2, frames), 'source_frame_index': (2, frames), 'person_count': (2, frames), 'person_tracked': (2, frames)}

def _validate_interval(path, row, frames=16, size=160):
    with np.load(path, allow_pickle=False) as d:
        for k, shape in _expected_shapes(frames, size).items():
            if k not in d or d[k].shape != shape:
                raise ValueError(f'{Path(path).name}: invalid {k}')
        if int(d['global_interval_index'].item()) != int(row['global_interval_index']):
            raise ValueError('global ID mismatch')
        if str(d['interval_id'].item()) != str(row['interval_id']):
            raise ValueError('native interval ID mismatch')
        if str(d['video_id'].item()) != str(row['video_id']):
            raise ValueError('video ID mismatch')
        if d['video_uint8'].dtype != np.uint8 or d['pose_coco2d'].dtype != np.float32 or d['imu_raw_phone_watch'].dtype != np.float32:
            raise ValueError('array dtype mismatch')
        for k in ('pose_joint_mask', 'frame_available', 'region_available', 'imu_channel_mask', 'person_tracked'):
            if d[k].dtype != np.bool_:
                raise ValueError(f'mask dtype mismatch: {k}')

def validate_resume(marker, contract, output, rows=None, frames=16, size=160, sensor_receipts=None):
    if rows is None:
        raise ValueError('resume rows required')
    marker = Path(marker)
    if not marker.is_file():
        raise ValueError('completion marker missing')
    state = json.loads(marker.read_text(encoding='utf-8'))
    if state.get('status') != 'complete' or state.get('contract') != contract:
        raise ValueError('resume contract mismatch')
    if sensor_receipts is not None and state.get('sensor_receipts') != sensor_receipts:
        raise ValueError('resume sensor source fingerprint mismatch')
    records = state.get('interval_files', [])
    if len(records) != len(rows):
        raise ValueError('completion marker interval count mismatch')
    indexed = {int(x['global_interval_index']): x for x in records}
    if len(indexed) != len(records):
        raise ValueError('duplicate marker interval ID')
    for row in rows:
        rec = indexed.get(int(row['global_interval_index']))
        path = _interval_path(output, row)
        if rec is None or rec.get('file') != path.name or rec.get('interval_id') != str(row['interval_id']):
            raise ValueError('completion marker native identity mismatch')
        if not path.is_file() or rec.get('sha256') != sha256_file(path):
            raise ValueError('interval cache corruption or hash mismatch')
        _validate_interval(path, row, frames, size)

def _write_npz(path, arrays, floor):
    path = Path(path)
    required = sum((int(getattr(v, 'nbytes', 0)) for v in arrays.values()))
    before = disk_guard(path.parent, required, floor)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    disk_guard(path.parent, 0, floor)
    os.replace(tmp, path)
    return {'file': path.name, 'bytes': path.stat().st_size, 'sha256': sha256_file(path), 'free_bytes_before': before}

def _interval_obj(row):
    return SimpleNamespace(subject=row['subject'], scene=row['scene'], session=row['session'], start_ms=int(row['start_ms']), end_ms=int(row['end_ms']), line_number=0)

def _load_sensors(root, row):
    data = []
    receipts = []
    for name, spec in SENSORS.items():
        try:
            directory = _sensor_directory(root, name, _interval_obj(row))
            files = sorted(directory.glob('*.csv'), key=lambda p: p.name.lower())
            times, values = load_sensor_session(directory, spec['rate'], spec['scale'])
            data.append((name, times, values))
            receipts.append({'sensor': name, 'status': 'present', 'directory': str(directory.resolve()), 'files': [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': sha256_file(p)} for p in files], 'samples': len(times), 'first_ms': float(times[0]), 'last_ms': float(times[-1])})
        except FileNotFoundError:
            data.append((name, np.empty(0, np.float64), np.empty((0, 3), np.float32)))
            receipts.append({'sensor': name, 'status': 'missing', 'files': [], 'samples': 0})
    return (data, receipts)

def _empty_record(grid, frames, size, sensors):
    imu, imask = binned_raw_imu(sensors, grid, 4)
    return {'video_uint8': np.zeros((2, frames, 3, 3, size, size), np.uint8), 'pose_coco2d': np.full((2, frames, 17, 3), np.nan, np.float32), 'pose_joint_mask': np.zeros((2, frames, 17), bool), 'frame_available': np.zeros((2, frames), bool), 'region_available': np.zeros((2, frames, 3), bool), 'roi_person_workspace_xyxy': np.full((2, frames, 2, 4), np.nan, np.float32), 'workspace_broad_fallback': np.zeros((2, frames), bool), 'frame_width_height': np.zeros((2, frames, 2), np.int32), 'imu_raw_phone_watch': imu, 'imu_channel_mask': imask, 'source_time_ms': grid, 'source_frame_index': np.full((2, frames), -1, np.int32), 'person_count': np.zeros((2, frames), np.int16), 'person_tracked': np.zeros((2, frames), bool)}

def _pilot_helpers():
    from session_kd import _roi_preprocessing as pilot
    return pilot

def _process_video(video, rows, sensors, model_path, device, frames, size, batch_size=16, end_ms=None, detector=None):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if end_ms is None:
        raise ValueError('original member end timestamp is required')
    import cv2
    pilot = _pilot_helpers()
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or count <= 0:
        cap.release()
        raise ValueError('invalid video probe')
    start = float(end_ms) - count / fps * 1000
    grids = [normalized_times(int(r['start_ms']), int(r['end_ms']), frames) for r in rows]
    records = [_empty_record(g, frames, size, sensors) for g in grids]
    consumers = {}
    for local, g in enumerate(grids):
        for w in range(2):
            for j, t in enumerate(g[w]):
                if start <= t <= float(end_ms):
                    idx = int(round((t - start) * fps / 1000))
                    if 0 <= idx < count:
                        records[local]['source_frame_index'][w, j] = idx
                        consumers.setdefault(idx, []).append((local, w, j))
    requested = sorted(consumers)
    model = detector if detector is not None else pilot.YOLO(str(model_path))
    previous = None
    batch_i = []
    batch_f = []
    decoded = inferred = peak = 0

    def consume():
        nonlocal previous, inferred, peak
        if not batch_f:
            return
        peak = max(peak, len(batch_f))
        results = model.predict(batch_f, device=device, imgsz=640, conf=0.1, verbose=False)
        if len(results) != len(batch_f):
            raise ValueError('pose result count mismatch')
        for idx, frame, res in zip(batch_i, batch_f, results, strict=True):
            boxes = np.empty((0, 4), np.float32) if res.boxes is None else res.boxes.xyxy.cpu().numpy()
            kp = np.empty((0, 17, 3), np.float32) if res.keypoints is None else res.keypoints.data.cpu().numpy()
            choice = pilot.select_person(boxes, previous)
            person = pose = work = None
            if choice >= 0:
                previous = boxes[choice].astype(np.float32)
                person = previous.copy()
                pose = kp[choice].astype(np.float32)
                work = pilot.workspace_box(pose, person)
            h, w0 = frame.shape[:2]
            scene, _, _ = pilot.letterbox_rgb(frame, size)
            for local, w, j in consumers[idx]:
                r = records[local]
                r['frame_available'][w, j] = True
                r['region_available'][w, j, 0] = True
                r['video_uint8'][w, j, 0] = scene
                r['frame_width_height'][w, j] = [w0, h]
                r['person_count'][w, j] = min(len(boxes), 32767)
                if person is not None:
                    r['person_tracked'][w, j] = True
                    r['pose_coco2d'][w, j, :, 0] = pose[:, 0] / w0
                    r['pose_coco2d'][w, j, :, 1] = pose[:, 1] / h
                    r['pose_coco2d'][w, j, :, 2] = pose[:, 2]
                    r['pose_joint_mask'][w, j] = np.isfinite(pose).all(1) & (pose[:, 2] > 0)
                    r['roi_person_workspace_xyxy'][w, j] = [person, work[0]]
                    r['workspace_broad_fallback'][w, j] = not work[1]
                    r['video_uint8'][w, j, 1], _, _ = pilot.letterbox_rgb(pilot.square_crop(frame, person, 1.15), size)
                    r['video_uint8'][w, j, 2], _, _ = pilot.letterbox_rgb(pilot.square_crop(frame, work[0], 1.4), size)
                    r['region_available'][w, j, 1:] = True
            inferred += 1
        batch_i.clear()
        batch_f.clear()
    try:
        it = iter(requested)
        target = next(it, None)
        for idx in range(requested[-1] + 1 if requested else 0):
            if not cap.grab():
                break
            if idx != target:
                continue
            ok, frame = cap.retrieve()
            if ok:
                decoded += 1
                batch_i.append(idx)
                batch_f.append(frame)
                if len(batch_f) >= batch_size:
                    consume()
            target = next(it, None)
            if target is None:
                break
        consume()
    finally:
        cap.release()
    if decoded != len(requested) or inferred != len(requested):
        raise ValueError(f'incomplete chronological decode/pose: requested={len(requested)} decoded={decoded} inferred={inferred}')
    if requested and (not any((r['frame_available'].any() for r in records))):
        raise ValueError('video produced no available requested frames')
    return (records, {'fps': fps, 'frame_count': count, 'start_ms': start, 'end_ms': int(end_ms), 'requested_unique_frames': len(requested), 'decoded_unique_frames': decoded, 'pose_inferred_frames': inferred, 'peak_pose_batch': peak})

def run(args):
    started = time.time()
    manifest_path = Path(args.input_manifest).resolve()
    manifest = validate_input_manifest(manifest_path)
    rows = manifest['rows_manifest']
    archive_dir = Path(args.archive_dir).resolve()
    inventory = Path(args.archive_inventory).resolve()
    root = Path(args.dataset_root).resolve()
    pose = Path(args.pose_model).resolve()
    members = _member_inventory(inventory, archive_dir)
    if args.frames != 16 or args.size != 160:
        raise ValueError('V1 contract requires 16 frames and 160px')
    config = {'frames': 16, 'size': 160, 'batch_size': int(args.batch_size), 'window_bounds': [list(x) for x in WINDOW_BOUNDS], 'regions': ['scene', 'person', 'workspace'], 'imu_slots': ['phone_acc_xyz_g', 'watch_acc_xyz_g', 'phone_gyro_xyz'], 'imu_points_per_visual_bin': 4}
    contract = build_run_contract(manifest_path, manifest, pose, inventory, config)
    out = Path(args.output).resolve()
    scratch = Path(args.scratch).resolve()
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    groups = {}
    for r in rows:
        vid = f'{r['selected_camera']}:{r['subject']}:{r['scene']}:{r['session']}'
        groups.setdefault(vid, []).append({**r, 'video_id': vid})
    queue = sorted(groups.items())
    if args.video_id:
        queue = [x for x in queue if x[0] == args.video_id]
    if args.limit_videos is not None:
        queue = queue[:int(args.limit_videos)]
    if not queue:
        raise ValueError('requested video selection is empty')
    stats = {'schema': SCHEMA, 'status': 'RUNNING', 'scope': 'full' if len(queue) == len(groups) else 'partial', 'expected_intervals': len(rows), 'expected_videos': len(groups), 'selected_videos': len(queue), 'completed_videos': 0, 'skipped_videos': 0, 'completed_intervals': 0, 'skipped_intervals': 0, 'errors': [], 'videos': [], 'started_unix': started}
    with OutputLock(out / '.lock'):
        rm = out / 'RUN_MANIFEST.json'
        if rm.exists():
            if json.loads(rm.read_text(encoding='utf-8')) != contract:
                raise ValueError('committed run contract changed')
        else:
            atomic_json(rm, contract)
        atomic_json(out / 'status.json', stats)
        for vid, selected in queue:
            if args.stop_file and Path(args.stop_file).exists():
                stats['status'] = 'STOPPED_AT_VIDEO_BOUNDARY'
                break
            camera, subject, scene, session = vid.split(':')
            hits = [n for n in members if f'video/trainval/{camera}/{subject}/{scene}/{session}/' in n]
            entry = {'video_id': vid, 'intervals': len(selected), 'started_unix': time.time(), 'free_bytes_before': shutil.disk_usage(out).free}
            if len(hits) != 1:
                entry.update(status='error', error=f'expected one archive member, found {len(hits)}')
                stats['errors'].append(entry.copy())
                stats['videos'].append(entry)
                atomic_json(out / 'status.json', stats)
                continue
            meta = members[hits[0]]
            vc = _video_contract(contract, meta, selected)
            marker = out / (vid.replace(':', '_') + '.complete.json')
            if marker.exists():
                _, current_sensor_receipts = _load_sensors(root, selected[0])
                validate_resume(marker, vc, out, selected, sensor_receipts=current_sensor_receipts)
                entry.update(status='skipped_verified', completed_unix=time.time(), free_bytes_after=shutil.disk_usage(out).free)
                stats['skipped_videos'] += 1
                stats['skipped_intervals'] += len(selected)
                stats['videos'].append(entry)
                atomic_json(out / 'status.json', stats)
                continue
            video = None
            try:
                print(f'BEGIN_VIDEO video_id={vid} member={meta['member']} unix={time.time():.6f}', file=sys.stderr, flush=True)
                disk_guard(scratch, int(meta['size']), args.free_floor)
                video = extract_member(meta, scratch)
                sensors, sensor_receipts = _load_sensors(root, selected[0])
                values, processing = _process_video(video, selected, sensors, pose, args.device, 16, 160, args.batch_size, meta['end_ms'])
                files = []
                for row, arrays in zip(selected, values, strict=True):
                    arrays.update(global_interval_index=np.asarray(int(row['global_interval_index']), np.int32), interval_id=np.asarray(str(row['interval_id'])), video_id=np.asarray(vid))
                    path = _interval_path(out, row)
                    receipt = _write_npz(path, arrays, args.free_floor)
                    _validate_interval(path, row)
                    receipt.update(global_interval_index=int(row['global_interval_index']), interval_id=str(row['interval_id']))
                    files.append(receipt)
                atomic_json(marker, {'status': 'complete', 'contract': vc, 'interval_files': files, 'sensor_receipts': sensor_receipts, 'processing': processing, 'completed_unix': time.time()})
                validate_resume(marker, vc, out, selected)
                entry.update(status='completed', processing=processing, sensor_receipts=sensor_receipts, completed_unix=time.time(), free_bytes_after=shutil.disk_usage(out).free)
                stats['completed_videos'] += 1
                stats['completed_intervals'] += len(selected)
            except DiskPaused as exc:
                entry.update(status='paused_disk', error=str(exc), completed_unix=time.time(), free_bytes_after=shutil.disk_usage(out).free)
                stats['videos'].append(entry)
                stats['status'] = 'PAUSED_DISK'
                atomic_json(out / 'status.json', stats)
                break
            except Exception as exc:
                entry.update(status='error', error=repr(exc), completed_unix=time.time(), free_bytes_after=shutil.disk_usage(out).free)
                stats['errors'].append(entry.copy())
            finally:
                if video is not None and video.exists():
                    assert_scratch_contained(video, scratch).unlink()
                print(f'END_VIDEO video_id={vid} status={entry.get('status', 'unknown')} unix={time.time():.6f}', file=sys.stderr, flush=True)
            stats['videos'].append(entry)
            atomic_json(out / 'status.json', stats)
        if stats['status'] == 'RUNNING':
            stats['status'] = 'COMPLETE' if len(queue) == len(groups) and (not stats['errors']) else 'PARTIAL_COMPLETE'
        stats['validated_intervals_this_invocation'] = stats['completed_intervals'] + stats['skipped_intervals']
        stats['interval_files_on_disk'] = len(list(out.glob('interval_*.npz')))
        stats['free_bytes_final'] = shutil.disk_usage(out).free
        stats['elapsed_seconds'] = time.time() - started
        atomic_json(out / 'status.json', stats)
    return stats

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-manifest', required=True, type=Path)
    p.add_argument('--archive-dir', required=True, type=Path)
    p.add_argument('--archive-inventory', required=True, type=Path)
    p.add_argument('--dataset-root', required=True, type=Path)
    p.add_argument('--pose-model', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--scratch', required=True, type=Path)
    p.add_argument('--frames', type=int, default=16)
    p.add_argument('--size', type=int, default=160)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--device', default='0')
    p.add_argument('--limit-videos', type=int)
    p.add_argument('--video-id')
    p.add_argument('--stop-file', type=Path)
    p.add_argument('--free-floor', type=int, default=DEFAULT_FLOOR)
    args = p.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        prior = None
        try:
            prior = json.loads((output / 'status.json').read_text(encoding='utf-8'))
        except Exception:
            pass
        failed = {'schema': SCHEMA, 'status': 'FAILED', 'error': repr(exc), 'failed_unix': time.time(), 'prior_status': prior}
        atomic_json(output / 'status.json', failed)
        print(json.dumps(failed, indent=2))
        raise
    print(json.dumps(result, indent=2))
    sys.exit(0 if result['status'] in {'COMPLETE', 'PARTIAL_COMPLETE'} and (not result.get('errors')) else 2)
