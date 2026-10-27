from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import hashlib
import json
import os
import shutil
from pathlib import Path
import numpy as np
TAIL = (2, 3, 16, 512)
FULL = 5406
PILOT_IDS = (3577, 3578)
FLOOR = 5 * 2 ** 30

def sha256_file(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def atomic_json(p, v):
    t = Path(str(p) + '.tmp')
    with t.open('w') as f:
        json.dump(v, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(t, p)

def guard(p, n=0):
    if shutil.disk_usage(p).free < FLOOR + n:
        raise RuntimeError('below 5GiB disk floor')

def read_visual_manifest(p):
    d = json.loads(Path(p).read_text())
    r = d.get('rows_manifest', d)
    if not isinstance(r, list) or len(r) != FULL or [int(x.get('global_interval_index', -1)) for x in r] != list(range(FULL)) or (len({str(x.get('interval_id')) for x in r}) != FULL):
        raise ValueError('manifest native IDs invalid')
    if any(({'label', 'action', 'class', 'label_id', 'operation'} & set(x) for x in r)):
        raise ValueError('manifest must be label-free')
    return r

def checkpoint_gate(checkpoint, *, status_path=None, pilot=False):
    checkpoint = Path(checkpoint)
    status_path = Path(status_path) if status_path else checkpoint.with_name('status.json')
    if not checkpoint.is_file() or not status_path.is_file():
        raise FileNotFoundError(checkpoint)
    s = json.loads(status_path.read_text())
    ok = s.get('status') == 'complete_visual_initialization_only' and int(s.get('epochs_completed', -1)) == 16
    if not ok and (not pilot):
        raise ValueError('requires completed 16-epoch visual initialization')
    return {'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': sha256_file(checkpoint), 'parent_status_sha256': sha256_file(status_path), 'parent_status': s, 'pilot': bool(pilot)}

def extract_visual_arrays(p):
    with np.load(p, allow_pickle=False) as d:
        v = np.array(d['video_uint8'])
        r = np.array(d['region_available'])
        f = np.array(d['frame_available'])
        t = np.array(d['source_time_ms'])
        i = int(d['global_interval_index'])
        ident = str(d['interval_id'])
    if v.shape != (2, 16, 3, 3, 160, 160) or v.dtype != np.uint8 or r.shape != (2, 16, 3) or (r.dtype != bool) or (f.shape != (2, 16)) or (t.shape != (2, 16)) or np.any(r & ~f[..., None]):
        raise ValueError('invalid visual input')
    return (v, r, f, t, i, ident)

def payload_hash(v, i, ids, m, t):
    h = hashlib.sha256()
    for x in (np.asarray(i, np.int64), np.asarray(ids, dtype='<U256'), np.asarray(m, bool), np.asarray(t, np.float64), np.asarray(v, np.float16)):
        h.update(x.tobytes())
    return h.hexdigest()

class BackboneCacheWriter:

    def __init__(self, output, rows, *, checkpoint_contract, input_manifest_hash, cache_contract_hash='', cache_audit_hash='', source_hashes=None, pilot=False, chunk_size=32, device='cpu', inference_precision='float32', batch_size=4):
        if chunk_size < 1:
            raise ValueError('chunk size')
        self.output = Path(output)
        self.rows = list(rows)
        self.pilot = pilot
        self.c = {'schema': 'MMACT_BACKBONE_CACHE_V1', 'shape': [len(rows), *TAIL], 'dtype': 'float16', 'checkpoint': checkpoint_contract, 'input_manifest_sha256': input_manifest_hash, 'cache_contract_sha256': cache_contract_hash, 'cache_audit_sha256': cache_audit_hash, 'native_mapping_sha256': hashlib.sha256(np.asarray([x['global_interval_index'] for x in rows], np.int64).tobytes()).hexdigest(), 'source_hashes': source_hashes or {}, 'pilot': bool(pilot), 'chunk_size': chunk_size, 'inference': {'device': str(device), 'precision': str(inference_precision), 'batch_size': int(batch_size)}, 'lineage': 'frozen supervised Student backbone; not a source-OOF teacher feature'}

    def run(self, extractor, paths, *, stop_file=None):
        o = self.output
        o.mkdir(parents=True, exist_ok=True)
        guard(o)
        cp = o / 'RUN_CONTRACT.json'
        dp = o / 'backbone_sequence.npy'
        sp = o / 'status.json'
        stop = Path(stop_file) if stop_file else o / 'STOP'
        if cp.exists() and json.loads(cp.read_text()) != self.c:
            raise ValueError('resume contract mismatch')
        if not cp.exists():
            atomic_json(cp, self.c)
        expected_shape = tuple(self.c['shape'])
        if not dp.exists():
            guard(o, int(np.prod(expected_shape)) * np.dtype(np.float16).itemsize)
        mm = np.lib.format.open_memmap(dp, mode='r+' if dp.exists() else 'w+', dtype=np.float16, shape=expected_shape)
        if mm.shape != expected_shape or mm.dtype != np.dtype(np.float16):
            raise ValueError('existing mmap header shape/dtype mismatch')
        idx = np.asarray([x['global_interval_index'] for x in self.rows], np.int64)
        width = max(map(len, [str(x['interval_id']) for x in self.rows]))
        ids = np.asarray([str(x['interval_id']) for x in self.rows], dtype=f'<U{width}')
        masks = np.zeros((len(idx), 2, 16, 3), bool)
        times = np.zeros((len(idx), 2, 16), np.float64)
        for a in range(0, len(idx), self.c['chunk_size']):
            b = min(len(idx), a + self.c['chunk_size'])
            marker = o / f'chunk_{a:06d}_{b:06d}.complete.json'
            loaded = []
            for j in range(a, b):
                v, r, _, t, k, ident = extract_visual_arrays(paths[j])
                masks[j] = r
                times[j] = t
                if k != idx[j] or ident != ids[j]:
                    raise ValueError('native identity mismatch')
                loaded.append((v, r, t))
            if marker.exists():
                s = json.loads(marker.read_text())
                if s.get('contract_sha256') != sha256_file(cp) or s.get('payload_sha256') != payload_hash(mm[a:b], idx[a:b], ids[a:b], masks[a:b], times[a:b]):
                    raise ValueError('committed chunk corruption')
            else:
                if stop.exists():
                    atomic_json(sp, {'status': 'STOPPED_AT_CHUNK_BOUNDARY', 'completed_rows': a, 'pilot': self.pilot})
                    return json.loads(sp.read_text())
                value = np.asarray(extractor(np.stack([x[0] for x in loaded]), np.stack([x[1] for x in loaded]), np.stack([x[2] for x in loaded])), np.float32)
                if value.shape != (b - a, *TAIL) or not np.isfinite(value).all():
                    raise ValueError('extractor output invalid')
                value = value.astype(np.float16)
                if not np.isfinite(value).all():
                    raise ValueError('nonfinite after float16 cast')
                guard(o, value.nbytes)
                mm[a:b] = value
                mm.flush()
                atomic_json(marker, {'status': 'complete', 'start': a, 'stop': b, 'global_indices': idx[a:b].tolist(), 'contract_sha256': sha256_file(cp), 'payload_sha256': payload_hash(value, idx[a:b], ids[a:b], masks[a:b], times[a:b])})
            atomic_json(sp, {'status': 'RUNNING', 'completed_rows': b, 'rows': len(idx), 'pilot': self.pilot})
        tmp = o / 'identity_metadata.npz.tmp'
        with tmp.open('wb') as f:
            np.savez_compressed(f, global_interval_indices=idx, interval_ids=ids, region_available=masks, source_time_ms=times)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, o / 'identity_metadata.npz')
        full = not self.pilot and len(idx) == FULL and np.array_equal(idx, np.arange(FULL))
        s = {'status': 'COMPLETE' if full else 'PILOT_PARTIAL_COMPLETE', 'complete': full, 'rows': len(idx), 'pilot': self.pilot, 'artifact_sha256': {x: sha256_file(o / x) for x in ('backbone_sequence.npy', 'identity_metadata.npz', 'RUN_CONTRACT.json')}}
        atomic_json(sp, s)
        return s
