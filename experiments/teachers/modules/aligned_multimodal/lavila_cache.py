from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
'Identity-safe access to the original LaViLa frame-token source cache.'
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from aligned_multimodal.stable_routing_protocol import ProtocolError
CHECKPOINT_SHA = 'bf32bdc77ea030592ee9b2c6e0fc633d4a7d67498a9e9d7348d13cac9f10c961'
EXPECTED_ROWS = 2914

def _ids(values):
    raw = np.asarray(values)
    if raw.ndim != 1 or any((not isinstance(v, (str, np.str_)) or not v.strip() for v in raw)):
        raise ProtocolError('missing or malformed LaViLa IDs')
    x = raw.astype(str)
    if len(set(x)) != len(x):
        raise ProtocolError('duplicate or malformed LaViLa IDs')
    return x

def read_ids(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as h:
        return _ids([r['sample_id'] for r in csv.DictReader(h)])

def align_indices(pixel_ids, requested_ids):
    pixels = _ids(pixel_ids)
    requested = _ids(requested_ids)
    lookup = {s: i for i, s in enumerate(pixels)}
    if any((s not in lookup for s in requested)):
        raise ProtocolError('requested ID missing from LaViLa pixel table')
    return np.asarray([lookup[s] for s in requested], dtype=np.int64)

def validate_cache(tokens, metadata, pixel_ids, master_ids):
    pixels = _ids(pixel_ids)
    master = _ids(master_ids)
    n = len(pixels)
    if not n or set(pixels) != set(master):
        raise ProtocolError('LaViLa pixel/master ID sets differ')
    if tokens.shape != (n, 48, 768) or tokens.dtype != np.float16:
        raise ProtocolError('LaViLa token schema differs')
    required = {'stage': 'LaViLa_frame_token_cache', 'status': 'complete', 'rows': n, 'shape': [n, 48, 768], 'frames_per_view': 16, 'view_order': ['scene', 'person', 'workspace'], 'spatial_pool': 'mean over 14x14 patch tokens', 'checkpoint_frames': 4, 'checkpoint_epoch': 5, 'checkpoint_sha256': CHECKPOINT_SHA, 'labels_used': False, 'test_rows_loaded': 0, 'submission_generated': False}
    if any((metadata.get(k) != v for k, v in required.items())):
        raise ProtocolError('LaViLa extraction metadata differs')
    for start in range(0, n, 128):
        if not np.isfinite(tokens[start:start + 128]).all():
            raise ProtocolError('nonfinite LaViLa tokens')

def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as h:
        for block in iter(lambda: h.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()

class LaViLaCache:

    def __init__(self, cache_dir, pixel_rows, master_manifest):
        cache_dir = Path(cache_dir)
        self.pixel_ids = read_ids(pixel_rows)
        self.master_ids = read_ids(master_manifest)
        if len(self.master_ids) != EXPECTED_ROWS:
            raise ProtocolError('expected canonical2914Train IDs')
        self.tokens = np.load(cache_dir / 'frame_tokens.npy', mmap_mode='r', allow_pickle=False)
        self.metadata = json.loads((cache_dir / 'summary.json').read_text(encoding='utf-8'))
        validate_cache(self.tokens, self.metadata, self.pixel_ids, self.master_ids)
        self.master_to_pixel = align_indices(self.pixel_ids, self.master_ids)
        root = WORK_ROOT
        paths = [cache_dir / 'frame_tokens.npy', cache_dir / 'summary.json', Path(pixel_rows), Path(master_manifest), root / 'lavila_frame_token_cache.py', root / 'lavila_teacher.py', Path(__file__)]
        self.provenance = {'type': 'frozen_external', 'model': 'LaViLa_TimeSformer_B', 'checkpoint_sha256': CHECKPOINT_SHA, 'input_sha256': {str(p.resolve()): _hash(p) for p in paths}, 'row_order': 'pixel rows.csv; explicit ID join to requested order', 'labels_consumed': False, 'positions_different_from_master': int(np.sum(self.master_to_pixel != np.arange(len(self.master_ids)))), 'attestation': 'current extraction source and pixel row table', 'limitation': 'no embedded token IDs; lavila_cache summary lacks historical row-table/argument hash; not a re-extraction proof'}

    def select(self, requested_ids):
        indices = align_indices(self.pixel_ids, requested_ids)
        return np.asarray(self.tokens[indices], dtype=np.float32)
