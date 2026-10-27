from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
'Original VJEPA raw-token cache with explicit master-order attestation.'
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from aligned_multimodal.stable_routing_protocol import ProtocolError
VIEW_NAMES = ('global_full_scene', 'global_full_person', 'global_full_workspace', 'global_early_scene', 'global_early_person', 'global_early_workspace', 'global_middle_scene', 'global_middle_person', 'global_middle_workspace', 'global_late_scene', 'global_late_person', 'global_late_workspace', 'hand_full_left', 'hand_full_right', 'hand_full_interaction', 'hand_early_left', 'hand_early_right', 'hand_early_interaction', 'hand_late_left', 'hand_late_right', 'hand_late_interaction', 'hand_motion_peak_left', 'hand_motion_peak_right', 'hand_motion_peak_interaction')

def read_ids(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as h:
        ids = np.asarray([row['sample_id'] for row in csv.DictReader(h)]).astype(str)
    if len(set(ids)) != len(ids):
        raise ProtocolError('duplicate token manifest IDs')
    return ids

def validate_raw(x, done, meta, master_ids, extraction_ids):
    n = len(master_ids)
    if len(set(master_ids)) != n or len(set(extraction_ids)) != len(extraction_ids) or set(master_ids) != set(extraction_ids):
        raise ProtocolError('raw extraction/master identity mismatch')
    if x.shape != (n, 24, 1024) or x.dtype != np.float16 or done.shape != (n,) or (done.dtype != np.bool_) or (not done.all()):
        raise ProtocolError('incomplete or malformed raw token cache')
    if meta.get('model_repo') != 'facebook/vjepa2-vitl-fpc16-256-ssv2' or meta.get('label_free_extraction') is not True or meta.get('complete') is not True or (meta.get('completed_samples') != n) or (meta.get('total_samples') != n) or (meta.get('view_count') != 24) or (meta.get('view_names') != list(VIEW_NAMES)):
        raise ProtocolError('raw token provenance/view order changed')
    for start in range(0, n, 128):
        if not np.isfinite(x[start:start + 128]).all():
            raise ProtocolError('nonfinite raw tokens')

class TokenCache:

    def __init__(self, cache_dir, master_manifest, extraction_manifest):
        cache_dir = Path(cache_dir)
        self.ids = read_ids(master_manifest)
        if len(self.ids) != 2914:
            raise ProtocolError('expected original2914master rows')
        self.tokens = np.load(cache_dir / 'features.npy', mmap_mode='r', allow_pickle=False)
        done = np.load(cache_dir / 'done.npy', allow_pickle=False)
        self.metadata = json.loads((cache_dir / 'cache_summary.json').read_text(encoding='utf-8'))
        validate_raw(self.tokens, done, self.metadata, self.ids, read_ids(extraction_manifest))
        self.lookup = {s: i for i, s in enumerate(self.ids)}
        self.provenance = {'type': 'frozen_external', 'row_order': 'teacher master order via original read_aligned_rows', 'limitation': 'historical source/manifest attestation; raw.npy has no embedded IDs', 'labels_loaded': False, 'model_repo': self.metadata['model_repo'], 'snapshot': self.metadata.get('snapshot')}
        hashes = {}
        for path in (cache_dir / 'features.npy', cache_dir / 'done.npy', cache_dir / 'cache_summary.json', Path(master_manifest), Path(extraction_manifest)):
            digest = hashlib.sha256()
            with path.open('rb') as h:
                for chunk in iter(lambda: h.read(1024 ** 2), b''):
                    digest.update(chunk)
            hashes[str(path.resolve())] = digest.hexdigest()
        self.provenance['input_sha256'] = hashes

    def select(self, ids):
        ids = np.asarray(ids).astype(str)
        if ids.ndim != 1 or len(set(ids)) != len(ids) or any((s not in self.lookup for s in ids)):
            raise ProtocolError('token ID selection invalid')
        return np.asarray(self.tokens[[self.lookup[s] for s in ids]], dtype=np.float32)
