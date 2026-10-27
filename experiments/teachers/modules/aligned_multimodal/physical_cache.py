from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
'Identity-safe access to the four frozen physical_tokens physical feature caches.'
import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
from aligned_multimodal.stable_routing_protocol import ProtocolError
from aligned_multimodal.foundation_provider import array_hash
EXPECTED_ROWS = 2914
TOKEN_SHAPES = ((2, 3, 768), (2, 3, 768), (3, 768), (3, 768))
DEFAULT_PATHS = (WORK_ROOT / 'runs/videomaev2_distilled_teacher/complete_features.npz', WORK_ROOT / 'runs/internvideo2_l_k400_teacher/complete_features.npz', WORK_ROOT / 'runs/videomaev2_depth_fold0/complete_features.npz', WORK_ROOT / 'runs/videomaev2_thermal_fold0/complete_features.npz')
FAMILY_NAMES = ('ir_vmae', 'ir_iv2', 'depth_vmae', 'thermal_vmae')

def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def _ids(values, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or len(raw) == 0 or any((not isinstance(v, (str, np.str_)) for v in raw)):
        raise ProtocolError(f'{name} must be a nonempty vector')
    out = raw.astype(str)
    if any((not value.strip() for value in out)) or len(set(out.tolist())) != len(out):
        raise ProtocolError(f'{name} must contain unique nonblank IDs')
    return out

def _values(values, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or len(raw) == 0 or any((not isinstance(v, (str, np.str_)) for v in raw)):
        raise ProtocolError(f'{name} must be a nonempty vector')
    out = raw.astype(str)
    if any((not value.strip() for value in out)):
        raise ProtocolError(f'{name} must contain nonblank values')
    return out

def _manifest(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows or 'sample_id' not in rows[0] or 'user_id' not in rows[0]:
        raise ProtocolError('canonical manifest lacks sample_id/user_id')
    ids = _ids([row.get('sample_id') for row in rows], 'canonical sample IDs')
    users = _values([row.get('user_id') for row in rows], 'canonical users')
    if len(ids) != EXPECTED_ROWS:
        raise ProtocolError(f'expected canonical {EXPECTED_ROWS} IDs')
    return (ids, users)

def _metadata(path: Path, family: str) -> tuple[dict, bool]:
    summary_path = path.with_name('cache_summary.json')
    if not summary_path.is_file():
        raise ProtocolError(f'missing pinned cache metadata: {summary_path}')
    try:
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
    except Exception as exc:
        raise ProtocolError(f'invalid cache metadata: {summary_path}') from exc
    required = {'ir_vmae': {'model_repo': 'OpenGVLab/VideoMAE2', 'model_file': 'distill/vit_b_k710_dl_from_giant.pth', 'clips': 'early/late x scene/person/workspace'}, 'ir_iv2': {'model_repo': 'OpenGVLab/InternVideo2_distillation_models', 'clips': 'early/late x scene/person/workspace', 'frames_per_clip': 8}, 'depth': {'modality': 'depth', 'clips': 'scene/person/workspace', 'samples': EXPECTED_ROWS}, 'thermal': {'modality': 'thermal', 'clips': 'scene/person/workspace', 'samples': EXPECTED_ROWS, 'available_samples': 2776}}[family]
    if any((summary.get(key) != value for key, value in required.items())):
        raise ProtocolError(f'{family} cache metadata is not pinned')
    if family == 'ir_iv2':
        identity = str(summary.get('model_file', '')).replace('\\', '/')
        suffix = '/snapshots/449f7ea1d7d3b70b6b5630e70d238b44d3b7aaac/stage1/L14/L14_ft_k710_ft_k400_f8/pytorch_model.bin'
    else:
        identity = str(summary.get('checkpoint', '')).replace('\\', '/')
        suffix = '/snapshots/706cc172d65ebd4dedbee3f9c0183a93df9fa125/distill/vit_b_k710_dl_from_giant.pth'
    if not identity.endswith(suffix):
        raise ProtocolError(f'{family} checkpoint identity differs')
    weak = family in {'depth', 'thermal'}
    return (summary, weak)

class PhysicalTokenCache:
    """Load and explicitly ID-align the original four physical_tokens feature caches."""

    def __init__(self, paths: Sequence[str | Path]=DEFAULT_PATHS, manifest: str | Path | None=None):
        if len(paths) != 4:
            raise ProtocolError('physical_tokens requires exactly four feature caches')
        self.paths = tuple((Path(p) for p in paths))
        self.manifest = Path(manifest) if manifest is not None else Path(__file__).resolve().parent / 'data/manifest.csv'
        self.master_ids, self.master_users = _manifest(self.manifest)
        self.ids = self.master_ids
        self.users = self.master_users
        arrays: list[np.ndarray] = []
        provenance: dict[str, dict] = {}
        weak_metadata = False
        for family, path, shape in zip(FAMILY_NAMES, self.paths, TOKEN_SHAPES, strict=True):
            if not path.is_file():
                raise ProtocolError(f'missing frozen feature cache: {path}')
            summary, weak = _metadata(path, 'depth' if family == 'depth_vmae' else 'thermal' if family == 'thermal_vmae' else family)
            weak_metadata |= weak
            with np.load(path, allow_pickle=False) as archive:
                if 'sample_ids' not in archive or 'features' not in archive:
                    raise ProtocolError(f'cache lacks sample_ids/features: {path}')
                ids = _ids(archive['sample_ids'], f'{family} sample IDs')
                if not np.array_equal(ids, self.master_ids):
                    raise ProtocolError(f'{family} IDs do not exactly match canonical order')
                if 'users' not in archive:
                    raise ProtocolError(f'{family} users missing')
                if 'users' in archive:
                    users = _values(archive['users'], f'{family} users')
                    if not np.array_equal(users, self.master_users):
                        raise ProtocolError(f'{family} users do not match canonical manifest')
                raw = np.asarray(archive['features'])
                if raw.dtype != np.dtype(np.float16) or raw.dtype.kind == 'c':
                    raise ProtocolError(f'{family} features must be float16 real values')
                if raw.shape != (EXPECTED_ROWS, *shape) or not np.isfinite(raw).all():
                    raise ProtocolError(f'{family} feature schema/nonfinite values differ')
                if family == 'thermal_vmae':
                    mask = np.asarray(archive['modality_available'])
                    if mask.shape != (EXPECTED_ROWS,) or mask.dtype.kind not in 'biu' or (not np.isin(mask, [0, 1]).all()) or (int(mask.sum()) != 2776):
                        raise ProtocolError('thermal availability identity differs')
                    if np.any(raw[~mask.astype(bool)] != 0):
                        raise ProtocolError('missing thermal features must be zero')
                arrays.append(raw.astype(np.float16, copy=False).reshape(EXPECTED_ROWS, -1))
            provenance[family] = {'path': str(path.resolve()), 'sha256': _sha(path), 'summary_path': str(path.with_name('cache_summary.json').resolve()), 'summary_sha256': _sha(path.with_name('cache_summary.json')), 'summary': summary, 'sample_id_sha256': array_hash(ids), 'labels_loaded': False, 'users_bound': True}
        self.tokens = np.ascontiguousarray(np.concatenate(arrays, axis=1), dtype=np.float16)
        if self.tokens.shape != (EXPECTED_ROWS, 18 * 768) or not np.isfinite(self.tokens).all():
            raise ProtocolError('physical concatenated feature schema differs')
        self._lookup = {sample_id: i for i, sample_id in enumerate(self.master_ids)}
        self.provenance = {'type': 'frozen_external', 'source': 'original_physical_tokens', 'families': list(FAMILY_NAMES), 'token_shape': [18, 768], 'paths': {name: value['path'] for name, value in provenance.items()}, 'input_sha256': {**{f'{name}_features': value['sha256'] for name, value in provenance.items()}, **{f'{name}_metadata': value['summary_sha256'] for name, value in provenance.items()}, 'canonical_manifest': _sha(self.manifest)}, 'cache_provenance': provenance, 'labels_loaded': False, 'features_normalized': False, 'availability_masks_used': False, 'limitation': 'frozen external cache; metadata lacks complete extraction attestation' if weak_metadata else 'frozen external cache; no task-label encoder fit'}

    def select(self, requested_ids) -> np.ndarray:
        ids = _ids(requested_ids, 'requested IDs')
        try:
            indices = [self._lookup[value] for value in ids]
        except KeyError as exc:
            raise ProtocolError('requested ID missing from physical_tokens cache') from exc
        return np.asarray(self.tokens[indices], dtype=np.float32).reshape(len(ids), 18, 768)
Cache = PhysicalTokenCache
