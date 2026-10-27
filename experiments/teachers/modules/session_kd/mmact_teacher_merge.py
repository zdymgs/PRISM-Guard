from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
EXTRA_NAMES = ('mc3_motion', 'native_skeleton', 'sequential_imu')
FOLDS = ('F2', 'F3', 'F4')

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()

def ordered_hash(values: Sequence[object]) -> str:
    return hashlib.sha256(('\n'.join(map(str, values)) + '\n').encode()).hexdigest()

def array_hash(value) -> str:
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(np.asarray(a.shape, np.int64).tobytes())
    h.update(a.tobytes())
    return h.hexdigest()

def safe_rebind(root: Path, value: str) -> Path:
    root = root.resolve()
    raw = Path(value)
    candidate = (root / raw).resolve() if not raw.is_absolute() else raw.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as e:
        raise ValueError(f'artifact escapes producer root: {value}') from e
    return candidate

def validate_probability(prob, avail, name):
    p = np.asarray(prob)
    a = np.asarray(avail, bool)
    if p.ndim != 3 or p.shape[:2] != a.shape or p.shape[2] != 37:
        raise ValueError(f'{name} shape invalid')
    if not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError(f'{name} nonfinite/negative')
    sums = p.sum(2)
    if not np.all(p[~a] == 0) or not np.all(sums[~a] == 0):
        raise ValueError(f'{name} unavailable probability nonzero')
    if not np.allclose(sums[a], 1, atol=1e-05):
        raise ValueError(f'{name} available probability not normalized')

def validate_identity(indices, ids, name):
    i = np.asarray(indices, np.int64)
    s = np.asarray(ids, str)
    if i.ndim != 1 or s.shape != i.shape or len(np.unique(i)) != len(i) or (len(np.unique(s)) != len(s)):
        raise ValueError(f'{name} identity invalid')

def verify_base_receipts(root: Path, manifest: Mapping, experts: Sequence[str]):
    receipts = manifest.get('receipts', [])
    expected = {(e, f) for e in experts for f in FOLDS}
    seen = set()
    logical = []
    if len(receipts) != 36:
        raise ValueError('base must contain 36 receipts')
    for r in receipts:
        expert = r.get('expert', r.get('expert_name'))
        key = (expert, r.get('prediction_fold'))
        if key in seen or key not in expected or set(r.get('fit_folds', [])) != set(FOLDS) - {key[1]}:
            raise ValueError('base receipt fold/expert invalid')
        seen.add(key)
        arts = r.get('artifacts', {})
        hashes = r.get('artifact_sha256', {})
        if not arts or set(arts) != set(hashes):
            raise ValueError('base receipt artifact contract invalid')
        rebound = []
        for rel, digest in hashes.items():
            path = safe_rebind(root, rel)
            if not path.is_file() or sha(path) != digest:
                raise ValueError('base artifact missing/hash mismatch')
            rebound.append({'original': rel, 'rebound': str(path), 'sha256': digest, 'role': arts[rel]})
        lineage_path = next((Path(x['rebound']) for x in rebound if x['role'] == 'identity_lineage'), None)
        if lineage_path is None:
            raise ValueError('base lineage artifact missing')
        with np.load(lineage_path, allow_pickle=False) as z:
            if ordered_hash(z['fit_interval_ids']) != r['fit_ids_sha256'] or ordered_hash(z['prediction_interval_ids']) != r['prediction_ids_sha256']:
                raise ValueError('base lineage IDs disagree with receipt')
            if 'normalizer_training_ids' in z.files and (not np.array_equal(z['normalizer_training_ids'].astype(str), z['fit_interval_ids'].astype(str))):
                raise ValueError('base normalizer IDs differ from fit IDs')
        base_manifest_path = root / 'manifest.json'
        logical.append({'class': 'oof', 'logical_expert': key[0], 'prediction_fold': key[1], 'status': 'complete', 'fit_folds': r['fit_folds'], 'fit_count': r['fit_count'], 'prediction_count': r['prediction_count'], 'config_sha256': r['config_sha256'], 'source_manifest_sha256': sha(base_manifest_path), 'producer_artifact_root': str(root.resolve()), 'original_physical_receipt': f'manifest.json#embedded-receipt:{key[0]}:{key[1]}', 'rebound_physical_receipt': str(base_manifest_path.resolve()), 'physical_receipt_sha256': sha(base_manifest_path), 'artifacts': rebound, 'fit_ids_sha256': r['fit_ids_sha256'], 'prediction_ids_sha256': r['prediction_ids_sha256']})
    if seen != expected:
        raise ValueError('base receipt coverage incomplete')
    return logical

def verify_extra_receipts(root: Path, manifest: Mapping, source_indices=None, source_folds=None, source_ids=None, validation_indices=None, validation_ids=None, target_indices=None, target_ids=None):
    physical = manifest.get('receipts', [])
    keys = [(r.get('fit_name'), r.get('family')) for r in physical]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate extra physical receipt')
    lookup = dict(zip(keys, physical))
    logical = []
    for fit in (*FOLDS, 'final_source'):
        mc3 = lookup.get((fit, 'mc3_motion'))
        native = lookup.get((fit, 'native_skeleton+sequential_imu'))
        if mc3 is None or native is None:
            raise ValueError(f'extra physical receipt missing for {fit}')
        for receipt, names in ((mc3, ('mc3_motion',)), (native, ('native_skeleton', 'sequential_imu'))):
            if receipt.get('status') != 'complete' or receipt.get('f0_labels_read') is not False:
                raise ValueError('extra receipt incomplete/F0 unsafe')
            arts = receipt.get('artifacts', {})
            if not arts:
                raise ValueError('extra receipt artifacts empty')
            rebound = []
            for rel, digest in arts.items():
                path = safe_rebind(root, rel)
                if not path.is_file() or sha(path) != digest:
                    raise ValueError('extra artifact missing/hash mismatch')
                rebound.append({'original': rel, 'rebound': str(path), 'sha256': digest})
            receipt_path = root / f'{receipt['contract']['family']}__{fit}.receipt.json'
            if not receipt_path.is_file():
                raise ValueError('physical receipt file missing')
            if json.loads(receipt_path.read_text()) != receipt:
                raise ValueError('embedded extra receipt differs from disk receipt')
            checkpoint = next((x for x in rebound if x['original'].endswith('.pt')), None)
            lineage = next((Path(x['rebound']) for x in rebound if x['original'].endswith('.lineage.npz')), None)
            prediction = next((Path(x['rebound']) for x in rebound if x['original'].endswith('.pred.npz')), None)
            if lineage is None or prediction is None:
                raise ValueError('extra lineage/prediction artifact missing')
            with np.load(lineage, allow_pickle=False) as z:
                fit_actual = z['fit_indices'].astype(np.int64)
                pred_actual = z['prediction_indices'].astype(np.int64)
                if ordered_hash(map(str, fit_actual)) != receipt['fit_indices_sha256']:
                    raise ValueError('extra fit lineage indices disagree with receipt')
                pred_hash = ordered_hash(map(str, pred_actual))
                if receipt.get('prediction_indices_sha256') is not None and pred_hash != receipt['prediction_indices_sha256']:
                    raise ValueError('extra prediction lineage indices disagree with receipt')
                if source_indices is not None:
                    source_indices = np.asarray(source_indices, np.int64)
                    source_folds = np.asarray(source_folds, str)
                    expected_fit = source_indices if fit == 'final_source' else source_indices[source_folds != fit]
                    expected_pred = np.r_[validation_indices, target_indices] if fit == 'final_source' else source_indices[source_folds == fit]
                    if not np.array_equal(fit_actual, expected_fit) or not np.array_equal(pred_actual, expected_pred):
                        raise ValueError('extra lineage does not match canonical fold split')
                    if not np.array_equal(z['fit_interval_ids'].astype(str), np.asarray(source_ids)[np.isin(source_indices, fit_actual)] if fit == 'final_source' else np.asarray(source_ids)[source_folds != fit]):
                        raise ValueError('extra fit IDs mismatch canonical identities')
                if 'normalizer_indices' not in z.files:
                    raise ValueError('extra normalizer lineage missing')
                norm_indices = z['normalizer_indices'].astype(np.int64)
                if not np.array_equal(norm_indices, fit_actual):
                    raise ValueError('extra normalizer indices differ from fit indices')
            norm_path = next((Path(x['rebound']) for x in rebound if x['original'].endswith('.norm.npz')), None)
            if norm_path is None:
                raise ValueError('extra normalizer artifact missing')
            with np.load(norm_path, allow_pickle=False) as n:
                if 'training_ids' in n.files and len(n['training_ids']) != len(fit_actual):
                    raise ValueError('normalizer training IDs count differs from fit')
            with np.load(prediction, allow_pickle=False) as z:
                logical_output_hash = {'mc3_motion': array_hash(z['probabilities'])} if len(names) == 1 else {'native_skeleton': array_hash(z['skeleton_probabilities']), 'sequential_imu': array_hash(z['imu_probabilities'])}
            for name in names:
                logical_config_hash = hashlib.sha256(json.dumps({'physical_contract': receipt['contract'], 'logical_head': name}, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                fit_folds = list(FOLDS) if fit == 'final_source' else [value for value in FOLDS if value != fit]
                logical.append({'class': 'final_source' if fit == 'final_source' else 'oof', 'logical_expert': name, 'prediction_fold': fit, 'producer_artifact_root': str(root.resolve()), 'status': 'complete', 'fit_folds': fit_folds, 'fit_count': len(fit_actual), 'prediction_count': len(pred_actual), 'fit_ids_sha256': receipt['fit_indices_sha256'], 'prediction_ids_sha256': pred_hash, 'config_sha256': logical_config_hash, 'source_manifest_sha256': sha(root / 'manifest.json'), 'f0_labels_read': False, 'original_physical_receipt': receipt_path.name, 'rebound_physical_receipt': str(receipt_path.resolve()), 'physical_receipt_sha256': sha(receipt_path), 'shared_checkpoint_group': f'native__{fit}' if len(names) == 2 else None, 'checkpoint_sha256': checkpoint['sha256'] if checkpoint else None, 'artifacts': rebound, 'logical_output_sha256': logical_output_hash[name], 'logical_configuration_sha256': logical_config_hash, 'fit_indices_sha256': receipt['fit_indices_sha256'], 'prediction_indices_sha256': pred_hash})
    oof = [x for x in logical if x['class'] == 'oof']
    final = [x for x in logical if x['class'] == 'final_source']
    if {(x['logical_expert'], x['prediction_fold']) for x in oof} != {(e, f) for e in EXTRA_NAMES for f in FOLDS}:
        raise ValueError('extra logical OOF coverage invalid')
    if {x['logical_expert'] for x in final} != set(EXTRA_NAMES):
        raise ValueError('extra final-source coverage invalid')
    return logical

def assert_same_identity(left, right, prefix):
    for split in ('source', 'validation', 'target'):
        for suffix in ('indices', 'interval_ids'):
            key = f'{split}_{suffix}'
            if not np.array_equal(left[key], right[key]):
                raise ValueError(f'{prefix} identity mismatch: {key}')

def validate_partial_extra_receipts(root: Path, source_indices, source_folds, validation_indices, target_indices) -> int:
    paths = sorted(root.glob('*.receipt.json'))
    seen = set()
    for path in paths:
        receipt = json.loads(path.read_text())
        key = (receipt.get('fit_name'), receipt.get('family'))
        if key in seen:
            raise ValueError('duplicate partial extra receipt')
        seen.add(key)
        if receipt.get('status') != 'complete' or not receipt.get('artifacts'):
            raise ValueError('partial extra receipt incomplete')
        for rel, digest in receipt['artifacts'].items():
            artifact = safe_rebind(root, rel)
            if not artifact.is_file() or sha(artifact) != digest:
                raise ValueError('partial extra artifact mismatch')
        lineage = root / f'{receipt['contract']['family']}__{receipt['fit_name']}.lineage.npz'
        if not lineage.is_file():
            raise ValueError('partial lineage missing')
        with np.load(lineage, allow_pickle=False) as z:
            actual_fit = z['fit_indices']
            actual_pred = z['prediction_indices']
            fit = receipt['fit_name']
            expected_fit = source_indices if fit == 'final_source' else source_indices[source_folds != fit]
            expected_pred = np.r_[validation_indices, target_indices] if fit == 'final_source' else source_indices[source_folds == fit]
            if not np.array_equal(actual_fit, expected_fit) or not np.array_equal(actual_pred, expected_pred):
                raise ValueError('partial receipt lineage differs from canonical split')
            if not np.array_equal(z['normalizer_indices'], actual_fit):
                raise ValueError('partial normalizer lineage differs from fit')
    return len(paths)

def _assert_values(actual, expected, name, atol=1e-07, rtol=1e-06):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape or not np.allclose(actual, expected, atol=atol, rtol=rtol):
        difference = float(np.max(np.abs(actual - expected))) if actual.shape == expected.shape else float('inf')
        raise ValueError(f'{name} receipt-to-bank mismatch; max_abs={difference}')

def verify_base_oof_values(root: Path, manifest: Mapping, bank: Mapping, atol=1e-07, rtol=1e-06):
    names = bank['expert_names'].astype(str).tolist()
    indices = bank['source_indices'].astype(np.int64)
    position = {int(v): i for i, v in enumerate(indices)}
    rebuilt = np.zeros_like(bank['source_probabilities'])
    available = np.zeros_like(bank['source_availability'])
    assigned = np.zeros_like(available, dtype=np.int8)
    for receipt in manifest['receipts']:
        expert = receipt.get('expert', receipt.get('expert_name'))
        column = names.index(str(expert))
        prediction_rel = next((rel for rel, role in receipt['artifacts'].items() if role == 'prediction'), None)
        if prediction_rel is None:
            raise ValueError('base receipt prediction artifact missing')
        path = safe_rebind(root, prediction_rel)
        if sha(path) != receipt['artifact_sha256'][prediction_rel]:
            raise ValueError('base prediction artifact hash mismatch')
        with np.load(path, allow_pickle=False) as z:
            rows = z['global_indices'].astype(np.int64)
            prob = z['probabilities'].astype(np.float32)
            mask = z['availability'].astype(bool)
        positions = np.asarray([position[int(row)] for row in rows])
        rebuilt[positions, column] = prob
        available[positions, column] = mask
        assigned[positions, column] += 1
    if not np.all(assigned == 1):
        raise ValueError('base source OOF reconstruction is not exactly once')
    _assert_values(rebuilt, bank['source_probabilities'], 'base source OOF probabilities', atol, rtol)
    if not np.array_equal(available, bank['source_availability']):
        raise ValueError('base source OOF availability mismatch')
    return {'scope': 'source OOF only', 'rows': len(indices), 'experts': len(names), 'exact_once': True, 'comparison': {'rtol': rtol, 'atol': atol}, 'validation_target_limitation': 'Base F1/F0 retained full-source prediction files are outside the OOF producer artifact root and are not safely rebound by this merger; their bank values remain bound by the base producer manifest hash, not independently reconstructed here.'}

def verify_extra_bank_values(root: Path, manifest: Mapping, bank: Mapping, atol=1e-07, rtol=1e-06):
    names = bank['names'].astype(str).tolist()
    reports = {}
    receipts = {(r['fit_name'], r['family']): r for r in manifest['receipts']}
    for split in ('source', 'validation', 'target'):
        indices = bank[f'{split}_indices'].astype(np.int64)
        position = {int(v): i for i, v in enumerate(indices)}
        rebuilt = np.zeros_like(bank[f'{split}_probabilities'])
        available = np.zeros_like(bank[f'{split}_availability'])
        assigned = np.zeros_like(available, dtype=np.int8)
        fits = FOLDS if split == 'source' else ('final_source',)
        for fit in fits:
            for family in ('mc3_motion', 'native_skeleton+sequential_imu'):
                receipt = receipts[fit, family]
                pred_rel = next((rel for rel in receipt['artifacts'] if rel.endswith('.pred.npz')), None)
                if pred_rel is None:
                    raise ValueError('extra prediction artifact missing')
                path = safe_rebind(root, pred_rel)
                if sha(path) != receipt['artifacts'][pred_rel]:
                    raise ValueError('extra prediction artifact hash mismatch')
                with np.load(path, allow_pickle=False) as z:
                    rows = z['indices'].astype(np.int64)
                    keep = np.asarray([int(row) in position for row in rows])
                    rows = rows[keep]
                    positions = np.asarray([position[int(row)] for row in rows])
                    if family == 'mc3_motion':
                        column = names.index('mc3_motion')
                        rebuilt[positions, column] = z['probabilities'][keep]
                        available[positions, column] = z['availability'][keep]
                        assigned[positions, column] += 1
                    else:
                        for name, pkey, mkey in (('native_skeleton', 'skeleton_probabilities', 'skeleton_availability'), ('sequential_imu', 'imu_probabilities', 'imu_availability')):
                            column = names.index(name)
                            rebuilt[positions, column] = z[pkey][keep]
                            available[positions, column] = z[mkey][keep]
                            assigned[positions, column] += 1
        if not np.all(assigned == 1):
            raise ValueError(f'extra {split} reconstruction is not exactly once')
        _assert_values(rebuilt, bank[f'{split}_probabilities'], f'extra {split} probabilities', atol, rtol)
        if not np.array_equal(available, bank[f'{split}_availability']):
            raise ValueError(f'extra {split} availability mismatch')
        reports[split] = {'rows': len(indices), 'experts': 3, 'exact_once': True}
    return {'scope': 'source OOF plus F1/F0 final-source predictions', 'comparison': {'rtol': rtol, 'atol': atol}, 'splits': reports}
