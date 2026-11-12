from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
ROOT = WORK_ROOT
from session_kd.mmact_teacher_merge import *

def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')
    os.replace(tmp, path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, default=ROOT / 'runs/mmact_teacher_oof')
    p.add_argument('--extra', type=Path, default=ROOT / 'runs/mmact_extra_teachers')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--preflight-only', action='store_true')
    a = p.parse_args()
    base_manifest = json.loads((a.base / 'manifest.json').read_text())
    status_path = a.extra / 'status.json'
    if not status_path.is_file() or json.loads(status_path.read_text()).get('status') != 'complete':
        with np.load(a.base / 'source_oof_bank.npz', allow_pickle=False) as z:
            partial_count = validate_partial_extra_receipts(a.extra, z['source_indices'], z['source_folds'], z['validation_indices'], z['target_indices'])
        raise RuntimeError(f'extra producer is incomplete; validated {partial_count} completed physical receipts; refusing V2 merge')
    extra_manifest_path = a.extra / 'manifest.json'
    if not extra_manifest_path.is_file():
        raise RuntimeError('extra manifest absent; refusing V2 merge')
    extra_manifest = json.loads(extra_manifest_path.read_text())
    if base_manifest.get('status') != 'complete' or extra_manifest.get('status') != 'complete':
        raise RuntimeError('producer manifest incomplete')
    base_declared = base_manifest.get('artifact_sha256', {})
    for name in ('source_oof_bank.npz', 'geometry_features.npz'):
        if base_declared.get(name) != sha(a.base / name):
            raise ValueError(f'base manifest hash mismatch: {name}')
    if extra_manifest.get('extra_bank_sha256') != sha(a.extra / 'extra_bank.npz'):
        raise ValueError('extra manifest hash mismatch: extra_bank.npz')
    with np.load(a.base / 'source_oof_bank.npz', allow_pickle=False) as z:
        base = {k: z[k].copy() for k in z.files}
    with np.load(a.extra / 'extra_bank.npz', allow_pickle=False) as z:
        if 'target_labels' in z.files:
            raise ValueError('extra bank contains target labels')
        extra = {k: z[k].copy() for k in z.files}
    with np.load(a.base / 'geometry_features.npz', allow_pickle=False) as z:
        geometry = {k: z[k].copy() for k in z.files}
    if any(('target' in key.lower() and 'label' in key.lower() for key in base)):
        raise ValueError('base bank target-label sentinel')
    if any(('target' in key.lower() and 'label' in key.lower() for key in extra)):
        raise ValueError('extra bank target-label sentinel')
    base_names = base['expert_names'].astype(str).tolist()
    extra_names = extra['names'].astype(str).tolist()
    if extra_names != list(EXTRA_NAMES):
        raise ValueError('extra expert order changed')
    if len(base_names) != 12 or len(set(base_names + extra_names)) != 15:
        raise ValueError('expert identities invalid')
    value_binding = {'base': verify_base_oof_values(a.base, base_manifest, base), 'extra': verify_extra_bank_values(a.extra, extra_manifest, extra)}
    for split, n in (('source', 3209), ('validation', 1081), ('target', 1116)):
        validate_identity(base[f'{split}_indices'], base[f'{split}_interval_ids'], f'base {split}')
        validate_identity(extra[f'{split}_indices'], extra[f'{split}_interval_ids'], f'extra {split}')
        if len(base[f'{split}_indices']) != n:
            raise ValueError('canonical split size changed')
        if not np.array_equal(base[f'{split}_indices'], geometry[f'{split}_indices']) or not np.array_equal(base[f'{split}_interval_ids'], geometry[f'{split}_interval_ids']):
            raise ValueError('geometry identity mismatch')
        validate_probability(base[f'{split}_probabilities'], base[f'{split}_availability'], f'base {split}')
        validate_probability(extra[f'{split}_probabilities'], extra[f'{split}_availability'], f'extra {split}')
        if base[f'{split}_probabilities'].shape != (n, 12, 37) or base[f'{split}_availability'].shape != (n, 12):
            raise ValueError('base exact bank dimensions invalid')
        if extra[f'{split}_probabilities'].shape != (n, 3, 37) or extra[f'{split}_availability'].shape != (n, 3):
            raise ValueError('extra exact bank dimensions invalid')
    assert_same_identity(base, extra, 'base/extra')
    logical = verify_base_receipts(a.base, base_manifest, base_names) + verify_extra_receipts(a.extra, extra_manifest, base['source_indices'], base['source_folds'], base['source_interval_ids'], base['validation_indices'], base['validation_interval_ids'], base['target_indices'], base['target_interval_ids'])
    oof = [x for x in logical if x['class'] == 'oof']
    final = [x for x in logical if x['class'] == 'final_source']
    if len(oof) != 45 or len(final) != 3:
        raise AssertionError('logical receipt cardinality mismatch')
    if a.preflight_only:
        print(json.dumps({'status': 'preflight_complete', 'would_merge_shape': [3209, 15, 37], 'logical_oof_receipts': 45, 'logical_final_source_receipts': 3}, indent=2))
        return
    if a.output.exists():
        raise FileExistsError('output must be new')
    a.output.mkdir(parents=True)
    arrays = {}
    for split in ('source', 'validation', 'target'):
        arrays[f'{split}_probabilities'] = np.concatenate((base[f'{split}_probabilities'], extra[f'{split}_probabilities']), axis=1)
        arrays[f'{split}_availability'] = np.concatenate((base[f'{split}_availability'], extra[f'{split}_availability']), axis=1)
        arrays[f'{split}_indices'] = base[f'{split}_indices']
        arrays[f'{split}_interval_ids'] = base[f'{split}_interval_ids']
    for key in ('source_labels', 'source_folds', 'validation_labels'):
        arrays[key] = base[key]
    arrays['expert_names'] = np.asarray(base_names + extra_names)
    for split, n in (('source', 3209), ('validation', 1081), ('target', 1116)):
        if arrays[f'{split}_probabilities'].shape != (n, 15, 37) or arrays[f'{split}_availability'].shape != (n, 15):
            raise AssertionError('merged exact 15-expert dimensions invalid')
    bank = a.output / 'teacher_bank.npz'
    np.savez_compressed(bank, **arrays)
    geo = a.output / 'geometry_features.npz'
    np.savez_compressed(geo, **geometry)
    manifest = {'status': 'complete', 'schema': 'MMACT_TEACHER_BANK_V2', 'expert_names': base_names + extra_names, 'shapes': {s: list(arrays[f'{s}_probabilities'].shape) for s in ('source', 'validation', 'target')}, 'logical_oof_receipt_count': 45, 'logical_final_source_receipt_count': 3, 'logical_receipts': logical, 'base_eval_ancestry': 'unchanged: F1/F0 base12 predictions remain the retained full-source-head outputs from MMACT_TEACHER_OOF_V1', 'producer_roots': {'base_original': str(a.base.resolve()), 'extra_original': str(a.extra.resolve())}, 'producer_manifest_sha256': {'base': sha(a.base / 'manifest.json'), 'extra': sha(extra_manifest_path)}, 'receipt_to_bank_value_verification': value_binding, 'merge_provenance': {'module': str((MODULE_ROOT / 'session_kd/mmact_teacher_merge.py').resolve()), 'cli': str(Path(__file__).resolve()), 'module_sha256': sha(MODULE_ROOT / 'session_kd/mmact_teacher_merge.py'), 'cli_sha256': sha(Path(__file__).resolve())}, 'artifact_sha256': {'teacher_bank.npz': sha(bank), 'geometry_features.npz': sha(geo)}, 'split_identity_sha256': {s: {'indices': hashlib.sha256(arrays[f'{s}_indices'].tobytes()).hexdigest(), 'ids': ordered_hash(arrays[f'{s}_interval_ids'])} for s in ('source', 'validation', 'target')}, 'labels_source': 'base bank only', 'target_labels_present': False, 'checkpoint_trees_copied': False}
    atomic_json(a.output / 'manifest.json', manifest)
    print(json.dumps({'status': 'complete', 'shapes': manifest['shapes'], 'output': str(a.output)}, indent=2))
