from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
ROOT = WORK_ROOT
from session_kd.mmact_strong_teacher import *

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()

def load_video(cache, n):
    features = np.full((n, 4, 2, 1024), np.nan, np.float16)
    masks = np.zeros((n, 4, 2), bool)
    for mi, model in enumerate(MODELS):
        files = sorted((cache / model).glob('cam*/subject*/scene*/session*.npz'))
        if len(files) != 941:
            raise ValueError(f'{model}: expected 941 caches, got {len(files)}')
        for path in files:
            ci = CAMERAS.index(path.parts[-4])
            with np.load(path, allow_pickle=False) as z:
                idx = z['global_interval_index'].astype(int)
                covered = z['video_covered']
                if np.any(masks[idx, ci, mi]):
                    raise ValueError(f'duplicate rows in {path}')
                features[idx, ci, mi] = z['features']
                masks[idx, ci, mi] = covered
    if not np.all(masks.any((1, 2))):
        raise ValueError('an annotation has no video feature')
    return (features, masks)

def load_imu(legacy, intervals, label_to_id):
    rows = list(csv.DictReader((legacy / 'segments.csv').open(encoding='utf-8')))
    values = np.load(legacy / 'imu_accgyro_96.npy', mmap_mode='r')
    lookup = {f'{r['person']}/{r['session']}:{r['segment_id'].rsplit(':', 1)[1]}': i for i, r in enumerate(rows)}
    out = np.zeros((len(intervals), 18), np.float32)
    present = np.zeros(len(intervals), bool)
    for i, r in enumerate(intervals):
        key = f'{r['subject']}/{r['scene']}/{r['session']}:{r['line_number']}'
        if key not in lookup:
            continue
        j = lookup[key]
        x = np.asarray(values[j], np.float32)
        out[i] = np.r_[x.mean(1), x.std(1)]
        present[i] = True
    if int(present.sum()) != 3989:
        raise ValueError(f'IMU exact join produced {present.sum()}, expected 3989')
    return (out, present)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, default=ROOT / 'runs/mmact_strong_features/cache')
    p.add_argument('--legacy', type=Path, default=ROOT / 'runs/mmact_cache')
    p.add_argument('--run-dir', type=Path, default=ROOT / 'runs/mmact_strong_teacher')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--pilot', action='store_true')
    a = p.parse_args()
    if a.run_dir.exists() and any(a.run_dir.iterdir()):
        raise FileExistsError('run directory must be empty')
    a.run_dir.mkdir(parents=True)
    started = time.perf_counter()
    audit = json.loads((a.cache.parent / 'cache_integrity_audit.json').read_text(encoding='utf-8'))
    if not (audit.get('safe_for_developmental_teacher_training') is True and audit.get('failure_count') == 0 and (audit.get('observed_markers') == 1882)):
        raise RuntimeError('independent cache audit reports failures')
    manifest = json.loads((a.cache / 'interval_manifest.json').read_text(encoding='utf-8'))
    intervals = manifest['intervals']
    if len(intervals) != 5406:
        raise ValueError('requires all 5,406 annotations')
    meta = json.loads((a.legacy / 'cache_metadata.json').read_text(encoding='utf-8'))
    label_to_id = meta['label_to_id']
    classes = [x for x, _ in sorted(label_to_id.items(), key=lambda kv: kv[1])]
    folds = np.asarray([subject_fold(r['subject']) for r in intervals])
    train = np.flatnonzero(np.isin(folds, ['F2', 'F3', 'F4']))
    val = np.flatnonzero(folds == 'F1')
    test = np.flatnonzero(folds == 'F0')
    y = np.asarray([label_to_id[r['label']] if folds[i] != 'F0' else -1 for i, r in enumerate(intervals)])
    if (len(train), len(val), len(test)) != (3243, 1080, 1083) or np.any(y[test] != -1):
        raise ValueError('split firewall failed')
    video, view_masks = load_video(a.cache, len(intervals))
    imu, imu_mask = load_imu(a.legacy, intervals, label_to_id)
    np.save(a.run_dir / 'availability_masks.npy', np.concatenate((view_masks.reshape(len(intervals), 8), imu_mask[:, None]), 1))
    np.savez(a.run_dir / 'split_indices.npz', train=train, validation=val, test=test)
    identities = [{'index': i, 'interval_id': f'{r['annotation_file']}:{r['line_number']}', 'subject': r['subject'], 'scene': r['scene'], 'session': r['session'], 'fold': folds[i]} for i, r in enumerate(intervals)]
    (a.run_dir / 'interval_identities.json').write_text(json.dumps(identities, indent=2) + '\n', encoding='utf-8')
    (a.run_dir / 'classes.json').write_text(json.dumps({'classes': classes, 'label_to_id': label_to_id}, indent=2) + '\n', encoding='utf-8')
    arrays = {}
    available = {}
    normalizers = {}
    for mi, model in enumerate(MODELS):
        for ci, cam in enumerate(CAMERAS):
            name = f'{model}__{cam}'
            raw = np.asarray(video[:, ci, mi], np.float32)
            mask = view_masks[:, ci, mi]
            mean, std = train_normalizer(raw, mask, train)
            arrays[name] = np.nan_to_num((raw - mean) / std, nan=0.0)
            available[name] = mask
            normalizers[name + '__mean'] = mean
            normalizers[name + '__std'] = std
        raw = masked_view_mean(video[:, :, mi], view_masks[:, :, mi])
        mask = view_masks[:, :, mi].any(1)
        name = f'{model}__viewmean'
        mean, std = train_normalizer(raw, mask, train)
        arrays[name] = np.c_[(raw - mean) / std, view_masks[:, :, mi].astype(np.float32)]
        available[name] = mask
        normalizers[name + '__mean'] = mean
        normalizers[name + '__std'] = std
    joint = np.concatenate((arrays[f'{MODELS[0]}__viewmean'], arrays[f'{MODELS[1]}__viewmean']), 1)
    arrays['joint_video'] = joint
    available['joint_video'] = np.ones(len(intervals), bool)
    mean, std = train_normalizer(imu, imu_mask, train)
    arrays['imu'] = np.c_[np.where(imu_mask[:, None], (imu - mean) / std, 0), imu_mask]
    available['imu'] = imu_mask
    normalizers['imu__mean'] = mean
    normalizers['imu__std'] = std
    np.savez(a.run_dir / 'normalizers_train_only.npz', **normalizers)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(6)
    epochs = min(a.epochs, 3) if a.pilot else min(a.epochs, 60)
    summaries = {}
    val_probs = []
    test_probs = []
    val_avail = []
    test_avail = []
    expert_names = []
    for offset, (name, x) in enumerate(arrays.items()):
        present = available[name]
        fit = train[present[train]]
        select_val = val[present[val]] if not np.all(present[val]) else val
        model, summary = train_head(x, y, fit, select_val, 37, device, a.run_dir / f'teacher_{name}.pt', a.run_dir / f'teacher_{name}_train.jsonl', SEED + offset, epochs)
        trlog = predict(model, x[train], device)
        vlog = predict(model, x[val], device)
        tlog = predict(model, x[test], device)
        summaries[name] = summary
        np.savez(a.run_dir / f'predictions_{name}.npz', train_logits=trlog, validation_logits=vlog, test_logits=tlog)
        expert_names.append(name)
        val_probs.append(softmax(vlog))
        test_probs.append(softmax(tlog))
        val_avail.append(present[val])
        test_avail.append(present[test])
    val_probs = np.stack(val_probs)
    test_probs = np.stack(test_probs)
    val_avail = np.stack(val_avail)
    test_avail = np.stack(test_avail)
    selection = sparse_forward_select(val_probs, val_avail, y[val], expert_names, 37)
    equal_indices = [i for i, n in enumerate(expert_names) if n != 'joint_video']
    equal_val = masked_equal_mean(val_probs[equal_indices], val_avail[equal_indices])
    equal_test = masked_equal_mean(test_probs[equal_indices], test_avail[equal_indices])
    sparse_val = weighted_available_mix(val_probs, val_avail, selection['indices'], selection['weights'])
    sparse_test = weighted_available_mix(test_probs, test_avail, selection['indices'], selection['weights'])
    single_i = selection['indices'][0]
    single_val = weighted_available_mix(val_probs, val_avail, [single_i], [1.0])
    single_test = weighted_available_mix(test_probs, test_avail, [single_i], [1.0])
    np.savez(a.run_dir / 'frozen_teacher_predictions.npz', validation_single=single_val, test_single=single_test, validation_equalmean=equal_val, test_equalmean=equal_test, validation_sparse=sparse_val, test_sparse=sparse_test, validation_joint=softmax(np.load(a.run_dir / 'predictions_joint_video.npz')['validation_logits']), test_joint=softmax(np.load(a.run_dir / 'predictions_joint_video.npz')['test_logits']))
    report = {'seed': SEED, 'epochs': epochs, 'pilot': a.pilot, 'split_counts': {'train': len(train), 'validation': len(val), 'test': len(test)}, 'classes': classes, 'expert_order': expert_names, 'head_summaries': summaries, 'selection': selection, 'validation_comparison': {'single': metrics(y[val], single_val, 37), 'equalmean': metrics(y[val], equal_val, 37), 'sparse': metrics(y[val], sparse_val, 37), 'joint': metrics(y[val], np.load(a.run_dir / 'predictions_joint_video.npz')['validation_logits'], 37)}, 'test_labels_used': False, 'test_metrics_computed': False, 'elapsed_seconds': time.perf_counter() - started}
    (a.run_dir / 'validation_selection.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    artifacts = [x for x in a.run_dir.iterdir() if x.is_file()]
    freeze = {**report, 'status': 'teacher_choices_and_f0_predictions_frozen_no_f0_evaluation', 'test_identity_order_sha256': ordered_hash([identities[i]['interval_id'] for i in test]), 'cache_audit_sha256': sha(a.cache.parent / 'cache_integrity_audit.json'), 'artifact_sha256': {x.name: sha(x) for x in artifacts}}
    (a.run_dir / 'pre_evaluation_freeze.json').write_text(json.dumps(freeze, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'run_dir': str(a.run_dir), 'pilot': a.pilot, 'elapsed_seconds': report['elapsed_seconds'], 'validation': report['validation_comparison']}, indent=2))
