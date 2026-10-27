from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import argparse
import atexit
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT = WORK_ROOT
from session_kd.mmact_extra_teachers import *
from session_kd.mmact_strong_teacher import train_head, predict, softmax, metrics

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()

def atomic_json(p, v):
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(v, indent=2) + '\n', encoding='utf8')
    os.replace(t, p)

def atomic_torch(p, v):
    t = p.with_suffix(p.suffix + '.tmp')
    torch.save(v, t)
    os.replace(t, p)

def acquire_lock(path):
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as e:
        raise RuntimeError(f'active lock: {path}') from e
    with os.fdopen(fd, 'w') as f:
        json.dump({'pid': os.getpid(), 'time': time.time()}, f)

def receipt_valid(path, expected, root):
    if not path.exists():
        return False
    try:
        r = json.loads(path.read_text())
    except Exception:
        return False
    artifacts = r.get('artifacts', {})
    if r.get('status') != 'complete' or r.get('contract') != expected or (not isinstance(artifacts, dict)) or (not artifacts):
        return False
    return all(((root / name).is_file() and sha(root / name) == digest for name, digest in artifacts.items()))

def native_predict(model, loader, device, limit=0):
    model.eval()
    sk = []
    imu = []
    avs = []
    avi = []
    idx = []
    with torch.no_grad():
        for bi, b in enumerate(loader):
            if limit and bi >= limit:
                break
            motion = {k: v.to(device) for k, v in b.items() if k not in {'label', 'skeleton_available', 'imu_available', 'global_index'}}
            out = model(motion, mask_ratio=0.0)
            sk.append(out['skeleton_logits'].float().cpu())
            imu.append(out['imu_logits'].float().cpu())
            avs.append(b['skeleton_available'])
            avi.append(b['imu_available'])
            idx.append(b['global_index'])
    return (torch.cat(sk).numpy(), torch.cat(imu).numpy(), torch.cat(avs).numpy(), torch.cat(avi).numpy(), torch.cat(idx).numpy())

def trainmodules(cache, fit_indices, fit_ids, fit_labels, val_indices, val_ids, val_labels, normalizer, out, device, epochs, pilot_batches, seed, stop_file):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = buildmodules_model().to(device)
    recipe = native_recipe()
    counts = np.bincount(fit_labels, minlength=37).astype(np.float32)
    weight = 1 / np.power(np.maximum(counts, 1), 0.35)
    weight /= weight.mean()
    weight = torch.from_numpy(weight).to(device)
    train = NativeMotionDataset(cache, fit_indices, fit_ids, fit_labels, normalizer)
    val = NativeMotionDataset(cache, val_indices, val_ids, val_labels, normalizer)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=2e-05)
    best = (-1.0, -1.0)
    history = []
    for epoch in range(epochs):
        if stop_file.exists():
            raise InterruptedError('STOP requested at epoch boundary')
        model.train()
        gen = torch.Generator().manual_seed(20260922 + epoch)
        loader = DataLoader(train, batch_size=64, shuffle=True, num_workers=0, generator=gen)
        total = 0.0
        seen = 0
        for bi, b in enumerate(loader):
            if pilot_batches and bi >= pilot_batches:
                break
            motion = {k: v.to(device) for k, v in b.items() if k not in {'label', 'skeleton_available', 'imu_available', 'global_index'}}
            y = b['label'].to(device)
            opt.zero_grad(set_to_none=True)
            outputs = model(motion, mask_ratio=0.0)
            loss, parts = ce_onlymodules_loss(outputs, y, b['skeleton_available'].to(device), b['imu_available'].to(device), weight)
            if not torch.isfinite(loss):
                raise FloatingPointError('native nonfinite loss')
            loss.backward()
            if not all((p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())):
                raise FloatingPointError('native nonfinite gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            total += float(loss.detach()) * len(y)
            seen += len(y)
        sched.step()
        vlog_s, vlog_i, avs, avi, _ = native_predict(model, DataLoader(val, batch_size=64, num_workers=0), device, pilot_batches)
        score_s = metrics(val_labels[:len(vlog_s)][avs], vlog_s[avs], 37) if avs.any() else {'macro_f1_fixed37': 0.0, 'accuracy': 0.0}
        score_i = metrics(val_labels[:len(vlog_i)][avi], vlog_i[avi], 37) if avi.any() else {'macro_f1_fixed37': 0.0, 'accuracy': 0.0}
        key = ((score_s['macro_f1_fixed37'] + score_i['macro_f1_fixed37']) / 2, (score_s['accuracy'] + score_i['accuracy']) / 2)
        history.append({'epoch': epoch + 1, 'train_rows': seen, 'optimizer_updates': bi if pilot_batches and bi >= pilot_batches else bi + 1, 'loss': total / max(seen, 1), 'skeleton': score_s, 'imu': score_i})
        atomic_json(out.with_suffix('.history.json'), history)
        atomic_json(out.parent / 'status.json', {'status': 'RUNNING', 'pid': os.getpid(), 'fit': out.stem, 'epoch': epoch + 1, 'epochs': epochs, 'f0_labels_read': False})
        if key > best:
            best = key
            atomic_torch(out, {'state_dict': model.state_dict(), 'epoch': epoch + 1, 'recipe': recipe, 'history': history, 'seed': seed})
    saved = torch.load(out, map_location=device, weights_only=False)
    model.load_state_dict(saved['state_dict'])
    return (model, saved)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, default=ROOT / 'runs/mmact_input_cache')
    p.add_argument('--oof-bank', type=Path, default=ROOT / 'runs/mmact_teacher_oof/source_oof_bank.npz')
    p.add_argument('--mc3', type=Path, default=ROOT / 'runs/mmact_mc3_features/mc3_18_512_fp16.npy')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--families', choices=('mc3', 'native', 'all'), default='all')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--preflight-only', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    device = torch.device(a.device)
    a.output.mkdir(parents=True, exist_ok=True)
    lock = a.output / '.lock'
    acquire_lock(lock)
    stop_file = a.output / 'STOP'
    atomic_json(a.output / 'status.json', {'status': 'RUNNING', 'pid': os.getpid(), 'f0_labels_read': False})
    atexit.register(lambda: lock.unlink(missing_ok=True))
    old_hook = sys.excepthook

    def failed(kind, value, traceback):
        atomic_json(a.output / 'status.json', {'status': 'FAILED', 'error_type': kind.__name__, 'error': str(value), 'f0_labels_read': False})
        old_hook(kind, value, traceback)
    sys.excepthook = failed
    with np.load(a.oof_bank, allow_pickle=False) as z:
        src_idx = z['source_indices']
        src_ids = z['source_interval_ids'].astype(str)
        src_y = z['source_labels']
        src_folds = z['source_folds'].astype(str)
        val_idx = z['validation_indices']
        val_ids = z['validation_interval_ids'].astype(str)
        val_y = z['validation_labels']
        tgt_idx = z['target_indices']
        tgt_ids = z['target_interval_ids'].astype(str)
        if 'target_labels' in z.files:
            raise ValueError('F0 labels forbidden')
        global_y = np.full(5406, -1, np.int64)
        global_y[src_idx] = src_y
        global_y[val_idx] = val_y
        mc3_manifest_path = ROOT / 'runs/mmact_mc3_features/input_manifest.json'
        mc3_manifest = json.loads(mc3_manifest_path.read_text())
        all_ids = np.empty(5406, dtype=object)
        for row in mc3_manifest['rows_manifest']:
            all_ids[int(row['global_interval_index'])] = str(row['interval_id'])
        for indices, ids in ((src_idx, src_ids), (val_idx, val_ids), (tgt_idx, tgt_ids)):
            if not np.array_equal(all_ids[indices].astype(str), ids):
                raise ValueError('MC3 feature identity metadata differs from verified OOF bank')
        source_hashes = {'oof_bank': sha(a.oof_bank), 'mc3': sha(a.mc3), 'mc3_input_manifest': sha(mc3_manifest_path), 'cache_manifest': sha(a.cache / 'RUN_MANIFEST.json'), 'cache_full_array_audit': sha(a.cache / 'full_array_audit.json'), 'module': sha(MODULE_ROOT / 'session_kd/mmact_extra_teachers.py'), 'cli': sha(Path(__file__).resolve()), 'strong_teacher': sha(MODULE_ROOT / 'session_kd/mmact_strong_teacher.py'), 'imu_adapter': sha(MODULE_ROOT / 'session_kd/mmact_imu.py'), 'skeleton_adapter': sha(MODULE_ROOT / 'session_kd/mmact_skeleton.py'), 'motion_adapter': sha(MODULE_ROOT / 'session_kd/mmact_motion_pretrain.py'), 'mobind': sha(MODULE_ROOT / 'session_kd/mmact_backbones/mobind.py'), 'cache_mmact': sha(MODULE_ROOT / 'session_kd/cache_mmact.py')}
    if a.preflight_only:
        mc3 = np.load(a.mc3, mmap_mode='r')
        files = list(a.cache.glob('interval_*.npz'))
        if mc3.shape != (5406, 512) or len(files) != 5406:
            raise ValueError('actual extra-teacher asset geometry/count mismatch')
        samples = []
        for index, expected in ((int(src_idx[0]), src_ids[0]), (int(val_idx[0]), val_ids[0]), (int(tgt_idx[0]), tgt_ids[0])):
            with np.load(a.cache / f'interval_{index:06d}.npz', allow_pickle=False) as z:
                samples.append({'index': index, 'id_ok': str(z['interval_id']) == str(expected), 'pose_shape': list(z['pose_coco2d'].shape), 'imu_shape': list(z['imu_raw_phone_watch'].shape)})
        if not all((x['id_ok'] for x in samples)):
            raise ValueError('sample cache identity mismatch')
        report = {'status': 'preflight_complete_no_fit', 'mc3_shape': list(mc3.shape), 'pixel_cache_files': len(files), 'samples': samples, 'source_rows': len(src_idx), 'validation_rows': len(val_idx), 'target_rows': len(tgt_idx), 'f0_labels_loaded': False, 'families': list(FAMILIES), 'native_recipe': native_recipe(), 'source_hashes': source_hashes}
        atomic_json(a.output / 'preflight.json', report)
        print(json.dumps(report, indent=2))
        return
    families = ['mc3'] if a.families == 'mc3' else ['native'] if a.families == 'native' else ['mc3', 'native']
    fits = ['F2'] if a.pilot else list(FITS)
    source_id_by_index = {int(index): sid for index, sid in zip(src_idx, src_ids)}
    receipts = []
    predictions = {}
    started = time.perf_counter()
    for family in families:
        for fit_name in fits:
            if stop_file.exists():
                result = {'status': 'STOPPED_AT_FIT_BOUNDARY', 'completed_receipts': len(receipts), 'full_bank_written': False, 'f0_labels_read': False}
                atomic_json(a.output / 'status.json', result)
                print(json.dumps(result))
                return
            fit, pred = fit_predict_indices(src_idx, src_folds, fit_name)
            stem = f'{family}__{fit_name}'
            receipt_path = a.output / f'{stem}.receipt.json'
            expected = {'family': family, 'fit_name': fit_name, 'fit_indices_sha256': ordered_hash(map(str, fit)), 'source_hashes': source_hashes, 'pilot': a.pilot}
            if receipt_valid(receipt_path, expected, a.output):
                receipts.append(json.loads(receipt_path.read_text()))
                continue
            if family == 'mc3':
                torch.manual_seed(20260922)
                np.random.seed(20260922)
                x = np.asarray(np.load(a.mc3, mmap_mode='r'), np.float32)
                mean = x[fit].mean(0)
                std = np.maximum(x[fit].std(0), 1e-06)
                xn = (x - mean) / std
                epochs = 3 if a.pilot else 60
                model, summary = train_head(xn, global_y, fit, val_idx, 37, device, a.output / f'{stem}.pt', a.output / f'{stem}.jsonl', 20260922, epochs)
                targets = pred if fit_name != 'final_source' else np.r_[val_idx, tgt_idx]
                logits = predict(model, xn[targets], device)
                probs = softmax(logits)
                avail = np.ones(len(targets), bool)
                np.savez(a.output / f'{stem}.pred.npz', indices=targets, probabilities=probs, availability=avail)
                np.savez(a.output / f'{stem}.norm.npz', mean=mean, std=std)
                rec = {'status': 'complete', 'contract': expected, 'family': 'mc3_motion', 'fit_name': fit_name, 'fit_indices_sha256': ordered_hash(map(str, fit)), 'prediction_indices_sha256': ordered_hash(map(str, targets)), 'fit_count': len(fit), 'prediction_count': len(targets), 'summary': summary, 'epochs': epochs, 'selection_fold': 'F1', 'f0_labels_read': False}
                np.savez_compressed(a.output / f'{stem}.lineage.npz', fit_indices=fit, fit_interval_ids=np.asarray([source_id_by_index[int(i)] for i in fit]), prediction_indices=targets, prediction_interval_ids=np.asarray([source_id_by_index[int(i)] for i in targets]) if fit_name != 'final_source' else np.r_[val_ids, tgt_ids], normalizer_indices=fit)
            else:
                norm_fit = fit[:128] if a.pilot else fit
                norm_fit_ids = np.asarray([source_id_by_index[int(i)] for i in norm_fit])
                norm = fitmodules_normalizer(a.cache, norm_fit, norm_fit_ids)
                epochs = 1 if a.pilot else 24
                model, saved = trainmodules(a.cache, fit, np.asarray([source_id_by_index[int(i)] for i in fit]), global_y[fit], val_idx, val_ids, val_y, norm, a.output / f'{stem}.pt', device, epochs, 2 if a.pilot else 0, 20260922, stop_file)
                targets = pred if fit_name != 'final_source' else np.r_[val_idx, tgt_idx]
                target_ids = np.asarray([source_id_by_index[int(i)] for i in targets]) if fit_name != 'final_source' else np.r_[val_ids, tgt_ids]
                ds = NativeMotionDataset(a.cache, targets, target_ids, np.zeros(len(targets), np.int64), norm)
                ls, li, avs, avi, indices = native_predict(model, DataLoader(ds, batch_size=64, num_workers=0), device, 2 if a.pilot else 0)
                ps = softmax(ls)
                pi = softmax(li)
                ps[~avs] = 0
                pi[~avi] = 0
                np.savez(a.output / f'{stem}.pred.npz', indices=indices, skeleton_probabilities=ps, imu_probabilities=pi, skeleton_availability=avs, imu_availability=avi)
                np.savez(a.output / f'{stem}.norm.npz', mean=norm.mean, std=norm.std, training_ids=np.asarray(norm.training_ids))
                rec = {'status': 'complete', 'contract': expected, 'family': 'native_skeleton+sequential_imu', 'fit_name': fit_name, 'fit_indices_sha256': ordered_hash(map(str, fit)), 'normalizer_indices_sha256': ordered_hash(map(str, norm_fit)), 'fit_count': len(fit), 'prediction_count': len(indices), 'epochs': epochs, 'selection_fold': 'F1', 'recipe': native_recipe(), 'initialization': 'from_scratch', 'f0_labels_read': False}
                np.savez_compressed(a.output / f'{stem}.lineage.npz', fit_indices=fit, fit_interval_ids=np.asarray([source_id_by_index[int(i)] for i in fit]), prediction_indices=indices, prediction_interval_ids=target_ids[:len(indices)], normalizer_indices=norm_fit, normalizer_interval_ids=norm_fit_ids)
            rec['artifacts'] = {name: sha(a.output / name) for name in (f'{stem}.pt', f'{stem}.pred.npz', f'{stem}.norm.npz', f'{stem}.lineage.npz')}
            atomic_json(receipt_path, rec)
            receipts.append(rec)
    if a.pilot:
        result = {'status': 'pilot_incomplete', 'families': families, 'fits': fits, 'receipts': len(receipts), 'elapsed_seconds': time.perf_counter() - started, 'full_bank_written': False}
        atomic_json(a.output / 'status.json', result)
        print(json.dumps(result, indent=2))
        return
    if set(families) != {'mc3', 'native'}:
        result = {'status': 'stage_complete_partial_not_full_bank', 'families': families, 'fits': fits, 'receipts': len(receipts), 'elapsed_seconds': time.perf_counter() - started, 'full_bank_written': False, 'f0_labels_read': False}
        atomic_json(a.output / 'status.json', result)
        print(json.dumps(result, indent=2))
        return
    split_defs = {'source': (src_idx, src_ids), 'validation': (val_idx, val_ids), 'target': (tgt_idx, tgt_ids)}
    arrays = {}
    for split, (indices, ids) in split_defs.items():
        prob = np.zeros((len(indices), 3, 37), np.float32)
        avail = np.zeros((len(indices), 3), bool)
        assigned = np.zeros((len(indices), 3), np.int8)
        pos = {int(v): i for i, v in enumerate(indices)}
        for fit_name in FITS:
            if fit_name == 'final_source' and split == 'source':
                continue
            if fit_name != 'final_source' and split != 'source':
                continue
            if 'mc3' in families:
                with np.load(a.output / f'mc3__{fit_name}.pred.npz') as z:
                    for row, p, m in zip(z['indices'], z['probabilities'], z['availability']):
                        if int(row) in pos:
                            prob[pos[int(row)], 0] = p
                            avail[pos[int(row)], 0] = m
                            assigned[pos[int(row)], 0] += 1
            if 'native' in families:
                with np.load(a.output / f'native__{fit_name}.pred.npz') as z:
                    for j, row in enumerate(z['indices']):
                        if int(row) in pos:
                            prob[pos[int(row)], 1] = z['skeleton_probabilities'][j]
                            prob[pos[int(row)], 2] = z['imu_probabilities'][j]
                            avail[pos[int(row)], 1] = z['skeleton_availability'][j]
                            avail[pos[int(row)], 2] = z['imu_availability'][j]
                            assigned[pos[int(row)], 1:3] += 1
        arrays[f'{split}_probabilities'] = prob
        arrays[f'{split}_availability'] = avail
        arrays[f'{split}_indices'] = indices
        arrays[f'{split}_interval_ids'] = ids
        if not np.all(assigned == 1):
            raise AssertionError(f'{split} rows are not exactly once per extra family')
        if not np.all(np.any(prob != 0, axis=2) == avail):
            raise AssertionError(f'{split} availability/probability coverage mismatch')
        if not np.isfinite(prob).all() or np.any(prob < 0):
            raise AssertionError('nonfinite/negative extra probability')
        sums = prob.sum(2)
        if not np.allclose(sums[avail], 1, atol=1e-05) or not np.all(sums[~avail] == 0):
            raise AssertionError('available probabilities must sum one; unavailable must be zero')
    np.savez_compressed(a.output / 'extra_bank.npz', **arrays, names=np.asarray(FAMILIES))
    manifest = {'status': 'complete', 'schema': 'MMACT_EXTRA_TEACHERS_V1', 'names': list(FAMILIES), 'shape': {k: list(v.shape) for k, v in arrays.items() if 'probabilities' in k}, 'f0_labels_read': False, 'f0_metrics_computed': False, 'ce_onlymodules': True, 'receipts': receipts, 'extra_bank_sha256': sha(a.output / 'extra_bank.npz'), 'elapsed_seconds': time.perf_counter() - started}
    atomic_json(a.output / 'manifest.json', manifest)
    atomic_json(a.output / 'status.json', {'status': 'complete', 'full_bank_written': True, 'f0_labels_read': False})
    print(json.dumps({k: manifest[k] for k in ('status', 'shape', 'elapsed_seconds')}, indent=2))
