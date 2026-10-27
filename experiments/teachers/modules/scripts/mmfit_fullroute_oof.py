from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT = WORK_ROOT
from session_kd.mmfit_rgb_benchmark import ModalityTeacher, MMFitDataset, fit_normalizer, make_loader, predict
from session_kd.mmfit_strong_benchmark import StrongMLP, ArrayDataset, make_loader as array_loader, fit_feature_normalizer, standardize, normalized_pooled_statistics, probabilities_from_logits, validate_split_firewall, sha256_file, ordered_id_hash, TEACHER_ORDER, predict_logits

def stopped(out):
    if any((p.exists() for p in (ROOT / 'STOP', ROOT / 'runs/STOP', out / 'STOP', out.parent / 'STOP'))):
        raise RuntimeError('STOP requested; last.pt is resumable')

def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT / 'runs/mmfit_fullroute_v3/oof')
    p.add_argument('--epochs-old', type=int, default=30)
    p.add_argument('--epochs-strong', type=int, default=60)
    p.add_argument('--pilot', action='store_true')
    a = p.parse_args()
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device('cuda')
    seed = 20260927
    rows = list(csv.DictReader((ROOT / 'runs/mmfit_rgb_benchmark/inputs.csv').open()))
    v1 = ROOT / 'runs/mmfit_rgb_benchmark'
    sf = ROOT / 'runs/mmfit_strong_features'
    ic = ROOT / 'runs/mmfit_unseen_cache'
    y = np.load(v1 / 'source_labels_test_minus1.npy')
    idx = validate_split_firewall(rows, y)
    persons = np.array([int(r['person']) for r in rows])
    ids = np.array([r['segment_id'] for r in rows])
    fold_persons = ((0,), (1,), (5, 6, 7))
    rgb = np.load(v1 / 'rgb_4x512_fp16.npy', mmap_mode='r')
    pose = np.load(v1 / 'pose_51x150_fp16.npy', mmap_mode='r')
    imu = np.load(ic / 'teacher_sensor_24x250.npy', mmap_mode='r')
    present = np.load(ic / 'modality_presence_8.npy', mmap_mode='r')
    strong = [np.load(sf / n, mmap_mode='r') for n in ('videomaev2_large_16x1024_fp16.npy', 'vjepa2_vitl_ssv2_16x1024_fp16.npy')]
    contract = {'seed': seed, 'fold_persons': fold_persons, 'old_epochs': a.epochs_old, 'strong_epochs': a.epochs_strong, 'pilot': a.pilot, 'encoder_initialization': 'frozen generic local encoders; scratch fold heads', 'held_label_epoch_selection': False, 'test_labels_read': False, 'inputs_sha256': sha256_file(v1 / 'inputs.csv'), 'feature_manifest_sha256': sha256_file(sf / 'cache_manifest.json'), 'normalizers': 'fold-fit-only', 'teacher_order': TEACHER_ORDER, 'temperature': 2.0, 'oof_scope': 'train only'}
    cp = out / 'contract.json'
    if cp.exists() and json.loads(cp.read_text()) != json.loads(json.dumps(contract)):
        raise ValueError('resume contract mismatch')
    save_json(cp, contract)
    receipts = []
    result = np.zeros((len(idx['train']), 6, 11), np.float32)
    train_position = {int(i): j for j, i in enumerate(idx['train'])}
    for fold, held_people in enumerate(fold_persons):
        hold = idx['train'][np.isin(persons[idx['train']], held_people)]
        fit = idx['train'][~np.isin(persons[idx['train']], held_people)]
        if a.pilot:
            fit = fit[np.linspace(0, len(fit) - 1, 384).astype(int)]
            hold = hold[np.linspace(0, len(hold) - 1, 64).astype(int)]
        assert not set(persons[fit]) & set(persons[hold])
        n = fit_normalizer(rgb, pose, imu, present, fit)
        arrays = []
        normalizers = dict(n)
        for k, x in enumerate(strong):
            mean, std = fit_feature_normalizer(x, fit)
            arrays.append(standardize(x, mean, std))
            normalizers[f'strong{k}_mean'] = mean
            normalizers[f'strong{k}_std'] = std
        np.savez(out / f'f{fold}_normalizer.npz', **normalizers)
        ps, ims, pm = normalized_pooled_statistics(pose, imu, present, n, np.arange(len(rows)))
        for expert, name in enumerate(TEACHER_ORDER):
            stopped(out)
            folder = out / f'f{fold}_{name}'
            folder.mkdir(exist_ok=True)
            pred_path = folder / 'hold_predictions.npz'
            epoch_budget = 1 if a.pilot else a.epochs_old if expert < 3 else a.epochs_strong
            receipt = {'fold': fold, 'expert': name, 'fit_persons': sorted(set(persons[fit].tolist())), 'held_persons': list(held_people), 'fit_rows': len(fit), 'held_rows': len(hold), 'fit_ids_sha256': ordered_id_hash(ids[fit]), 'hold_ids_sha256': ordered_id_hash(ids[hold]), 'fit_indices_file': str(out / f'f{fold}_fit_indices.npy'), 'hold_indices_file': str(out / f'f{fold}_hold_indices.npy'), 'normalizer_sha256': sha256_file(out / f'f{fold}_normalizer.npz'), 'fixed_epochs': epoch_budget, 'held_labels_read_by_training': False, 'held_labels_used_for_selection': False, 'checkpoint_initialization': 'scratch', 'encoder_ancestry': 'generic pretrained encoder; no MM-Fit-label encoder updates'}
            np.save(out / f'f{fold}_fit_indices.npy', fit)
            np.save(out / f'f{fold}_hold_indices.npy', hold)
            if pred_path.exists():
                with np.load(pred_path) as z:
                    assert np.array_equal(z['ids'], ids[hold])
                    prob = z['probabilities']
                old = json.loads((folder / 'receipt.json').read_text())
                if old['fixed_epochs'] != epoch_budget:
                    raise ValueError('OOF budget changed')
                receipt = old
            else:
                local_seed = seed + fold * 20 + expert
                torch.manual_seed(local_seed)
                np.random.seed(local_seed)
                torch.cuda.manual_seed_all(local_seed)
                if expert < 3:
                    model = ModalityTeacher(name).to(device)
                    ds = MMFitDataset(rgb, pose, imu, present, fit, y, n)
                    hs = MMFitDataset(rgb, pose, imu, present, hold, None, n)
                    loader = make_loader(ds, 256, True, local_seed)
                    hl = make_loader(hs, 256, False, local_seed)
                else:
                    parts = ([arrays[0]], [arrays[1]], [*arrays, ps, ims, pm])[expert - 3]
                    model = StrongMLP(sum((x.shape[1] for x in parts))).to(device)
                    loader = array_loader(ArrayDataset(parts, fit, y), True, local_seed)
                    hl = array_loader(ArrayDataset(parts, hold), False, local_seed)
                opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
                sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epoch_budget)
                counts = np.bincount(y[fit], minlength=11).astype(np.float32)
                cw = 1 / np.sqrt(np.maximum(counts, 1))
                cw /= cw.mean()
                cw = torch.tensor(cw, device=device)
                start = 0
                updates = 0
                last = folder / 'last.pt'
                if last.exists():
                    z = torch.load(last, map_location='cpu', weights_only=False)
                    model.load_state_dict(z['state_dict'])
                    opt.load_state_dict(z['optimizer'])
                    sch.load_state_dict(z['scheduler'])
                    start = z['epoch']
                    updates = z['updates']
                    torch.set_rng_state(z['rng'])
                    torch.cuda.set_rng_state_all(z['cuda_rng'])
                    loader.generator.set_state(z['loader_rng'])
                began = time.perf_counter()
                with (folder / 'train.jsonl').open('a') as log:
                    for epoch in range(start + 1, epoch_budget + 1):
                        stopped(out)
                        model.train()
                        loss_sum = 0.0
                        seen = 0
                        for batch in loader:
                            stopped(out)
                            xs = [x.to(device) for x in batch[:-1]]
                            truth = batch[-1].to(device)
                            opt.zero_grad(set_to_none=True)
                            logits = model(*xs) if expert < 3 else model(xs[0])
                            loss = F.cross_entropy(logits, truth, weight=cw)
                            if not torch.isfinite(loss):
                                raise FloatingPointError('nonfinite OOF loss')
                            loss.backward()
                            if not all((torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)):
                                raise FloatingPointError('nonfinite OOF gradient')
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                            opt.step()
                            updates += 1
                            loss_sum += float(loss.detach()) * len(truth)
                            seen += len(truth)
                        sch.step()
                        torch.save({'state_dict': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': sch.state_dict(), 'epoch': epoch, 'updates': updates, 'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(), 'loader_rng': loader.generator.get_state()}, folder / 'last.tmp')
                        os.replace(folder / 'last.tmp', last)
                        log.write(json.dumps({'epoch': epoch, 'loss': loss_sum / seen, 'updates': updates, 'seconds': time.perf_counter() - began}) + '\n')
                        log.flush()
                logits = predict(model, hl, device) if expert < 3 else predict_logits(model, hl, device)
                prob = probabilities_from_logits(logits)
                np.savez(pred_path, ids=ids[hold], indices=hold, probabilities=prob)
                receipt.update({'updates': updates, 'checkpoint_sha256': sha256_file(last), 'prediction_sha256': sha256_file(pred_path), 'finite_gradients': True})
                save_json(folder / 'receipt.json', receipt)
                del model, opt, loader, hl
                torch.cuda.empty_cache()
            receipts.append(receipt)
            for j, i in enumerate(hold):
                result[train_position[int(i)], expert] = prob[j]
            print(json.dumps({'fold': fold, 'expert': name, 'completed': True, 'rows': len(hold)}), flush=True)
    if not a.pilot:
        if not np.allclose(result.sum(2), 1, atol=1e-05):
            raise ValueError('OOF coverage incomplete')
        np.savez(out / 'train_oof_bank.npz', indices=idx['train'], ids=ids[idx['train']], probabilities=result)
    save_json(out / 'audit.json', {'status': 'pilot_complete' if a.pilot else 'complete', 'receipts': receipts, 'subject_disjoint': True, 'normalizer_fit_excludes_held': True, 'test_labels_read': False, 'train_oof_bank_sha256': None if a.pilot else sha256_file(out / 'train_oof_bank.npz')})
