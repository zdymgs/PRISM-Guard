from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import argparse
import csv
import json
import os
import subprocess
import sys
import time
import hashlib
from pathlib import Path

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 ** 2), b''):
            h.update(b)
    return h.hexdigest()

def child(out):
    import numpy as np
    import torch
    from aligned_multimodal.lavila_cache import LaViLaCache, CHECKPOINT_SHA
    from aligned_multimodal.lavila_teacher import CHECKPOINT, PIXELS, EXTERNAL, load_visual, prepare_video
    root = WORK_ROOT
    here = root / 'aligned_multimodal'
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    if digest(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('checkpoint identity differs')
    cache = LaViLaCache(here / 'runs/lavila_frame_token_cache', PIXELS / 'rows.csv', here / 'data/manifest.csv')
    with (here / 'data/manifest.csv').open(encoding='utf-8-sig') as f:
        users = {r['sample_id']: r['user_id'] for r in csv.DictReader(f)}
    blocked = {'user1', 'user2', 'user21'}
    candidates = []
    for i, j in enumerate(cache.master_to_pixel):
        if i == j or users[cache.master_ids[i]] in blocked or users[cache.pixel_ids[i]] in blocked:
            continue
        diff = cache.tokens[j, :16].astype(np.float32) - cache.tokens[i, :16].astype(np.float32)
        candidates.append((float(np.linalg.norm(diff)), i, int(j)))
    chosen = []
    used_users = set()
    for _, i, j in sorted(candidates, reverse=True):
        user = users[cache.master_ids[i]]
        if user not in used_users:
            chosen.append((i, j))
            used_users.add(user)
        if len(chosen) == 2:
            break
    if len(chosen) != 2:
        raise ValueError('two distinct eligible probes required')
    images = np.load(PIXELS / 'images.npy', mmap_mode='r', allow_pickle=False)
    completed = np.load(PIXELS / 'completed.npy', allow_pickle=False)
    if images.shape != (2914, 2, 16, 3, 160, 160) or images.dtype != np.uint8 or completed.shape != (2914,) or (not np.all(completed == 1)):
        raise ValueError('pixel cache shape/dtype/completion differs')
    pixels = [np.asarray(images[j, :, :, 0]).copy() for i, j in chosen]
    paths = [CHECKPOINT, Path(__file__).resolve().parent / 'lavila_teacher.py', Path(__file__).resolve().parent / 'lavila_frame_token_cache.py', Path(__file__), MODULE_ROOT.parent / 'README.md', PIXELS / 'completed.npy', PIXELS / 'summary.json', EXTERNAL / 'lavila/models/timesformer.py', EXTERNAL / 'lavila/models/openai_model.py']
    registry = {'files_sha256': {str(p): digest(p) for p in paths}, 'cache_provenance': cache.provenance, 'probes': [{'canonical_index': i, 'pixel_index': j, 'sample_id': str(cache.master_ids[i]), 'wrong_positional_sample_id': str(cache.pixel_ids[i]), 'pixel_sha256': hashlib.sha256(pixels[k].tobytes()).hexdigest()} for k, (i, j) in enumerate(chosen)], 'device': 'cpu', 'threads': 1, 'fitting': False, 'labels_used': False}
    (out / 'registry.json').write_text(json.dumps(registry, indent=2), encoding='utf-8')
    torch.set_num_threads(1)
    visual, projection, checkpoint, frames = load_visual(CHECKPOINT, torch.device('cpu'), 16)
    del projection, checkpoint
    results = []
    arrays = {}
    for k, ((i, j), pixel) in enumerate(zip(chosen, pixels)):
        with torch.inference_mode():
            video = prepare_video(pixel[None], torch.device('cpu'), 16)
            tokens = visual.forward_features(video.permute(0, 2, 1, 3, 4).contiguous(), cls_at_last=False).float()
            actual = tokens[:, 1:].reshape(1, 16, visual.patches_per_frame, 768).mean(2)[0].numpy()
        expected = cache.tokens[j, :16].astype(np.float32)
        wrong = cache.tokens[i, :16].astype(np.float32)
        a, e, w = (v.astype(np.float64).reshape(-1) for v in (actual, expected, wrong))
        error = float(np.linalg.norm(a - e))
        wrong_error = float(np.linalg.norm(a - w))
        relative = error / max(float(np.linalg.norm(e)), 1e-12)
        cosine = float(a @ e / max(float(np.linalg.norm(a) * np.linalg.norm(e)), 1e-12))
        passed = bool(cosine >= 0.995 and relative <= 0.05 and (wrong_error > 0) and (error <= 0.1 * wrong_error))
        result = {'probe': k, 'cosine': cosine, 'relative_l2': relative, 'error_vs_wrong_ratio': error / max(wrong_error, 1e-12), 'passed': passed}
        results.append(result)
        arrays.update({f'actual{k}': actual, f'expected{k}': expected, f'wrong{k}': wrong})
        print(json.dumps(result), flush=True)
    np.savez_compressed(out / 'tokens.npz', **arrays)
    report = {'probes': results, 'passed': all((r['passed'] for r in results)), 'seconds': time.monotonic() - start, 'fitting': False, 'test_rows_loaded': 0, 'scope': 'two sampled scene rows only, not whole-cache certification'}
    (out / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', required=True)
    p.add_argument('--child', action='store_true')
    a = p.parse_args()
    out = Path(a.out_dir)
    if a.child:
        child(out)
        return
    log = Path(str(out) + '.process.log')
    report = Path(str(out) + '.process.json')
    if out.exists() or log.exists() or report.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    started = time.monotonic()
    with log.open('x', encoding='utf-8') as f:
        process = subprocess.Popen([sys.executable, '-u', '-m', 'aligned_multimodal.cpu_spotcheck', '--child', '--out-dir', str(out)], stdout=f, stderr=subprocess.STDOUT, env=env)
        try:
            code = process.wait(timeout=180)
            status = 'complete' if code == 0 else 'failed'
        except subprocess.TimeoutExpired:
            process.kill()
            code = process.wait(timeout=10)
            status = 'watchdog_timeout'
    result = {'status': status, 'exit_code': code, 'seconds': time.monotonic() - started, 'pid': process.pid}
    report.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if code:
        raise SystemExit(1)
