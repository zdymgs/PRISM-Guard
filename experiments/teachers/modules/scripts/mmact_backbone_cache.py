from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
import numpy as np
import torch
ROOT = WORK_ROOT
from session_kd.mmact_backbone_cache import *
from session_kd.mmact_visual import MMACTVisualStudent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, default=ROOT / 'runs/mmact_mc3_features/input_manifest.json')
    p.add_argument('--interval-root', type=Path, default=ROOT / 'runs/mmact_input_cache')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--status', type=Path)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--chunk-size', type=int, default=32)
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--stop-file', type=Path)
    a = p.parse_args()
    if not 1 <= a.batch_size <= 4:
        raise ValueError('batch size must be 1..4')
    rows = read_visual_manifest(a.manifest)
    gate = checkpoint_gate(a.checkpoint, status_path=a.status, pilot=a.pilot)
    rows = [rows[i] for i in PILOT_IDS] if a.pilot else rows
    paths = [a.interval_root / f'interval_{int(r['global_interval_index']):06d}.npz' for r in rows]
    a.output.mkdir(parents=True, exist_ok=True)
    lock = a.output / '.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, json.dumps({'pid': os.getpid(), 'created': time.time()}).encode())
        os.close(fd)
    except FileExistsError:
        raise RuntimeError('output lock exists')
    try:
        saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
        if not isinstance(saved, dict) or 'model' not in saved:
            raise ValueError("checkpoint lacks 'model'")
        model = MMACTVisualStudent(classes=37, kinetics_pretrained=False)
        model.load_state_dict(saved['model'], strict=True)
        device = torch.device(a.device)
        model.to(device).eval()

        def extractor(video, region, times):
            out = []
            with torch.no_grad():
                for j in range(0, len(video), a.batch_size):
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                        value = model.encode_backbone_sequence(torch.from_numpy(video[j:j + a.batch_size]).to(device))
                    out.append(value.float().cpu().numpy())
            return np.concatenate(out)
        audit = a.interval_root / 'full_array_audit.json'
        run_contract = a.interval_root / 'RUN_MANIFEST.json'
        if json.loads(audit.read_text()).get('status') != 'PASS':
            raise ValueError('full cache audit required')
        w = BackboneCacheWriter(a.output, rows, checkpoint_contract=gate, input_manifest_hash=sha256_file(a.manifest), cache_contract_hash=sha256_file(run_contract), cache_audit_hash=sha256_file(audit), source_hashes={'module': sha256_file(MODULE_ROOT / 'session_kd/mmact_backbone_cache.py'), 'cli': sha256_file(Path(__file__).resolve()), 'visual_adapter': sha256_file(MODULE_ROOT / 'session_kd/mmact_visual.py'), 'vendor_visual': sha256_file(MODULE_ROOT / 'session_kd/mmact_backbones/visual.py'), 'vendor_constants': sha256_file(MODULE_ROOT / 'session_kd/mmact_backbones/constants.py')}, pilot=a.pilot, chunk_size=a.chunk_size, device=a.device, inference_precision='cuda_fp16_autocast' if device.type == 'cuda' else 'cpu_float32', batch_size=a.batch_size)
        print(json.dumps(w.run(extractor, paths, stop_file=a.stop_file), indent=2))
    except Exception as e:
        atomic_json(a.output / 'status.json', {'status': 'FAILED', 'error_type': type(e).__name__, 'error': str(e), 'pilot': a.pilot})
        raise
    finally:
        lock.unlink(missing_ok=True)
