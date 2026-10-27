from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import argparse
import csv
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
ROOT = WORK_ROOT
from session_kd.mmfit_fullroute import build_student, SharedRGBReader, EARLY_LATE

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--segment-id', required=True)
    p.add_argument('--run-root', type=Path, default=ROOT / 'runs/mmfit_fullroute_v3')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    rows = list(csv.DictReader((ROOT / 'runs/mmfit_rgb_benchmark/inputs.csv').open()))
    matches = [(i, r) for i, r in enumerate(rows) if r['segment_id'] == a.segment_id]
    if len(matches) != 1:
        raise ValueError('segment ID not unique/found')
    i, r = matches[0]
    pose = np.load(ROOT / f'data/MM-Fit/mm-fit/{r['workout']}/{r['workout']}_pose_2d.npy', mmap_mode='r')
    reader = SharedRGBReader(Path(str(DATA_ROOT / 'Downloads')) / f'{r['workout']}_rgb.mp4', pose, int(pose[0, 0, 0]))
    pixels, valid = reader.window(int(r['start_frame']))
    reader.close()
    native = a.run_root / 'native_motion_refclock'
    motion = {path.stem: torch.as_tensor(np.array(np.load(path, mmap_mode='r')[i:i + 1]), device='cuda') for path in native.glob('*.npy') if path.stem != 'completed'}
    model = build_student().cuda()
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'], strict=True)
    model.eval()
    images = torch.as_tensor(pixels[None], device='cuda')
    mask = torch.as_tensor(valid[None], device='cuda')
    began = time.perf_counter()
    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            sequence = model.visual.encode_backbone_sequence(images)
        sequence = sequence.float().half().float()
        result = model.forward_from_backbone_sequence(sequence, mask, mask.float(), motion['global_time_position'], motion)['logits'].float().cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError('raw deployment nonfinite')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.output, ids=np.array([a.segment_id]), logits=result)
    print(json.dumps({'checkpoint': str(a.checkpoint.resolve()), 'segment_id': a.segment_id, 'logits': result.tolist(), 'seconds': time.perf_counter() - began, 'single_student': True, 'teacher_execution': False, 'labels_read': False}))
