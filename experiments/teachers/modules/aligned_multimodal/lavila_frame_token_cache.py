from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from aligned_multimodal.lavila_teacher import CHECKPOINT, PIXELS, checkpoint_sha256, load_visual, prepare_video
HERE = WORK_ROOT / 'aligned_multimodal'
OUTPUT = HERE / 'runs/lavila_frame_token_cache'

def read_rows(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pixel-cache', type=Path, default=PIXELS)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    args.pixel_cache = args.pixel_cache.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.pixel_cache / 'rows.csv')
    images = np.load(args.pixel_cache / 'images.npy', mmap_mode='r')
    output = np.lib.format.open_memmap(args.output_dir / 'frame_tokens.npy', mode='w+', dtype=np.float16, shape=(len(rows), 3 * args.frames, 768))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    visual, _, checkpoint, checkpoint_frames = load_visual(args.checkpoint, device, args.frames)
    records = [(row, view) for row in range(len(rows)) for view in range(3)]
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            batch_records = records[start:start + args.batch_size]
            values = np.stack([images[row, :, :, view] for row, view in batch_records])
            video = prepare_video(values, device, args.frames)
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda', dtype=torch.float16):
                tokens = visual.forward_features(video.permute(0, 2, 1, 3, 4).contiguous(), cls_at_last=False).float()
                spatial = tokens[:, 1:].reshape(len(batch_records), args.frames, visual.patches_per_frame, 768).mean(dim=2)
            values_numpy = spatial.cpu().numpy().astype(np.float16)
            for index, (row, view) in enumerate(batch_records):
                left = view * args.frames
                output[row, left:left + args.frames] = values_numpy[index]
            if start % (args.batch_size * 20) == 0:
                print(json.dumps({'stage': 'cache', 'encoded': min(start + len(batch_records), len(records)), 'total': len(records)}), flush=True)
    output.flush()
    report = {'stage': 'LaViLa_frame_token_cache', 'status': 'complete', 'rows': len(rows), 'shape': list(output.shape), 'frames_per_view': args.frames, 'view_order': ['scene', 'person', 'workspace'], 'spatial_pool': 'mean over 14x14 patch tokens', 'checkpoint_frames': checkpoint_frames, 'checkpoint_epoch': int(checkpoint.get('epoch', -1)), 'checkpoint_sha256': checkpoint_sha256(args.checkpoint), 'labels_used': False, 'test_rows_loaded': 0, 'submission_generated': False}
    (args.output_dir / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
