from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import argparse
from pathlib import Path
ROOT = WORK_ROOT
from session_kd.mmfit_rgb_benchmark_cache import build_input_cache
from session_kd._mmfit_windows import ALL_WORKOUTS

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/MM-Fit/mm-fit')
    parser.add_argument('--rgb-root', type=Path, default=ROOT / 'data/MM-Fit/rgb')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=Path.home() / '.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')
    parser.add_argument('--workouts', nargs='*', default=list(ALL_WORKOUTS))
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 128:
        raise ValueError('batch-size must be 1..128')
    build_input_cache(args.data_root, args.rgb_root, args.output, args.weights, tuple(args.workouts), args.batch_size)
