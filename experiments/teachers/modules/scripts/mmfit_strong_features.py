from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import argparse
import json
from pathlib import Path
ROOT = WORK_ROOT
from session_kd.mmfit_strong_features import MODEL_ORDER, build_cache, merge_model_cache, pilot_model

def main() -> None:
    parser = argparse.ArgumentParser(description='Build label-firewalled MMFit strong RGB features')
    parser.add_argument('--stage', choices=('pilot', 'cache', 'both', 'merge'), default='both')
    parser.add_argument('--models', nargs='+', choices=MODEL_ORDER, default=list(MODEL_ORDER))
    parser.add_argument('--inputs', type=Path, default=ROOT / 'runs/mmfit_rgb_benchmark/inputs.csv')
    parser.add_argument('--rgb-root', type=Path, default=ROOT / 'data/MM-Fit/rgb')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/mmfit_strong_features')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--pilot-count', type=int, default=32)
    parser.add_argument('--pilot-workout', default='w01')
    parser.add_argument('--merge-source', type=Path)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 32:
        raise ValueError('batch-size must be 1..32')
    if args.pilot_count < 16:
        raise ValueError('pilot-count must be at least 16')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'merge':
        if args.merge_source is None or len(args.models) != 1:
            raise ValueError('merge requires --merge-source and exactly one --models value')
        result = merge_model_cache(args.output, args.merge_source, args.models[0])
        print(json.dumps({'status': result['status'], 'ready': result['ready']}), flush=True)
        return
    for model in args.models:
        if args.stage in ('pilot', 'both'):
            report = pilot_model(model, args.inputs, args.rgb_root, args.batch_size, args.pilot_count, args.pilot_workout)
            path = args.output / f'pilot_{model}.json'
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            print(json.dumps(report, ensure_ascii=False), flush=True)
            if report['estimated_full_hours_at_pilot_rate'] > 3.0:
                raise RuntimeError(f'{model} pilot extrapolation exceeds 3 hours')
        if args.stage in ('cache', 'both'):
            print(json.dumps({'purpose': f'cache frozen 16-frame {model} features for all V1 IDs'}), flush=True)
            result = build_cache(model, args.inputs, args.rgb_root, args.output, args.batch_size)
            print(json.dumps(result, ensure_ascii=False), flush=True)
