"""Dependency-light argument parsing for native student entry points."""

import argparse
import os
from pathlib import Path


def parser(description, *, baseline=False):
    p = argparse.ArgumentParser(description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Paper adaptation: batch 64; AdamW wd 0.02; motion/fusion lr 1e-4; "
               "visual-head lr 5e-5; minimum lr 5e-6; temperature 1; gradient clip 1. "
               "Default seeds: CUHK-X 20260826, MMAct 20260922, MM-Fit 20260927.")
    p.add_argument("--dataset", choices=("cuhkx", "mmact", "mmfit"), required=True)
    p.add_argument("--inputs", type=Path, required=True, help="prism.student.v1 tensor manifest")
    p.add_argument("--targets", type=Path, required=True,
                   help="record_id + direct_probability JSON" if baseline else
                        "record_id + targets JSON emitted by scripts/build_targets.py")
    p.add_argument("--c0", type=Path, required=True, help="local common source checkpoint")
    p.add_argument("--output", type=Path,
                   default=Path(os.environ["PRISM_OUTPUT_ROOT"]) if os.environ.get("PRISM_OUTPUT_ROOT") else None,
                   required=not bool(os.environ.get("PRISM_OUTPUT_ROOT")), help="empty output directory")
    p.add_argument("--cache-root", type=Path, default=os.environ.get("PRISM_CACHE_ROOT"),
                   help="relative tensor paths resolve here, otherwise beside the input manifest")
    p.add_argument("--fold", type=int, choices=(0,1,2), help="required for CUHK-X; 2 denotes H2")
    p.add_argument("--seed", type=int, help="dataset's original seed when omitted")
    p.add_argument("--epochs", type=int, default=40, help="adaptation epochs")
    p.add_argument("--device", default="cuda", help="PyTorch device")
    if baseline:
        p.add_argument("--method", choices=("kd-soft", "confidence-pl"), required=True)
    else:
        p.set_defaults(method="prism-d")
    return p


def main(argv=None, *, baseline=False):
    args = parser("Run the paper's native student baseline." if baseline else
                  "Distill PRISM targets into the full native student.", baseline=baseline).parse_args(argv)
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.dataset == "cuhkx" and args.fold is None:
        raise ValueError("CUHK-X requires --fold 0, 1 or 2")
    if baseline:
        from experiments.baselines.training import run
        return run(args)
    if args.dataset != "cuhkx":
        from experiments.baselines.training import run
        return run(args, distill=True)
    from .distill import run_cuhkx
    return run_cuhkx(args)
