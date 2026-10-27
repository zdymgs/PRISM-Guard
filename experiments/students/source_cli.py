"""Argument parser for source initialization; runtime imports follow parsing."""
import argparse
from pathlib import Path

def parse_args(argv=None):
    p=argparse.ArgumentParser(description="Train a source-side student initialization stage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="CUHK-X source: seed 20260811, visual16 + motion24 + fusion4/20. "
               "MMAct: seed20260922, visual16 + motion24 + fusion4/20. "
               "MM-Fit: seed20260927, visual2 + motion24 + fusion24. "
               "Source labels and source-fitted teacher tensors are supplied separately.")
    p.add_argument("--dataset",choices=("cuhkx","mmact","mmfit"),required=True)
    p.add_argument("--stage",choices=("visual","motion","fusion"),required=True)
    p.add_argument("--inputs",type=Path,required=True)
    p.add_argument("--labels",type=Path,required=True,help="source-only record_id,label JSON")
    p.add_argument("--validation-inputs",type=Path)
    p.add_argument("--validation-labels",type=Path)
    p.add_argument("--pretrained-weights",type=Path,help="local MC3 Kinetics state for visual stage")
    p.add_argument("--visual-checkpoint",type=Path)
    p.add_argument("--motion-checkpoint",type=Path)
    p.add_argument("--cache-root",type=Path)
    p.add_argument("--fold",type=int,choices=(0,1,2))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda")
    args=p.parse_args(argv)
    if args.stage=="visual" and args.pretrained_weights is None:p.error("visual stage requires --pretrained-weights")
    if args.stage=="fusion" and (args.visual_checkpoint is None or args.motion_checkpoint is None):p.error("fusion requires both source checkpoints")
    if (args.validation_inputs is None)!=(args.validation_labels is None):p.error("validation inputs and labels are paired")
    if args.dataset=="mmact" and args.validation_inputs is None:p.error("MMAct source stages require the F1 selection manifest and labels")
    if args.dataset=="mmfit" and args.stage=="fusion" and args.validation_inputs is None:p.error("MM-Fit fusion requires source validation")
    if args.dataset=="cuhkx" and args.fold is None:p.error("CUHK-X source training requires --fold")
    return args

def main(argv=None):
    args=parse_args(argv)
    from .source import run_source
    run_source(args)
