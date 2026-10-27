"""Extract the four frozen physical teacher feature caches from local models."""
import argparse
import importlib
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backbone", choices=("ir-videomaev2", "ir-internvideo2", "depth-videomaev2", "thermal-videomaev2"))
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--pretrained-root", required=True, type=Path)
    parser.add_argument("--external-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trial-batch", type=int,
                        help="Default 8 for InternVideo2 and 16 for VideoMAEv2.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=40)
    parser.add_argument("--reuse-cache", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    for key, path in (("TEACHER_WORK", args.work_root), ("PRETRAINED", args.pretrained_root), ("EXTERNAL", args.external_root)):
        os.environ["PRISM_" + key + "_ROOT"] = str(path.resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parent / "modules"))
    from _release_paths import configure_local_resources
    configure_local_resources()
    name = ("internvideo2_l_teacher" if args.backbone == "ir-internvideo2" else
            "videomaev2_distilled_teacher" if args.backbone == "ir-videomaev2" else
            "videomaev2_modality_teacher")
    args.output_dir = args.output
    args.trial_batch = args.trial_batch or (8 if args.backbone == "ir-internvideo2" else 16)
    args.modality = args.backbone.split("-", 1)[0]
    args.output.mkdir(parents=True, exist_ok=True)
    module = importlib.import_module("aligned_multimodal." + name)
    module.extract(args, args.output / "complete_features.npz")


if __name__ == "__main__":
    main()
