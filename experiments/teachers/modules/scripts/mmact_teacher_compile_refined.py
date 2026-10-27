from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import argparse
import json
import sys
from pathlib import Path
from session_kd.mmact_teacher_compile_refined import compile

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--input-manifest', type=Path, required=True)
    p.add_argument('--merged-manifest', type=Path, required=True)
    p.add_argument('--base-selection', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    print(json.dumps(compile(a.bank, a.geometry, a.input_manifest, a.merged_manifest, a.base_selection, a.output, a.threads), indent=2))
