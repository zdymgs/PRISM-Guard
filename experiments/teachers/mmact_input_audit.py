"""Validate the complete MMAct native-input arrays before teacher fitting."""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent / "modules"))
    from scripts.audit_mmact_input_arrays import audit
    audit(args.cache_root, args.input_manifest, args.output)


if __name__ == "__main__":
    main()
