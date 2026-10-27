"""Run a paper teacher or feature-cache command with local resources."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent


def main():
    entries = json.loads((HERE / "entries.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entry", choices=sorted(entries))
    parser.add_argument("--work-root", type=Path, required=True,
                        help="User-owned teacher workspace with data manifests and generated caches.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, default=Path("external"))
    raw = sys.argv[1:]
    split = raw.index("--") if "--" in raw else len(raw)
    args = parser.parse_args(raw[:split])
    arguments = raw[split + 1:]
    for key, value in (("WORK", args.work_root), ("DATA", args.data_root),
                       ("PRETRAINED", args.pretrained_root), ("EXTERNAL", args.external_root)):
        os.environ["PRISM_TEACHER_WORK_ROOT" if key == "WORK" else "PRISM_" + key + "_ROOT"] = str(value.resolve())
    native = HERE / "modules"
    sys.path.insert(0, str(native))
    os.environ["PYTHONPATH"] = str(native) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    from _release_paths import configure_local_resources
    configure_local_resources()
    sys.argv = [args.entry, *arguments]
    importlib.import_module(entries[args.entry]).main()


if __name__ == "__main__":
    main()
