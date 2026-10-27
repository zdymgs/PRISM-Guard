"""Adapters for copied preprocessing builders; supports any N test rows."""
from __future__ import annotations
import importlib
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE / "modules") not in sys.path: sys.path.insert(0, str(HERE / "modules"))

def run(module_name: str, argv: list[str], manifest: Path | None = None) -> None:
    mod = importlib.import_module(module_name)
    # The copied canonicalizer strips timestamps from some Skeleton filenames.
    # Recover the actual timestamped filename before aligning sensors.
    if hasattr(mod, "frame_map"):
        original_frame_map = mod.frame_map
        def frame_map_compatible(path, modality):
            raw = original_frame_map(path, modality)
            if modality != "skeleton" or not raw:
                return raw
            # Preserve the historical shared IR acquisition grid, including when
            # a Skeleton directory has additional frames not recorded in IR.
            sibling = Path(path).parent / "IR"
            try:
                ir = original_frame_map(sibling, "ir")
            except FileNotFoundError:
                ir = {}
            if ir:
                by_suffix = {key.rsplit("_", 1)[-1]: key for key in ir}
                if len(by_suffix) != len(ir):
                    raise ValueError('Ambiguous IR acquisition counters')
                mapped = {by_suffix[key.rsplit("_", 1)[-1]]: value
                          for key, value in raw.items()
                          if key.rsplit("_", 1)[-1] in by_suffix}
                if mapped and all(not key.isdigit() for key in mapped):
                    return mapped
            # Without IR, use real Skeleton timestamps, never a synthetic time.
            restored = {value.stem[len('Color_'):]: value for value in raw.values()
                        if value.stem.startswith('Color_')}
            if len(restored) == len(raw) and all(not key.isdigit() for key in restored):
                return restored
            if all(not key.isdigit() for key in raw):
                return raw
            raise RuntimeError(f'Counter-only Skeleton lacks an absolute timeline: {path}')
        mod.frame_map = frame_map_compatible
    if manifest is not None and hasattr(mod, "read_manifest"):
        def read_any(path):
            with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if not rows: raise RuntimeError("empty manifest")
            return rows
        mod.read_manifest = read_any
    old = sys.argv
    try:
        sys.argv = [module_name, *map(str, argv)]
        mod.main()
    finally:
        sys.argv = old
