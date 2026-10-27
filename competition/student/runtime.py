"""Fresh-data challenge runtime.

This module deliberately owns no predictions or test caches.  ``bundle.load_bundle``
is the only model input; preprocessing is performed in a newly-created work dir.
The preprocessing builders in ``legacy`` are invoked
as subprocesses so their relative imports cannot accidentally resolve old caches.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
# Disable Ultralytics online checks/telemetry before any legacy import.
os.environ['YOLO_OFFLINE'] = 'true'
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import PurePosixPath, PureWindowsPath
import cv2
import numpy as np
from pathlib import Path
from typing import Iterable

HERE = Path(__file__).resolve().parent
MODULES = HERE / "modules"
from robust_input import prepare_trials, write_manifest as write_prepared_manifest


def image_is_readable(path: Path) -> bool:
    """Decode bytes rather than passing a Unicode filename to Windows OpenCV."""
    try:
        content = np.fromfile(path, dtype=np.uint8)
        return bool(content.size) and cv2.imdecode(content, cv2.IMREAD_UNCHANGED) is not None
    except (OSError, cv2.error):
        return False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_test_csv(path: Path) -> list[dict[str, str]]:
    """Read official submission input without accepting labels or predictions."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or set(fields) not in ({"path"}, {"path", "prediction"}):
            raise ValueError("test CSV must have path and optionally empty prediction; extra/duplicate columns forbidden")
        rows = list(reader)
    if not rows or any(set(row) != set(fields) for row in rows):
        raise ValueError("test CSV must have path and optionally empty prediction; extra columns forbidden")
    if any(row.get("prediction", "") is None or row.get("prediction", "").strip() for row in rows):
        raise ValueError("input test CSV must not contain populated predictions")
    paths = [row["path"] or "" for row in rows]
    if any(not p.strip() or p != p.strip() or ".." in PurePosixPath(p.replace('\\', '/')).parts for p in paths):
        raise ValueError("invalid or traversal path in test CSV")
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate test paths")
    return rows


def _resolve_sample(root: Path, official_path: str) -> Path:
    rel = Path(official_path.replace('\\', '/').rstrip("/"))
    if rel.is_absolute() or PureWindowsPath(official_path).drive or '..' in rel.parts:
        raise ValueError(f"absolute test path is forbidden: {official_path}")
    candidates = [root / rel]
    if rel.parts and rel.parts[0] == root.name:
        candidates.append(root.joinpath(*rel.parts[1:]))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir() and os.path.commonpath((str(root.resolve()), str(resolved))) == str(root.resolve()):
            return resolved
    raise FileNotFoundError(f"test sample directory not found: {official_path}")


def _run(script: str, args: Iterable[str], cwd: Path) -> None:
    path = MODULES / script
    if not path.is_file():
        raise FileNotFoundError(f"missing packaged builder: {path}")
    cmd = [sys.executable, str(path), *map(str, args)]
    subprocess.run(cmd, cwd=str(cwd), check=True)


class PipelineStageError(RuntimeError):
    def __init__(self, stage: str, error: BaseException) -> None:
        self.stage = stage
        self.original = error
        super().__init__(f"{stage}: {type(error).__name__}: {error}")


def _pipeline(
    manifest: Path,
    manifest_rows: list[dict[str, str]],
    output: Path,
    student_path: Path,
    yolo_path: Path,
    args: argparse.Namespace,
    torch_module,
) -> tuple[Path, dict[str, object]]:
    """Run one recoverable batch; errors retain the responsible stage name."""
    output.mkdir(parents=True, exist_ok=True)
    from stage_runner import run as stage

    def guarded(name: str, function, *function_args) -> None:
        try:
            function(*function_args)
        except Exception as exc:
            raise PipelineStageError(name, exc) from exc

    pose_rows = [
        row for row in manifest_rows
        if row["depth_color_usable"] == "1"
        and row["ir_usable"] == "1"
        and row["skeleton_usable"] == "1"
    ]
    if pose_rows:
        guarded(
            "pose_cache",
            stage,
            "build_adaptive_yolo11_pose_skeleton_cache",
            ["--manifest", manifest, "--model", yolo_path, "--output-dir", output / "pose",
             "--device", ("cpu" if args.device == "cpu" or (args.device == "auto" and not torch_module.cuda.is_available()) else "0")],
        )
        guarded(
            "roi_geometry",
            stage,
            "build_multiscale_dir_rois",
            ["--manifest", manifest, "--pose-run", output / "pose", "--output-dir", output / "roi"],
        )
    else:
        (output / "roi").mkdir(parents=True, exist_ok=True)

    guarded(
        "pixel_cache", stage, "build_test_pixel_cache",
        ["--manifest", manifest, "--roi-run", output / "roi", "--output-dir", output / "pixels"], manifest,
    )
    guarded(
        "motion_source_cache", stage, "build_test_motion_source_cache",
        ["--manifest", manifest, "--roi-run", output / "roi", "--output-dir", output / "motion_source"], manifest,
    )
    guarded(
        "motion_window_cache", _run, "build_motion_window_cache.py",
        ["--pixel-cache", output / "pixels", "--motion-source-run", output / "motion_source", "--output-dir", output / "motion"], HERE,
    )
    infer_args = [
        "--checkpoint", student_path, "--sequence-cache", output / "sequence",
        "--motion-cache", output / "motion", "--pixel-cache", output / "pixels",
        "--manifest", manifest, "--output", output / "submission.csv",
        "--batch-size", args.batch_size, "--device", args.device,
    ]
    guarded(
        "student_forward", _run_student,
        [sys.executable, str(HERE / "infer_student.py"), *map(str, infer_args)],
    )
    metrics_path = output / "inference_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    return output / "submission.csv", metrics


def _run_student(command: list[str]) -> None:
    subprocess.run(command, cwd=str(HERE), check=True)


def _read_predictions(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, int] = {}
    for row in rows:
        prediction = int(row["prediction"])
        if prediction < 0 or prediction > 39 or row["path"] in result:
            raise RuntimeError(f"invalid or duplicate prediction row: {row}")
        result[row["path"]] = prediction
    return result


def _write_submission(path: Path, rows: list[dict[str, str]], predictions: dict[str, int]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("path", "prediction"))
        for row in rows:
            if row["path"] in predictions:
                writer.writerow((row["path"], predictions[row["path"]]))


def run(args: argparse.Namespace) -> Path:
    # bundle.py is supplied by the packager; no fallback to a historical checkpoint.
    started = time.perf_counter()
    if any((args.output / name).exists() for name in ('submission.csv', 'report.json', 'diagnostics.json')):
        raise FileExistsError('Use a new output directory; prior results will not be overwritten')
    from bundle import load_bundle  # type: ignore
    student, yolo_bytes = load_bundle(args.bundle)
    if not isinstance(student, dict) or not isinstance(yolo_bytes, (bytes, bytearray)):
        raise TypeError("load_bundle must return (student checkpoint dict, YOLO bytes)")
    rows = read_test_csv(args.test_csv)
    root = args.data_root.resolve()
    work = Path(tempfile.mkdtemp(prefix="cuhkx_inference_", dir=str(args.output.resolve())))
    try:
        # Isolate library settings/caches from the developer's home directory.
        settings = work / 'library_config'
        settings.mkdir()
        os.environ['YOLO_CONFIG_DIR'] = str(settings)
        os.environ['HF_HOME'] = str(settings / 'huggingface')
        os.environ['TORCH_HOME'] = str(settings / 'torch')
        os.environ['HF_HUB_OFFLINE'] = '1'
        import torch
        student['model_state'] = {k: v.half() if v.is_floating_point() else v
                                  for k, v in student['model_state'].items()}
        student_path = work / "student.pt"
        yolo_path = work / "yolo11n-pose.pt"
        torch.save(student, student_path)
        yolo_path.write_bytes(bytes(yolo_bytes))
        manifest, prepared_rows, diagnostics = prepare_trials(
            rows, root, work / "manifest.csv", work / "fallbacks"
        )

        diagnostic_by_path = {item["path"]: item for item in diagnostics}
        predictions: dict[str, int] = {}
        batch_metrics: list[dict[str, object]] = []
        isolation_runs = 0
        full_batch_output: Path | None = None

        def process(batch_rows: list[dict[str, str]], label: str) -> None:
            nonlocal isolation_runs, full_batch_output
            if not batch_rows:
                return
            isolation_runs += 1
            batch_dir = work / "batches" / f"{isolation_runs:04d}_{label}"
            batch_manifest = write_prepared_manifest(batch_dir / "manifest.csv", batch_rows)
            try:
                submission, metrics = _pipeline(
                    batch_manifest, batch_rows, batch_dir, student_path, yolo_path, args, torch
                )
                try:
                    batch_predictions = _read_predictions(submission)
                except Exception as exc:
                    raise PipelineStageError("output_contract", exc) from exc
                expected = {row["official_path"] for row in batch_rows}
                if set(batch_predictions) != expected:
                    raise PipelineStageError("output_contract", RuntimeError("batch output paths differ from manifest"))
                predictions.update(batch_predictions)
                batch_metrics.append({"label": label, "rows": len(batch_rows), **metrics})
                if len(batch_rows) == len(prepared_rows) and len(prepared_rows) == len(rows):
                    full_batch_output = submission
                for manifest_row in batch_rows:
                    item = diagnostic_by_path[manifest_row["official_path"]]
                    item.update({
                        "status": "predicted", "stage": "complete", "classified": True,
                        "prediction": batch_predictions[manifest_row["official_path"]],
                        "used_remaining_modalities": bool(item["missing_modalities"] or item["fallbacks"]),
                    })
            except PipelineStageError as exc:
                if len(batch_rows) == 1:
                    manifest_row = batch_rows[0]
                    item = diagnostic_by_path[manifest_row["official_path"]]
                    item.update({
                        "status": "unrecoverable", "stage": exc.stage, "classified": False,
                        "unrecoverable_reason": str(exc),
                    })
                    return
                middle = len(batch_rows) // 2
                process(batch_rows[:middle], label + "a")
                process(batch_rows[middle:], label + "b")

        process(prepared_rows, "all")
        final = args.output.resolve() / "submission.csv"
        if full_batch_output is not None and len(predictions) == len(rows):
            shutil.copy2(full_batch_output, final)
        else:
            _write_submission(final, rows, predictions)

        failure_count = sum(item["status"] == "unrecoverable" for item in diagnostics)
        diagnostics_payload = {
            "input_rows": len(rows), "predicted_rows": len(predictions),
            "unrecoverable_rows": failure_count, "all_input_rows_accounted_for": len(diagnostics) == len(rows),
            "policy": "normal missing modalities are neutralized/masked; unrecoverable trials are recorded and omitted from submission.csv",
            "trials": diagnostics,
        }
        (args.output.resolve() / "diagnostics.json").write_text(
            json.dumps(diagnostics_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (args.output.resolve() / "report.json").write_text(json.dumps({
            "rows": len(rows), "predicted_rows": len(predictions), "unrecoverable_rows": failure_count,
            "partial": bool(args.max_rows or failure_count), "labels_read": False,
            "history": "fresh raw preprocessing, trial-local missing-modality recovery and frozen Student forward",
            "bundle_sha256": _sha256(args.bundle), "input_csv_sha256": _sha256(args.test_csv),
            "manifest_sha256": _sha256(manifest), "submission_sha256": _sha256(final),
            "elapsed_seconds": time.perf_counter() - started,
            "device_requested": args.device,
            "torch": torch.__version__, "numpy": __import__('numpy').__version__
            , "student_metrics": batch_metrics, "isolation_runs": isolation_runs
            , "python_executable": sys.executable, "library_caches_isolated": True
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return final
    finally:
        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Run fixed challenge inference from user-supplied raw trials and a deployment bundle.")
    p.add_argument("--bundle", type=Path, default=os.environ.get("PRISM_COMPETITION_BUNDLE"),
                   required=not bool(os.environ.get("PRISM_COMPETITION_BUNDLE")),
                   help="User-supplied challenge single-checkpoint bundle; alternatively set PRISM_COMPETITION_BUNDLE")
    p.add_argument("--test-csv", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--keep-work", action="store_true")
    p.add_argument("--max-rows", type=int, default=0, help="smoke limit; never use for official submission")
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    if a.batch_size < 1 or a.max_rows < 0:
        p.error('batch-size must be positive and max-rows nonnegative')
    if a.max_rows:
        original = read_test_csv
        def limited(path): return original(path)[:a.max_rows]
        globals()["read_test_csv"] = limited
    print(run(a))


if __name__ == "__main__":
    main()
