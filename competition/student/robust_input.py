"""Trial-local input auditing and deterministic missing-modality fallbacks.

The frozen challenge model consumes visual, Skeleton and IMU tensors.  This module
does not alter complete trials.  For incomplete trials it creates only the
minimal neutral inputs needed by the existing builders: zero Skeleton frames,
IR-compatible grayscale frames from Depth/Thermal, and Depth timeline markers.
Radar is audited but is not presented to the frozen model because challenge has no
Radar encoder.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MODALITIES = ("Depth_Color", "IR", "Thermal", "IMU", "Radar", "Skeleton")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
MANIFEST_FIELDS = (
    "split", "sample_id", "official_path", "class_id", "class_name",
    "user_id", "trial_id", "depth_color_path", "ir_path", "thermal_path",
    "skeleton_path", "imu_path", "radar_path", "official_sample_id",
    "p46_ir_readable", "depth_color_usable", "ir_usable", "skeleton_usable",
    "depth_color_present", "ir_present", "skeleton_present",
)


def _inside(root: Path, candidate: Path) -> bool:
    return os.path.commonpath((str(root.resolve()), str(candidate.resolve()))) == str(root.resolve())


def resolve_sample(root: Path, official_path: str) -> Path:
    rel = Path(official_path.replace("\\", "/").rstrip("/"))
    candidates = [root / rel]
    if rel.parts and rel.parts[0] == root.name:
        candidates.append(root.joinpath(*rel.parts[1:]))
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_dir() and _inside(root, resolved):
            return resolved
    raise FileNotFoundError(f"test sample directory not found: {official_path}")


def _decode(path: Path) -> np.ndarray | None:
    try:
        content = np.fromfile(path, dtype=np.uint8)
        if not content.size:
            return None
        return cv2.imdecode(content, cv2.IMREAD_UNCHANGED)
    except (OSError, cv2.error):
        return None


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"could not encode fallback image: {path}")
    encoded.tofile(path)


def _canonical_image_id(path: Path, modality: str) -> str | None:
    stem = path.stem
    if modality == "IR" and stem.startswith("IR_"):
        return stem[3:]
    if modality == "Depth_Color" and stem.startswith("Depth_") and stem.endswith("_Color"):
        return stem[len("Depth_") : -len("_Color")]
    return None


def _image_inventory(path: Path, modality: str) -> tuple[list[tuple[str, Path, np.ndarray]], dict[str, Any]]:
    files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES) if path.is_dir() else []
    decoded: list[tuple[str, Path, np.ndarray]] = []
    invalid_names = unreadable = 0
    for index, file_path in enumerate(files):
        frame_id = _canonical_image_id(file_path, modality)
        if modality == "Thermal":
            frame_id = f"thermal_{index:08d}"
        if frame_id is None:
            invalid_names += 1
            continue
        image = _decode(file_path)
        if image is None:
            unreadable += 1
            continue
        decoded.append((frame_id, file_path, image))
    return decoded, {
        "present": path.is_dir(), "file_count": len(files), "usable_file_count": len(decoded),
        "invalid_name_count": invalid_names, "unreadable_file_count": unreadable,
        "usable": bool(decoded),
    }


def _skeleton_inventory(path: Path) -> tuple[list[tuple[str, Path]], dict[str, Any]]:
    base = path / "predictions" if (path / "predictions").is_dir() else path
    files = sorted(base.glob("*.json")) if base.is_dir() else []
    valid: list[tuple[str, Path]] = []
    invalid = 0
    for file_path in files:
        if not file_path.stem.startswith("Color_"):
            invalid += 1
            continue
        try:
            json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            invalid += 1
            continue
        valid.append((file_path.stem[len("Color_"):], file_path))
    return valid, {
        "present": path.is_dir(), "file_count": len(files), "usable_file_count": len(valid),
        "invalid_or_unreadable_file_count": invalid, "usable": bool(valid),
    }


def _imu_inventory(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    csv_files = sorted(path.glob("*.csv")) if path.is_dir() else []
    times: list[float] = []
    accepted = rejected = 0
    if path.is_dir():
        import sys
        legacy = Path(__file__).resolve().parent / "modules"
        if str(legacy) not in sys.path:
            sys.path.insert(0, str(legacy))
        from skeleton_imu_preprocessing import load_full_imu_trial
        try:
            by_device, audit = load_full_imu_trial(path)
            accepted = int(audit.accepted_rows)
            rejected = int(audit.rejected_rows + audit.unknown_device_rows)
            for point_times, _ in by_device.values():
                times.extend(np.asarray(point_times, dtype=np.float64).tolist())
        except Exception:
            rejected = max(rejected, 1)
    array = np.asarray(sorted(set(times)), dtype=np.float64)
    return array, {
        "present": path.is_dir(), "file_count": len(csv_files), "usable_file_count": len(csv_files) if len(array) else 0,
        "accepted_rows": accepted, "rejected_rows": rejected, "usable": bool(len(array)),
    }


def _generic_inventory(path: Path) -> dict[str, Any]:
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else []
    nonempty = sum(int(p.stat().st_size > 0) for p in files)
    return {"present": path.is_dir(), "file_count": len(files), "usable_file_count": nonempty, "usable": bool(nonempty)}


def _synthetic_ids(count: int, start: float = 946684800.0, end: float | None = None) -> list[str]:
    count = max(1, count)
    if end is None or end <= start:
        times = start + np.arange(count, dtype=np.float64) * 0.1
    else:
        times = np.linspace(start, end, count, dtype=np.float64)
    result = []
    for index, value in enumerate(times):
        stamp = datetime.fromtimestamp(float(value), tz=timezone.utc).strftime("%Y-%m-%d_%H-%M-%S.%f")
        result.append(f"{stamp}_{index:08d}")
    return result


def _make_skeleton(directory: Path, frame_ids: list[str], source_files: list[Path] | None = None) -> Path:
    predictions = directory / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    for index, frame_id in enumerate(frame_ids):
        target = predictions / f"Color_{frame_id}.json"
        if source_files:
            try:
                payload = source_files[min(index, len(source_files) - 1)].read_text(encoding="utf-8")
                json.loads(payload)
            except (OSError, json.JSONDecodeError, UnicodeError):
                payload = "[]"
        else:
            payload = "[]"
        target.write_text(payload, encoding="utf-8")
    return directory


def _make_ir(directory: Path, frames: list[tuple[str, Path, np.ndarray]], frame_ids: list[str] | None = None) -> tuple[Path, list[str]]:
    directory.mkdir(parents=True, exist_ok=True)
    ids = frame_ids or [item[0] for item in frames]
    if not frames:
        raise ValueError("visual fallback requires at least one readable frame")
    for index, frame_id in enumerate(ids):
        image = frames[min(index, len(frames) - 1)][2]
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.asarray(image, dtype=np.uint8)
        _write_png(directory / f"IR_{frame_id}.png", image)
    return directory, ids


def _make_depth_markers(directory: Path, frame_ids: list[str]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    marker = np.zeros((2, 2, 3), dtype=np.uint8)
    for frame_id in frame_ids:
        _write_png(directory / f"Depth_{frame_id}_Color.png", marker)
    return directory


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_manifest(path: Path, rows: list[dict[str, str]]) -> Path:
    return _write_manifest(path, rows)


def prepare_trials(
    rows: list[dict[str, str]], data_root: Path, manifest_path: Path, fallback_root: Path
) -> tuple[Path, list[dict[str, str]], list[dict[str, Any]]]:
    """Return a recoverable-only manifest plus one diagnostic record per input row."""
    root = data_root.resolve()
    prepared: list[dict[str, str]] = []
    diagnostics: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for index, row in enumerate(rows):
        official_path = row["path"]
        diagnostic: dict[str, Any] = {
            "index": index, "path": official_path, "sample_id": None,
            "modalities": {}, "available_modalities": [], "missing_modalities": [],
            "fallbacks": [], "timeline_source": None, "stage": "input_preflight",
            "status": "pending", "prediction": None, "classified": False,
            "unrecoverable_reason": None,
        }
        diagnostics.append(diagnostic)
        try:
            sample = resolve_sample(root, official_path)
            sid_base = sample.name
            sid = sid_base if sid_base not in used_ids else f"{sid_base}__{index:06d}"
            used_ids.add(sid)
            diagnostic["sample_id"] = sid

            actual: dict[str, str] = {}
            for name in MODALITIES:
                candidate = sample / name
                if candidate.is_dir():
                    resolved = candidate.resolve()
                    if not _inside(root, resolved):
                        raise ValueError(f"modality symlink escapes data root: {candidate}")
                    actual[name] = str(resolved)
                else:
                    actual[name] = ""

            ir_frames, ir_info = _image_inventory(Path(actual["IR"]) if actual["IR"] else Path("__missing__"), "IR")
            depth_frames, depth_info = _image_inventory(Path(actual["Depth_Color"]) if actual["Depth_Color"] else Path("__missing__"), "Depth_Color")
            thermal_frames, thermal_info = _image_inventory(Path(actual["Thermal"]) if actual["Thermal"] else Path("__missing__"), "Thermal")
            skeleton_frames, skeleton_info = _skeleton_inventory(Path(actual["Skeleton"]) if actual["Skeleton"] else Path("__missing__"))
            imu_times, imu_info = _imu_inventory(Path(actual["IMU"]) if actual["IMU"] else Path("__missing__"))
            radar_info = _generic_inventory(Path(actual["Radar"]) if actual["Radar"] else Path("__missing__"))
            infos = {"Depth_Color": depth_info, "IR": ir_info, "Thermal": thermal_info,
                     "IMU": imu_info, "Radar": radar_info, "Skeleton": skeleton_info}
            diagnostic["modalities"] = infos
            diagnostic["available_modalities"] = [name for name in MODALITIES if infos[name]["usable"]]
            diagnostic["missing_modalities"] = [name for name in MODALITIES if not infos[name]["usable"]]

            fallback = fallback_root / f"{index:06d}_{sid}"
            paths = dict(actual)
            skeleton_actual_usable = bool(skeleton_frames)
            ir_actual_usable = bool(ir_frames)
            depth_actual_usable = bool(depth_frames)
            thermal_actual_usable = bool(thermal_frames)
            imu_actual_usable = bool(len(imu_times))

            # Counter-only Skeleton without any IR acquisition grid needs a real,
            # deterministic time axis before it reaches the frozen motion builder.
            skeleton_ids = [item[0] for item in skeleton_frames]
            counter_only_without_ir = bool(skeleton_ids) and all(value.isdigit() for value in skeleton_ids) and not actual["IR"]
            need_skeleton = not skeleton_actual_usable or counter_only_without_ir
            timeline_ids: list[str] = []
            timeline_source: str | None = None
            visual_source: list[tuple[str, Path, np.ndarray]] = []

            if ir_actual_usable:
                timeline_ids = [item[0] for item in ir_frames]
                timeline_source = "IR"
                visual_source = ir_frames
            elif depth_actual_usable:
                timeline_ids = [item[0] for item in depth_frames]
                timeline_source = "Depth_Color"
                visual_source = depth_frames
            elif thermal_actual_usable:
                timeline_ids = _synthetic_ids(len(thermal_frames))
                timeline_source = "Thermal"
                visual_source = thermal_frames
            elif skeleton_actual_usable and not counter_only_without_ir:
                timeline_ids = skeleton_ids
                timeline_source = "Skeleton"
            elif imu_actual_usable:
                timeline_ids = _synthetic_ids(min(max(len(imu_times) // 20, 16), 64), float(imu_times[0]), float(imu_times[-1]))
                timeline_source = "IMU"

            if need_skeleton:
                if not timeline_ids:
                    raise RuntimeError("no usable Skeleton, visual or IMU timeline")
                source_files = [item[1] for item in skeleton_frames] if skeleton_frames else None
                paths["Skeleton"] = str(_make_skeleton(fallback / "Skeleton", timeline_ids, source_files).resolve())
                diagnostic["fallbacks"].append("zero_or_normalized_skeleton_with_recovered_timeline")

            # Only activate a cross-modal visual fallback when Skeleton itself had
            # to be recovered.  This preserves the historical challenge behaviour for
            # every original complete/known-corrupt Kaggle trial.
            generated_ir = False
            if need_skeleton and not ir_actual_usable and visual_source:
                paths["IR"], timeline_ids = _make_ir(fallback / "IR", visual_source, timeline_ids)
                paths["IR"] = str(paths["IR"].resolve())
                generated_ir = True
                diagnostic["fallbacks"].append(f"IR_compatible_grayscale_from_{timeline_source}")
            if need_skeleton and (ir_actual_usable or generated_ir) and not depth_actual_usable:
                paths["Depth_Color"] = str(_make_depth_markers(fallback / "Depth_Color", timeline_ids).resolve())
                diagnostic["fallbacks"].append("depth_timeline_markers")

            skeleton_ready = bool(paths["Skeleton"]) and (skeleton_actual_usable or need_skeleton)
            compatible_signal = skeleton_actual_usable or ir_actual_usable or (
                need_skeleton and (depth_actual_usable or thermal_actual_usable)
            ) or imu_actual_usable
            if not skeleton_ready or not compatible_signal:
                raise RuntimeError("no modality can form a valid frozen-challenge input")

            ir_path = paths["IR"]
            ir_usable = generated_ir or (bool(ir_path) and bool(ir_frames) and ir_info["unreadable_file_count"] == 0 and ir_info["invalid_name_count"] == 0)
            depth_usable = bool(paths["Depth_Color"])
            skeleton_usable = bool(paths["Skeleton"])
            manifest_row = {
                "split": "test", "sample_id": f"test/anonymous/{sid}",
                "official_path": official_path, "class_id": "-1", "class_name": "",
                "user_id": "anonymous", "trial_id": sid,
                "depth_color_path": paths["Depth_Color"], "ir_path": ir_path,
                "thermal_path": paths["Thermal"], "skeleton_path": paths["Skeleton"],
                "imu_path": paths["IMU"], "radar_path": paths["Radar"],
                "official_sample_id": sid, "p46_ir_readable": str(int(ir_usable)),
                "depth_color_usable": str(int(depth_usable)), "ir_usable": str(int(ir_usable)),
                "skeleton_usable": str(int(skeleton_usable)),
                "depth_color_present": str(int(bool(actual["Depth_Color"]))),
                "ir_present": str(int(bool(actual["IR"]))),
                "skeleton_present": str(int(bool(actual["Skeleton"]))),
            }
            prepared.append(manifest_row)
            diagnostic["timeline_source"] = timeline_source or "Skeleton"
            diagnostic["stage"] = "prepared"
            diagnostic["status"] = "prepared"
        except Exception as exc:
            diagnostic["status"] = "unrecoverable"
            diagnostic["unrecoverable_reason"] = f"{type(exc).__name__}: {exc}"
    return _write_manifest(manifest_path, prepared), prepared, diagnostics
