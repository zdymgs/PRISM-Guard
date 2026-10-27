"""Target construction and smoothing for the original dataset-native paths."""

from pathlib import Path
import hashlib
import json
import numpy as np
from .data.io import load_unlabeled_bundle
from .targets import smooth_targets
from .utils import write_config_snapshot, canonical_config_hash
from .chain import check_classes, construct, target_clusters


def build_targets(config, path="direct"):
    """Construct a requested native path and Eq. (11) smoothed targets."""
    root = Path(config["output_root"])
    value = load_unlabeled_bundle(Path(config["posterior_input"]))
    check_classes(value, config["classes"])
    native = None
    if path == "direct" and "expert_bank" not in value:
        chosen = np.asarray(value["direct_probability"]).argmax(1)
        method, scope = "direct", "DIRECT"
    else:
        native = construct(config, value)
        method = native["selected_path"] if path == "auto" else path
        chosen = native["predictions"]["selected" if path == "auto" and "selected" in native["predictions"] else method]
        scope = "dataset-native chain"
    write_config_snapshot(config, root)
    q = smooth_targets(chosen, config["classes"], float(config.get("eta", 0.0005)))
    result = {
        "method": method,
        "scope": scope,
        "record_id": value["record_id"],
        "class_count": config["classes"],
        "prediction": chosen.tolist(),
        "targets": q.tolist(),
        "clusters": target_clusters(config, value) if native is not None else value.get("session", value["record_id"]),
    }
    if native is not None:
        result["available_paths"] = list(native["predictions"])
        result["selected_path"] = native["selected_path"]
        result["selection_records"] = native["selection_records"]
        result["gates"] = native["gates"]
    path = root / "targets.json"
    raw = json.dumps(result, allow_nan=False)
    if path.exists() and path.read_text(encoding="utf-8") != raw:
        raise FileExistsError("targets already frozen")
    path.write_text(raw, encoding="utf-8")
    freeze = {
        "config_hash": canonical_config_hash(config),
        "targets_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "evaluation_labels_read": False,
        "scope": scope,
    }
    (root / "target_freeze.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    return result
