"""The sole evaluation-truth reader for the normalized public interface."""

from pathlib import Path
import json
import numpy as np


def load_eval_labels(path: Path, output_root: Path):
    """Audit before access and reject a second pass even after failure."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "label_access_started.json").open("x", encoding="utf-8") as handle:
        json.dump({"event": "independent_evaluation_label_read", "loader_calls": 1}, handle)
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(value) != {"record_id", "label"}:
        raise ValueError("truth schema must be record_id,label")
    ids = np.asarray(value["record_id"]).astype(str)
    labels = np.asarray(value["label"])
    if (
        labels.ndim != 1
        or labels.shape != ids.shape
        or not np.issubdtype(labels.dtype, np.integer)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("invalid truth identities")
    return ids, labels
