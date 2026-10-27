"""Check that formal analysis follows the frozen source-selected predictions."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_analysis_uses_supplied_selected_predictions(tmp_path):
    ids = np.array([f"synthetic-{i}" for i in range(6)])
    labels = np.tile([0, 1], 3)
    arrays = {
        "evaluation_ids": ids,
        "evaluation_outer_fold": np.repeat(np.arange(3), 2),
        "evaluation_clusters": np.repeat(["a", "b", "c"], 2),
    }
    for name in ("direct", "selected", "equal_mean", "group", "sequence", "direct_sequence",
                 "direct_viterbi_naive", "online_forward"):
        arrays["evaluation_prediction__" + name] = labels if name in {"direct", "selected"} else 1 - labels
    predictions = tmp_path / "predictions.npz"
    np.savez(predictions, **arrays)
    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "class_id", "user_id"])
        writer.writerows((name, int(label), "synthetic") for name, label in zip(ids, labels))
    manifest = tmp_path / "classes.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_label"])
        writer.writerows((i, f"Class_{i}") for i in range(40))
    output = tmp_path / "analysis.json"
    script = Path(__file__).resolve().parents[1] / "experiments/analysis/analysis_stats.py"
    subprocess.run([sys.executable, str(script), str(predictions), str(metadata), str(manifest), str(output)],
                   check=True, capture_output=True, text=True)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["check"]["selected"] == result["check"]["direct"]
    assert all(row["delta"] == 0 for row in result["A2_per_class"])
    for fold in result["A4_rescue_harm"].values():
        assert fold["selected"] == {"rescue": 0, "harm": 0, "changed": 0}
