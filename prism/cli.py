"""Command-line interfaces for target construction, evaluation and synthetic checks."""

from pathlib import Path
import argparse
import hashlib
import json
import os
import numpy as np
import yaml
from .utils import canonical_config_hash


def read_config(path, overrides=None):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("data_root", "cache_root", "output_root"):
        value[key] = os.environ.get("PRISM_" + key.upper(), value.get(key, "../prism-output"))
    if overrides:
        value.update({key: item for key, item in overrides.items() if item is not None})
    if "posterior_input" not in value:
        value["posterior_input"] = str(Path(value["cache_root"]) / "posterior.json")
    value.setdefault("source_input", str(Path(value["cache_root"]) / "source.json"))
    value.setdefault("selection_input", str(Path(value["cache_root"]) / "selection.json"))
    root = Path(__file__).resolve().parents[1]
    if Path(value["output_root"]).resolve().is_relative_to(root):
        raise ValueError("output_root must be outside the source repository")
    return value


def _input_arguments(parser):
    for name in ("data-root", "cache-root", "output-root", "source-input", "selection-input", "posterior-input"):
        parser.add_argument("--" + name, help="Override the configured " + name.replace("-", " ") + ".")


def _input_overrides(args):
    return {name: getattr(args, name) for name in (
        "data_root", "cache_root", "output_root", "source_input", "selection_input", "posterior_input"
    )}


def build_main():
    from .pipeline import build_targets
    from .chain import PATHS

    p = argparse.ArgumentParser(description="Build smoothed targets with a dataset-native path.")
    p.add_argument("--config", required=True)
    p.add_argument("--path", choices=PATHS, default="direct")
    _input_arguments(p)
    a = p.parse_args()
    result = build_targets(read_config(a.config, _input_overrides(a)), path=a.path)
    print(
        json.dumps(
            {
                "rows": len(result["record_id"]),
                "method": result["method"],
                "scope": result["scope"],
            }
        )
    )


def evaluate_main():
    from .eval.io import load_eval_labels
    from .eval.metrics import metrics, paired
    from .eval.bootstrap import cluster_bootstrap

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--reference-predictions", required=True)
    _input_arguments(p)
    a = p.parse_args()
    c = read_config(a.config, _input_overrides(a))
    root = Path(c["output_root"])
    freeze = json.loads((root / "target_freeze.json").read_text())
    if (
        canonical_config_hash(c) != freeze["config_hash"]
        or (root / "config_hash.txt").read_text().strip() != freeze["config_hash"]
    ):
        raise ValueError("configuration hash changed before evaluation")
    if hashlib.sha256((root / "targets.json").read_bytes()).hexdigest() != freeze["targets_sha256"]:
        raise ValueError("prediction artifact changed")
    predictions = json.loads((root / "targets.json").read_text())
    ref = json.loads(Path(a.reference_predictions).read_text())
    if ref["record_id"] != predictions["record_id"]:
        raise ValueError("reference identity mismatch")
    if (root / "results.json").exists():
        raise FileExistsError("evaluation already completed")
    ids, y = load_eval_labels(Path(a.labels), root)
    if ids.tolist() != predictions["record_id"]:
        raise ValueError("truth identity mismatch")
    pred = np.asarray(predictions["prediction"], int)
    base = np.asarray(ref["prediction"], int)
    classes = predictions["class_count"]
    result = {
        "dataset": c["dataset"],
        "methods": {
            "candidate": dict(
                metrics(y, pred, classes),
                versus_direct=paired(y, base, pred),
                bootstrap=cluster_bootstrap(
                    y, base, {"candidate": pred}, np.asarray(predictions["clusters"]), classes
                )["candidate"],
            )
        },
        "config_hash": freeze["config_hash"],
        "evaluation_label_loader_calls": 1,
    }
    (root / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "rows": len(y), "label_loader_calls": 1}))


def smoke_main(kind):
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic-smoke", action="store_true")
    a = p.parse_args()
    if not a.synthetic_smoke:
        p.error(
            "This entry point provides synthetic checks; use --synthetic-smoke."
        )
    import torch

    if kind == "distill":
        from .student.distill import synthetic_distillation_smoke

        losses = synthetic_distillation_smoke(
            np.eye(3), np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]], np.float32)
        )
        print(json.dumps({"scope": "synthetic linear probe", "losses": losses}))
    else:
        from .baselines.objectives import kd_soft_loss, pl_conf_loss, entropy_loss

        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        q = torch.tensor([[0.95, 0.05], [0.1, 0.9]])
        ce, kept = pl_conf_loss(x, q)
        print(
            json.dumps(
                {
                    "scope": "synthetic objective check only",
                    "kd_soft": float(kd_soft_loss(x, q).detach()),
                    "pl_conf": float(ce.detach()),
                    "retained": kept,
                    "tent_entropy": float(entropy_loss(x).detach()),
                }
            )
        )


def check_expected_main():
    p = argparse.ArgumentParser()
    p.add_argument("--result", required=True)
    p.add_argument("--expected", required=True)
    p.add_argument(
        "--section", choices=["targets", "students"], required=True
    )
    p.add_argument("--dataset", required=True)
    a = p.parse_args()
    result = json.loads(Path(a.result).read_text())
    expected = yaml.safe_load(Path(a.expected).read_text())[a.section][a.dataset]
    if "splits" in result:
        result = result["splits"]["evaluation_all"]
    actual = result.get("methods", result)
    out = []
    for name, value in expected.items():
        row = actual.get(name)
        if row is None:
            out.append({"method": name, "status": "missing"})
            continue
        keys = [k for k in row if k.startswith("macro_f1_fixed")]
        if len(keys) != 1:
            out.append({"method": name, "status": "missing_fixed_class_metric"})
            continue
        score = 100 * row[keys[0]]
        out.append(
            {
                "method": name,
                "expected_percent": value,
                "actual_percent": score,
                "difference_pp": score - value,
                "status": "match" if abs(score - value) <= 0.05001 else "different",
            }
        )
    print(
        json.dumps(
            {"comparisons": out, "all_match": all(r["status"] == "match" for r in out)}, indent=2
        )
    )


def table_main():
    p = argparse.ArgumentParser()
    p.add_argument("--result", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    result = json.loads(Path(a.result).read_text())
    methods = result.get("methods", {})
    lines = ["| Method | Accuracy (%) | Fixed-class Macro-F1 (%) |", "|---|---:|---:|"]
    for name, row in methods.items():
        keys = [k for k in row if k.startswith("macro_f1_fixed")]
        if len(keys) == 1:
            lines.append(f'| {name} | {100*row["accuracy"]:.4f} | {100*row[keys[0]]:.4f} |')
    Path(a.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
