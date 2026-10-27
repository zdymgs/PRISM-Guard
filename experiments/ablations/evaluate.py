"""Evaluate immutable control predictions and export the paper's Figure 4 metrics."""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPERIMENTS = {
    "CUHK-X": ("E1R_clean23", "all_2470", 40),
    "MMAct": ("E3_component_ablation", "F0", 37),
    "MM-Fit": ("E2_information_decomposition", "TEST", 11),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, nargs="+",
                        help="One frozen NPZ, or the three CUHK-X outer-fold NPZ files.")
    parser.add_argument("--labels", type=Path, required=True,
                        help="Evaluation-only JSON with record_id and label arrays.")
    parser.add_argument("--dataset", required=True, choices=tuple(EXPERIMENTS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    import numpy as np
    from prism.eval.bootstrap import cluster_bootstrap
    from prism.eval.io import load_eval_labels
    from prism.eval.metrics import metrics, paired

    if len(args.predictions) not in ({1, 3} if args.dataset == "CUHK-X" else {1}):
        parser.error("supply one NPZ, or three outer-fold NPZ files for CUHK-X")
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("output must be outside the source repository")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a new or empty evaluation output directory")
    output.mkdir(parents=True, exist_ok=True)
    bundles, manifests, input_hashes = [], [], []
    for path in args.predictions:
        freeze_path = path.parent / "PRE_EVALUATION_FREEZE.json"
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if (freeze["status"] != "complete" or freeze["dataset"] != args.dataset
                or freeze["evaluation_labels_read"]):
            raise ValueError("incompatible construction freeze")
        for name, expected in freeze["artifact_sha256"].items():
            if sha256(path.parent / name) != expected:
                raise ValueError("frozen construction artifact changed: " + name)
        if path.name not in freeze["artifact_sha256"]:
            raise ValueError("prediction file is not included in the construction freeze")
        with np.load(path, allow_pickle=False) as bundle:
            values = {name: bundle[name] for name in bundle.files}
        if any("label" in name.lower() or "truth" in name.lower() for name in values):
            raise ValueError("frozen predictions must not contain evaluation labels")
        methods = sorted(name.removeprefix("evaluation_prediction__")
                         for name in values if name.startswith("evaluation_prediction__"))
        if methods != freeze["methods"]:
            raise ValueError("frozen method set changed")
        if manifests and (methods != manifests[0]["methods"]
                          or set(values) != set(bundles[0])):
            raise ValueError("outer folds have incompatible prediction schemas")
        bundles.append(values)
        manifests.append(freeze)
        input_hashes.append(sha256(path))
    pooled = {key: np.concatenate([bundle[key] for bundle in bundles]) for key in bundles[0]}
    ids = pooled["evaluation_ids"].astype(str)
    if len(set(ids)) != len(ids):
        raise ValueError("evaluation identities overlap between supplied folds")
    predictions = {key.removeprefix("evaluation_prediction__"): value
                   for key, value in pooled.items() if key.startswith("evaluation_prediction__")}
    if "direct" not in predictions or "selected" not in predictions:
        raise ValueError("direct and source-selected predictions are required")
    experiment, split, classes = EXPERIMENTS[args.dataset]
    if any(f["class_count"] != classes for f in manifests):
        raise ValueError("class count does not match dataset")
    if pooled["evaluation_clusters"].shape != ids.shape or pooled["evaluation_outer_fold"].shape != ids.shape:
        raise ValueError("invalid cluster/fold vectors")
    for name, values in predictions.items():
        if (values.shape != ids.shape or not np.issubdtype(values.dtype, np.integer)
                or np.any((values < 0) | (values >= classes))):
            raise ValueError("invalid frozen prediction vector: " + name)
    if len(bundles) == 3 and set(np.unique(pooled["evaluation_outer_fold"])) != {0, 1, 2}:
        raise ValueError("three CUHK-X inputs must cover outer folds 0, 1 and 2")
    # The only evaluation-label read occurs after all construction hashes pass.
    label_ids, labels = load_eval_labels(args.labels, output)
    lookup = {record: int(label) for record, label in zip(label_ids, labels)}
    missing = set(ids) - set(lookup)
    if missing:
        raise ValueError("evaluation labels are missing prediction identities")
    truth = np.asarray([lookup[record] for record in ids], dtype=np.int64)
    bootstrap = cluster_bootstrap(
        truth, predictions["direct"], predictions, pooled["evaluation_clusters"],
        classes, repetitions=2000, seed=20261015,
    )
    rows, details = [], {}
    for method, pred in predictions.items():
        score = metrics(truth, pred, classes)
        delta = bootstrap[method]
        row = {
            "dataset": args.dataset, "experiment": experiment, "split": split,
            "method": method, "n": score["n"], "correct": score["correct"],
            "accuracy": score["accuracy"], "macro_f1": score[f"macro_f1_fixed{classes}"],
            "delta_accuracy": delta["accuracy_difference"],
            "accuracy_ci95_lower": delta["accuracy_ci95"][0],
            "accuracy_ci95_upper": delta["accuracy_ci95"][1],
            "delta_macro_f1": delta["macro_f1_difference"],
            "macro_f1_ci95_lower": delta["macro_f1_ci95"][0],
            "macro_f1_ci95_upper": delta["macro_f1_ci95"][1],
        }
        rows.append(row)
        details[method] = {**score, "versus_direct": paired(truth, predictions["direct"], pred),
                           "delta_vs_direct_cluster_bootstrap": delta}
        if args.dataset == "CUHK-X" and method == "selected":
            rows.append({**row, "experiment": "E5_selected_path_completed"})
    with (output / "metrics_long.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pooled_path = output / "pooled_predictions.npz"
    with pooled_path.open("xb") as handle:
        np.savez_compressed(handle, **pooled)
    result = {
        "dataset": args.dataset, "rows": len(ids), "class_count": classes,
        "bootstrap_repetitions": 2000, "bootstrap_seed": 20261015,
        "prediction_sha256": input_hashes, "evaluation_label_loader_calls": 1,
        "methods": details,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(ids), "methods": len(predictions),
                      "metrics": str(output / "metrics_long.csv")}))


if __name__ == "__main__":
    main()
