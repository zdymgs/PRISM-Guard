"""Freeze the paper's prespecified controls from normalized private expert banks."""

from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def external_output(path):
    path = Path(path).resolve()
    if path.is_relative_to(ROOT):
        raise ValueError("output must be outside the source repository")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("use a new or empty output directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source-input", required=True, type=Path)
    parser.add_argument("--posterior-input", required=True, type=Path,
                        help="Unlabeled target-bank JSON; labels are rejected before decoding.")
    parser.add_argument("--selection-input", type=Path,
                        help="Separate labeled selection-bank JSON for MMAct and MM-Fit.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), default=0,
                        help="CUHK-X outer-fold index; default 0.")
    args = parser.parse_args(argv)

    import numpy as np
    import yaml
    from prism.chain import check_classes, construct, target_clusters
    from prism.data.io import load_unlabeled_bundle

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    if dataset not in {"CUHK-X", "MMAct", "MM-Fit"}:
        parser.error("dataset must be CUHK-X, MMAct or MM-Fit")
    if dataset != "CUHK-X" and args.selection_input is None:
        parser.error("MMAct and MM-Fit require --selection-input")
    config["source_input"] = str(args.source_input)
    if args.selection_input:
        config["selection_input"] = str(args.selection_input)
    inputs = {"config": args.config, "source": args.source_input,
              "target": args.posterior_input}
    if args.selection_input:
        inputs["selection"] = args.selection_input
    hashes = {role: sha256(path) for role, path in inputs.items()}
    target = load_unlabeled_bundle(args.posterior_input)
    check_classes(target, config["classes"])
    target_ids = set(map(str, target["record_id"]))
    source = json.loads(args.source_input.read_text(encoding="utf-8"))
    source_ids = set(map(str, source["record_id"]))
    if source_ids & target_ids:
        raise ValueError("source and target identities must be disjoint")
    if args.selection_input:
        selection = json.loads(args.selection_input.read_text(encoding="utf-8"))
        selection_ids = set(map(str, selection["record_id"]))
        if selection_ids & (source_ids | target_ids):
            raise ValueError("source, selection and target identities must be disjoint")
    output = external_output(args.output)
    result = construct(config, target, include_controls=True)
    if hashes != {role: sha256(path) for role, path in inputs.items()}:
        raise RuntimeError("input changed during construction")

    formal_paths = {"direct", "group", "sequence", "direct_sequence", "direct_sequence_nogate",
                    "online_forward", "direct_sequence_gated", "selected"}
    predictions = {name: value for name, value in result["predictions"].items()
                   if name in formal_paths}
    predictions.update(result["controls"])
    # Selection is frozen by the original main-path search, before controls.
    if "selected" not in predictions:
        predictions["selected"] = result["predictions"][result["selected_path"]].copy()
    ids = np.asarray(target["record_id"], dtype=str)
    payload = {
        "evaluation_ids": ids,
        "evaluation_clusters": np.asarray(target_clusters(config, target), dtype=str),
        "evaluation_outer_fold": np.full(len(ids), args.fold, dtype=np.int64),
    }
    for name, values in predictions.items():
        values = np.asarray(values)
        if values.shape != ids.shape or not np.issubdtype(values.dtype, np.integer):
            raise ValueError("invalid prediction vector: " + name)
        if np.any((values < 0) | (values >= config["classes"])):
            raise ValueError("prediction class outside configured range: " + name)
        payload["evaluation_prediction__" + name] = values.astype(np.int64)
    prediction_path = output / "frozen_predictions.npz"
    with prediction_path.open("xb") as handle:
        np.savez_compressed(handle, **payload)
    policy = {
        "dataset": dataset, "outer_fold": args.fold,
        "selected_path": result["selected_path"],
        "selection_records": result["selection_records"],
        "gates": result["gates"], "control_policies": result["control_policies"],
    }
    policy_path = output / "selection.json"
    policy_path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    freeze = {
        "status": "complete", "dataset": dataset, "class_count": config["classes"],
        "evaluation_labels_read": False, "methods": sorted(predictions),
        "selected_path": result["selected_path"], "rows": len(ids),
        "outer_fold": args.fold, "input_sha256": hashes,
        "bootstrap_repetitions": 2000, "bootstrap_seed": 20261015,
        "random_peer_seeds": list(range(2026101501, 2026101506)),
        "artifact_sha256": {p.name: sha256(p) for p in (prediction_path, policy_path)},
    }
    (output / "PRE_EVALUATION_FREEZE.json").write_text(
        json.dumps(freeze, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rows": len(ids), "methods": len(predictions),
                      "selected_path": result["selected_path"]}))


if __name__ == "__main__":
    main()
