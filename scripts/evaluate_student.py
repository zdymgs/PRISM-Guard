"""Evaluate one frozen student batch or pooled CUHK-X outer-fold predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cuhkx", "mmact", "mmfit"], required=True)
    parser.add_argument("--predictions", nargs="+", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    from prism.eval.metrics import metrics

    if args.output.exists():
        raise FileExistsError(args.output)
    ids, predictions, hashes = [], [], {}
    for path in args.predictions:
        hashes[path.name + ":" + str(len(hashes))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as value:
            key = "record_id" if "record_id" in value else "ids"
            names = value[key].astype(str).tolist()
            logits = value["logits"]
            classes = {"cuhkx": 40, "mmact": 37, "mmfit": 11}[args.dataset]
            if logits.shape != (len(names), classes) or not np.isfinite(logits).all():
                raise ValueError("invalid frozen student logits")
            ids.extend(names)
            predictions.extend(logits.argmax(1).tolist())
    if len(set(ids)) != len(ids):
        raise ValueError("prediction identities overlap")
    truth = json.loads(args.labels.read_text(encoding="utf-8"))
    if len(set(truth["record_id"])) != len(truth["record_id"]):
        raise ValueError("duplicate label identities")
    label_map = dict(zip(truth["record_id"], truth["label"], strict=True))
    if set(label_map) != set(ids):
        raise ValueError("prediction and label identity sets differ")
    result = {"dataset": args.dataset, "records": len(ids),
              "prediction_sha256": hashes,
              "metrics": metrics(np.asarray([label_map[name] for name in ids]),
                                 np.asarray(predictions), classes)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"]))


if __name__ == "__main__":
    main()
