"""Export teacher probabilities and label-free metadata to PRISM JSON inputs."""
import argparse
import json
from pathlib import Path

MMFIT_ORDER = ("rgb", "pose", "imu", "videomaev2_large", "vjepa2", "joint")
CUHKX_ORDER = tuple("head_" + n + "_logits" for n in
                    ("early", "late", "window_mean", "early_late", "temporal_delta", "kinetics")) + tuple(
    "mechanism_" + n + "_logits" for n in
    ("baseline", "drop_scene", "drop_person", "drop_workspace", "drop_early", "drop_late",
     "swap_early_late", "collapse_early_late", "swap_person_workspace", "collapse_view_identity")) + (
    "skeleton", "thermal", "all_token", "hand_interaction",
    "ir_thermal", "lavila_frame_token", "physical_token")
MMACT_ORDER = tuple(name for model in ("videomaev2_large", "vjepa2_vitl_ssv2")
                    for name in (*(model + "__cam" + str(i) for i in range(1, 5)), model + "__viewmean")) + (
    "joint_video", "imu", "mc3_motion", "native_skeleton", "sequential_imu")
METADATA = {
    "cuhkx": {"record_id", "users", "dates", "starts", "inner_fold"},
    "mmact": {"record_id", "subject", "scene", "session", "start_ms", "end_ms", "ordinal", "geometry"},
    "mmfit": {"record_id", "workout", "time", "descriptor"},
}


def read_metadata(path, dataset):
    """Check keys before decoding values, including any forbidden label value."""
    text = Path(path).read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    position = 0

    def whitespace(index):
        while index < len(text) and text[index].isspace():
            index += 1
        return index

    position = whitespace(position)
    if text[position:position + 1] != "{":
        raise ValueError("Metadata must be a JSON object")
    position += 1
    result = {}
    while True:
        position = whitespace(position)
        if text[position:position + 1] == "}":
            position += 1
            break
        key, position = decoder.raw_decode(text, position)
        if key not in METADATA[dataset] or key in result:
            raise ValueError("Non-whitelisted or duplicate metadata field")
        position = whitespace(position)
        if text[position:position + 1] != ":":
            raise ValueError("Missing metadata colon")
        result[key], position = decoder.raw_decode(text, whitespace(position + 1))
        position = whitespace(position)
        if text[position:position + 1] == ",":
            position += 1
        elif text[position:position + 1] == "}":
            position += 1
            break
        else:
            raise ValueError("Invalid metadata delimiter")
    if whitespace(position) != len(text):
        raise ValueError("Trailing metadata content")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(METADATA), required=True)
    parser.add_argument("--bank", type=Path, action="append", required=True,
                        help="CUHK-X: seven family directories in the README order; other datasets: one NPZ.")
    parser.add_argument("--role", choices=("source", "selection", "posterior"), required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--labels", type=Path, help="Source/selection integer NPY or JSON vector, in metadata order.")
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--geometry", type=Path, help="MMAct geometry_features.npz, using the matching split.")
    parser.add_argument("--ids", type=Path, help="MM-Fit selection/posterior ID JSON vector in NPZ row order.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.role == "posterior" and args.labels is not None:
        parser.error("posterior export does not accept labels")
    if args.role != "posterior" and args.labels is None:
        parser.error("source and selection exports require --labels")
    import numpy as np
    metadata = read_metadata(args.metadata, args.dataset)
    ids = np.asarray(metadata["record_id"]).astype(str)
    if ids.ndim != 1 or not len(ids) or len(set(ids)) != len(ids):
        raise ValueError("Metadata IDs must be unique and nonempty")
    parts = []
    names = []
    availability = None
    inner = None
    archive_ids = None
    for bank_path in args.bank:
        if args.dataset == "cuhkx":
            if len(args.bank) != 7 or args.role == "selection":
                parser.error("CUHK-X requires seven banks and role source or posterior")
            path = bank_path / f"fold{args.fold}_banks.npz"
            prefix = "inner" if args.role == "source" else "outer"
            with np.load(path, allow_pickle=False) as bank:
                bank_ids = bank[prefix + "_sample_ids"].astype(str)
                part = bank[prefix + "_probability_bank"]
                part_names = bank["expert_names"].astype(str).tolist()
                if args.role == "source":
                    part_inner = bank["inner_fold"]
                    if inner is not None and not np.array_equal(inner, part_inner):
                        raise ValueError("CUHK-X family inner partitions differ")
                    inner = part_inner
        elif args.dataset == "mmact":
            if len(args.bank) != 1:
                parser.error("MMAct requires one complete merged bank")
            prefix = {"source": "source", "selection": "validation", "posterior": "target"}[args.role]
            with np.load(bank_path, allow_pickle=False) as bank:
                bank_ids = bank[prefix + "_interval_ids"].astype(str)
                part = bank[prefix + "_probabilities"]
                availability = bank[prefix + "_availability"]
                part_names = bank["expert_names"].astype(str).tolist()
            if args.geometry:
                with np.load(args.geometry, allow_pickle=False) as geometry:
                    if not np.array_equal(geometry[prefix + "_interval_ids"].astype(str), bank_ids):
                        raise ValueError("Geometry and bank IDs differ")
                    metadata["geometry"] = geometry[prefix + "_features"].tolist()
        else:
            if len(args.bank) != 1:
                parser.error("MM-Fit requires one bank")
            with np.load(bank_path, allow_pickle=False) as bank:
                if args.role == "source":
                    bank_ids = bank["ids"].astype(str)
                    part = bank["probabilities"]
                else:
                    if args.ids is None:
                        parser.error("MM-Fit selection/posterior requires --ids")
                    bank_ids = np.asarray(json.loads(args.ids.read_text(encoding="utf-8"))).astype(str)
                    part = np.stack([bank[name] for name in MMFIT_ORDER], axis=1)
                part_names = list(MMFIT_ORDER)
        if archive_ids is not None and not np.array_equal(archive_ids, bank_ids):
            raise ValueError("Family bank row order differs")
        archive_ids = bank_ids
        parts.append(part)
        names.extend(part_names)
    if not np.array_equal(ids, archive_ids):
        raise ValueError("Metadata must follow the explicit bank ID order")
    bank = np.concatenate(parts, axis=1)
    count, classes = {"cuhkx": (23, 40), "mmact": (15, 37), "mmfit": (6, 11)}[args.dataset]
    if bank.shape != (len(ids), count, classes) or len(set(names)) != count:
        raise ValueError("Paper expert count, order or class shape is invalid")
    if tuple(names) != {"cuhkx": CUHKX_ORDER, "mmact": MMACT_ORDER, "mmfit": MMFIT_ORDER}[args.dataset]:
        raise ValueError("Teacher expert order differs from the paper pipeline")
    if not np.isfinite(bank).all() or (bank < 0).any():
        raise ValueError("Nonfinite or negative expert probabilities")
    if availability is None:
        if not np.allclose(bank.sum(2), 1):
            raise ValueError("Each expert probability must sum to one")
    elif not np.allclose(bank.sum(2)[availability.astype(bool)], 1):
        raise ValueError("Available expert probabilities must sum to one")
    result = dict(metadata, expert_bank=bank.tolist(), expert_bank_dtype=str(bank.dtype), expert_names=names)
    if availability is not None:
        result["availability"] = availability.tolist()
    if inner is not None:
        result["inner_fold"] = inner.tolist()
    if args.labels:
        labels = (np.load(args.labels, allow_pickle=False) if args.labels.suffix == ".npy" else
                  np.asarray(json.loads(args.labels.read_text(encoding="utf-8"))))
        if labels.shape != (len(ids),) or labels.dtype.kind not in "iu" or np.any((labels < 0) | (labels >= classes)):
            raise ValueError("Source/selection labels must be an aligned class vector")
        result["label"] = labels.tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, separators=(",", ":"))
        handle.write("\n")


if __name__ == "__main__":
    main()
