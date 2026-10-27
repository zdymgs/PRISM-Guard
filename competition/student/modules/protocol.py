from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = PROJECT_DIR / "data" / "subject_folds" / "fold_0.csv"
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "single_split.csv"
DEFAULT_SUMMARY = PROJECT_DIR / "data" / "single_split_summary.json"

HARD_CLASS_IDS = (
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    18,
    19,
    20,
    21,
    22,
    24,
    25,
    26,
    35,
    37,
    38,
    39,
)
HARD_CLASS_TO_INDEX = {class_id: index for index, class_id in enumerate(HARD_CLASS_IDS)}

VAL_SUBJECT_IDS = ("user1", "user2", "user8", "user9")

EXPECTED = {
    "train_trials": 2257,
    "val_trials": 657,
    "train_detail_trials": 1094,
    "val_detail_trials": 290,
    "train_detail_frames": 33872,
    "val_detail_frames": 9226,
    "train_subjects": 14,
    "val_subjects": 4,
}


def split_for_user(user_id: str) -> str:
    return "val" if user_id in VAL_SUBJECT_IDS else "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the one-split, subject-disjoint video detail protocol."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def freeze_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if not rows:
        raise RuntimeError("video source split is empty")
    output: list[dict[str, str]] = []
    for row in rows:
        split = split_for_user(row["user_id"])
        class_id = int(row["class_id"])
        detail_index = HARD_CLASS_TO_INDEX.get(class_id, -1)
        enriched = dict(row)
        enriched["source_id"] = (
            f"{row['class_name']}/{row['user_id']}/{row['trial_id']}"
        )
        enriched["p46_split"] = split
        enriched["detail_selected"] = "1" if detail_index >= 0 else "0"
        enriched["detail_index"] = str(detail_index)
        output.append(enriched)

    output.sort(
        key=lambda row: (
            0 if row["p46_split"] == "train" else 1,
            int(row["class_id"]),
            row["user_id"],
            row["trial_id"],
        )
    )
    train = [row for row in output if row["p46_split"] == "train"]
    val = [row for row in output if row["p46_split"] == "val"]
    train_detail = [row for row in train if row["detail_selected"] == "1"]
    val_detail = [row for row in val if row["detail_selected"] == "1"]
    train_subjects = sorted({row["user_id"] for row in train})
    val_subjects = sorted({row["user_id"] for row in val})
    overlap = sorted(set(train_subjects) & set(val_subjects))
    if overlap:
        raise RuntimeError(f"video train/val subject leakage: {overlap}")
    observed = {
        "train_trials": len(train),
        "val_trials": len(val),
        "train_detail_trials": len(train_detail),
        "val_detail_trials": len(val_detail),
        "train_detail_frames": sum(int(row["num_aligned_frames"]) for row in train_detail),
        "val_detail_frames": sum(int(row["num_aligned_frames"]) for row in val_detail),
        "train_subjects": len(train_subjects),
        "val_subjects": len(val_subjects),
    }
    if observed != EXPECTED:
        raise RuntimeError(f"video protocol changed: expected={EXPECTED}, observed={observed}")

    summary: dict[str, Any] = {
        "protocol": "single_subject_disjoint_split_14train_4val",
        "source_split": (
            "all 2914 trials reassigned by frozen subject IDs; validation users are "
            + ",".join(VAL_SUBJECT_IDS)
        ),
        "split_selection": (
            "four-subject grouped stratification chosen before corrected training; "
            "both sides cover all Detail21 classes and validation has at least three "
            "trials per Detail21 class"
        ),
        "cross_validation": False,
        "inner_calibration_split": False,
        "hard_class_ids": list(HARD_CLASS_IDS),
        **observed,
        "train_subject_ids": train_subjects,
        "val_subject_ids": val_subjects,
        "train_detail_per_class": dict(
            sorted(Counter(int(row["class_id"]) for row in train_detail).items())
        ),
        "val_detail_per_class": dict(
            sorted(Counter(int(row["class_id"]) for row in val_detail).items())
        ),
        "validation_role": (
            "val selects epoch/configuration and is therefore development validation, "
            "not an untouched final test set"
        ),
    }
    return output, summary


def atomic_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".building")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".building")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    rows, summary = freeze_rows(read_rows(args.source.resolve()))
    summary["source_manifest"] = str(args.source.resolve())
    summary["output_manifest"] = str(args.output.resolve())
    atomic_csv(args.output.resolve(), rows)
    atomic_json(args.summary.resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
