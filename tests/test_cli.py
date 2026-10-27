import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml


REPOSITORY = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def run_script(name, *args, cwd):
    result = subprocess.run(
        [sys.executable, str(REPOSITORY / "scripts" / name), *map(str, args)],
        cwd=cwd, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_direct_cli_build_then_evaluate_has_zero_gain(tmp_path):
    ids = ["a", "b", "c", "d"]
    probability = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8], [.2, .6, .2]])
    posterior = write_json(tmp_path / "posterior.json", {
        "record_id": ids, "direct_probability": probability.tolist(),
        "session": ["one", "one", "two", "two"],
    })
    labels = write_json(tmp_path / "labels.json", {"record_id": ids, "label": [0, 1, 2, 0]})
    reference = write_json(tmp_path / "reference.json", {
        "record_id": ids, "prediction": probability.argmax(1).tolist(),
    })
    output = tmp_path / "output"
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "dataset": "synthetic", "classes": 3,
        "posterior_input": str(posterior), "output_root": str(output),
    }), encoding="utf-8")

    run_script("build_targets.py", "--config", config, cwd=tmp_path)
    run_script(
        "evaluate.py", "--config", config, "--labels", labels,
        "--reference-predictions", reference, cwd=tmp_path,
    )

    targets = json.loads((output / "targets.json").read_text(encoding="utf-8"))
    result = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert targets["method"] == "direct"
    assert targets["prediction"] == probability.argmax(1).tolist()
    candidate = result["methods"]["candidate"]
    assert candidate["versus_direct"]["rescues"] == 0
    assert candidate["versus_direct"]["harms"] == 0
    assert candidate["versus_direct"]["net"] == 0
    assert candidate["bootstrap"]["macro_f1_difference"] == pytest.approx(0.)


def test_expected_comparison_requires_explicit_expected_argument(tmp_path):
    result = subprocess.run(
        [
            sys.executable, "-c",
            "from prism.cli import check_expected_main; check_expected_main()",
            "--result", str(tmp_path / "result.json"),
            "--section", "targets", "--dataset", "CUHK-X",
        ],
        cwd=REPOSITORY, text=True, capture_output=True,
    )

    assert result.returncode == 2
    assert "--expected" in result.stderr
    assert "required" in result.stderr


def test_cuhkx_rich_json_auto_cli_builds_all_six_paths(tmp_path):
    source_labels = np.tile(np.arange(3), 6)
    bank = np.full((18, 2, 40), .001, dtype=np.float32)
    for row, label in enumerate(source_labels):
        bank[row, :, label] = .961
    bank /= bank.sum(2, keepdims=True)
    source = write_json(tmp_path / "source.json", {
        "record_id": [f"source-{i}" for i in range(18)],
        "expert_bank": bank.tolist(), "expert_bank_dtype": "float32",
        "label": source_labels.tolist(), "inner_fold": np.repeat(np.arange(3), 6).tolist(),
        "users": np.repeat(["a", "b", "c"], 6).tolist(),
        "dates": ["source-day"] * 18,
        "starts": (np.tile(np.arange(6), 3) * 5 + np.repeat(np.arange(3), 6) * 100).tolist(),
    })
    posterior = write_json(tmp_path / "posterior.json", {
        "record_id": [f"target-{i}" for i in range(6)],
        "expert_bank": bank[:6].tolist(), "expert_bank_dtype": "float32",
        "users": ["held"] * 6, "dates": ["target-day"] * 6,
        "starts": (np.arange(6) * 5).tolist(),
    })
    output = tmp_path / "native-output"
    config = tmp_path / "native.yaml"
    config.write_text(yaml.safe_dump({
        "dataset": "CUHK-X", "classes": 40, "source_input": str(source),
        "posterior_input": str(posterior), "output_root": str(output),
    }), encoding="utf-8")

    run_script("build_targets.py", "--config", config, "--path", "auto", cwd=tmp_path)

    targets = json.loads((output / "targets.json").read_text(encoding="utf-8"))
    paths = [
        "direct", "group", "sequence", "direct_sequence",
        "direct_sequence_nogate", "online_forward",
    ]
    assert targets["available_paths"] == paths
    assert targets["selected_path"] in paths
    assert targets["method"] == targets["selected_path"]
    assert len(targets["prediction"]) == 6
    assert np.asarray(targets["targets"]).shape == (6, 40)
    np.testing.assert_array_equal(np.asarray(targets["targets"]).argmax(1), targets["prediction"])
    assert set(targets["gates"]) == {"group_gate", "sequence_gate", "direct_sequence_gate"}
    assert [row["path"] for row in targets["selection_records"]] == paths


@pytest.mark.parametrize("dataset,filename,classes", [("MMAct", "mmact.yaml", 37), ("MM-Fit", "mmfit.yaml", 11)])
def test_selection_cli_accepts_source_selection_and_target(tmp_path, dataset, filename, classes):
    config = yaml.safe_load((REPOSITORY / "configs" / filename).read_text(encoding="utf-8"))
    count = max(config["direct_expert_indices"]) + 1 if dataset == "MMAct" else 6
    experts = [f"expert-{i}" for i in range(count)]
    labels = np.tile([0, 1], 4)
    bank = np.full((8, len(experts), classes), .001, dtype=np.float32)
    for i, label in enumerate(labels):
        bank[i, :, label] = .9
    bank /= bank.sum(2, keepdims=True)
    for split in ("source", "selection", "target"):
        data = {
            "record_id": [f"{split}-{i}" for i in range(8)],
            "expert_bank": bank.tolist(), "expert_bank_dtype": "float32",
        }
        if split != "target":
            data["label"] = labels.tolist()
        if dataset == "MMAct":
            data.update({
                "expert_names": experts, "availability": np.ones((8, len(experts)), bool).tolist(),
                "geometry": np.eye(2)[labels].tolist(), "subject": [split] * 8,
                "scene": ["room"] * 8, "session": ["one"] * 4 + ["two"] * 4,
                "start_ms": (np.arange(8) * 1000).tolist(),
                "end_ms": (np.arange(8) * 1000 + 500).tolist(),
                "ordinal": np.tile(np.arange(4), 2).tolist(),
            })
        else:
            data.update({
                "workout": [split] * 8, "time": np.arange(8).tolist(),
                "descriptor": np.eye(2)[labels].tolist(),
            })
        path = write_json(tmp_path / f"{split}.json", data)
        config["posterior_input" if split == "target" else split + "_input"] = str(path)
    config["output_root"] = str(tmp_path / "output")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_script("build_targets.py", "--config", path, "--path", "auto", cwd=tmp_path)
    result = json.loads((tmp_path / "output/targets.json").read_text(encoding="utf-8"))
    assert {"direct", "group", "sequence", "direct_sequence", "selected"} <= set(result["available_paths"])
    assert len(result["prediction"]) == 8
    assert np.asarray(result["targets"]).shape == (8, classes)
    if dataset == "MM-Fit":
        assert "repeat_precedence" in result["available_paths"]
