import json

import numpy as np
import pytest

from prism.data.io import load_unlabeled_bundle
from prism.gate import accept, select_source_path
from prism.pipeline import build_targets


def test_gate_rejects_evaluation_label_role():
    with pytest.raises(PermissionError):
        accept([0], [1], [.8], .5, label_role="evaluation")


def test_path_selection_rejects_evaluation_label_role():
    with pytest.raises(PermissionError):
        select_source_path([0], {"group": [1]}, [1], label_role="evaluation")


def test_unlabeled_bundle_rejects_label_key(tmp_path):
    path = tmp_path / "posterior.json"
    path.write_text(json.dumps({
        "record_id": ["clip"], "direct_probability": [[.8, .2]], "label": [0],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="non-whitelisted"):
        load_unlabeled_bundle(path)


def test_build_targets_rejects_posterior_class_mismatch(tmp_path):
    posterior = tmp_path / "posterior.json"
    posterior.write_text(json.dumps({
        "record_id": ["clip"], "direct_probability": np.full((1, 40), 1 / 40).tolist(),
    }), encoding="utf-8")
    config = {
        "dataset": "MM-Fit", "classes": 11,
        "posterior_input": str(posterior), "output_root": str(tmp_path / "output"),
    }

    with pytest.raises(ValueError, match="class count 40.*classes 11"):
        build_targets(config)
