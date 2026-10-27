"""Array input adapters for the original dataset-native target chains."""

from pathlib import Path
import json
import numpy as np

from .core.sequence import RecordingMetadata

PATHS = (
    "direct", "auto", "group", "direct_sequence", "sequence",
    "direct_sequence_nogate", "online_forward", "repeat_precedence",
    "direct_sequence_gated", "historical_selected",
)


def check_classes(bundle, classes):
    """Check both supplied posterior matrices and expert-bank class axes."""
    for key in ("direct_probability", "expert_bank"):
        if key in bundle:
            actual = np.asarray(bundle[key]).shape[-1]
            if actual != classes:
                raise ValueError(
                    f"posterior class count {actual} does not match config classes {classes}"
                )


def _source_input(path, role, classes):
    if role not in ("source", "selection"):
        raise PermissionError("evaluation labels cannot construct or select targets")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if "expert_bank" in value:
        value["expert_bank"] = np.asarray(value["expert_bank"], dtype=value["expert_bank_dtype"])
    check_classes(value, classes)
    return value


def _recording_metadata(value):
    return RecordingMetadata(
        np.asarray(value["record_id"]).astype(str),
        np.asarray(value["users"]).astype(str),
        np.asarray(value["dates"]).astype(str),
        np.asarray(value["starts"], dtype=float),
    )


def construct(config, target, *, include_controls=False):
    """Construct native paths using separately supplied source/selection inputs."""
    source_role = config.get("source_label_role", "source")
    source = _source_input(config["source_input"], source_role, config["classes"])
    dataset = config["dataset"].lower()
    if dataset == "cuhk-x":
        from ._chains.cuhkx import build_cuhkx_targets

        return build_cuhkx_targets(
            source["expert_bank"], target["expert_bank"], source["label"],
            source["inner_fold"], _recording_metadata(source), _recording_metadata(target),
            label_role=source_role, include_controls=include_controls,
        )
    selection_role = config.get("selection_label_role", "selection")
    selection = _source_input(config["selection_input"], selection_role, config["classes"])
    if dataset == "mmact":
        from ._chains.mmact import build_targets
    elif dataset == "mm-fit":
        from ._chains.mmfit import build_targets
    else:
        raise ValueError(f"unknown dataset: {config['dataset']}")
    return build_targets(source, selection, target, config, label_role=selection_role,
                         include_controls=include_controls)


def target_clusters(config, target):
    """Use the original session/workout resampling units for the target cohort."""
    dataset = config["dataset"].lower()
    if dataset == "cuhk-x":
        from .core.sequence import build_sessions

        ids = np.asarray(target["record_id"]).astype(str)
        blocks = build_sessions(np.arange(len(ids)), _recording_metadata(target), 30., "anonymous_date")
        clusters = np.array([f"singleton:{i}" for i in ids], dtype=object)
        for block in blocks:
            clusters[block] = "session:" + str(ids[int(block[0])])
        return clusters.astype(str).tolist()
    if dataset == "mmact":
        return ["|".join(map(str, row)) for row in zip(target["subject"], target["scene"], target["session"])]
    if dataset == "mm-fit":
        return list(map(str, target["workout"]))
    return target.get("session", target["record_id"])
