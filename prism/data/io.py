"""Normalized private-cache interfaces; no evaluation-label reader here."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

METADATA_COLUMNS = ("record_id", "subject", "timestamp", "session", "position")


def read_metadata(path: Path):
    """CSV column projection occurs in the parser, before materialization."""
    columns = pd.read_csv(path, nrows=0).columns
    if "record_id" not in columns:
        raise ValueError("record_id metadata column required")
    allowed = [name for name in METADATA_COLUMNS if name in columns]
    return pd.read_csv(path, usecols=allowed)


def load_unlabeled_bundle(path: Path):
    """Load an explicit normalized posterior schema, rejecting label fields."""
    allowed = {
        "record_id",
        "direct_probability",
        "group_probability",
        "session",
        "position",
        "source_transition",
        "source_start",
        "source_end",
        "expert_bank",
        "expert_bank_dtype",
        "expert_names",
        "availability",
        "geometry",
        "users",
        "dates",
        "starts",
        "subject",
        "scene",
        "start_ms",
        "end_ms",
        "ordinal",
        "workout",
        "time",
        "descriptor",
    }
    # Check each top-level key before decoding its value, so a forbidden label
    # column is never materialized even in a contaminated JSON object.
    text = Path(path).read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    i = 0

    def whitespace(index):
        while index < len(text) and text[index].isspace():
            index += 1
        return index

    i = whitespace(i)
    if i >= len(text) or text[i] != "{":
        raise ValueError("JSON object required")
    i += 1
    value = {}
    while True:
        i = whitespace(i)
        if i < len(text) and text[i] == "}":
            i += 1
            break
        key, i = decoder.raw_decode(text, i)
        if not isinstance(key, str) or key not in allowed or key in value:
            raise ValueError("non-whitelisted or duplicate posterior field")
        i = whitespace(i)
        if i >= len(text) or text[i] != ":":
            raise ValueError("missing JSON colon")
        item, i = decoder.raw_decode(text, whitespace(i + 1))
        value[key] = item
        i = whitespace(i)
        if i < len(text) and text[i] == ",":
            i += 1
            continue
        if i < len(text) and text[i] == "}":
            i += 1
            break
        raise ValueError("invalid JSON object delimiter")
    if whitespace(i) != len(text):
        raise ValueError("trailing JSON content")
    if "expert_bank" in value:
        value["expert_bank"] = np.asarray(value["expert_bank"], dtype=value["expert_bank_dtype"])
    key = "direct_probability" if "direct_probability" in value else "expert_bank"
    probability = np.asarray(value[key], dtype=np.float64)
    ids = np.asarray(value["record_id"]).astype(str)
    dimensions = 2 if key == "direct_probability" else 3
    if probability.ndim != dimensions or len(ids) != len(probability) or len(set(ids)) != len(ids):
        raise ValueError("invalid posterior identities")
    if (
        not np.isfinite(probability).all()
        or (probability < 0).any()
        or (key == "direct_probability" and not np.allclose(probability.sum(1), 1))
    ):
        raise ValueError("invalid posterior rows")
    return value
