from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

class ProtocolError(ValueError):
    """Raised when an experiment violates a stability/provenance contract."""

@dataclass(frozen=True)
class ArtifactNode:
    node_id: str
    parents: tuple[str, ...] = ()
    supervised_train_subjects: frozenset[str] = frozenset()
    provenance: str = 'derived'
    has_task_labels: bool = False
    oof_labels_only: bool = False

def assert_prediction_provenance(prediction_subject: str, prediction_artifacts: Sequence[str], nodes: Mapping[str, ArtifactNode]) -> None:
    """Reject unknown/malformed/cyclic ancestry and train-subject leakage.

    ``oof_labels_only`` never counts as evidence of independence: OOF nodes are
    treated as unknown unless they also have an explicit, complete provenance DAG.
    """
    if not prediction_subject or not prediction_artifacts:
        raise ProtocolError('prediction subject and at least one artifact are required')
    if not nodes:
        raise ProtocolError('artifact ancestry is unknown')
    visiting: set[str] = set()
    visited: set[str] = set()
    train_subjects: set[str] = set()

    def walk(node_id: str) -> None:
        if node_id not in nodes:
            raise ProtocolError(f'missing artifact parent/node: {node_id}')
        if node_id in visiting:
            raise ProtocolError(f'cycle in artifact ancestry at {node_id}')
        if node_id in visited:
            return
        node = nodes[node_id]
        if node.node_id != node_id:
            raise ProtocolError(f'artifact key/node_id mismatch: {node_id}')
        if node.oof_labels_only:
            raise ProtocolError(f'OOF labels are not provenance proof: {node_id}')
        allowed = {'frozen_external', 'raw_input', 'supervised', 'derived', 'unknown'}
        if node.provenance not in allowed or node.provenance == 'unknown':
            raise ProtocolError(f'unknown artifact ancestry: {node_id}')
        if node.provenance == 'frozen_external' and (node.has_task_labels or node.supervised_train_subjects):
            raise ProtocolError(f'frozen_external contains task labels/training subjects: {node_id}')
        if node.provenance == 'raw_input' and (node.has_task_labels or node.supervised_train_subjects):
            raise ProtocolError(f'raw_input contains task labels/training subjects: {node_id}')
        if node.has_task_labels and (not node.supervised_train_subjects):
            raise ProtocolError(f'labeled artifact lacks explicit training subjects: {node_id}')
        if node.provenance == 'supervised' and (not node.supervised_train_subjects):
            raise ProtocolError(f'supervised artifact lacks training subjects: {node_id}')
        if node.provenance == 'derived' and (not node.parents and (not node.supervised_train_subjects)):
            raise ProtocolError(f'unsubstantiated derived leaf: {node_id}')
        visiting.add(node_id)
        train_subjects.update(node.supervised_train_subjects)
        for parent in node.parents:
            walk(parent)
        visiting.remove(node_id)
        visited.add(node_id)
    for artifact in prediction_artifacts:
        walk(artifact)
    if prediction_subject in train_subjects:
        raise ProtocolError('prediction subject intersects supervised training ancestry')

def canonical_experiment_json(spec: Mapping[str, Any]) -> str:
    return json.dumps(spec, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)

def experiment_sha256(spec: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_experiment_json(spec).encode('utf-8')).hexdigest()

def register_experiment(registry_path: str | Path, spec: Mapping[str, Any], output_paths: Sequence[str | Path]=()) -> str:
    """Write a canonical registration; never overwrite an existing output."""
    for output in output_paths:
        if Path(output).exists():
            raise FileExistsError(f'refusing to overwrite existing output: {output}')
    path = Path(registry_path)
    digest = experiment_sha256(spec)
    if path.exists():
        try:
            prior = json.loads(path.read_text(encoding='utf-8'))
        except Exception as exc:
            raise ProtocolError('invalid existing experiment registry') from exc
        try:
            stored_spec = prior['spec']
            recomputed = experiment_sha256(stored_spec)
        except Exception as exc:
            raise ProtocolError('invalid existing experiment registry') from exc
        if prior.get('sha256') != recomputed or recomputed != digest:
            raise FileExistsError('experiment registry already contains a different experiment')
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'sha256': digest, 'spec': json.loads(canonical_experiment_json(spec))}
    try:
        with path.open('x', encoding='utf-8', newline='') as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n')
    except FileExistsError:
        raise FileExistsError('experiment registry was created concurrently')
    return digest
