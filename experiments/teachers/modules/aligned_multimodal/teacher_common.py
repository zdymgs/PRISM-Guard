from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, top_k_accuracy_score
HERE = WORK_ROOT / 'aligned_multimodal'
REPO_ROOT = WORK_ROOT
MASTER_MANIFEST = HERE / 'data' / 'manifest.csv'
FOLD_DIR = HERE / 'data' / 'subject_folds'
NUM_CLASSES = 40

@dataclass(frozen=True)
class Protocol:
    manifest: pd.DataFrame
    sample_ids: np.ndarray
    labels: np.ndarray
    users: np.ndarray
    fold_id: np.ndarray

    def train_indices(self, fold: int) -> np.ndarray:
        return np.flatnonzero(self.fold_id != fold)

    def val_indices(self, fold: int) -> np.ndarray:
        return np.flatnonzero(self.fold_id == fold)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_protocol() -> Protocol:
    manifest = pd.read_csv(MASTER_MANIFEST)
    if manifest['sample_id'].duplicated().any():
        raise ValueError('master manifest contains duplicated sample_id')
    if len(manifest) != 2914:
        raise ValueError(f'expected 2914 labelled trials, found {len(manifest)}')
    sample_ids = manifest['sample_id'].astype(str).to_numpy(dtype=str)
    sample_to_row = {sample_id: i for i, sample_id in enumerate(sample_ids)}
    fold_id = np.full(len(manifest), -1, dtype=np.int8)
    seen_val_users: list[set[str]] = []
    for fold in range(3):
        frame = pd.read_csv(FOLD_DIR / f'fold_{fold}.csv')
        if set(frame['sample_id'].astype(str)) != set(sample_ids):
            raise ValueError(f'fold {fold} does not contain the master sample set')
        val = frame.loc[frame['split'].astype(str).eq('val')]
        val_users = set(val['user_id'].astype(str))
        seen_val_users.append(val_users)
        for sample_id in val['sample_id'].astype(str):
            row = sample_to_row[sample_id]
            if fold_id[row] != -1:
                raise ValueError(f'sample {sample_id} appears in two validation folds')
            fold_id[row] = fold
    if (fold_id < 0).any():
        raise ValueError('some samples never appear in validation')
    if any((seen_val_users[i] & seen_val_users[j] for i in range(3) for j in range(i))):
        raise ValueError('validation user sets overlap')
    labels = manifest['class_id'].to_numpy(dtype=np.int64)
    if labels.min() != 0 or labels.max() != NUM_CLASSES - 1:
        raise ValueError('teacher requires contiguous 40-class labels')
    return Protocol(manifest=manifest, sample_ids=sample_ids, labels=labels, users=manifest['user_id'].astype(str).to_numpy(dtype=str), fold_id=fold_id)

def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities

def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int=15) -> float:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = (confidence > lo) & (confidence <= hi)
        if selected.any():
            value += selected.mean() * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
    return float(value)

def classification_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.shape != (len(labels), NUM_CLASSES):
        raise ValueError(f'expected logits {(len(labels), NUM_CLASSES)}, got {logits.shape}')
    probabilities = softmax(logits)
    predictions = probabilities.argmax(axis=1)
    return {'accuracy': float(accuracy_score(labels, predictions)), 'balanced_accuracy': float(balanced_accuracy_score(labels, predictions)), 'macro_f1': float(f1_score(labels, predictions, average='macro', zero_division=0)), 'top5_accuracy': float(top_k_accuracy_score(labels, probabilities, k=5, labels=np.arange(NUM_CLASSES))), 'log_loss': float(log_loss(labels, probabilities, labels=np.arange(NUM_CLASSES))), 'ece_15': expected_calibration_error(probabilities, labels)}

def fold_metrics(logits: np.ndarray, protocol: Protocol) -> dict[str, Any]:
    result: dict[str, Any] = {'overall': classification_metrics(logits, protocol.labels), 'folds': []}
    for fold in range(3):
        indices = protocol.val_indices(fold)
        values = classification_metrics(logits[indices], protocol.labels[indices])
        values['fold'] = fold
        values['samples'] = int(len(indices))
        result['folds'].append(values)
    accuracies = [item['accuracy'] for item in result['folds']]
    result['fold_accuracy_mean'] = float(np.mean(accuracies))
    result['fold_accuracy_std'] = float(np.std(accuracies))
    return result

def save_oof_artifact(output_dir: Path, name: str, logits: np.ndarray, protocol: Protocol, metadata: dict[str, Any] | None=None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    logits = np.asarray(logits, dtype=np.float32)
    if not np.isfinite(logits).all():
        raise ValueError(f'{name}: non-finite logits')
    metrics = fold_metrics(logits, protocol)
    payload = {'teacher': name, 'protocol': 'teacher subject-disjoint 3-fold OOF', 'samples': int(len(protocol.labels)), 'classes': NUM_CLASSES, 'metrics': metrics, 'metadata': metadata or {}}
    np.savez_compressed(output_dir / f'{name}_oof.npz', sample_ids=protocol.sample_ids, labels=protocol.labels, users=protocol.users, fold_id=protocol.fold_id, logits=logits, probabilities=softmax(logits).astype(np.float32))
    (output_dir / f'{name}_metrics.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload
