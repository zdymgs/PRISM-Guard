from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import numpy as np
import pandas as pd
from session_kd.mmact_strong_teacher import CAMERAS, MODELS, masked_view_mean, softmax, subject_fold
SOURCE_FOLDS = ('F2', 'F3', 'F4')
VALIDATION_FOLD = 'F1'
SEALED_FOLD = 'F0'
EXPERT_NAMES = tuple((name for model in MODELS for name in (*[f'{model}__{camera}' for camera in CAMERAS], f'{model}__viewmean'))) + ('joint_video', 'imu')

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def ordered_hash(values: Sequence[str]) -> str:
    return hashlib.sha256(('\n'.join(map(str, values)) + '\n').encode()).hexdigest()

def canonical_json_hash(value: Mapping[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()

def load_identities(path: Path) -> list[dict[str, object]]:
    rows = json.loads(path.read_text(encoding='utf-8'))
    if len(rows) != 5406 or [int(row['index']) for row in rows] != list(range(5406)):
        raise ValueError('interval identities must contain ordered indices 0..5405')
    if any((str(row['fold']) != subject_fold(str(row['subject'])) for row in rows)):
        raise ValueError('identity fold does not match subject')
    ids = [str(row['interval_id']) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('interval identities are not unique')
    return rows

def validate_mc3_identity_manifest(path: Path, identities: Sequence[Mapping[str, object]]) -> str:
    manifest = json.loads(path.read_text(encoding='utf-8'))
    rows = manifest.get('rows_manifest', [])
    if len(rows) != len(identities):
        raise ValueError('MC3 manifest row count differs from identities')
    for index, (left, right) in enumerate(zip(rows, identities, strict=True)):
        if int(left['global_interval_index']) != index or str(left['interval_id']) != str(right['interval_id']):
            raise ValueError(f'MC3 identity mismatch at row {index}')
    return file_sha256(path)

def _annotation_label(line: str) -> str:
    value = line.strip()
    if not value or '-' not in value:
        raise ValueError('invalid MMACT annotation line')
    return value.rsplit('-', 1)[1].strip().lower()

def load_nonsealed_labels(dataset_root: Path, identities: Sequence[Mapping[str, object]], label_to_id: Mapping[str, int], allowed_folds: Iterable[str]=(*SOURCE_FOLDS, VALIDATION_FOLD)) -> tuple[np.ndarray, dict[str, str]]:
    """Load labels only for explicitly allowed folds; F0 files are never opened."""
    allowed = frozenset(allowed_folds)
    if SEALED_FOLD in allowed:
        raise ValueError('F0 label access is forbidden')
    labels = np.full(len(identities), -1, dtype=np.int64)
    grouped: dict[str, list[tuple[int, int]]] = {}
    for index, row in enumerate(identities):
        if str(row['fold']) not in allowed:
            continue
        rel, line = str(row['interval_id']).rsplit(':', 1)
        grouped.setdefault(rel, []).append((index, int(line)))
    hashes: dict[str, str] = {}
    for rel, requests in sorted(grouped.items()):
        path = dataset_root / Path(rel)
        lines = path.read_text(encoding='utf-8').splitlines()
        hashes[rel] = file_sha256(path)
        for index, line_number in requests:
            label = _annotation_label(lines[line_number - 1])
            if label not in label_to_id:
                raise ValueError(f'unknown label {label!r} in {rel}:{line_number}')
            labels[index] = int(label_to_id[label])
    expected = np.asarray([str(row['fold']) in allowed for row in identities])
    if np.any(labels[expected] < 0) or np.any(labels[~expected] != -1):
        raise AssertionError('label access scope mismatch')
    return (labels, hashes)

def load_video_features(cache: Path, row_count: int=5406) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    features = np.full((row_count, 4, 2, 1024), np.nan, np.float16)
    masks = np.zeros((row_count, 4, 2), dtype=bool)
    files_used: list[Path] = []
    for model_index, model in enumerate(MODELS):
        files = sorted((cache / model).glob('cam*/subject*/scene*/session*.npz'))
        if len(files) != 941:
            raise ValueError(f'{model}: expected 941 feature shards, got {len(files)}')
        for path in files:
            camera_index = CAMERAS.index(path.parts[-4])
            with np.load(path, allow_pickle=False) as archive:
                indices = archive['global_interval_index'].astype(np.int64)
                covered = archive['video_covered'].astype(bool)
                if np.any(masks[indices, camera_index, model_index]):
                    raise ValueError(f'duplicate covered rows in {path}')
                features[indices, camera_index, model_index] = archive['features']
                masks[indices, camera_index, model_index] = covered
            files_used.append(path)
    if not np.all(masks.any(axis=(1, 2))):
        raise ValueError('at least one annotation has no video feature')
    return (features, masks, files_used)

def load_imu_features(legacy: Path, identities: Sequence[Mapping[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    """Join IMU using identity columns only; segment labels are never accessed."""
    table = pd.read_csv(legacy / 'segments.csv', usecols=['person', 'session', 'segment_id'], dtype={'person': str, 'session': str, 'segment_id': str})
    lookup = {f'{row.person}/{row.session}:{row.segment_id.rsplit(':', 1)[1]}': index for index, row in enumerate(table.itertuples(index=False))}
    values = np.load(legacy / 'imu_accgyro_96.npy', mmap_mode='r')
    output = np.zeros((len(identities), 18), dtype=np.float32)
    present = np.zeros(len(identities), dtype=bool)
    for index, row in enumerate(identities):
        line = str(row['interval_id']).rsplit(':', 1)[1]
        key = f'{row['subject']}/{row['scene']}/{row['session']}:{line}'
        if key not in lookup:
            continue
        value = np.asarray(values[lookup[key]], dtype=np.float32)
        output[index] = np.r_[value.mean(1), value.std(1)]
        present[index] = True
    if int(present.sum()) != 3989:
        raise ValueError(f'IMU exact join produced {present.sum()}, expected 3989')
    return (output, present)

@dataclass(frozen=True)
class RawInputs:
    video: np.ndarray
    video_mask: np.ndarray
    imu: np.ndarray
    imu_mask: np.ndarray

def _normalizer(raw: np.ndarray, mask: np.ndarray, fit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    chosen = np.asarray(raw[fit][mask[fit]], dtype=np.float64)
    if not len(chosen):
        raise ValueError('no available fit rows for normalizer')
    return (chosen.mean(0).astype(np.float32), np.maximum(chosen.std(0), 1e-06).astype(np.float32))

def build_expert_inputs(raw: RawInputs, fit_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Fit every normalization statistic on the current cross-fit rows only."""
    arrays: dict[str, np.ndarray] = {}
    available: dict[str, np.ndarray] = {}
    normalizers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    view_values: dict[str, np.ndarray] = {}
    for model_index, model in enumerate(MODELS):
        for camera_index, camera in enumerate(CAMERAS):
            name = f'{model}__{camera}'
            value = np.asarray(raw.video[:, camera_index, model_index], dtype=np.float32)
            mask = raw.video_mask[:, camera_index, model_index]
            mean, std = _normalizer(value, mask, fit_indices)
            arrays[name] = np.nan_to_num((value - mean) / std, nan=0.0)
            available[name] = mask
            normalizers[name] = (mean, std)
        name = f'{model}__viewmean'
        value = masked_view_mean(raw.video[:, :, model_index], raw.video_mask[:, :, model_index])
        mask = raw.video_mask[:, :, model_index].any(1)
        mean, std = _normalizer(value, mask, fit_indices)
        arrays[name] = np.c_[(value - mean) / std, raw.video_mask[:, :, model_index].astype(np.float32)]
        available[name] = mask
        normalizers[name] = (mean, std)
        view_values[model] = arrays[name]
    arrays['joint_video'] = np.concatenate([view_values[model] for model in MODELS], axis=1)
    available['joint_video'] = np.ones(len(raw.imu), dtype=bool)
    normalizers['joint_video'] = (np.zeros(arrays['joint_video'].shape[1], dtype=np.float32), np.ones(arrays['joint_video'].shape[1], dtype=np.float32))
    mean, std = _normalizer(raw.imu, raw.imu_mask, fit_indices)
    arrays['imu'] = np.c_[np.where(raw.imu_mask[:, None], (raw.imu - mean) / std, 0.0), raw.imu_mask]
    available['imu'] = raw.imu_mask
    normalizers['imu'] = (mean, std)
    if tuple(arrays) != EXPERT_NAMES:
        raise AssertionError('expert order changed')
    return (arrays, available, normalizers)

def make_receipt_contract(*, expert: str, prediction_fold: str, fit_ids: Sequence[str], prediction_ids: Sequence[str], selection_ids: Sequence[str], input_hashes: Mapping[str, str], config: Mapping[str, object]) -> dict[str, object]:
    fit_folds = tuple(sorted(set((str(config_fold) for config_fold in config['fit_folds']))))
    if prediction_fold in fit_folds:
        raise ValueError('prediction fold appears in fit folds')
    return {'schema': 'mmact_source_oof_receipt', 'posterior_provenance': 'source_oof_crossfit_verified', 'expert': expert, 'prediction_fold': prediction_fold, 'fit_folds': list(fit_folds), 'fit_ids_sha256': ordered_hash(fit_ids), 'fit_count': len(fit_ids), 'prediction_ids_sha256': ordered_hash(prediction_ids), 'prediction_count': len(prediction_ids), 'epoch_selection_fold': VALIDATION_FOLD, 'epoch_selection_ids_sha256': ordered_hash(selection_ids), 'epoch_selection_count': len(selection_ids), 'input_hashes': dict(sorted(input_hashes.items())), 'config_sha256': canonical_json_hash(config)}

def receipt_matches(path: Path, contract: Mapping[str, object]) -> bool:
    if not path.exists():
        return False
    try:
        saved = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return False
    return all((saved.get(key) == value for key, value in contract.items())) and saved.get('status') == 'complete'

def validate_source_oof_manifest(manifest: Mapping[str, object], *, expected_experts: Sequence[str], expected_source_ids: Sequence[str]) -> None:
    """Reject string-only provenance and require verifiable fold receipts."""
    if manifest.get('status') != 'complete' or manifest.get('provenance') != 'source_oof_crossfit_verified':
        raise ValueError('source OOF manifest lacks verified complete provenance')
    if manifest.get('source_ids_sha256') != ordered_hash(expected_source_ids):
        raise ValueError('source identity hash mismatch')
    receipts = manifest.get('receipts')
    if not isinstance(receipts, list):
        raise ValueError('source OOF manifest lacks receipts')
    expected = {(expert, fold) for expert in expected_experts for fold in SOURCE_FOLDS}
    observed: set[tuple[str, str]] = set()
    for receipt in receipts:
        if not isinstance(receipt, dict) or receipt.get('posterior_provenance') != 'source_oof_crossfit_verified':
            raise ValueError('receipt lacks verified cross-fit provenance')
        key = (str(receipt.get('expert')), str(receipt.get('prediction_fold')))
        fit_folds = set(map(str, receipt.get('fit_folds', [])))
        if key[1] in fit_folds or fit_folds != set(SOURCE_FOLDS) - {key[1]}:
            raise ValueError('receipt fold exclusion is invalid')
        if key in observed:
            raise ValueError('duplicate expert/fold receipt')
        observed.add(key)
    if observed != expected:
        raise ValueError('expert/fold receipt coverage is incomplete')

def pilot_status(*, completed_receipts: int, experts: Sequence[str], prediction_folds: Sequence[str], epochs: int) -> dict[str, object]:
    """Return a status that cannot be mistaken for a complete OOF bank."""
    return {'status': 'pilot_incomplete', 'full_bank_written': False, 'completed_receipts': int(completed_receipts), 'required_full_receipts': len(EXPERT_NAMES) * len(SOURCE_FOLDS), 'experts': list(experts), 'prediction_folds': list(prediction_folds), 'epochs': int(epochs)}

def availability_from_video_imu(video: np.ndarray, imu_mask: np.ndarray) -> np.ndarray:
    if video.ndim != 3 or video.shape[1:] != (4, 2) or imu_mask.shape != (len(video),):
        raise ValueError('availability inputs must be [N,4,2] video and [N] IMU')
    values = []
    for model_index, _model in enumerate(MODELS):
        values.extend((video[:, camera_index, model_index] for camera_index in range(4)))
        values.append(video[:, :, model_index].any(1))
    values.extend([np.ones(len(video), dtype=bool), imu_mask.astype(bool)])
    return np.stack(values, axis=1)

def geometry_features(raw: RawInputs) -> np.ndarray:
    """Return label-free, per-row L2-normalized two-model view-mean geometry."""
    parts = [masked_view_mean(raw.video[:, :, model_index], raw.video_mask[:, :, model_index]) for model_index in range(len(MODELS))]
    values = np.nan_to_num(np.concatenate(parts, axis=1).astype(np.float32), nan=0.0)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError('geometry contains a zero row')
    return values / norms
