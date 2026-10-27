from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
'irthermal frozen IR+thermal source-only provider.\n\nThe three input caches are treated as external, immutable feature artifacts.  In\nparticular, this module never reads labels from an NPZ (labels are supplied to\n``fit_predict`` only).\n'
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from aligned_multimodal.foundation_provider import array_hash
from aligned_multimodal.stable_routing_protocol import ArtifactNode, ProtocolError, assert_prediction_provenance
HERE = WORK_ROOT / 'aligned_multimodal'
ROOT = WORK_ROOT
IR1_PATH = ROOT / 'runs/videomaev2_distilled_teacher/complete_features.npz'
IR2_PATH = ROOT / 'runs/internvideo2_l_k400_teacher/complete_features.npz'
THERMAL_PATH = ROOT / 'runs/videomaev2_thermal_fold0/complete_features.npz'
SUMMARY_PATHS = {'ir1': IR1_PATH.with_name('cache_summary.json'), 'ir2': IR2_PATH.with_name('cache_summary.json'), 'thermal': THERMAL_PATH.with_name('cache_summary.json')}
EXCLUDED_USERS = frozenset(('user1', 'user2', 'user21'))

def _flat(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    if z.ndim < 2 or not np.isfinite(z).all():
        raise ProtocolError('non-finite or malformed frozen features')
    z = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-08)
    return np.ascontiguousarray(z.reshape(len(z), -1), dtype=np.float32)

def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def load_raw(master_ids):
    """Load and align the frozen caches, returning ``(X, thermal_mask, metadata)``."""
    ids = np.asarray(master_ids).astype(str)
    if ids.ndim != 1 or len(ids) != 2914 or len(np.unique(ids)) != len(ids):
        raise ProtocolError('master_ids must be a unique nonempty vector')
    arrays = []
    provenance = {}
    for name, path in (('ir1', IR1_PATH), ('ir2', IR2_PATH), ('thermal', THERMAL_PATH)):
        if not path.exists():
            raise ProtocolError(f'missing frozen cache: {path}')
        summary_path = SUMMARY_PATHS[name]
        if not summary_path.exists():
            raise ProtocolError(f'missing pinned cache metadata: {summary_path}')
        try:
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
        except Exception as exc:
            raise ProtocolError(f'invalid pinned cache metadata: {summary_path}') from exc
        required = {'ir1': {'model_repo': 'OpenGVLab/VideoMAE2', 'model_file': 'distill/vit_b_k710_dl_from_giant.pth', 'clips': 'early/late x scene/person/workspace'}, 'ir2': {'model_repo': 'OpenGVLab/InternVideo2_distillation_models', 'clips': 'early/late x scene/person/workspace', 'frames_per_clip': 8}, 'thermal': {'modality': 'thermal', 'clips': 'scene/person/workspace', 'samples': 2914, 'available_samples': 2776}}[name]
        if any((summary.get(k) != v for k, v in required.items())):
            raise ProtocolError(f'{name} cache metadata is not the pinned source')
        if name == 'ir1' and (not str(summary.get('checkpoint', '')).replace('\\', '/').endswith('/snapshots/706cc172d65ebd4dedbee3f9c0183a93df9fa125/distill/vit_b_k710_dl_from_giant.pth')):
            raise ProtocolError('IR1 checkpoint identity changed')
        if name == 'ir2' and (not str(summary.get('model_file', '')).replace('\\', '/').endswith('/snapshots/449f7ea1d7d3b70b6b5630e70d238b44d3b7aaac/stage1/L14/L14_ft_k710_ft_k400_f8/pytorch_model.bin')):
            raise ProtocolError('IR2 checkpoint identity changed')
        if name == 'thermal' and summary.get('checkpoint') != provenance['ir1']['summary'].get('checkpoint'):
            raise ProtocolError('thermal checkpoint differs from pinned IR1')
        with np.load(path, allow_pickle=False) as z:
            got = np.asarray(z['sample_ids']).astype(str)
            if not np.array_equal(got, ids):
                raise ProtocolError(f'{name} sample_ids do not exactly match master order')
            features = _flat(z['features'])
            if len(features) != len(ids):
                raise ProtocolError(f'{name} feature row count mismatch')
            if name == 'thermal':
                if 'modality_available' not in z:
                    raise ProtocolError('thermal cache lacks modality_available')
                raw_mask = np.asarray(z['modality_available'])
                if raw_mask.shape != (len(ids),) or raw_mask.dtype.kind not in 'biu' or (not np.isin(raw_mask, (0, 1)).all()) or (int(raw_mask.sum()) != 2776):
                    raise ProtocolError('thermal modality mask shape mismatch')
                mask = raw_mask.astype(bool)
                raw_features = np.asarray(z['features'], dtype=np.float32)
                if raw_features.shape[1:] != (3, 768) or np.any(raw_features[~mask] != 0):
                    raise ProtocolError('thermal unavailable rows must already be zero')
            elif np.asarray(z['features']).shape[1:] != (2, 3, 768):
                raise ProtocolError(f'{name} feature shape mismatch')
            arrays.append(features)
        provenance[name] = {'path': str(path), 'sha256': _file_hash(path), 'summary_path': str(summary_path), 'summary_sha256': _file_hash(summary_path), 'summary': summary, 'sample_id_sha256': array_hash(got), 'labels_loaded': False}
    x = np.ascontiguousarray(np.concatenate(arrays, axis=1), dtype=np.float32)
    if x.shape != (len(ids), 11520) or not np.isfinite(x).all():
        raise ProtocolError('irthermal feature schema is not (N,11520)')
    metadata = {'type': 'frozen_external', 'source': 'original_ir_thermal', 'paths': {k: v['path'] for k, v in provenance.items()}, 'input_sha256': {f'{k}_features': v['sha256'] for k, v in provenance.items()} | {f'{k}_metadata': v['summary_sha256'] for k, v in provenance.items()}, 'cache_provenance': provenance, 'thermal_unavailable_rows_are_zero': True, 'limitation': 'external frozen cache; no label or test artifact loaded'}
    return (x, mask.astype(bool), metadata)

def _weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=40).astype(np.float64)
    present = counts > 0
    reference = counts[present].mean()
    w = np.zeros(40, dtype=np.float64)
    w[present] = np.power(reference / counts[present], 0.75)
    out = w[labels]
    return out / out.mean()

def align_scores(scores, classes):
    """Fixed source-only extension for newly nested missing-class contexts."""
    scores = np.asarray(scores, dtype=np.float64)
    classes = np.asarray(classes, dtype=np.int64)
    if len(classes) == 2 and (scores.ndim == 1 or scores.shape[1] == 1):
        margin = scores.reshape(-1)
        scores = np.column_stack((-margin, margin))
    elif scores.ndim == 1:
        scores = scores[:, None]
    if scores.ndim != 2 or scores.shape[1] != len(classes) or (not np.isfinite(scores).all()):
        raise ProtocolError('invalid Ridge class scores')
    floor = scores.min(1, keepdims=True) - np.maximum(np.ptp(scores, axis=1, keepdims=True), 1.0)
    out = np.repeat(floor, 40, axis=1)
    out[:, classes] = scores
    return out

class Provider:

    def __init__(self, matrix, ids, users, mask, provenance):
        self.x = np.asarray(matrix, dtype=np.float32)
        self.ids = np.asarray(ids).astype(str)
        self.users = np.asarray(users).astype(str)
        self.mask = np.asarray(mask, dtype=bool)
        self.provenance = dict(provenance or {})
        if self.x.ndim != 2 or self.x.shape[1] != 11520 or len(self.ids) != len(self.x) or (self.users.shape != self.ids.shape) or (self.mask.shape != (len(self.ids),)) or (len(np.unique(self.ids)) != len(self.ids)) or (not np.isfinite(self.x).all()):
            raise ProtocolError('invalid irthermal provider universe')

    def fit_predict(self, source, source_labels, target, *, context):
        source, target = (np.asarray(source), np.asarray(target))
        labels = np.asarray(source_labels)
        for ind in (source, target):
            if ind.ndim != 1 or ind.dtype.kind not in 'iu' or (not len(ind)) or (len(np.unique(ind)) != len(ind)) or np.any(ind < 0) or np.any(ind >= len(self.ids)):
                raise ProtocolError('invalid irthermal source/target indices')
        if labels.shape != source.shape or labels.dtype.kind not in 'iu' or np.any((labels < 0) | (labels >= 40)) or (len(np.unique(labels)) < 2) or np.intersect1d(source, target).size or set(self.users[source]) & set(self.users[target]) or (set(self.users[np.r_[source, target]]) & EXCLUDED_USERS):
            raise ProtocolError('invalid source-only irthermal context')
        labels = labels.astype(np.int64, copy=True)
        with threadpool_limits(limits=1):
            scaler = StandardScaler().fit(self.x[source])
            model = RidgeClassifier(alpha=3000.0, class_weight=None, solver='lsqr', tol=1e-05, max_iter=5000)
            model.fit(scaler.transform(self.x[source]), labels, sample_weight=_weights(labels))
            scores = np.asarray(model.decision_function(scaler.transform(self.x[target])), dtype=np.float64)
        classes = np.asarray(model.classes_, dtype=np.int64)
        aligned = align_scores(scores, classes)
        probability = softmax(aligned, axis=1).astype(np.float32)[:, None, :]
        node = f'{context}.ridge'
        receipt = {'context': context, 'source_ids': self.ids[source].tolist(), 'target_ids': self.ids[target].tolist(), 'source_users': self.users[source].tolist(), 'target_users': self.users[target].tolist(), 'source_label_sha256': array_hash(labels), 'source_class_counts': np.bincount(labels, minlength=40).tolist(), 'classes': classes.tolist(), 'missing_classes': sorted(set(range(40)) - set(classes.tolist())), 'provenance': self.provenance, 'prediction_nodes': [node], 'recipe': {'alpha': 3000.0, 'solver': 'lsqr', 'tol': 1e-05, 'max_iter': 5000, 'power': 0.75, 'temperature': 1.0, 'cpu_threads': 1, 'features': 11520, 'class_alignment': 'source_score_floor'}, 'artifact_dag': {'raw.external': {'node_id': 'raw.external', 'parents': [], 'provenance': 'frozen_external', 'has_task_labels': False, 'supervised_train_subjects': []}, node: {'node_id': node, 'parents': ['raw.external'], 'provenance': 'supervised', 'has_task_labels': True, 'supervised_train_subjects': sorted(set(self.users[source]))}}, 'target_labels_received': False, 'calibration': False, 'selection': False, 'provenance_checked': True}
        payload = {'mean': scaler.mean_.copy(), 'scale': scaler.scale_.copy(), 'var': scaler.var_.copy(), 'n_samples_seen': np.asarray(scaler.n_samples_seen_).copy(), 'coef': model.coef_.copy(), 'intercept': model.intercept_.copy(), 'classes': classes.copy()}
        payload['logits'] = aligned.copy()
        for user in np.unique(self.users[target]):
            assert_prediction_provenance(str(user), [node], {'raw.external': ArtifactNode('raw.external', provenance='frozen_external'), node: ArtifactNode(node, ('raw.external',), frozenset(self.users[source]), 'supervised', True)})
        return (probability, receipt, payload)
