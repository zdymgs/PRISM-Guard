from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
'Contextual reconstruction of sixteen actual multimodal foundation expert slots.\n\nThe provider stores no label array. Each call receives source labels only;\ntarget labels cannot enter model fitting or temperature calibration.\n'
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from sklearn.model_selection import GroupKFold
from aligned_multimodal.foundation_kernels import FEATURE_RECIPES, normalize_tokens, feature_sets, fixed_perturbations, fit_ridge, aligned_scores, fit_temperature, softmax
from aligned_multimodal.stable_routing_protocol import ArtifactNode, ProtocolError, assert_prediction_provenance
PERTURBATIONS = ('baseline', 'drop_scene', 'drop_person', 'drop_workspace', 'drop_early', 'drop_late', 'swap_early_late', 'collapse_early_late', 'swap_person_workspace', 'collapse_view_identity')
EXPERT_NAMES = tuple((f'head_{name}_logits' for name in FEATURE_RECIPES)) + tuple((f'mechanism_{name}_logits' for name in PERTURBATIONS))

def array_hash(array):
    x = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode())
    h.update(str(x.shape).encode())
    h.update(x.tobytes())
    return h.hexdigest()

def _indices(values, n):
    x = np.asarray(values)
    if x.ndim != 1 or not np.issubdtype(x.dtype, np.integer) or (not len(x)) or (len(np.unique(x)) != len(x)) or np.any((x < 0) | (x >= n)):
        raise ProtocolError('indices must be unique nonempty in-range integer vectors')
    return x.astype(np.int64)

def graph_subset(nodes, roots):
    result = {}

    def visit(key):
        if key in result:
            return
        node = nodes[key]
        result[key] = {'node_id': key, 'parents': list(node.parents), 'provenance': node.provenance, 'has_task_labels': node.has_task_labels, 'supervised_train_subjects': sorted(node.supervised_train_subjects)}
        for parent in node.parents:
            visit(parent)
    for root in roots:
        visit(root)
    return result

class FoundationProvider:

    def __init__(self, raw_features, kinetics_logits, sample_ids):
        self.ids = np.asarray(sample_ids).astype(str).copy()
        if self.ids.ndim != 1 or len(set(self.ids)) != len(self.ids) or len(raw_features) != len(self.ids):
            raise ProtocolError('foundation raw inputs need unique aligned IDs')
        self.matrices = feature_sets(raw_features, kinetics_logits)
        self.normalized = normalize_tokens(raw_features)
        for matrix in self.matrices.values():
            matrix.flags.writeable = False
        self.normalized.flags.writeable = False
        self.models = {}
        self.calibrators = {}
        self.nodes = {'raw.p85_external': ArtifactNode('raw.p85_external', provenance='frozen_external')}
        self.records = {}
        self.calibration_arrays = {}
        self.model_fit_count = 0
        self.deadline = None

    def _key(self, name, train, labels, subjects):
        payload = {'feature': name, 'indices': train.tolist(), 'labels': labels.tolist(), 'subjects': subjects.tolist(), 'recipe': FEATURE_RECIPES[name]}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _check_time(self):
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise TimeoutError('foundation pilot exceeded600seconds')

    def _model(self, name, train, labels, subjects):
        key = self._key(name, train, labels, subjects)
        node_id = f'ridge.{name}.{key}'
        if key not in self.models:
            self._check_time()
            started = time.monotonic()
            recipe = FEATURE_RECIPES[name]
            model = fit_ridge(self.matrices[name][train], labels, recipe['alpha'], recipe['power'])
            self.models[key] = model
            self.model_fit_count += 1
            self.nodes[node_id] = ArtifactNode(node_id, parents=('raw.p85_external',), provenance='supervised', has_task_labels=True, supervised_train_subjects=frozenset(subjects))
            self.records[node_id] = {'kind': 'source_weighted_ridge', 'feature': name, **recipe, 'train_indices': train.tolist(), 'train_ids': self.ids[train].tolist(), 'train_subjects': sorted(set(subjects)), 'source_label_sha256': array_hash(labels), 'scaler_mean_sha256': array_hash(model.named_steps['scale'].mean_), 'coefficient_sha256': array_hash(model.named_steps['ridge'].coef_), 'classes': model.named_steps['ridge'].classes_.tolist(), 'seconds': time.monotonic() - started}
            print(json.dumps({'event': 'foundation_head_fit', 'feature': name, 'train_rows': len(train), 'unique_fit_count': self.model_fit_count, 'seconds': self.records[node_id]['seconds']}), flush=True)
        return (self.models[key], node_id)

    def _temperature(self, name, train, labels, subjects):
        key = self._key(name, train, labels, subjects)
        node_id = f'temperature.{name}.{key}'
        if key not in self.calibrators:
            groups = np.unique(subjects)
            if len(groups) < 2:
                raise ProtocolError('source calibration needs at least two subjects')
            raw_oof = np.zeros((len(train), 40))
            covered = np.zeros(len(train), int)
            parents = []
            partitions = []
            for fold, (fit_local, held_local) in enumerate(GroupKFold(min(3, len(groups))).split(train, labels, subjects)):
                model, parent = self._model(name, train[fit_local], labels[fit_local], subjects[fit_local])
                for subject in np.unique(subjects[held_local]):
                    assert_prediction_provenance(str(subject), [parent], self.nodes)
                raw_oof[held_local] = aligned_scores(model, self.matrices[name][train[held_local]])
                covered[held_local] += 1
                parents.append(parent)
                partitions.append({'fold': fold, 'fit_indices': train[fit_local].tolist(), 'held_indices': train[held_local].tolist(), 'fit_ids': self.ids[train[fit_local]].tolist(), 'held_ids': self.ids[train[held_local]].tolist(), 'fit_subjects': sorted(set(subjects[fit_local])), 'held_subjects': sorted(set(subjects[held_local])), 'model_node': parent})
            if not np.all(covered == 1):
                raise ProtocolError('calibration OOF coverage incomplete')
            self._check_time()
            temperature = fit_temperature(raw_oof, labels)
            self.calibrators[key] = temperature
            self.nodes[node_id] = ArtifactNode(node_id, parents=tuple(parents), provenance='supervised', has_task_labels=True, supervised_train_subjects=frozenset(subjects))

            def nll(t):
                z = raw_oof / t
                return float(np.mean(logsumexp(z, axis=1) - z[np.arange(len(labels)), labels]))
            self.records[node_id] = {'kind': 'source_oof_temperature', 'feature': name, 'temperature': temperature, 'fit_ids': self.ids[train].tolist(), 'fit_subjects': sorted(set(subjects)), 'source_label_sha256': array_hash(labels), 'oof_score_sha256': array_hash(raw_oof), 'source_nll_before': nll(1), 'source_nll_after': nll(temperature), 'partitions': partitions}
            self.calibration_arrays[node_id] = {'sample_ids': self.ids[train].copy(), 'scores': raw_oof, 'source_labels': labels.copy()}
        return (self.calibrators[key], node_id)

    def fit_predict(self, train_indices, source_labels, source_subjects, target_indices, target_subjects, *, context, requested_names=None, deadline=None):
        train = _indices(train_indices, len(self.ids))
        target = _indices(target_indices, len(self.ids))
        labels = np.asarray(source_labels)
        subjects = np.asarray(source_subjects).astype(str)
        held_subjects = np.asarray(target_subjects).astype(str)
        if labels.shape != (len(train),) or not np.issubdtype(labels.dtype, np.number) or (not np.isfinite(labels).all()) or np.any(labels != np.floor(labels)) or np.any((labels < 0) | (labels >= 40)):
            raise ProtocolError('source labels must be aligned integer40-class targets')
        labels = labels.astype(np.int64, copy=True)
        if subjects.shape != (len(train),) or held_subjects.shape != (len(target),):
            raise ProtocolError('subjects do not align with context rows')
        if np.intersect1d(train, target).size or set(subjects) & set(held_subjects):
            raise ProtocolError('source/target context is not subject-disjoint')
        if any((not str(s).strip() or str(s).lower() in ('nan', 'none') for s in np.r_[subjects, held_subjects])):
            raise ProtocolError('subject metadata missing')
        names = EXPERT_NAMES if requested_names is None else tuple(requested_names)
        if len(set(names)) != len(names) or any((name not in EXPERT_NAMES for name in names)):
            raise ProtocolError('requested champion expert is unavailable; no fallback allowed')
        self.deadline = deadline
        self._check_time()
        start = time.monotonic()
        before = self.model_fit_count
        values = {}
        pred_nodes = {}
        models = {}
        for name in FEATURE_RECIPES:
            model, head_node = self._model(name, train, labels, subjects)
            models[name] = (model, head_node)
            temperature, cal_node = self._temperature(name, train, labels, subjects)
            expert = f'head_{name}_logits'
            values[expert] = softmax(aligned_scores(model, self.matrices[name][target]) / temperature)
            pred = f'{context}.{expert}'
            self.nodes[pred] = ArtifactNode(pred, parents=(head_node, cal_node))
            pred_nodes[expert] = pred
        mean = self.normalized[train].mean(axis=0)
        mean_node = f'{context}.source_imputation_mean'
        self.nodes[mean_node] = ArtifactNode(mean_node, parents=('raw.p85_external',))
        self.records[mean_node] = {'kind': 'source_only_unlabeled_mean', 'train_indices': train.tolist(), 'train_ids': self.ids[train].tolist(), 'fit_subjects': sorted(set(subjects)), 'mean_sha256': array_hash(mean)}
        model, head_node = models['early_late']
        perturbed = fixed_perturbations(self.normalized[target], mean)
        if tuple(perturbed) != PERTURBATIONS:
            raise ProtocolError('mechanism condition order changed')
        for condition, matrix in perturbed.items():
            expert = f'mechanism_{condition}_logits'
            values[expert] = softmax(aligned_scores(model, matrix.reshape(len(target), -1)))
            pred = f'{context}.{expert}'
            self.nodes[pred] = ArtifactNode(pred, parents=(head_node, mean_node))
            pred_nodes[expert] = pred
        roots = [pred_nodes[name] for name in names]
        for subject in np.unique(held_subjects):
            assert_prediction_provenance(str(subject), roots, self.nodes)
        bank = np.stack([values[name] for name in names], axis=1)
        if bank.shape != (len(target), len(names), 40) or not np.isfinite(bank).all() or (not np.allclose(bank.sum(2), 1)):
            raise ProtocolError('foundation output schema invalid')
        if not np.array_equal(values['p85_head_early_late_logits'].argmax(1), values['p86_mechanism_baseline_logits'].argmax(1)):
            raise ProtocolError('shared early_late/mechanism baseline head diverged')
        graph = graph_subset(self.nodes, roots)
        receipt = {'context': context, 'source_indices': train.tolist(), 'target_indices': target.tolist(), 'source_ids': self.ids[train].tolist(), 'target_ids': self.ids[target].tolist(), 'source_subjects': sorted(set(subjects)), 'target_subjects': sorted(set(held_subjects)), 'expert_names': list(names), 'prediction_nodes': roots, 'artifact_dag': graph, 'fit_receipts': {key: self.records[key] for key in graph if key in self.records}, 'new_ridge_fits': self.model_fit_count - before, 'total_cached_ridge_fits': self.model_fit_count, 'seconds': time.monotonic() - start, 'provenance_checked': True, 'complete_p315': False}
        arrays = {'source_imputation_mean': mean}
        for i, key in enumerate((k for k in graph if k in self.calibration_arrays)):
            data = self.calibration_arrays[key]
            arrays[f'calibration{i}_scores'] = data['scores']
            arrays[f'calibration{i}_labels'] = data['source_labels']
            arrays[f'calibration{i}_ids'] = data['sample_ids']
            receipt.setdefault('calibration_array_nodes', {})[str(i)] = key
        self._check_time()
        return (bank, receipt, arrays)
