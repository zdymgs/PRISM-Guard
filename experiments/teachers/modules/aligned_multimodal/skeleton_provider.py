from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
'Original sensor skeleton with source-internal checkpoint selection.'
from dataclasses import asdict
import numpy as np
from sklearn.model_selection import GroupKFold
from aligned_multimodal.skeleton_data import SkeletonDataset
from aligned_multimodal.skeleton_training import train_trajectory
from aligned_multimodal.foundation_provider import array_hash
from aligned_multimodal.stable_routing_protocol import ArtifactNode, ProtocolError, assert_prediction_provenance

def select_epoch(logits, labels):
    logits, labels = (np.asarray(logits), np.asarray(labels))
    if logits.shape != (15, len(labels), 40) or not np.isfinite(logits).all():
        raise ProtocolError('invalid source selection logits')
    correct = (logits.argmax(2) == labels[None]).sum(1)
    return (int(np.argmax(correct)) + 1, correct)

class SkeletonProvider:

    def __init__(self, cache_dir, sample_ids):
        self.cache_dir = cache_dir
        self.ids = np.asarray(sample_ids).astype(str)
        if len(set(self.ids)) != len(self.ids):
            raise ProtocolError('duplicate universe IDs')

    def fit_predict(self, source, labels, users, target, target_users, *, context, deadline=None, device='cuda'):
        source, target = (np.asarray(source, int), np.asarray(target, int))
        labels, users, target_users = (np.asarray(labels), np.asarray(users).astype(str), np.asarray(target_users).astype(str))
        if len(source) != len(labels) or len(source) != len(users) or len(target) != len(target_users) or (not len(target)) or (len(set(source)) != len(source)) or (len(set(target)) != len(target)) or (set(source) & set(target)) or (set(users) & set(target_users)) or np.any(source < 0) or np.any(target < 0) or np.any(source >= len(self.ids)) or np.any(target >= len(self.ids)) or (labels.dtype.kind not in 'iu') or np.any((labels < 0) | (labels >= 40)):
            raise ProtocolError('invalid source-only context')
        if len(set(users)) < 3:
            raise ProtocolError('three source subjects required')
        raw = context + '.raw'
        nodes = {raw: ArtifactNode(raw, provenance='raw_input')}
        receipts, selection_nodes = ([], [])
        oof = np.zeros((15, len(source), 40), np.float32)
        coverage = np.zeros(len(source), int)
        for k, (tr, va) in enumerate(GroupKFold(3).split(source, groups=users)):
            node = context + f'.selection{k}'
            nodes[node] = ArtifactNode(node, (raw,), frozenset(users[tr]), 'supervised', True)
            for subject in set(users[va]):
                assert_prediction_provenance(subject, [node], nodes)
            train = SkeletonDataset(self.cache_dir, self.ids, source[tr], labels[tr], True)
            predict = SkeletonDataset(self.cache_dir, self.ids, source[va])
            trajectory, _, diagnostics = train_trajectory(train, predict, labels[tr], deadline=deadline, device=device)
            if trajectory.shape != (15, len(va), 40) or not np.isfinite(trajectory).all():
                raise ProtocolError('selection trajectory incomplete')
            oof[:, va] = trajectory
            coverage[va] += 1
            selection_nodes.append(node)
            receipts.append({'node': node, 'train_ids': self.ids[source[tr]].tolist(), 'train_users': users[tr].tolist(), 'prediction_ids': self.ids[source[va]].tolist(), 'prediction_users': users[va].tolist(), 'diagnostics': diagnostics})
        if not np.all(coverage == 1):
            raise ProtocolError('source selection coverage')
        epoch, correct = select_epoch(oof, labels)
        selection = context + '.epoch_choice'
        nodes[selection] = ArtifactNode(selection, tuple(selection_nodes), frozenset(users), 'supervised', True)
        refit = context + '.refit'
        nodes[refit] = ArtifactNode(refit, (raw, selection), frozenset(users), 'supervised', True)
        train = SkeletonDataset(self.cache_dir, self.ids, source, labels, True)
        predict = SkeletonDataset(self.cache_dir, self.ids, target)
        trajectory, state, diagnostics = train_trajectory(train, predict, labels, epochs=epoch, collect_each_epoch=False, deadline=deadline, device=device)
        if trajectory.shape != (1, len(target), 40) or not np.isfinite(trajectory).all():
            raise ProtocolError('refit trajectory invalid')
        logits = trajectory[-1]
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        for subject in set(target_users):
            assert_prediction_provenance(subject, [refit], nodes)
        dag = {k: {**asdict(v), 'supervised_train_subjects': sorted(v.supervised_train_subjects)} for k, v in nodes.items()}
        receipt = {'context': context, 'source_ids': self.ids[source].tolist(), 'source_users': users.tolist(), 'target_ids': self.ids[target].tolist(), 'target_users': target_users.tolist(), 'selected_epoch': epoch, 'source_epoch_correct': correct.tolist(), 'source_label_sha256': array_hash(labels), 'selection_logit_sha256': array_hash(oof), 'selection_fits': receipts, 'refit_diagnostics': diagnostics, 'artifact_dag': dag, 'prediction_nodes': [refit], 'provenance_checked': True, 'target_labels_received': False, 'historical_task_checkpoint_loaded': False}
        return (p[:, None, :], receipt, {'logits': logits, 'selection_logits': oof, 'selection_labels': labels, 'selection_ids': self.ids[source]}, state)
