from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
'Thermal population-aware source-only selection and refit provider.'
from dataclasses import asdict
import numpy as np
from sklearn.model_selection import GroupKFold
from aligned_multimodal.thermal_data import ThermalDataset
from aligned_multimodal.thermal_training import train_trajectory
from aligned_multimodal.foundation_provider import array_hash
from aligned_multimodal.stable_routing_protocol import ArtifactNode, ProtocolError, assert_prediction_provenance

def select_epoch(logits, labels):
    logits, labels = (np.asarray(logits), np.asarray(labels))
    if logits.shape != (15, len(labels), 40) or not np.isfinite(logits).all():
        raise ProtocolError('invalid source selection logits')
    correct = (logits.argmax(2) == labels[None]).sum(1)
    return (int(np.argmax(correct)) + 1, correct)

class ThermalProvider:

    def __init__(self, paths, thermal_ids):
        self.paths = dict(paths)
        self.ids = np.asarray(thermal_ids).astype(str)
        if len(set(self.ids)) != len(self.ids) or any((s not in paths for s in self.ids)):
            raise ProtocolError('invalid full thermal universe')

    def fit_predict(self, source, labels, users, target_ids, target_users, *, context, deadline=None, device='cuda', fit_callback=None):
        source = np.asarray(source)
        labels = np.asarray(labels)
        users = np.asarray(users).astype(str)
        target_ids = np.asarray(target_ids).astype(str)
        target_users = np.asarray(target_users).astype(str)
        if source.ndim != 1 or source.dtype.kind not in 'iu' or len(set(source)) != len(source) or np.any(source < 0) or np.any(source >= len(self.ids)) or (labels.shape != source.shape) or (users.shape != source.shape) or (labels.dtype.kind not in 'iu') or np.any((labels < 0) | (labels >= 40)) or (target_ids.ndim != 1) or (target_users.shape != target_ids.shape) or (not len(target_ids)) or (len(set(target_ids)) != len(target_ids)) or (set(self.ids[source]) & set(target_ids)) or (set(users) & set(target_users)) or (len(set(users)) < 3):
            raise ProtocolError('invalid source-only thermal context')
        raw = context + '.raw'
        nodes = {raw: ArtifactNode(raw, provenance='raw_input')}
        oof = np.zeros((15, len(source), 40), np.float32)
        coverage = np.zeros(len(source), int)
        fits = []
        selection_nodes = []
        for k, (tr, va) in enumerate(GroupKFold(3).split(source, groups=users)):
            node = context + f'.selection{k}'
            selection_nodes.append(node)
            nodes[node] = ArtifactNode(node, (raw,), frozenset(users[tr]), 'supervised', True)
            for u in set(users[va]):
                assert_prediction_provenance(u, [node], nodes)
            train = ThermalDataset(self.paths, self.ids, source[tr], augment=True)
            predict = ThermalDataset(self.paths, self.ids, source[va])
            logits, fit_state, d = train_trajectory(train, predict, labels[tr], deadline=deadline, device=device)
            if logits.shape != (15, len(va), 40) or not np.isfinite(logits).all():
                raise ProtocolError('incomplete thermal selection trajectory')
            oof[:, va] = logits
            coverage[va] += 1
            if fit_callback is not None:
                fit_callback(f'selection{k}', logits, fit_state, d, self.ids[source[tr]], self.ids[source[va]])
            fits.append({'node': node, 'train_ids': self.ids[source[tr]].tolist(), 'train_users': users[tr].tolist(), 'prediction_ids': self.ids[source[va]].tolist(), 'prediction_users': users[va].tolist(), 'diagnostics': d})
        if not np.all(coverage == 1):
            raise ProtocolError('source thermal OOF coverage')
        epoch, correct = select_epoch(oof, labels)
        choice = context + '.epoch_choice'
        refit = context + '.refit'
        nodes[choice] = ArtifactNode(choice, tuple(selection_nodes), frozenset(users), 'supervised', True)
        nodes[refit] = ArtifactNode(refit, (raw, choice), frozenset(users), 'supervised', True)
        present = np.asarray([s in self.paths for s in target_ids])
        target_dataset = ThermalDataset(self.paths, target_ids, np.flatnonzero(present))
        trajectory, state, d = train_trajectory(ThermalDataset(self.paths, self.ids, source, augment=True), target_dataset, labels, epochs=epoch, collect_each_epoch=False, deadline=deadline, device=device)
        if trajectory.shape != (1, int(present.sum()), 40) or not np.isfinite(trajectory).all():
            raise ProtocolError('invalid thermal refit trajectory')
        if fit_callback is not None:
            fit_callback('refit', trajectory, state, d, self.ids[source], target_ids[present])
        logits = np.zeros((len(target_ids), 40), np.float32)
        logits[present] = trajectory[-1]
        probability = np.exp(logits - logits.max(1, keepdims=True))
        probability /= probability.sum(1, keepdims=True)
        for u in set(target_users):
            assert_prediction_provenance(u, [refit], nodes)
        receipt = {'context': context, 'source_ids': self.ids[source].tolist(), 'source_users': users.tolist(), 'target_ids': target_ids.tolist(), 'target_users': target_users.tolist(), 'selected_epoch': epoch, 'source_epoch_correct': correct.tolist(), 'selection_fits': fits, 'refit_diagnostics': d, 'source_label_sha256': array_hash(labels), 'selection_logit_sha256': array_hash(oof), 'artifact_dag': {k: {**asdict(v), 'supervised_train_subjects': sorted(v.supervised_train_subjects)} for k, v in nodes.items()}, 'prediction_nodes': [refit], 'provenance_checked': True, 'missing_logits': 'zeros -> uniform40', 'target_labels_received': False, 'historical_task_checkpoint_loaded': False}
        return (probability[:, None, :], receipt, {'logits': logits, 'thermal_present': present, 'selection_ids': self.ids[source], 'selection_labels': labels, 'selection_logits': oof}, state)
