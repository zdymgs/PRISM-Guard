from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
'Actual token/protocol slots; no proxy-bank composition or target labels.'
from dataclasses import asdict
import copy
import numpy as np
from aligned_multimodal.token_training import train_member
from aligned_multimodal.foundation_provider import array_hash
from aligned_multimodal.stable_routing_protocol import ArtifactNode, ProtocolError, assert_prediction_provenance
EXPERTS = ('all_token', 'hand_interaction')
TOKEN_INDICES = (tuple(range(24)), (14, 17, 20, 23))
BASE_SEEDS = ((14201, 14217, 14233), (14401, 14417, 14433))

class TokenProvider:

    def __init__(self, tokens, sample_ids, users, *, cache_provenance=None):
        self.cache_provenance = copy.deepcopy(cache_provenance)
        self.x = np.asarray(tokens)
        self.ids = np.asarray(sample_ids).astype(str)
        self.users = np.asarray(users).astype(str)
        if self.ids.ndim != 1 or self.users.shape != self.ids.shape or len(set(self.ids)) != len(self.ids) or (self.x.shape != (len(self.ids), 24, 1024)) or (self.x.dtype.kind != 'f') or (not np.isfinite(self.x).all()):
            raise ProtocolError('invalid token provider universe')

    def fit_predict(self, source, source_labels, target, *, outer_fold, context, deadline=None, device='cuda', fit_fn=None, fit_callback=None):
        if fit_fn is None:
            p = self.cache_provenance or {}
            hashes = p.get('input_sha256', {})
            if fit_callback is None or p.get('type') != 'frozen_external' or p.get('labels_loaded') is not False or (p.get('model_repo') != 'facebook/vjepa2-vitl-fpc16-256-ssv2') or (len(hashes) != 5) or any((len(v) != 64 or any((c not in '0123456789abcdef' for c in v)) for v in hashes.values())):
                raise ProtocolError('real token fit requires bound cache provenance and durable member callback')
        source, target = (np.asarray(source), np.asarray(target))
        labels = np.asarray(source_labels)
        for ix in (source, target):
            if ix.ndim != 1 or ix.dtype.kind not in 'iu' or (not len(ix)) or (len(set(ix)) != len(ix)) or np.any(ix < 0) or np.any(ix >= len(self.ids)):
                raise ProtocolError('invalid source/target indices')
        su = set(self.users[source])
        tu = set(self.users[target])
        if su & tu or (su | tu) & {'user1', 'user2', 'user21'} or labels.shape != source.shape or (labels.dtype.kind not in 'iu') or np.any((labels < 0) | (labels >= 40)):
            raise ProtocolError('source-only token context invalid')
        if isinstance(outer_fold, bool) or not isinstance(outer_fold, (int, np.integer)) or outer_fold not in (0, 1, 2):
            raise ProtocolError('invalid original outer fold')
        fitter = train_member if fit_fn is None else fit_fn
        external = context + '.external_vjepa'
        nodes = {external: ArtifactNode(external, provenance='frozen_external')}
        bank = []
        all_members = []
        receipts = []
        roots = []
        for expert, columns, base_seeds in zip(EXPERTS, TOKEN_INDICES, BASE_SEEDS):
            train_x = self.x[source][:, columns, :]
            target_x = self.x[target][:, columns, :]
            members = []
            member_nodes = []
            for base_seed in base_seeds:
                seed = int(base_seed + 1000 * outer_fold)
                logits, state, diagnostics = fitter(train_x, labels.copy(), target_x, seed=seed, deadline=deadline, device=device)
                logits = np.asarray(logits, dtype=np.float32)
                if logits.shape != (len(target), 40) or not np.isfinite(logits).all():
                    raise ProtocolError('invalid token member logits')
                node = context + f'.{expert}.seed{seed}'
                member_nodes.append(node)
                nodes[node] = ArtifactNode(node, (external,), frozenset(su), 'supervised', True)
                record = {'node': node, 'expert': expert, 'seed': seed, 'token_indices': list(columns), 'source_ids': self.ids[source].tolist(), 'target_ids': self.ids[target].tolist(), 'diagnostics': diagnostics}
                if fit_callback is not None:
                    fit_callback(expert, seed, logits, state, record)
                receipts.append(record)
                members.append(logits)
            members = np.stack(members)
            all_members.append(members)
            mean = members.mean(0)
            p = np.exp(mean - mean.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            bank.append(p.astype(np.float32))
            root = context + f'.{expert}.mean_logits'
            nodes[root] = ArtifactNode(root, tuple(member_nodes))
            roots.append(root)
        for user in tu:
            assert_prediction_provenance(user, roots, nodes)
        receipt = {'context': context, 'outer_fold': int(outer_fold), 'expert_names': list(EXPERTS), 'cache_provenance': copy.deepcopy(self.cache_provenance), 'injected_test_fitter': fit_fn is not None, 'member_callback_used': fit_callback is not None, 'source_ids': self.ids[source].tolist(), 'source_users': self.users[source].tolist(), 'target_ids': self.ids[target].tolist(), 'target_users': self.users[target].tolist(), 'source_label_sha256': array_hash(labels), 'source_class_counts': np.bincount(labels, minlength=40).tolist(), 'fits': receipts, 'aggregation': 'softmax(mean_three_logits)', 'target_labels_received': False, 'checkpoint_selection': False, 'prediction_nodes': roots, 'provenance_checked': True, 'artifact_dag': {k: {**asdict(v), 'supervised_train_subjects': sorted(v.supervised_train_subjects)} for k, v in nodes.items()}}
        return (np.stack(bank, axis=1), receipt, {'member_logits': np.stack(all_members)})
