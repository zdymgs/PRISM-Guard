from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
'Source-only physical/physical_tokens physical-token provider.'
import copy
import hashlib
import time
from dataclasses import asdict
from typing import Any
import numpy as np
from aligned_multimodal.foundation_provider import array_hash
from aligned_multimodal.physical_training import FIXED_RECIPE, train_member
from aligned_multimodal.physical_cache import PhysicalTokenCache, _manifest, FAMILY_NAMES
from aligned_multimodal.stable_routing_protocol import ArtifactNode, ProtocolError, assert_prediction_provenance
EXCLUDED_USERS = frozenset(('user1', 'user2', 'user21'))
BASE_SEEDS = (23801, 23817, 23833)
EXPERT_NAME = 'physical_token'

def _indices(value: Any, n: int, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in 'iu' or raw.dtype.kind == 'b' or (not len(raw)) or (len(set(raw.tolist())) != len(raw)) or np.any(raw < 0) or np.any(raw >= n):
        raise ProtocolError(f'invalid {name} indices')
    return raw.astype(np.int64, copy=False)

def _sha(path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

class PhysicalProvider:
    """One deterministic three-member physical_tokens probability expert."""

    def __init__(self, cache: PhysicalTokenCache, ids, users):
        self.cache = cache
        self.ids = np.asarray(ids)
        self.users = np.asarray(users)
        if self.ids.ndim != 1 or self.ids.dtype.kind not in 'OUS' or (not len(self.ids)) or any((not isinstance(x, (str, np.str_)) or not x.strip() for x in self.ids)) or (len(set(self.ids.astype(str).tolist())) != len(self.ids)) or (self.users.ndim != 1) or (self.users.shape != self.ids.shape) or any((not isinstance(x, (str, np.str_)) or not x.strip() for x in self.users)):
            raise ProtocolError('invalid physical provider universe')
        self.ids = self.ids.astype(str)
        self.users = self.users.astype(str)
        master = {sid: i for i, sid in enumerate(cache.master_ids.astype(str))}
        if any((sid not in master for sid in self.ids)):
            raise ProtocolError('provider IDs are not a cache subset')
        positions = np.asarray([master[sid] for sid in self.ids], dtype=np.int64)
        if not np.array_equal(cache.master_users[positions].astype(str), self.users):
            raise ProtocolError('provider users differ from canonical cache binding')
        self.x = np.asarray(cache.select(self.ids), dtype=np.float32)
        if self.x.shape != (len(self.ids), 18, 768) or not np.isfinite(self.x).all():
            raise ProtocolError('invalid physical provider features')
        self.cache_provenance = copy.deepcopy(cache.provenance)

    def _revalidate_cache(self) -> None:
        if not isinstance(self.cache, PhysicalTokenCache) or self.cache_provenance.get('families') != list(FAMILY_NAMES) or self.cache_provenance.get('type') != 'frozen_external' or (len(self.cache.paths) != 4):
            raise ProtocolError('real physical fit requires original physical cache')
        if self.cache.provenance != self.cache_provenance:
            raise ProtocolError('physical cache provenance changed')
        expected = {**{f'{n}_features': _sha(p) for n, p in zip(self.cache.provenance['families'], self.cache.paths)}, **{f'{n}_metadata': _sha(p.with_name('cache_summary.json')) for n, p in zip(self.cache.provenance['families'], self.cache.paths)}, 'canonical_manifest': _sha(self.cache.manifest)}
        if expected != self.cache_provenance.get('input_sha256'):
            raise ProtocolError('physical frozen cache input changed')
        master_ids, master_users = _manifest(self.cache.manifest)
        bound = dict(zip(master_ids, master_users))
        if any((i not in bound or bound[i] != u for i, u in zip(self.ids, self.users))):
            raise ProtocolError('physical user identity differs from pinned manifest')

    def fit_predict(self, source, source_labels, target, *, outer_fold, context, deadline=None, device='cuda', fit_fn=None, fit_callback=None):
        source = _indices(source, len(self.ids), 'source')
        target = _indices(target, len(self.ids), 'target')
        labels = np.asarray(source_labels)
        if set(source.tolist()) & set(target.tolist()) or labels.ndim != 1 or labels.shape != source.shape or (labels.dtype.kind not in 'iu') or (labels.dtype.kind == 'b') or np.any((labels < 0) | (labels >= 40)):
            raise ProtocolError('invalid physical source-only context')
        if isinstance(outer_fold, (bool, np.bool_)) or not isinstance(outer_fold, (int, np.integer)) or int(outer_fold) not in (0, 1, 2):
            raise ProtocolError('invalid physical outer fold')
        if not isinstance(context, str) or not context.strip():
            raise ProtocolError('invalid physical context')
        source_users = self.users[source]
        target_users = self.users[target]
        if set(source_users.tolist()) & set(target_users.tolist()) or bool((set(source_users.tolist()) | set(target_users.tolist())) & EXCLUDED_USERS):
            raise ProtocolError('invalid physical source/target users')
        real_fit = fit_fn is None
        if real_fit:
            if not callable(fit_callback):
                raise ProtocolError('real physical fit requires durable member callback')
            self._revalidate_cache()
        fitter = train_member if real_fit else fit_fn
        nodes: dict[str, ArtifactNode] = {}
        external = context + '.external_physical'
        nodes[external] = ArtifactNode(external, provenance='frozen_external')
        member_logits = []
        fits = []
        for base in BASE_SEEDS:
            seed = int(base + 1000 * int(outer_fold))
            node = context + f'.seed{seed}'
            nodes[node] = ArtifactNode(node, (external,), frozenset(source_users.tolist()), 'supervised', True)
            logits, state, diagnostics = fitter(self.x[source], labels.copy(), self.x[target], seed=seed, deadline=deadline, device=device)
            logits = np.asarray(logits, dtype=np.float32)
            if logits.shape != (len(target), 40) or not np.isfinite(logits).all():
                raise ProtocolError('invalid physical member logits')
            record = {'node': node, 'seed': seed, 'source_ids': self.ids[source].tolist(), 'source_users': source_users.tolist(), 'target_ids': self.ids[target].tolist(), 'target_users': target_users.tolist(), 'diagnostics': copy.deepcopy(diagnostics)}
            if real_fit or callable(fit_callback):
                fit_callback(seed, logits, state, record)
            member_logits.append(logits)
            fits.append(record)
        member = np.stack(member_logits, axis=0).astype(np.float32, copy=False)
        mean_logits = member.mean(axis=0, dtype=np.float32)
        if not np.isfinite(mean_logits).all():
            raise ProtocolError('physical mean logits overflow')
        score = mean_logits.astype(np.float64)
        score -= score.max(axis=1, keepdims=True)
        probability = np.exp(score)
        probability /= probability.sum(axis=1, keepdims=True)
        probability = probability.astype(np.float32)
        if not np.isfinite(probability).all() or not np.allclose(probability.sum(1), 1, rtol=1e-06, atol=1e-07):
            raise ProtocolError('physical probability invalid')
        probability_node = context + '.probability'
        nodes[probability_node] = ArtifactNode(probability_node, tuple((context + f'.seed{int(b + 1000 * int(outer_fold))}' for b in BASE_SEEDS)))
        for user in set(target_users.tolist()):
            assert_prediction_provenance(user, [probability_node], nodes)
        receipt = {'context': context, 'outer_fold': int(outer_fold), 'expert_name': EXPERT_NAME, 'seed_bases': list(BASE_SEEDS), 'seeds': [int(b + 1000 * int(outer_fold)) for b in BASE_SEEDS], 'recipe': copy.deepcopy(FIXED_RECIPE), 'cache_provenance': copy.deepcopy(self.cache_provenance), 'source_ids': self.ids[source].tolist(), 'source_users': source_users.tolist(), 'target_ids': self.ids[target].tolist(), 'target_users': target_users.tolist(), 'source_label_sha256': array_hash(labels), 'source_class_counts': np.bincount(labels.astype(np.int64), minlength=40).tolist(), 'fits': fits, 'member_count': 3, 'aggregation': 'mean_float32_logits_then_softmax_float64', 'target_labels_received': False, 'checkpoint_selection': False, 'injected_test_fitter': not real_fit, 'member_callback_used': callable(fit_callback), 'prediction_node': probability_node, 'provenance_checked': True, 'artifact_dag': {k: {**asdict(v), 'supervised_train_subjects': sorted(v.supervised_train_subjects)} for k, v in nodes.items()}}
        return (probability[:, None, :], receipt, {'member_logits': member, 'mean_logits': mean_logits})
Provider = PhysicalProvider
