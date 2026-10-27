from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
"Source-only provider for the frozen lavila LaViLa frame-token head.\n\nThe provider deliberately owns no labels and materialises features only through\n``LaViLaCache.select``.  This keeps the cache's ID join (rather than an array\nposition assumption) at the boundary of every fitting context.\n"
from dataclasses import asdict
import copy
from typing import Any
import numpy as np
from aligned_multimodal.foundation_provider import array_hash
from aligned_multimodal.lavila_training import FIXED_RECIPE, train_member
from aligned_multimodal.lavila_cache import CHECKPOINT_SHA
from aligned_multimodal.probe_contract import validate_spotcheck
from aligned_multimodal.stable_routing_protocol import ArtifactNode, ProtocolError, assert_prediction_provenance
EXCLUDED_USERS = frozenset(('user1', 'user2', 'user21'))
BASE_SEED = 15801

def _indices(value: Any, n: int, name: str) -> np.ndarray:
    ix = np.asarray(value)
    if ix.ndim != 1 or ix.dtype.kind not in 'iu' or ix.dtype.kind == 'b' or (not len(ix)) or (len(set(ix.tolist())) != len(ix)) or np.any(ix < 0) or np.any(ix >= n):
        raise ProtocolError(f'invalid {name} indices')
    return ix.astype(np.int64, copy=False)

def _hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all((c in '0123456789abcdefABCDEF' for c in value))

class ClassProvider:
    """One deterministic, single-seed lavila/lavila probability expert."""

    def __init__(self, cache, ids, users, probe_evidence):
        self.cache = cache
        self.ids = np.asarray(ids).astype(str)
        self.users = np.asarray(users).astype(str)
        if self.ids.ndim != 1 or not len(self.ids) or len(set(self.ids.tolist())) != len(self.ids) or (self.users.shape != self.ids.shape) or any((not x.strip() for x in self.ids)):
            raise ProtocolError('invalid LaViLa provider universe')
        try:
            selected = cache.select(self.ids)
        except Exception as exc:
            if isinstance(exc, ProtocolError):
                raise
            raise ProtocolError('LaViLa cache selection failed') from exc
        self.x = np.asarray(selected)
        if self.x.shape != (len(self.ids), 48, 768) or self.x.dtype.kind not in 'fiu' or (not np.isfinite(self.x).all()):
            raise ProtocolError('invalid LaViLa provider features')
        self.x = self.x.astype(np.float32, copy=False)
        if not np.isfinite(self.x).all():
            raise ProtocolError('LaViLa features overflow float32')
        self.probe_evidence = copy.deepcopy(probe_evidence)
        self.cache_provenance = copy.deepcopy(getattr(cache, 'provenance', None))
        if not isinstance(self.cache_provenance, dict):
            self.cache_provenance = {}

    def _validate_probe(self) -> None:
        evidence = self.probe_evidence
        hashes = self.cache_provenance.get('input_sha256', {})
        if not isinstance(evidence, dict) or evidence.get('validated') is not True:
            raise ProtocolError('real LaViLa fit requires validated probe evidence')
        if not isinstance(hashes, dict) or len(hashes) != 7 or any((not _hex64(v) for v in hashes.values())) or (evidence.get('input_sha256') != hashes) or (self.cache_provenance.get('type') != 'frozen_external') or (self.cache_provenance.get('model') != 'LaViLa_TimeSformer_B') or (self.cache_provenance.get('checkpoint_sha256') != CHECKPOINT_SHA) or (evidence.get('checkpoint_sha256') != CHECKPOINT_SHA) or (evidence.get('probe_count') != 2):
            raise ProtocolError('LaViLa probe input hashes do not match cache hashes')
        for key in ('registry_sha256', 'tokens_sha256', 'summary_sha256', 'process_sha256'):
            if not _hex64(evidence.get(key)):
                raise ProtocolError('LaViLa probe evidence is missing durable hashes')
        if not isinstance(evidence.get('probe_dir'), str):
            raise ProtocolError('LaViLa probe path missing')
        actual = validate_spotcheck(self.cache, evidence['probe_dir'], dict(zip(self.ids, self.users)))
        if actual != evidence:
            raise ProtocolError('LaViLa probe evidence changed')

    def fit_predict(self, source, source_labels, target, *, outer_fold, context, deadline=None, device='cuda', fit_fn=None, fit_callback=None):
        source = _indices(source, len(self.ids), 'source')
        target = _indices(target, len(self.ids), 'target')
        labels = np.asarray(source_labels)
        if set(source.tolist()) & set(target.tolist()) or labels.ndim != 1 or labels.shape != source.shape or (labels.dtype.kind not in 'iu') or (labels.dtype.kind == 'b') or np.any((labels < 0) | (labels >= 40)):
            raise ProtocolError('source-only LaViLa context invalid')
        if isinstance(outer_fold, (bool, np.bool_)) or not isinstance(outer_fold, (int, np.integer)):
            raise ProtocolError('invalid original outer fold')
        if int(outer_fold) not in (0, 1, 2):
            raise ProtocolError('invalid original outer fold')
        if not isinstance(context, str) or not context.strip():
            raise ProtocolError('invalid LaViLa context')
        source_users = self.users[source]
        target_users = self.users[target]
        if set(source_users.tolist()) & set(target_users.tolist()) or bool((set(source_users.tolist()) | set(target_users.tolist())) & EXCLUDED_USERS):
            raise ProtocolError('source-only LaViLa context invalid')
        real_fit = fit_fn is None
        if real_fit:
            self._validate_probe()
            if not callable(fit_callback):
                raise ProtocolError('real LaViLa fit requires durable member callback')
        fitter = train_member if real_fit else fit_fn
        seed = BASE_SEED + 1000 * int(outer_fold)
        external = context + '.external_lavila'
        seed_node = context + f'.seed{seed}'
        probability_node = context + '.probability'
        nodes = {external: ArtifactNode(external, provenance='frozen_external'), seed_node: ArtifactNode(seed_node, (external,), frozenset(source_users.tolist()), 'supervised', True), probability_node: ArtifactNode(probability_node, (seed_node,))}
        logits, state, diagnostics = fitter(self.x[source], labels.copy(), self.x[target], seed=seed, deadline=deadline, device=device)
        logits = np.asarray(logits, dtype=np.float32)
        if logits.shape != (len(target), 40) or not np.isfinite(logits).all():
            raise ProtocolError('invalid LaViLa member logits')
        record = {'node': seed_node, 'seed': seed, 'source_ids': self.ids[source].tolist(), 'source_users': source_users.tolist(), 'target_ids': self.ids[target].tolist(), 'target_users': target_users.tolist(), 'diagnostics': copy.deepcopy(diagnostics)}
        fit_callback_used = callable(fit_callback)
        if fit_callback_used:
            fit_callback(seed, logits, state, record)
        shifted = logits - logits.max(axis=1, keepdims=True)
        probability = np.exp(shifted).astype(np.float32, copy=False)
        probability /= probability.sum(axis=1, keepdims=True, dtype=np.float32)
        if not np.isfinite(probability).all():
            raise ProtocolError('invalid LaViLa probabilities')
        for user in set(target_users.tolist()):
            assert_prediction_provenance(user, [probability_node], nodes)
        receipt = {'context': context, 'outer_fold': int(outer_fold), 'expert_name': 'lavila_frame_token', 'seed': seed, 'recipe': copy.deepcopy(FIXED_RECIPE), 'cache_provenance': copy.deepcopy(self.cache_provenance), 'probe_evidence': copy.deepcopy(self.probe_evidence), 'probe_limitations': 'two sampled scene rows only; not whole-cache identity certification', 'injected_test_fitter': not real_fit, 'member_callback_used': fit_callback_used, 'source_ids': self.ids[source].tolist(), 'source_users': source_users.tolist(), 'target_ids': self.ids[target].tolist(), 'target_users': target_users.tolist(), 'source_label_sha256': array_hash(labels), 'source_class_counts': np.bincount(labels.astype(np.int64), minlength=40).tolist(), 'fits': [record], 'diagnostics': copy.deepcopy(diagnostics), 'aggregation': 'softmax(single_float32_logits)', 'target_labels_received': False, 'checkpoint_selection': False, 'prediction_node': probability_node, 'provenance_checked': True, 'artifact_dag': {k: {**asdict(v), 'supervised_train_subjects': sorted(v.supervised_train_subjects)} for k, v in nodes.items()}}
        return (probability[:, None, :].astype(np.float32, copy=False), receipt, {'logits': logits})
Provider = ClassProvider
