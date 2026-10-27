from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import hashlib
import json
import os
from pathlib import Path
from itertools import combinations
import numpy as np
from session_kd.mmact_relations import RelationConfig, fit_source_transition, _decode, build_repeat_probabilities
CLASSES = 37
MATCH_THRESHOLD = 0.55
SKIP_PENALTY = 0.2
TRANSITION_WEIGHT = 0.45
EMISSION_PROB = 0.65
EMISSION_HARD = 0.35
Q_WRONG = 0.0005

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def _prob(bank, mask, name):
    bank = np.asarray(bank, np.float64)
    mask = np.asarray(mask, bool)
    if bank.ndim != 3 or bank.shape[2] != CLASSES or mask.shape != bank.shape[:2]:
        raise ValueError(f'{name} bank shape mismatch')
    if not np.isfinite(bank).all() or np.any(bank < 0):
        raise ValueError(f'{name} has invalid probabilities')
    sums = bank.sum(2)
    if np.any(mask & ~np.isclose(sums, 1, atol=1e-05)) or np.any(~mask & (np.abs(bank).sum(2) > 1e-10)):
        raise ValueError(f'{name} availability/probability mismatch')
    return (bank, mask)

def weighted_available_mix(bank, mask, expert_names, selection):
    bank, mask = _prob(bank, mask, 'expert')
    names = list(map(str, expert_names))
    wanted = list(selection['names'])
    weights = np.asarray(selection['weights'], np.float64)
    if len(wanted) != len(weights) or not np.isclose(weights.sum(), 1):
        raise ValueError('invalid fixed selection weights')
    try:
        idx = np.asarray([names.index(x) for x in wanted])
    except ValueError as e:
        raise ValueError('selected expert absent from bank') from e
    active = mask[:, idx] * weights[None]
    den = active.sum(1)
    if np.any(den <= 0):
        raise ValueError('row has no selected direct expert')
    out = (bank[:, idx] * active[:, :, None]).sum(1) / den[:, None]
    return out / out.sum(1, keepdims=True)

def combined_similarity(geometry, posterior):
    geometry = np.asarray(geometry, np.float64)
    posterior = np.asarray(posterior, np.float64)
    if geometry.ndim != 2 or len(geometry) != len(posterior) or (not np.isfinite(geometry).all()):
        raise ValueError('geometry mismatch')
    if not np.allclose(np.linalg.norm(geometry, axis=1), 1, atol=1e-05):
        raise ValueError('geometry must be L2 normalized')
    cosine = np.clip(geometry @ geometry.T, -1, 1)
    cosine = (cosine + 1) / 2
    bc = np.sqrt(posterior) @ np.sqrt(posterior).T
    return 0.5 * cosine + 0.5 * bc

def _monotone(sim, left, right):
    matrix = sim[np.ix_(left, right)]
    m, n = matrix.shape
    score = np.full((m + 1, n + 1), -np.inf)
    action = np.zeros((m + 1, n + 1), np.int8)
    score[0, 0] = 0
    score[1:, 0] = -SKIP_PENALTY * np.arange(1, m + 1)
    score[0, 1:] = -SKIP_PENALTY * np.arange(1, n + 1)
    action[1:, 0] = 1
    action[0, 1:] = 2
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            opts = (score[i - 1, j - 1] + matrix[i - 1, j - 1] - MATCH_THRESHOLD, score[i - 1, j] - SKIP_PENALTY, score[i, j - 1] - SKIP_PENALTY)
            action[i, j] = np.argmax(opts)
            score[i, j] = opts[action[i, j]]
    found = []
    i = m
    j = n
    while i or j:
        move = int(action[i, j])
        if move == 0:
            if matrix[i - 1, j - 1] >= MATCH_THRESHOLD:
                found.append((int(left[i - 1]), int(right[j - 1]), float(matrix[i - 1, j - 1])))
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    return tuple(reversed(found))

def geometry_posterior_peers(meta, geometry, direct):
    n = len(direct)
    sim = combined_similarity(geometry, direct)
    peers = [[] for _ in range(n)]
    subject = np.asarray(meta['subject'], str)
    scene = np.asarray(meta['scene'], str)
    session = np.asarray(meta['session'], str)
    starts = np.asarray(meta['start_ms'], float)
    ids = np.asarray(meta['interval_id'], str)
    for sub, scn in sorted(set(zip(subject, scene))):
        scope = np.flatnonzero((subject == sub) & (scene == scn))
        sessions = sorted(set(session[scope]))
        for a, b in combinations(sessions, 2):
            left = scope[session[scope] == a]
            right = scope[session[scope] == b]
            left = left[np.lexsort((ids[left], starts[left]))]
            right = right[np.lexsort((ids[right], starts[right]))]
            for x, y, value in _monotone(sim, left, right):
                peers[x].append((y, value))
                peers[y].append((x, value))
    out = []
    peer_ids = []
    peer_sim = []
    for i, p in enumerate(peers):
        p = sorted(p, key=lambda x: (ids[x[0]], -x[1]))
        out.append([x[0] for x in p])
        peer_ids.append(tuple((ids[x[0]] for x in p)))
        peer_sim.append(tuple((x[1] for x in p)))
    return (out, tuple(peer_ids), tuple(peer_sim))

def metadata_rows(manifest, indices, ids):
    rows = manifest.get('rows_manifest')
    if not isinstance(rows, list):
        raise ValueError('input manifest rows missing')
    by_index = {int(r['global_interval_index']): r for r in rows}
    result = []
    for index, identity in zip(np.asarray(indices, int), np.asarray(ids, str), strict=True):
        row = by_index.get(int(index))
        if row is None or str(row['interval_id']) != identity:
            raise ValueError('bank/input manifest identity mismatch')
        result.append(row)
    return {'subject': np.asarray([r['subject'] for r in result]), 'scene': np.asarray([r['scene'] for r in result]), 'session': np.asarray([r['session'] for r in result]), 'start_ms': np.asarray([r['start_ms'] for r in result], np.int64), 'end_ms': np.asarray([r['end_ms'] for r in result], np.int64), 'interval_id': np.asarray(ids, str), 'ordinal': np.asarray([_ordinal(r['interval_id']) for r in result], np.int64)}

def _ordinal(identity):
    try:
        return int(str(identity).rsplit(':', 1)[1])
    except Exception:
        raise ValueError(f'interval_id lacks annotation ordinal: {identity}')

def sequence_from_group(meta, group, transition):
    hard = np.eye(CLASSES, dtype=np.float64)[group.argmax(1)]
    emission = EMISSION_PROB * group + EMISSION_HARD * hard
    seq = emission.copy()
    classes = group.argmax(1)
    block = np.full(len(group), -1, np.int64)
    keys = list(zip(meta['subject'], meta['scene'], meta['session']))
    groups = {}
    for i, key in enumerate(keys):
        groups.setdefault(tuple(key), []).append(i)
    bid = 0
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda i: (meta['ordinal'][i], meta['start_ms'][i], meta['interval_id'][i]))
        chunks = []
        cur = []
        for i in ordered:
            if cur and meta['ordinal'][i] != meta['ordinal'][cur[-1]] + 1:
                chunks.append(cur)
                cur = []
            cur.append(i)
        if cur:
            chunks.append(cur)
        for chunk in chunks:
            ix = np.asarray(chunk)
            block[ix] = bid
            bid += 1
            if len(ix) > 1:
                classes[ix] = _decode(emission[ix], transition, TRANSITION_WEIGHT)
                seq[ix] = 0.49 * emission[ix] + 0.51 * np.eye(CLASSES)[classes[ix]]
    return (seq / seq.sum(1, keepdims=True), classes, block)
