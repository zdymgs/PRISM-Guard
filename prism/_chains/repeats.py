"""Frozen session grouping and expert-vote repeat matching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.alignment import align_probabilities
from ..core.sequence import build_sessions

VOTE_SMOOTHING = 1e-5

@dataclass(frozen=True)
class RepeatGraphConfig:
    maximum_session_rank_distance: int
    maximum_start_gap_seconds: float
    minimum_probability_similarity: float
    minimum_path_overlap: float
    minimum_length_ratio: float
    consensus_weight: float
    alignment_gap_penalty: float = 0.2
    maximum_group_size: int = 3

CONFIG = RepeatGraphConfig(
    maximum_session_rank_distance=2,
    maximum_start_gap_seconds=300.0,
    minimum_probability_similarity=0.75,
    minimum_path_overlap=0.20,
    minimum_length_ratio=0.80,
    consensus_weight=0.50,
    alignment_gap_penalty=0.20,
    maximum_group_size=3,
)

def date_session_lists(indices: np.ndarray, metadata, short_gap: float) -> list[list[np.ndarray]]:
    sessions = build_sessions(indices, metadata, short_gap, "anonymous_date")
    result: list[list[np.ndarray]] = []
    for date in sorted(set(metadata.dates[indices].tolist())):
        selected = [session for session in sessions if metadata.dates[int(session[0])] == date]
        selected.sort(key=lambda session: float(np.nanmin(metadata.starts[session])))
        if len(selected) >= 2:
            result.append(selected)
    return result

def cluster_sessions(
    sessions: list[np.ndarray],
    probability: np.ndarray,
    decoded: np.ndarray,
    metadata,
    config: RepeatGraphConfig,
) -> list[list[np.ndarray]]:
    candidates: list[tuple[float, int, int]] = []
    for first in range(len(sessions)):
        stop = min(len(sessions), first + config.maximum_session_rank_distance + 1)
        for second in range(first + 1, stop):
            a, b = sessions[first], sessions[second]
            start_gap = abs(float(np.nanmin(metadata.starts[b])) - float(np.nanmin(metadata.starts[a])))
            if start_gap > config.maximum_start_gap_seconds:
                continue
            length_ratio = min(len(a), len(b)) / max(len(a), len(b))
            if length_ratio < config.minimum_length_ratio or abs(len(a) - len(b)) > 3:
                continue
            _, similarity = align_probabilities(probability[a], probability[b], config.alignment_gap_penalty)
            first_set, second_set = set(map(int, decoded[a])), set(map(int, decoded[b]))
            overlap = len(first_set & second_set) / max(len(first_set | second_set), 1)
            if similarity < config.minimum_probability_similarity or overlap < config.minimum_path_overlap:
                continue
            # A small rank-distance prior resolves ties without imposing a hard
            # 90-second block; interrupted recordings can still reunite.
            score = similarity + overlap + 0.08 * length_ratio - 0.01 * (second - first)
            candidates.append((score, first, second))
    candidates.sort(reverse=True)
    groups: list[list[int]] = []
    membership: dict[int, int] = {}
    for _, first, second in candidates:
        left, right = membership.get(first), membership.get(second)
        if left is None and right is None:
            membership[first] = membership[second] = len(groups)
            groups.append([first, second])
        elif left is not None and right is None and len(groups[left]) < config.maximum_group_size:
            groups[left].append(second)
            membership[second] = left
        elif left is None and right is not None and len(groups[right]) < config.maximum_group_size:
            groups[right].append(first)
            membership[first] = right
        elif left is not None and right is not None and left != right:
            if len(groups[left]) + len(groups[right]) <= config.maximum_group_size:
                merged = groups[left] + groups[right]
                groups[left], groups[right] = merged, []
                for value in merged:
                    membership[value] = left
    return [[sessions[index] for index in group] for group in groups if len(group) >= 2]

def expert_evidence(
    sample_ids: np.ndarray,
    base: np.ndarray,
    lookup: dict[str, np.ndarray],
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    teachers = np.stack([lookup[value] for value in sample_ids.astype(str)])
    hard = teachers.argmax(axis=2)
    vote = np.stack([(hard == class_id).mean(axis=1) for class_id in range(40)], axis=1)
    vote += VOTE_SMOOTHING
    vote /= vote.sum(axis=1, keepdims=True)
    mean = teachers.mean(axis=1)
    mean /= mean.sum(axis=1, keepdims=True)
    base_one_hot = np.zeros((len(base), 40), dtype=np.float32)
    base_one_hot[np.arange(len(base)), base] = 1.0
    choices = {
        "mean": mean,
        "vote": vote,
        "mix": 0.5 * mean + 0.5 * base_one_hot,
        "votemix": 0.5 * vote + 0.5 * base_one_hot,
    }
    return vote.astype(np.float64), choices[mode].astype(np.float64)

def graph(ids,base,bank,metadata,mode='aligned',seed=None):
    meta=metadata;lookup={str(i):p for i,p in zip(ids,bank)};vote,_=expert_evidence(ids,base,lookup,'vote');peers=[[] for _ in ids];rng=np.random.default_rng(seed);replacement_count=0
    for sessions in date_session_lists(np.arange(len(ids)),meta,30.):
        for group in cluster_sessions(sessions,vote,base,meta,CONFIG):
            ref=max(group,key=len)
            for other in group:
                if other is ref:continue
                pairs=list(zip(range(min(len(ref),len(other))),range(min(len(ref),len(other))))) if mode=='position' else align_probabilities(vote[ref],vote[other],CONFIG.alignment_gap_penalty)[0]
                for i,j in pairs:
                    left,right=int(ref[i]),int(other[j]);peers[left].append(right);peers[right].append(left)
    if mode=='shuffle':
        session_by_row={}
        # The donor stays in the original compatible session but at a random other position.
        for session in build_sessions(np.arange(len(ids)),meta,30.,'anonymous_date'):
            for row in session:session_by_row[int(row)]=session
        original=peers;peers=[]
        for i,pp in enumerate(original):
            new=[]
            for j in pp:
                candidates=[int(k) for k in session_by_row[j] if k!=i and k!=j and k not in new]
                if not candidates:candidates=[int(k) for k in session_by_row[j] if k!=i and k not in new]
                if not candidates:raise ValueError('cannot preserve shuffled peer count')
                k=int(rng.choice(candidates));replacement_count+=int(k!=j);new.append(k)
            peers.append(new)
        assert [len(p) for p in original]==[len(p) for p in peers]
    return peers,meta,replacement_count
