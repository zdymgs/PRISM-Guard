"""Monotone posterior alignment."""
from __future__ import annotations
import numpy as np

def align_probabilities(
    first: np.ndarray, second: np.ndarray, gap_penalty: float
) -> tuple[list[tuple[int, int]], float]:
    similarity = np.sqrt(first[:, None, :] * second[None, :, :]).sum(axis=2)
    rows, columns = similarity.shape
    score = np.full((rows + 1, columns + 1), -np.inf, dtype=np.float64)
    trace = np.zeros((rows + 1, columns + 1), dtype=np.int8)
    score[0, 0] = 0.0
    for i in range(1, rows + 1):
        score[i, 0] = score[i - 1, 0] - gap_penalty; trace[i, 0] = 1
    for j in range(1, columns + 1):
        score[0, j] = score[0, j - 1] - gap_penalty; trace[0, j] = 2
    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            values = (
                score[i - 1, j - 1] + similarity[i - 1, j - 1],
                score[i - 1, j] - gap_penalty,
                score[i, j - 1] - gap_penalty,
            )
            trace[i, j] = int(np.argmax(values)); score[i, j] = max(values)
    pairs: list[tuple[int, int]] = []
    i, j = rows, columns
    while i or j:
        action = int(trace[i, j])
        if i and j and action == 0:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i and (not j or action == 1):
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    mean_similarity = float(np.mean([similarity[i, j] for i, j in pairs])) if pairs else 0.0
    return pairs, mean_similarity
