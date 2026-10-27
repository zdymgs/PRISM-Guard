"""Minimal immutable constants needed by the vendored native model graph."""

from __future__ import annotations

import numpy as np

KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)

H36M_PARENTS = np.asarray(
    (0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15),
    dtype=np.int64,
)

PART_JOINTS = (
    tuple(range(17)),
    (8, 9, 10),
    (0, 1, 4, 7, 8, 9),
    (11, 12, 13),
    (14, 15, 16),
    (4, 5, 6),
    (1, 2, 3),
    (9, 10, 13, 16),
)
