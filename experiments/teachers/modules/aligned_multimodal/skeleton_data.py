from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import json
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from torch.utils.data import Dataset
try:
    from aligned_multimodal.aligned_data import AlignedMultimodalDataset, _flip_skeleton, resolve_project_path
except ImportError:
    from aligned_multimodal.aligned_data import AlignedMultimodalDataset, _flip_skeleton, resolve_project_path

class _TemporalSampler:
    """Minimal state required by the canonical AlignedMultimodalDataset method."""
    num_frames = 12

    def __init__(self, augment: bool) -> None:
        self.augment = bool(augment)
        self.temporal_view = 0.5

    def positions(self, length: int) -> list[int]:
        return AlignedMultimodalDataset._sample_positions(self, int(length))

class SkeletonDataset(Dataset):
    """Read-only P0-compatible skeleton samples from ``skeleton_float32.npy``.

    ``sample_ids`` is the full canonical universe; ``indices`` select entries
    from that universe. Selected IDs are joined by ID to the cache metadata,
    so cache order may differ from universe order. Labels are never inferred.
    """

    def __init__(self, cache_dir: str | Path, sample_ids: Sequence[str], indices: Sequence[int], labels: Sequence[int] | None=None, augment: bool=False) -> None:
        self.cache_dir = resolve_project_path(cache_dir)
        self.sample_ids = tuple((str(sample_id) for sample_id in sample_ids))
        self.indices = tuple((self._as_index(index) for index in indices))
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError('sample_ids must be unique')
        if len(set(self.indices)) != len(self.indices):
            raise ValueError('indices must be unique')
        if any((index < 0 or index >= len(self.sample_ids) for index in self.indices)):
            raise IndexError('requested universe index is out of bounds')
        if labels is not None:
            if len(labels) != len(self.indices):
                raise ValueError('labels must align one-for-one with indices')
            self.labels: tuple[int, ...] | None = tuple((int(label) for label in labels))
        else:
            self.labels = None
        self.augment = bool(augment)
        self._sampler = _TemporalSampler(self.augment)
        metadata_path = self.cache_dir / 'metadata.json'
        skeleton_path = self.cache_dir / 'skeleton_float32.npy'
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        if not skeleton_path.is_file():
            raise FileNotFoundError(skeleton_path)
        try:
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f'invalid cache metadata: {metadata_path}') from exc
        metadata_ids = metadata.get('sample_ids')
        metadata_offsets = metadata.get('offsets')
        if not isinstance(metadata_ids, list) or not isinstance(metadata_offsets, list):
            raise ValueError('metadata must contain sample_ids and offsets lists')
        if len(metadata_ids) != len(metadata_offsets):
            raise ValueError('metadata sample_ids/offsets length mismatch')
        metadata_ids = tuple((str(sample_id) for sample_id in metadata_ids))
        if len(set(metadata_ids)) != len(metadata_ids):
            raise ValueError('metadata sample_ids must be unique')
        if str(metadata.get('skeleton_strategy', 'first')) != 'first':
            raise ValueError("cache skeleton_strategy must be 'first'")
        try:
            array = np.load(skeleton_path, mmap_mode='r', allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f'invalid skeleton cache: {skeleton_path}') from exc
        if not isinstance(array, np.memmap):
            raise ValueError('skeleton cache must be loaded as a read-only mmap')
        if array.dtype != np.dtype(np.float32) or array.ndim != 3 or array.shape[1:] != (17, 4):
            raise ValueError(f'skeleton cache must have dtype float32 and shape (N,17,4), got {array.dtype}/{array.shape}')
        self._skeleton = array
        total_frames = int(array.shape[0])
        if int(metadata.get('total_frames', total_frames)) != total_frames:
            raise ValueError('metadata total_frames does not match skeleton cache')
        offsets: list[tuple[int, int]] = []
        for item in metadata_offsets:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError('each metadata offset must be [offset, length]')
            offset, length = (self._as_index(value) for value in item)
            if offset < 0 or length <= 0 or offset + length > total_frames:
                raise ValueError(f'cache offset out of bounds: {offset}, {length}')
            offsets.append((offset, length))
        self._offsets = tuple(offsets)
        metadata_rows = {sample_id: row for row, sample_id in enumerate(metadata_ids)}
        selected_ids = tuple((self.sample_ids[index] for index in self.indices))
        missing = [sample_id for sample_id in selected_ids if sample_id not in metadata_rows]
        if missing:
            raise KeyError(f'requested IDs missing from cache metadata: {missing[0]!r}')
        self._selected_ids = selected_ids
        self._cache_rows = tuple((metadata_rows[sample_id] for sample_id in selected_ids))

    @staticmethod
    def _as_index(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f'index/offset/length must be integer, got {type(value).__name__}')
        return int(value)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, object]:
        index = self._as_index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        cache_index = self._cache_rows[index]
        offset, length = self._offsets[cache_index]
        positions = np.asarray(self._sampler.positions(length), dtype=np.int64)
        skeleton = np.array(self._skeleton[offset + positions], dtype=np.float32, copy=True)
        flip = bool(torch.rand(1).item() < 0.5) if self.augment else False
        if flip:
            skeleton_tensor = torch.stack([_flip_skeleton(frame) for frame in torch.from_numpy(skeleton)])
        else:
            skeleton_tensor = torch.from_numpy(skeleton)
        result: dict[str, object] = {'skeleton': skeleton_tensor, 'sample_id': self._selected_ids[index]}
        if self.labels is not None:
            result['label'] = self.labels[index]
        return result
