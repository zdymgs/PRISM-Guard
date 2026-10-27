from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import csv
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from torch.utils.data import Dataset
from thermal_baseline.thermal_oof_data import ThermalOOFDataset, IMAGE_EXTENSIONS

def load_path_map(manifest, sample_ids):
    """Read only ID/path fields; never consume manifest class labels."""
    ids = tuple((str(x) for x in sample_ids))
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate canonical IDs')
    mapping = {}
    with Path(manifest).open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            sid = row['sample_id']
            if sid in mapping:
                raise ValueError('duplicate thermal ID')
            mapping[sid] = Path(row['trial_dir'])
    result = {}
    for sid in ids:
        if sid not in mapping:
            continue
        path = mapping[sid]
        if not path.is_dir():
            raise FileNotFoundError(path)
        if not any((p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in path.iterdir())):
            raise ValueError(f'empty thermal trial: {sid}')
        result[sid] = path
    return result

class ThermalDataset(Dataset):
    """Source labels supplied explicitly; missing thermal IDs are rejected.

    Canonical preprocessing is delegated without reading its manifest. The
    canonical loader's required class_id is a constant discarded sentinel, not
    any sample's label. Output exposes a label only when explicitly provided.
    """

    def __init__(self, paths, sample_ids, indices, labels=None, augment=False):
        ids = np.asarray(sample_ids).astype(str)
        ix = np.asarray(indices)
        if ix.dtype.kind not in 'iu' or ix.ndim != 1 or np.any(ix < 0) or np.any(ix >= len(ids)):
            raise ValueError('invalid indices')
        if len(set(ids)) != len(ids) or len(set(ix)) != len(ix):
            raise ValueError('duplicate IDs/indices')
        self.sample_ids = ids[ix]
        if any((s not in paths for s in self.sample_ids)):
            raise ValueError('missing thermal must be routed outside dataset')
        self.labels = None
        if labels is not None:
            y = np.asarray(labels)
            if y.shape != ix.shape or y.dtype.kind not in 'iu' or np.any((y < 0) | (y >= 40)):
                raise ValueError('invalid source labels')
            self.labels = y.copy()
        self.reader = ThermalOOFDataset.__new__(ThermalOOFDataset)
        self.reader.num_frames = 12
        self.reader.image_height = 144
        self.reader.image_width = 192
        self.reader.normalization = 'legacy'
        self.reader.augment = bool(augment)
        self.reader.samples = [SimpleNamespace(sample_id=s, trial_dir=Path(paths[s]), class_id=-1) for s in self.sample_ids]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        value = self.reader[index]
        result = {'clip': value['clip'], 'sample_id': value['sample_id']}
        if self.labels is not None:
            result['label'] = int(self.labels[index])
        return result
