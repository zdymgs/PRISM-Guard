from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
'Authoritative Train ID/user joins for canonical-context thermal refitting.'
import csv
from pathlib import Path
import numpy as np
from aligned_multimodal.thermal_data import load_path_map
from aligned_multimodal.stable_routing_protocol import ProtocolError

class ThermalPopulation:

    def __init__(self, manifest, canonical_ids, canonical_labels, canonical_users):
        self.canonical_ids = np.asarray(canonical_ids).astype(str)
        self.canonical_users = np.asarray(canonical_users).astype(str)
        canonical_labels = np.asarray(canonical_labels)
        if self.canonical_ids.ndim != 1 or len(set(self.canonical_ids)) != len(self.canonical_ids) or self.canonical_users.shape != self.canonical_ids.shape or (canonical_labels.shape != self.canonical_ids.shape):
            raise ProtocolError('malformed canonical Train population')
        rows = list(csv.DictReader(Path(manifest).open(encoding='utf-8-sig', newline='')))
        self.ids = np.asarray([r['sample_id'] for r in rows])
        self.users = np.asarray([r['user_id'] for r in rows])
        self.labels = np.asarray([int(r['class_id']) for r in rows], dtype=np.int64)
        if len(set(self.ids)) != len(self.ids) or np.any((self.labels < 0) | (self.labels >= 40)):
            raise ProtocolError('malformed thermal Train population')
        lookup = {s: i for i, s in enumerate(self.ids)}
        for i, s in enumerate(self.canonical_ids):
            if s in lookup:
                j = lookup[s]
                if self.users[j] != self.canonical_users[i] or self.labels[j] != canonical_labels[i]:
                    raise ProtocolError('canonical/thermal label or user disagreement')
        self.paths = load_path_map(manifest, self.ids)
        self.present = np.asarray([s in self.paths for s in self.canonical_ids])
        self.excluded = frozenset({'user1', 'user2', 'user21'})

    def context(self, canonical_source, canonical_target):
        source, target = (np.asarray(canonical_source), np.asarray(canonical_target))
        for ix in (source, target):
            if ix.ndim != 1 or ix.dtype.kind not in 'iu' or len(set(ix)) != len(ix) or np.any(ix < 0) or np.any(ix >= len(self.canonical_ids)):
                raise ProtocolError('invalid canonical context indices')
        su = set(self.canonical_users[source])
        tu = set(self.canonical_users[target])
        if not su or not tu or su & tu or (su | tu) & self.excluded:
            raise ProtocolError('context subject exclusion violated')
        thermal_source = np.flatnonzero(np.isin(self.users, list(su)))
        if set(self.users[thermal_source]) != su:
            raise ProtocolError('source subject has no thermal rows')
        return (thermal_source, self.labels[thermal_source].copy(), self.users[thermal_source].copy(), self.canonical_ids[target].copy(), self.canonical_users[target].copy())
