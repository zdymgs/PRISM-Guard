from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import json
from pathlib import Path
import numpy as np
from aligned_multimodal.stable_routing_protocol import ArtifactNode, ProtocolError, register_experiment, assert_prediction_provenance
ROOT = WORK_ROOT
HERE = WORK_ROOT / 'aligned_multimodal'
CACHE = HERE / 'runs/videomae_large_multiclip_full40/complete_features.npz'
CACHE_SUMMARY = CACHE.with_name('cache_summary.json')

def load_raw(ids):
    meta = json.loads(CACHE_SUMMARY.read_text(encoding='utf-8'))
    if meta.get('model') != 'MCG-NJU/videomae-large-finetuned-kinetics' or meta.get('trials') != 2914:
        raise ProtocolError('foundation external cache provenance/schema changed')
    with np.load(CACHE, allow_pickle=False) as archive:
        cache_ids = archive['sample_ids'].astype(str)
        if len(cache_ids) != 2914 or len(set(cache_ids)) != len(cache_ids):
            raise ProtocolError('foundation cache IDs not unique')
        if archive['window_names'].astype(str).tolist() != ['early', 'late'] or archive['view_names'].astype(str).tolist() != ['scene', 'person', 'workspace']:
            raise ProtocolError('foundation view/window order changed')
        positions = {sid: i for i, sid in enumerate(cache_ids)}
        if len(set(ids)) != len(ids) or any((sid not in positions for sid in ids)):
            raise ProtocolError('foundation/master ID join failed')
        order = np.asarray([positions[sid] for sid in ids], int)
        return (archive['features'][order], archive['kinetics_logits'][order], {'cache_rows': len(cache_ids), 'selected_rows': len(ids), 'order': 'explicit unique-ID join', 'cache_label_fields_read': False, 'model': meta['model'], 'checkpoint_snapshot': meta.get('snapshot'), 'provenance': 'historical extraction source attestation; external model frozen, no task-label fitting'})
