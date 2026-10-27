from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import hashlib
import csv
import json
from pathlib import Path
import numpy as np
from aligned_multimodal.lavila_cache import CHECKPOINT_SHA
from aligned_multimodal.stable_routing_protocol import ProtocolError

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()

def _expected_probe_sources():
    from aligned_multimodal.lavila_teacher import CHECKPOINT, PIXELS, EXTERNAL
    here = WORK_ROOT / 'aligned_multimodal'
    return {str(p.resolve()) for p in (CHECKPOINT, Path(__file__).resolve().parent / 'lavila_teacher.py', Path(__file__).resolve().parent / 'lavila_frame_token_cache.py', Path(__file__).resolve().parent / 'cpu_spotcheck.py', MODULE_ROOT.parent / 'README.md', PIXELS / 'completed.npy', PIXELS / 'summary.json', EXTERNAL / 'lavila/models/timesformer.py', EXTERNAL / 'lavila/models/openai_model.py')}

def validate_spotcheck(cache, probe_dir, users_by_id):
    folder = Path(probe_dir)
    registry_path = folder / 'registry.json'
    summary_path = folder / 'summary.json'
    tokens_path = folder / 'tokens.npz'
    r = json.loads(registry_path.read_text())
    s = json.loads(summary_path.read_text())
    process_path = Path(str(folder) + '.process.json')
    process = json.loads(process_path.read_text())
    if process.get('status') != 'complete' or process.get('exit_code') != 0 or (not 0 <= process.get('seconds', -1) <= 180) or (s.get('passed') is not True) or (s.get('fitting') is not False) or (s.get('test_rows_loaded') != 0) or (r.get('fitting') is not False) or (r.get('labels_used') is not False) or (r.get('device') != 'cpu') or (r.get('threads') != 1):
        raise ProtocolError('invalid LaViLa CPU probe status/scope')
    if r['cache_provenance'] != cache.provenance:
        raise ProtocolError('probe cache lineage differs')
    if set(r['files_sha256']) != _expected_probe_sources():
        raise ProtocolError('probe source inventory differs')
    if CHECKPOINT_SHA not in r['files_sha256'].values():
        raise ProtocolError('probe lacks pinned encoder checkpoint')
    for inventory in (r['files_sha256'], cache.provenance['input_sha256']):
        for path, digest in inventory.items():
            if sha(path) != digest:
                raise ProtocolError('probe input/source changed')
    manifests = [Path(p) for p in cache.provenance['input_sha256'] if Path(p).name == 'manifest.csv']
    if len(manifests) != 1:
        raise ProtocolError('unambiguous canonical manifest required')
    with manifests[0].open(encoding='utf-8-sig', newline='') as handle:
        records = list(csv.DictReader(handle))
    manifest_ids = [r.get('sample_id') for r in records]
    bound_users = {r.get('sample_id'): r.get('user_id') for r in records}
    if manifest_ids != list(cache.master_ids) or len(bound_users) != len(manifest_ids) or any((not isinstance(v, str) or not v.strip() for v in (*manifest_ids, *bound_users.values()))) or (not users_by_id) or any((key not in bound_users or bound_users[key] != value for key, value in users_by_id.items())):
        raise ProtocolError('user identities differ from pinned canonical manifest')
    users_by_id = bound_users
    probes = r['probes']
    if len(probes) != 2 or len(s['probes']) != 2:
        raise ProtocolError('two probe records required')
    rows = [Path(p) for p in cache.provenance['input_sha256'] if Path(p).name == 'rows.csv']
    if len(rows) != 1:
        raise ProtocolError('unambiguous pixel row table required')
    pixels = np.load(rows[0].with_name('images.npy'), mmap_mode='r', allow_pickle=False)
    if pixels.shape != (len(cache.pixel_ids), 2, 16, 3, 160, 160) or pixels.dtype != np.uint8:
        raise ProtocolError('probe pixel schema differs')
    seen = []
    with np.load(tokens_path, allow_pickle=False) as z:
        if set(z.files) != {f'{name}{k}' for k in range(2) for name in ('actual', 'expected', 'wrong')}:
            raise ProtocolError('probe array inventory differs')
        for k, p in enumerate(probes):
            i, j = (p['canonical_index'], p['pixel_index'])
            if isinstance(i, bool) or isinstance(j, bool) or (not isinstance(i, int)) or (not isinstance(j, int)) or (not 0 <= i < len(cache.master_ids)) or (not 0 <= j < len(cache.pixel_ids)) or (int(cache.master_to_pixel[i]) != j) or (i == j) or (str(cache.master_ids[i]) != p['sample_id']) or (str(cache.pixel_ids[j]) != p['sample_id']) or (str(cache.pixel_ids[i]) != p['wrong_positional_sample_id']):
                raise ProtocolError('probe ID mapping differs')
            if p['sample_id'] not in users_by_id or p['wrong_positional_sample_id'] not in users_by_id:
                raise ProtocolError('probe user identity missing')
            user = users_by_id[p['sample_id']]
            if user in {'user1', 'user2', 'user21'} or users_by_id[p['wrong_positional_sample_id']] in {'user1', 'user2', 'user21'}:
                raise ProtocolError('excluded user in probe')
            seen.append(user)
            block = np.asarray(pixels[j, :, :, 0]).copy()
            if hashlib.sha256(block.tobytes()).hexdigest() != p['pixel_sha256']:
                raise ProtocolError('probe pixel block changed')
            actual, expected, wrong = (z[f'{name}{k}'] for name in ('actual', 'expected', 'wrong'))
            if any((v.shape != (16, 768) or not np.isfinite(v).all() for v in (actual, expected, wrong))):
                raise ProtocolError('probe token arrays invalid')
            if not np.array_equal(expected, cache.tokens[j, :16].astype(np.float32)) or not np.array_equal(wrong, cache.tokens[i, :16].astype(np.float32)):
                raise ProtocolError('probe references differ from current cache')
            a, e, w = (v.astype(np.float64).reshape(-1) for v in (actual, expected, wrong))
            error = np.linalg.norm(a - e)
            wrong_error = np.linalg.norm(a - w)
            cosine = float(a @ e / max(np.linalg.norm(a) * np.linalg.norm(e), 1e-12))
            relative = float(error / max(np.linalg.norm(e), 1e-12))
            ratio = float(error / max(wrong_error, 1e-12))
            if not (cosine >= 0.995 and relative <= 0.05 and (wrong_error > 0) and (ratio <= 0.1)):
                raise ProtocolError('probe thresholds fail')
            report = s['probes'][k]
            if report.get('passed') is not True or not all((np.isclose(report[key], v, rtol=1e-10, atol=1e-12) for key, v in (('cosine', cosine), ('relative_l2', relative), ('error_vs_wrong_ratio', ratio)))):
                raise ProtocolError('probe reported metrics differ')
    if len(set(seen)) != 2:
        raise ProtocolError('probe users not distinct')
    return {'validated': True, 'probe_count': 2, 'scope': 'two sampled scene rows only, not whole-cache certification', 'input_sha256': dict(cache.provenance['input_sha256']), 'checkpoint_sha256': CHECKPOINT_SHA, 'registry_sha256': sha(registry_path), 'summary_sha256': sha(summary_path), 'tokens_sha256': sha(tokens_path), 'process_sha256': sha(process_path), 'probe_dir': str(folder.resolve())}
