from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import hashlib
import json
import os
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from session_kd.mmact_group import build_group_features, fit_group_head, class_aware_hard_targets
from session_kd.mmact_relations import fit_source_transition
from session_kd.mmact_teacher_compile import CLASSES, Q_WRONG, file_hash, _prob, weighted_available_mix, metadata_rows, geometry_posterior_peers, sequence_from_group
EXTRA = ('mc3_motion', 'native_skeleton', 'sequential_imu')
ALPHAS = (0.0, 0.1, 0.2, 0.3)
MARGINS = (0.0, 0.05, 0.1, 0.15)
STRATEGIES = ('direct', 'group', 'sequence', 'precedence')

def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def fixed_f1(y, p):
    return float(f1_score(y, p, labels=np.arange(CLASSES), average='macro', zero_division=0))

def direct_candidates(bank, mask, names, base_selection):
    base_names = list(base_selection['names'])
    base = np.asarray(base_selection['weights'], float)
    result = []
    for alpha in ALPHAS:
        chosen = base_names + list(EXTRA)
        weights = np.r_[base * (1 - alpha), np.full(3, alpha / 3)]
        result.append({'alpha': alpha, 'names': chosen, 'weights': weights.tolist(), 'probabilities': weighted_available_mix(bank, mask, names, {'names': chosen, 'weights': weights})})
    return result

def select_direct(candidates, labels):
    records = []
    for rank, c in enumerate(candidates):
        pred = c['probabilities'].argmax(1)
        records.append({'alpha': c['alpha'], 'names': c['names'], 'weights': c['weights'], 'macro_f1_fixed37': fixed_f1(labels, pred), 'accuracy': float(np.mean(labels == pred)), 'rank': rank})
    best = max(records, key=lambda r: (r['macro_f1_fixed37'], r['accuracy'], -r['rank']))
    return (best, records)

def apply_margin(direct, candidate, margin):
    base = direct.argmax(1)
    proposed = candidate.argmax(1)
    score = candidate[np.arange(len(base)), proposed] - candidate[np.arange(len(base)), base]
    changed = (proposed != base) & (score >= margin)
    out = direct.copy()
    out[changed] = candidate[changed]
    return (out, changed)

def select_policy(direct, group, sequence, labels):
    precedence = sequence.copy()
    use_group = group.argmax(1) != direct.argmax(1)
    precedence[use_group] = group[use_group]
    values = {'direct': direct, 'group': group, 'sequence': sequence, 'precedence': precedence}
    records = []
    for srank, strategy in enumerate(STRATEGIES):
        for margin in MARGINS:
            out, changed = (direct.copy(), np.zeros(len(direct), bool)) if strategy == 'direct' else apply_margin(direct, values[strategy], margin)
            pred = out.argmax(1)
            routes = {'direct': int((~changed).sum())}
            if strategy == 'direct':
                routes = {'direct': len(direct)}
            elif strategy == 'precedence':
                group_route = changed & (group.argmax(1) != direct.argmax(1))
                routes.update(group=int(group_route.sum()), sequence=int((changed & ~group_route).sum()))
            else:
                routes[strategy] = int(changed.sum())
            records.append({'strategy': strategy, 'margin': margin, 'macro_f1_fixed37': fixed_f1(labels, pred), 'accuracy': float(np.mean(labels == pred)), 'changed': int(changed.sum()), 'route_counts': routes, 'rank': srank})
    best = max(records, key=lambda r: (r['macro_f1_fixed37'], r['accuracy'], -r['changed'], -r['rank'], -r['margin']))
    return (best, records, values)

def apply_policy(direct, values, policy):
    if policy['strategy'] == 'direct':
        return (direct.copy(), np.zeros(len(direct), bool))
    return apply_margin(direct, values[policy['strategy']], policy['margin'])

def validate_merged_contract(bank_path, geometry_path, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get('status') != 'complete' or manifest.get('schema') != 'MMACT_TEACHER_BANK_V2':
        raise ValueError('merged V2 manifest incomplete')
    declared = manifest.get('artifact_sha256', {})
    for path in (Path(bank_path), Path(geometry_path)):
        if declared.get(path.name) != file_hash(path):
            raise ValueError('merged artifact hash mismatch')
    receipts = manifest.get('logical_receipts')
    if not isinstance(receipts, list) or len(receipts) != 48:
        raise ValueError('requires 45 OOF and 3 final-source receipts')
    oof = {(str(r.get('logical_expert')), str(r.get('prediction_fold'))) for r in receipts if r.get('prediction_fold') in {'F2', 'F3', 'F4'}}
    final = {str(r.get('logical_expert')) for r in receipts if r.get('prediction_fold') == 'final_source'}
    expected_names = tuple(manifest.get('expert_names', ()))
    if oof != {(e, f) for e in expected_names for f in ('F2', 'F3', 'F4')} or final != set(EXTRA):
        raise ValueError('logical receipt coverage mismatch')
    for receipt in receipts:
        if receipt.get('status') != 'complete':
            raise ValueError('incomplete logical receipt')
        for key in ('producer_artifact_root', 'original_physical_receipt', 'rebound_physical_receipt', 'physical_receipt_sha256', 'config_sha256', 'source_manifest_sha256'):
            if key not in receipt:
                raise ValueError('logical receipt provenance missing')
        root = Path(receipt['producer_artifact_root']).resolve()
        bound = Path(receipt['rebound_physical_receipt']).resolve()
        if root not in bound.parents and bound != root:
            raise ValueError('receipt artifact escapes root')
        if not bound.is_file() or file_hash(bound) != receipt['physical_receipt_sha256']:
            raise ValueError('physical receipt hash mismatch')
        if str(receipt.get('logical_expert')) in EXTRA:
            if not receipt.get('logical_configuration_sha256') or not receipt.get('logical_output_sha256'):
                raise ValueError('extra logical-head identity hashes missing')
            if str(receipt.get('logical_expert')) in {'native_skeleton', 'sequential_imu'} and (not receipt.get('shared_checkpoint_group')):
                raise ValueError('native shared-checkpoint group missing')
        fold = str(receipt.get('prediction_fold'))
        if fold in {'F2', 'F3', 'F4'}:
            if set(map(str, receipt.get('fit_folds', ()))) != {'F2', 'F3', 'F4'} - {fold}:
                raise ValueError('OOF receipt fold exclusion invalid')
            for key in ('fit_ids_sha256', 'prediction_ids_sha256', 'fit_count', 'prediction_count'):
                if key not in receipt:
                    raise ValueError('OOF identity lineage missing')
        elif fold == 'final_source':
            if set(map(str, receipt.get('fit_folds', ()))) != {'F2', 'F3', 'F4'} or receipt.get('f0_labels_read', False):
                raise ValueError('final-source lineage invalid')
    return manifest

def compile(bank_path, geometry_path, input_manifest, merged_manifest, base_selection, output, threads=2):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('V2 output must be new and empty')
    output.mkdir(parents=True, exist_ok=True)
    manifest = validate_merged_contract(bank_path, geometry_path, merged_manifest)
    with np.load(bank_path, allow_pickle=False) as z:
        bank = {k: z[k] for k in z.files}
    with np.load(geometry_path, allow_pickle=False) as z:
        geometry = {k: z[k] for k in z.files}
    if any(('target' in k.lower() and 'label' in k.lower() for k in bank)):
        raise ValueError('target label sentinel')
    names = tuple(bank['expert_names'].astype(str))
    if len(names) != 15 or names[-3:] != EXTRA or len(set(names)) != 15 or (tuple(manifest.get('expert_names', ())) != names):
        raise ValueError('V2 expert order invalid')
    input_data = json.loads(Path(input_manifest).read_text())
    base = json.loads(Path(base_selection).read_text())['selection']
    splits = {}
    for split, n in zip(('source', 'validation', 'target'), (3243, 1080, 1083)):
        p, m = _prob(bank[split + '_probabilities'], bank[split + '_availability'], split)
        if p.shape != (n, 15, 37):
            raise ValueError('V2 bank shape invalid')
        idx = bank[split + '_indices']
        ids = bank[split + '_interval_ids'].astype(str)
        if not np.array_equal(idx, geometry[split + '_indices']) or not np.array_equal(ids, geometry[split + '_interval_ids'].astype(str)):
            raise ValueError('geometry identity mismatch')
        meta = metadata_rows(input_data, idx, ids)
        splits[split] = {'bank': p, 'mask': m, 'idx': idx, 'ids': ids, 'meta': meta, 'geometry': geometry[split + '_features']}
    all_indices = np.concatenate([splits[x]['idx'] for x in ('source', 'validation', 'target')])
    if len(np.unique(all_indices)) != 5406 or set(map(int, all_indices)) != set(range(5406)):
        raise ValueError('split index partition invalid')
    from session_kd.mmact_strong_teacher import subject_fold
    for split, expected in (('source', {'F2', 'F3', 'F4'}), ('validation', {'F1'}), ('target', {'F0'})):
        if {subject_fold(x) for x in splits[split]['meta']['subject']} != expected:
            raise ValueError('split subject folds invalid')
    labels = bank['source_labels'].astype(int)
    vlabels = bank['validation_labels'].astype(int)
    if labels.shape != (3243,) or vlabels.shape != (1080,) or np.any((labels < 0) | (labels >= 37)) or np.any((vlabels < 0) | (vlabels >= 37)):
        raise ValueError('source/validation labels invalid')
    dc = direct_candidates(splits['validation']['bank'], splits['validation']['mask'], names, base)
    direct_choice, direct_records = select_direct(dc, vlabels)
    selection = {'names': direct_choice['names'], 'weights': direct_choice['weights']}
    validation_id_hash = hashlib.sha256(('\n'.join(splits['validation']['ids']) + '\n').encode()).hexdigest()
    input_bank_hashes = {'bank': file_hash(bank_path), 'geometry': file_hash(geometry_path)}
    for record in direct_records:
        record.update(validation_ids_sha256=validation_id_hash, input_bank_hashes=input_bank_hashes)
    atomic_json(output / 'validation_selection.json', {'schema': 'MMACT_TEACHER_V2_DIRECT_SELECTION', 'selection': selection, 'alpha': direct_choice['alpha'], 'candidates': direct_records, 'f0_labels_read': False})
    for split in splits:
        s = splits[split]
        s['direct'] = weighted_available_mix(s['bank'], s['mask'], names, selection)
        peers, peer_ids, _ = geometry_posterior_peers(s['meta'], s['geometry'], s['direct'])
        s['peer_ids'] = peer_ids
        s['features'] = build_group_features(s['bank'], s['direct'].argmax(1), peers, expert_mask=s['mask'])
    joint = np.concatenate((splits['source']['features'], splits['validation']['features']))
    joint_y = np.r_[labels, np.full(1080, -1)]
    joint_ids = np.r_[splits['source']['ids'], splits['validation']['ids']]
    with threadpool_limits(limits=threads):
        head = fit_group_head(joint, joint_y, joint_ids, fit_indices=np.arange(3243), holdout_indices=np.arange(3243, 4323), expert_names=names, class_count=37, posterior_provenance='source_oof', regularization_c=0.03)
        groups = {k: head.predict_proba(v['features']) for k, v in splits.items()}
    sm = splits['source']['meta']
    transition = fit_source_transition(subject_ids=sm['subject'], scene_ids=sm['scene'], session_ids=sm['session'], starts=sm['start_ms'], ends=sm['end_ms'], interval_ids=sm['interval_id'], labels=labels, num_classes=37, annotation_ordinals=sm['ordinal'])
    sequences = {}
    decoded = {}
    for split in splits:
        sequences[split], decoded[split], _ = sequence_from_group(splits[split]['meta'], groups[split], transition)
    policy, policy_records, _ = select_policy(splits['validation']['direct'], groups['validation'], sequences['validation'], vlabels)
    for record in policy_records:
        record.update(validation_ids_sha256=validation_id_hash, input_bank_hashes=input_bank_hashes)
    outputs = {}
    arrays = {'expert_names': np.asarray(names)}
    for split in splits:
        direct = splits[split]['direct']
        precedence = sequences[split].copy()
        use = groups[split].argmax(1) != direct.argmax(1)
        precedence[use] = groups[split][use]
        values = {'direct': direct, 'group': groups[split], 'sequence': sequences[split], 'precedence': precedence}
        final, accepted = apply_policy(direct, values, policy)
        chosen = final.argmax(1)
        outputs[split] = final
        arrays.update({split + '_indices': splits[split]['idx'], split + '_interval_ids': splits[split]['ids'], split + '_direct_probabilities': direct.astype(np.float32), split + '_direct_hard_targets': class_aware_hard_targets(direct.argmax(1), 37, Q_WRONG), split + '_group_probabilities': groups[split].astype(np.float32), split + '_sequence_probabilities': sequences[split].astype(np.float32), split + '_sequence_decoded_class': decoded[split].astype(np.int16), split + '_precedence_probabilities': precedence.astype(np.float32), split + '_soft_targets': final.astype(np.float32), split + '_hard_targets': class_aware_hard_targets(chosen, 37, Q_WRONG), split + '_chosen_class': chosen.astype(np.int16), split + '_confidence': final.max(1).astype(np.float32), split + '_accepted': accepted, split + '_peer_ids': np.asarray([json.dumps(x) for x in splits[split]['peer_ids']])})
        route = np.full(len(direct), 'direct', dtype='<U8')
        if policy['strategy'] == 'precedence':
            route[accepted & use] = 'group'
            route[accepted & ~use] = 'sequence'
        elif policy['strategy'] != 'direct':
            route[accepted] = policy['strategy']
        arrays[split + '_route'] = route
    np.savez_compressed(output / 'frozen_teacher_targets.npz', **arrays)
    joblib.dump({'group_head': head, 'transition': transition, 'policy': policy}, output / 'trusted_local_models.joblib')
    sources = (Path(__file__), Path(__file__).with_name('mmact_teacher_compile.py'), Path(__file__).with_name('mmact_group.py'), Path(__file__).with_name('mmact_relations.py'))
    hashes = {str(Path(x).resolve()): file_hash(x) for x in (bank_path, geometry_path, input_manifest, merged_manifest, base_selection, *sources)}
    hashes.update({p.name: file_hash(p) for p in output.iterdir() if p.is_file()})
    freeze = {'schema': 'MMACT_TEACHER_COMPILE_V2', 'status': 'complete', 'experts': 15, 'expert_names': list(names), 'direct_selection': selection, 'direct_alpha': direct_choice['alpha'], 'direct_candidates': direct_records, 'policy_selection': policy, 'policy_candidates': policy_records, 'source_validation_labels_only': True, 'f0_labels_read': False, 'f0_metrics_computed': False, 'artifact_sha256': hashes, 'boundary': 'Validated only after a complete 48-receipt merged-bank contract; no F0 label or metric is consumed.'}
    atomic_json(output / 'PRE_EVALUATION_FREEZE_V2.json', freeze)
    return freeze
