from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import joblib
ROOT = WORK_ROOT
from session_kd.mmfit_fullroute import bout_repeat_peers
from session_kd.mmact_group import build_group_features, fit_group_head, class_aware_hard_targets
from session_kd.mmfit_relations import fit_source_transition, decode_sequence, SequenceConfig
from session_kd.mmfit_strong_benchmark import sha256_file, ordered_id_hash, validate_split_firewall, TEACHER_ORDER
from session_kd.mmfit_rgb_benchmark import metrics

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-root', type=Path, default=ROOT / 'runs/mmfit_fullroute_v3')
    p.add_argument('--output-name', default='teachers_v3')
    a = p.parse_args()
    out = a.run_root / a.output_name
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'PRE_EVALUATION_FREEZE.json').exists():
        raise FileExistsError('teacher output already frozen')
    rows = list(csv.DictReader((ROOT / 'runs/mmfit_rgb_benchmark/inputs.csv').open()))
    y = np.load(ROOT / 'runs/mmfit_rgb_benchmark/source_labels_test_minus1.npy')
    idx = validate_split_firewall(rows, y)
    ids = np.array([r['segment_id'] for r in rows])
    people = np.array([int(r['person']) for r in rows])
    workouts = np.array([r['workout'] for r in rows])
    times = np.array([float(r['center_frame']) / 30 for r in rows])
    oofroot = a.run_root / 'oof'
    audit = json.loads((oofroot / 'audit.json').read_text())
    if audit['status'] != 'complete' or audit['test_labels_read']:
        raise ValueError('OOF incomplete')
    with np.load(oofroot / 'train_oof_bank.npz') as z:
        if not np.array_equal(z['indices'], idx['train']) or not np.array_equal(z['ids'], ids[idx['train']]):
            raise ValueError('OOF native ID order')
        trainbank = z['probabilities']
    if sha256_file(oofroot / 'train_oof_bank.npz') != audit['train_oof_bank_sha256']:
        raise ValueError('OOF changed')
    coverage = np.zeros((len(rows), 6), int)
    updates = []
    for receipt in audit['receipts']:
        f = receipt['fold']
        name = receipt['expert']
        k = TEACHER_ORDER.index(name)
        fit = np.load(oofroot / f'f{f}_fit_indices.npy')
        hold = np.load(oofroot / f'f{f}_hold_indices.npy')
        if set(people[fit]) & set(people[hold]) or np.any(y[fit] < 0):
            raise ValueError('teacher label ancestor overlaps held person')
        if set(fit) & set(idx['validation']) or set(fit) & set(idx['test']):
            raise ValueError('OOF teacher trained on nontrain')
        if receipt['fit_ids_sha256'] != ordered_id_hash(ids[fit]) or receipt['hold_ids_sha256'] != ordered_id_hash(ids[hold]):
            raise ValueError('receipt ID hash')
        folder = oofroot / f'f{f}_{name}'
        if receipt['checkpoint_sha256'] != sha256_file(folder / 'last.pt') or receipt['prediction_sha256'] != sha256_file(folder / 'hold_predictions.npz'):
            raise ValueError('OOF artifact modified')
        log = [json.loads(line) for line in (folder / 'train.jsonl').read_text().splitlines()]
        expected = receipt['fixed_epochs'] * int(np.ceil(len(fit) / 256))
        if receipt['updates'] != expected or log[-1]['updates'] != expected or (not receipt['finite_gradients']):
            raise ValueError('OOF effective update count')
        coverage[hold, k] += 1
        updates.append(expected)
    if len(audit['receipts']) != 18 or not np.all(coverage[idx['train']] == 1) or np.any(coverage[idx['validation']]) or np.any(coverage[idx['test']]):
        raise ValueError('OOF exact coverage')
    ancestry = {'status': 'passed', 'logical_receipts': 18, 'physical_checkpoints': 18, 'subject_disjoint': True, 'held_label_checkpoint_selection': False, 'normalizers_exclude_held': True, 'scratch_heads': True, 'encoder_ancestors': 'generic fixed representations; no MM-Fit encoder fine tuning', 'effective_updates': updates, 'test_labels_read': False, 'group_source_training': 'train OOF only; no claim that fitted train group output is OOF', 'validation_boundary': 'separate validation labels select policies; reused terminal heads were epoch-selected on this validation in V2, so validation is not a fresh untouched confirmation'}
    (out / 'OOF_ANCESTRY_ACCEPTANCE.json').write_text(json.dumps(ancestry, indent=2) + '\n')
    v2 = ROOT / 'runs/mmfit_strong_benchmark'
    vf = json.loads((v2 / 'pre_evaluation_freeze.json').read_text())
    banks = {'train': trainbank}
    for split, filename in (('validation', 'teacher_bank_source_validation.npz'), ('test', 'teacher_bank_unlabeled_test.npz')):
        expected = vf['artifact_sha256'][filename]
        if sha256_file(v2 / filename) != expected:
            raise ValueError('terminal bank freeze mismatch')
        with np.load(v2 / filename) as z:
            banks[split] = np.stack([z[k] for k in TEACHER_ORDER], axis=1)
    weights = np.array([0, 0.25, 0, 0.25, 0, 0.5], np.float32)
    strong = np.load(ROOT / 'runs/mmfit_strong_features/videomaev2_large_16x1024_fp16.npy', mmap_mode='r')
    direct = {}
    peer = {}
    gx = {}
    peer_reports = {}
    average = {}
    for split in ('train', 'validation', 'test'):
        ii = idx[split]
        direct[split] = np.einsum('k,nkc->nc', weights, banks[split])
        peer[split], peer_reports[split] = bout_repeat_peers(workouts[ii], times[ii], strong[ii], direct[split])
        edge_table = np.full((len(ii), 8), -1, np.int64)
        for j, pp in enumerate(peer[split]):
            edge_table[j, :len(pp)] = pp
            if any((workouts[ii[k]] != workouts[ii[j]] or abs(times[ii[k]] - times[ii[j]]) < 10 for k in pp)):
                raise ValueError('invalid native repeat edge')
        np.savez(out / f'{split}_repeat_edges.npz', indices=ii, ids=ids[ii], peer_local_indices=edge_table)
        gx[split] = build_group_features(banks[split], direct[split].argmax(1), peer[split])
        average[split] = direct[split].copy()
        for j, pp in enumerate(peer[split]):
            if pp:
                average[split][j] = 0.75 * direct[split][j] + 0.25 * direct[split][list(pp)].mean(0)
    combined = np.concatenate((gx['train'], gx['validation']))
    ntrain = len(idx['train'])
    head = fit_group_head(combined, np.r_[y[idx['train']], y[idx['validation']]], np.r_[ids[idx['train']], ids[idx['validation']]], fit_indices=np.arange(ntrain), holdout_indices=np.arange(ntrain, len(combined)), expert_names=TEACHER_ORDER, class_count=11, posterior_provenance='source_oof', regularization_c=0.03)
    joblib.dump(head, out / 'learned_group.joblib')
    group = {s: head.predict_proba(gx[s]).astype(np.float32) for s in ('validation', 'test')}
    transition = fit_source_transition(workouts[idx['train']], times[idx['train']], y[idx['train']], num_classes=11)
    np.savez(out / 'transition_train_only.npz', start=transition.start_log_probability, end=transition.end_log_probability, transition=transition.transition_log_probability)
    val = idx['validation']
    records = []
    best = None
    layers = {}
    for split in ('validation', 'test'):
        ii = idx[split]
        layers[split] = {'direct': direct[split], 'old_average_peer_targets': average[split], 'group': group[split]}
        for tw in (0.1, 0.3):
            emission = 0.65 * group[split] + 0.35 * direct[split]
            seq = decode_sequence(workouts[ii], times[ii], emission, transition, sample_ids=ids[ii], config=SequenceConfig(gap_seconds=1.5, transition_weight=tw, path_mix=0.51, emission_scale=0.2)).probabilities.astype(np.float32)
            layers[split][f'sequence_{tw}'] = seq
            route = group[split].argmax(1) != average[split].argmax(1)
            precedence = seq.copy()
            precedence[route] = group[split][route]
            layers[split][f'precedence_{tw}'] = precedence
    for name, prob in layers['validation'].items():
        if name == 'old_average_peer_targets':
            continue
        for threshold in (0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95):
            if name == 'direct' and threshold != 0:
                continue
            base = direct['validation'].argmax(1)
            proposal = prob.argmax(1)
            margin = prob[np.arange(len(base)), proposal] - prob[np.arange(len(base)), base]
            accepted = (proposal != base) & (margin >= threshold)
            final = direct['validation'].copy()
            final[accepted] = prob[accepted]
            m = metrics(y[val], final.argmax(1))
            rescue = int(np.sum((base != y[val]) & (final.argmax(1) == y[val])))
            harm = int(np.sum((base == y[val]) & (final.argmax(1) != y[val])))
            rec = {'name': name, 'margin': threshold, 'changed': int(accepted.sum()), 'rescues': rescue, 'harms': harm, 'accuracy': m['accuracy'], 'macro_f1_fixed11': m['macro_f1_fixed11']}
            records.append(rec)
            key = (m['macro_f1_fixed11'], m['accuracy'], -int(accepted.sum()))
            if best is None or key > best[0]:
                best = (key, rec)
    selected = best[1]
    output = {}
    for split in ('train', 'validation', 'test'):
        output[f'{split}_indices'] = idx[split]
        output[f'{split}_ids'] = ids[idx[split]]
        output[f'{split}_direct'] = direct[split]
        if split == 'train':
            continue
        base = direct[split].argmax(1)
        prob = layers[split][selected['name']]
        proposal = prob.argmax(1)
        margin = prob[np.arange(len(base)), proposal] - prob[np.arange(len(base)), base]
        accepted = (base != proposal) & (margin >= selected['margin'])
        final = direct[split].copy()
        final[accepted] = prob[accepted]
        output[f'{split}_final'] = final
        output[f'{split}_acceptance_mask'] = accepted
        for name, values in layers[split].items():
            output[f'{split}_{name}'] = values
        output[f'{split}_direct_hard'] = class_aware_hard_targets(base, 11)
        output[f'{split}_relation_hard'] = class_aware_hard_targets(final.argmax(1), 11)
    np.savez(out / 'frozen_targets.npz', **output)
    selection = {'selected': selected, 'validation_candidates': records, 'peer_reports': peer_reports, 'group_C': 0.03, 'source_only_revision': 'initial teachers/ snapshot retained; fixed averaging removed from learned-route selection, high-confidence thresholds added before any new TEST evaluation', 'direct_weights': weights.tolist(), 'emission': '0.65 learned group + 0.35 direct; self-loop Viterbi; scale .2', 'precedence': 'group when group argmax differs from native-peer fixed average; else sequence', 'hard_target': 'q_chosen=.995, q_other=.0005; KL T=1; all target IDs eligible', 'test_labels_read': False}
    (out / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    files = [out / n for n in ('frozen_targets.npz', 'learned_group.joblib', 'transition_train_only.npz', 'selection.json', 'OOF_ANCESTRY_ACCEPTANCE.json', 'train_repeat_edges.npz', 'validation_repeat_edges.npz', 'test_repeat_edges.npz')]
    freeze = {'status': 'complete', 'test_labels_read': False, 'test_id_sha256': ordered_id_hash(ids[idx['test']]), 'artifact_sha256': {x.name: sha256_file(x) for x in files}, 'oof_audit_sha256': sha256_file(oofroot / 'audit.json')}
    (out / 'PRE_EVALUATION_FREEZE.json').write_text(json.dumps(freeze, indent=2) + '\n')
    print(json.dumps(selection, indent=2))
