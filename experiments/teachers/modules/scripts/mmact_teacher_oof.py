from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
ROOT = WORK_ROOT
from session_kd.mmact_teacher_oof import EXPERT_NAMES, SEALED_FOLD, SOURCE_FOLDS, VALIDATION_FOLD, RawInputs, availability_from_video_imu, build_expert_inputs, canonical_json_hash, file_sha256, geometry_features, load_identities, load_imu_features, load_nonsealed_labels, load_video_features, make_receipt_contract, ordered_hash, receipt_matches, pilot_status, validate_mc3_identity_manifest, validate_source_oof_manifest
from session_kd.mmact_strong_teacher import predict, softmax, train_head

def _artifact_receipt_valid(receipt_path: Path, contract: dict[str, object], directory: Path) -> bool:
    if not receipt_matches(receipt_path, contract):
        return False
    saved = json.loads(receipt_path.read_text(encoding='utf-8'))
    hashes = saved.get('artifact_sha256', {})
    return isinstance(hashes, dict) and bool(hashes) and all(((directory / name).is_file() and file_sha256(directory / name) == digest for name, digest in hashes.items()))

def _existing_eval_arrays(strong_run: Path, identities, availability: np.ndarray):
    split_rows = {'validation': np.asarray([i for i, row in enumerate(identities) if row['fold'] == VALIDATION_FOLD], dtype=np.int64), 'target': np.asarray([i for i, row in enumerate(identities) if row['fold'] == SEALED_FOLD], dtype=np.int64)}
    output = {}
    hashes = {}
    for split, indices in split_rows.items():
        bank = np.zeros((len(indices), len(EXPERT_NAMES), 37), dtype=np.float32)
        for expert_index, expert in enumerate(EXPERT_NAMES):
            path = strong_run / f'predictions_{expert}.npz'
            hashes[path.name] = file_sha256(path)
            key = 'validation_logits' if split == 'validation' else 'test_logits'
            with np.load(path, allow_pickle=False) as archive:
                probability = softmax(archive[key])
            mask = availability[indices, expert_index]
            bank[mask, expert_index] = probability[mask]
        output[f'{split}_probabilities'] = bank
        output[f'{split}_availability'] = availability[indices]
        output[f'{split}_indices'] = indices
        output[f'{split}_interval_ids'] = np.asarray([identities[i]['interval_id'] for i in indices])
    return (output, hashes)

def _availability_from_raw(raw: RawInputs) -> np.ndarray:
    return availability_from_video_imu(raw.video_mask, raw.imu_mask)

def run(args: argparse.Namespace) -> dict[str, object]:
    if not args.pilot and args.epochs != 60:
        raise ValueError('full OOF construction requires the frozen 60 epochs')
    if args.threads <= 0:
        raise ValueError('threads must be positive')
    if args.device == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA was requested but is unavailable')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt_dir = output / 'receipts'
    receipt_dir.mkdir(exist_ok=True)
    checkpoint_dir = output / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    prediction_dir = output / 'fold_predictions'
    prediction_dir.mkdir(exist_ok=True)
    lineage_dir = output / 'lineage'
    lineage_dir.mkdir(exist_ok=True)
    identities_path = args.strong_run / 'interval_identities.json'
    identities = load_identities(identities_path)
    mc3_hash = validate_mc3_identity_manifest(args.mc3_manifest, identities)
    classes_document = json.loads((args.strong_run / 'classes.json').read_text(encoding='utf-8'))
    label_to_id = {str(k): int(v) for k, v in classes_document['label_to_id'].items()}
    labels, annotation_hashes = load_nonsealed_labels(args.dataset_root, identities, label_to_id)
    folds = np.asarray([row['fold'] for row in identities])
    source_indices = np.flatnonzero(np.isin(folds, SOURCE_FOLDS))
    validation_indices = np.flatnonzero(folds == VALIDATION_FOLD)
    target_indices = np.flatnonzero(folds == SEALED_FOLD)
    if tuple(map(len, (source_indices, validation_indices, target_indices))) != (3209, 1081, 1116):
        raise ValueError('canonical split sizes changed')
    if np.any(labels[source_indices] < 0) or np.any(labels[validation_indices] < 0) or np.any(labels[target_indices] != -1):
        raise ValueError('label firewall failed')
    started = time.perf_counter()
    video, video_mask, _video_files = load_video_features(args.cache)
    imu, imu_mask = load_imu_features(args.legacy, identities)
    raw = RawInputs(video, video_mask, imu, imu_mask)
    availability = _availability_from_raw(raw)
    interval_ids = np.asarray([row['interval_id'] for row in identities])
    input_hashes = {'identities': file_sha256(identities_path), 'classes': file_sha256(args.strong_run / 'classes.json'), 'cache_integrity_audit': file_sha256(args.cache.parent / 'cache_integrity_audit.json'), 'cache_interval_manifest': file_sha256(args.cache / 'interval_manifest.json'), 'mc3_input_manifest': mc3_hash, 'legacy_cache_metadata': file_sha256(args.legacy / 'cache_metadata.json'), 'legacy_segments': file_sha256(args.legacy / 'segments.csv'), 'legacy_imu': file_sha256(args.legacy / 'imu_accgyro_96.npy'), 'source:session_kd.mmact_teacher_oof': file_sha256(MODULE_ROOT / 'session_kd/mmact_teacher_oof.py'), 'source:scripts.mmact_teacher_oof': file_sha256(Path(__file__).resolve()), 'source:session_kd.mmact_strong_teacher': file_sha256(MODULE_ROOT / 'session_kd/mmact_strong_teacher.py')}
    input_hashes.update({f'annotation:{name}': digest for name, digest in annotation_hashes.items()})
    experts = EXPERT_NAMES[:1] if args.pilot else EXPERT_NAMES
    prediction_folds = SOURCE_FOLDS[:1] if args.pilot else SOURCE_FOLDS
    epochs = min(args.epochs, 3) if args.pilot else 60
    completed_receipts = []
    for fold_index, prediction_fold in enumerate(prediction_folds):
        fit_folds = tuple((value for value in SOURCE_FOLDS if value != prediction_fold))
        fit_all = np.flatnonzero(np.isin(folds, fit_folds))
        held = np.flatnonzero(folds == prediction_fold)
        arrays, masks, normalizers = build_expert_inputs(raw, fit_all)
        for expert_index, expert in enumerate(experts):
            fit = fit_all[masks[expert][fit_all]]
            selection = validation_indices[masks[expert][validation_indices]]
            config = {'schema': 'mmact_source_oof_config', 'expert': expert, 'prediction_fold': prediction_fold, 'fit_folds': list(fit_folds), 'epoch_selection_fold': VALIDATION_FOLD, 'epochs': epochs, 'seed': 20260913 + EXPERT_NAMES.index(expert) * 10 + fold_index, 'device_class': args.device, 'threads': args.threads, 'regularized_head_recipe': 'mmact_strong_teacher.train_head'}
            contract = make_receipt_contract(expert=expert, prediction_fold=prediction_fold, fit_ids=interval_ids[fit].tolist(), prediction_ids=interval_ids[held].tolist(), selection_ids=interval_ids[selection].tolist(), input_hashes=input_hashes, config=config)
            stem = f'{expert}__{prediction_fold}'
            receipt_path = receipt_dir / f'{stem}.json'
            if _artifact_receipt_valid(receipt_path, contract, output):
                completed_receipts.append(json.loads(receipt_path.read_text(encoding='utf-8')))
                continue
            checkpoint = checkpoint_dir / f'{stem}.pt'
            log = checkpoint_dir / f'{stem}.jsonl'
            model, summary = train_head(arrays[expert], labels, fit, selection, 37, device, checkpoint, log, int(config['seed']), epochs)
            held_logits = predict(model, arrays[expert][held], device)
            held_probability = softmax(held_logits)
            held_available = masks[expert][held]
            held_probability[~held_available] = 0.0
            prediction_path = prediction_dir / f'{stem}.npz'
            np.savez_compressed(prediction_path, probabilities=held_probability.astype(np.float32), availability=held_available, global_indices=held, interval_ids=interval_ids[held], labels=labels[held])
            normalizer_path = checkpoint_dir / f'{stem}_normalizer.npz'
            mean, std = normalizers[expert]
            np.savez(normalizer_path, mean=mean, std=std)
            lineage_path = lineage_dir / f'{stem}.npz'
            np.savez_compressed(lineage_path, fit_indices=fit, fit_interval_ids=interval_ids[fit], fit_folds=folds[fit], prediction_indices=held, prediction_interval_ids=interval_ids[held], prediction_folds=folds[held], selection_indices=selection, selection_interval_ids=interval_ids[selection], selection_folds=folds[selection])
            artifact_paths = (checkpoint, log, normalizer_path, prediction_path, lineage_path)
            artifact_roles = ('checkpoint', 'train_log', 'normalizer', 'prediction', 'identity_lineage')
            artifact_keys = [str(path.relative_to(output)).replace('\\', '/') for path in artifact_paths]
            receipt = {**contract, 'status': 'complete', 'summary': summary, 'available_fit_count': int(len(fit)), 'available_prediction_count': int(held_available.sum()), 'labels_access_scope': [*SOURCE_FOLDS, VALIDATION_FOLD], 'f0_labels_read': False, 'f0_metrics_computed': False, 'artifacts': dict(zip(artifact_keys, artifact_roles, strict=True)), 'artifact_sha256': {key: file_sha256(path) for key, path in zip(artifact_keys, artifact_paths, strict=True)}}
            receipt_path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
            completed_receipts.append(receipt)
    if args.pilot:
        result = {**pilot_status(completed_receipts=len(completed_receipts), experts=experts, prediction_folds=prediction_folds, epochs=epochs), 'device': args.device, 'threads': args.threads, 'elapsed_seconds': time.perf_counter() - started, 'f0_labels_read': False, 'f0_metrics_computed': False}
        (output / 'pilot_status.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        return result
    source_bank = np.zeros((len(source_indices), len(EXPERT_NAMES), 37), dtype=np.float32)
    source_available = np.zeros((len(source_indices), len(EXPERT_NAMES)), dtype=bool)
    source_position = {int(index): position for position, index in enumerate(source_indices)}
    receipts = []
    for expert_index, expert in enumerate(EXPERT_NAMES):
        seen = np.zeros(len(source_indices), dtype=np.int8)
        for prediction_fold in SOURCE_FOLDS:
            stem = f'{expert}__{prediction_fold}'
            receipt = json.loads((receipt_dir / f'{stem}.json').read_text(encoding='utf-8'))
            receipts.append(receipt)
            with np.load(prediction_dir / f'{stem}.npz', allow_pickle=False) as archive:
                indices = archive['global_indices'].astype(np.int64)
                positions = np.asarray([source_position[int(index)] for index in indices])
                source_bank[positions, expert_index] = archive['probabilities']
                source_available[positions, expert_index] = archive['availability']
                seen[positions] += 1
        if not np.all(seen == 1):
            raise AssertionError(f'{expert}: source rows are not exactly once')
    eval_arrays, eval_hashes = _existing_eval_arrays(args.strong_run, identities, availability)
    bank_path = output / 'source_oof_bank.npz'
    np.savez_compressed(bank_path, source_probabilities=source_bank, source_availability=source_available, source_indices=source_indices, source_interval_ids=interval_ids[source_indices], source_labels=labels[source_indices], source_folds=folds[source_indices], validation_probabilities=eval_arrays['validation_probabilities'], validation_availability=eval_arrays['validation_availability'], validation_indices=eval_arrays['validation_indices'], validation_interval_ids=eval_arrays['validation_interval_ids'], validation_labels=labels[validation_indices], target_probabilities=eval_arrays['target_probabilities'], target_availability=eval_arrays['target_availability'], target_indices=eval_arrays['target_indices'], target_interval_ids=eval_arrays['target_interval_ids'], expert_names=np.asarray(EXPERT_NAMES))
    geometry = geometry_features(raw)
    geometry_path = output / 'geometry_features.npz'
    np.savez_compressed(geometry_path, source_features=geometry[source_indices], source_indices=source_indices, source_interval_ids=interval_ids[source_indices], validation_features=geometry[validation_indices], validation_indices=validation_indices, validation_interval_ids=interval_ids[validation_indices], target_features=geometry[target_indices], target_indices=target_indices, target_interval_ids=interval_ids[target_indices])
    manifest = {'status': 'complete', 'provenance': 'source_oof_crossfit_verified', 'protocol': 'MMACT_TEACHER_OOF_V1', 'expert_names': list(EXPERT_NAMES), 'source_ids_sha256': ordered_hash(interval_ids[source_indices].tolist()), 'validation_ids_sha256': ordered_hash(interval_ids[validation_indices].tolist()), 'target_ids_sha256': ordered_hash(interval_ids[target_indices].tolist()), 'source_shape': list(source_bank.shape), 'validation_shape': list(eval_arrays['validation_probabilities'].shape), 'target_shape': list(eval_arrays['target_probabilities'].shape), 'labels_access_scope': [*SOURCE_FOLDS, VALIDATION_FOLD], 'f0_labels_read': False, 'f0_metrics_computed': False, 'epoch_selection_reuse': 'F1 labels select checkpoints for all 36 cross-fit heads, matching the existing supervised-head recipe; F1 is not OOF training data.', 'interim_bank_scope': 'existing 12-head family only; not the final mature multimodal teacher family', 'input_hashes': input_hashes, 'existing_prediction_sha256': eval_hashes, 'receipts': receipts, 'artifact_sha256': {bank_path.name: file_sha256(bank_path), geometry_path.name: file_sha256(geometry_path)}, 'elapsed_seconds': time.perf_counter() - started}
    validate_source_oof_manifest(manifest, expected_experts=EXPERT_NAMES, expected_source_ids=interval_ids[source_indices].tolist())
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return {k: manifest[k] for k in ('status', 'protocol', 'source_shape', 'validation_shape', 'target_shape', 'elapsed_seconds')}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, default=ROOT / 'runs/mmact_strong_features/cache')
    parser.add_argument('--legacy', type=Path, default=ROOT / 'runs/mmact_cache')
    parser.add_argument('--strong-run', type=Path, default=ROOT / 'runs/mmact_strong_teacher')
    parser.add_argument('--mc3-manifest', type=Path, default=ROOT / 'runs/mmact_mc3_features/input_manifest.json')
    parser.add_argument('--dataset-root', type=Path, default=ROOT / 'data/MMAct')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))
