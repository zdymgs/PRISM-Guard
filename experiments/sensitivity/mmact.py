import argparse
import os
import sys
import time
import json
import csv
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
STARTED = time.perf_counter()
parser = argparse.ArgumentParser()
parser.add_argument('--bank', type=Path, required=True)
parser.add_argument('--frozen', type=Path, required=True)
parser.add_argument('--metadata', type=Path, required=True)
parser.add_argument('--truth', type=Path, required=True)
parser.add_argument('--classes', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
ROOT = args.output
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import yaml
from prism.core.activity import _blocks, _decode, fit_source_transition
from prism.core.activity_mix import weighted_available_mix
from prism._chains.mmact import PATHS, MARGINS, apply_margin, _gated_choice, _score, _forward_filter

CONFIG = yaml.safe_load((REPO / 'configs/mmact.yaml').read_text(encoding='utf-8'))
RUNS = ROOT / 'runs_mmact'
BANK = args.bank
FROZEN = args.frozen
METADATA = args.metadata
TRUTH = args.truth
CLASSES = args.classes
DISPLAY_PATH = {
    'direct': 'D', 'group': 'D→G', 'sequence': 'D→G→S',
    'direct_sequence_gated': 'D→S', 'direct_sequence_nogate': 'D→S（无门控）',
    'online_forward': 'Online forward',
}


def load_inputs():
    rows = json.loads(METADATA.read_text(encoding='utf-8'))['rows_manifest']
    by_index = {r['global_interval_index']: r for r in rows}
    splits = {}
    with np.load(BANK, allow_pickle=False) as bank, np.load(FROZEN, allow_pickle=False) as frozen:
        names = bank['expert_names'].astype(str)
        mixture = {'names': names[CONFIG['direct_expert_indices']].tolist(),
                   'weights': CONFIG['direct_expert_weights']}
        for split in ('source', 'validation', 'target'):
            indices = bank[split + '_indices'].astype(int)
            ids = bank[split + '_interval_ids'].astype(str)
            selected = [by_index[int(i)] for i in indices]
            meta = {k: np.asarray([r[k] for r in selected])
                    for k in ('subject', 'scene', 'session', 'start_ms', 'end_ms')}
            meta['interval_id'] = ids
            meta['ordinal'] = np.asarray([int(s.rsplit(':', 1)[1]) for s in ids])
            data = {'indices': indices, 'ids': ids, 'meta': meta}
            if split != 'source':
                data['direct'] = weighted_available_mix(
                    bank[split + '_probabilities'], bank[split + '_availability'], names, mixture)
                data['group'] = frozen[split + '_group_probabilities'].astype(float)
            splits[split] = data
        splits['source']['labels'] = bank['source_labels'].astype(int)
        splits['validation']['labels'] = bank['validation_labels'].astype(int)
    return splits


def candidates(data, transition, config):
    meta, direct, group = data['meta'], data['direct'], data['group']
    chunks = _blocks(list(zip(meta['subject'], meta['scene'], meta['session'])),
                     meta['start_ms'], meta['interval_id'], meta['ordinal'])
    hard = np.eye(37)[group.argmax(1)]
    emission = config['group_sequence_emission_alpha'] * group + (
        1 - config['group_sequence_emission_alpha']) * hard
    sequence = emission.copy()
    rounded_direct = direct.astype(np.float32).astype(float)
    decoded = rounded_direct.argmax(1)
    for chunk in chunks:
        if len(chunk) > 1:
            group_path = _decode(emission[chunk], transition, config['transition_lambda'])
            sequence[chunk] = (config['gate_replay_direct_mass'] * emission[chunk]
                               + config['gate_replay_MAP_mass'] * np.eye(37)[group_path])
            decoded[chunk] = _decode(rounded_direct[chunk], transition, config['transition_lambda'])
    sequence /= sequence.sum(1, keepdims=True)
    sequence = sequence.astype(np.float32).astype(float)
    proposal = (config['gate_replay_direct_mass'] * direct
                + config['gate_replay_MAP_mass'] * np.eye(37)[decoded])
    return {'direct': direct, 'group': group, 'sequence': sequence,
            'direct_sequence_gated': proposal, 'direct_sequence_nogate': decoded,
            'online_forward': _forward_filter(rounded_direct, transition, chunks).argmax(1)}


def construct(splits, key, value):
    config = dict(CONFIG)
    config[key] = value
    sm = splits['source']['meta']
    transition = fit_source_transition(
        subject_ids=sm['subject'], scene_ids=sm['scene'], session_ids=sm['session'],
        starts=sm['start_ms'], ends=sm['end_ms'], interval_ids=sm['interval_id'],
        labels=splits['source']['labels'], num_classes=37,
        annotation_ordinals=sm['ordinal'], smoothing=config['transition_smoothing'])
    val = candidates(splits['validation'], transition, config)
    target = candidates(splits['target'], transition, config)
    y = splits['validation']['labels']
    gate, _ = _gated_choice(val['direct'], val['direct_sequence_gated'], y)
    records = []
    for rank, path in enumerate(PATHS):
        if path in ('group', 'sequence', 'direct_sequence_gated'):
            for margin in MARGINS:
                p, changed = apply_margin(val['direct'], val[path], margin)
                records.append(dict(path=path, rank=rank, margin=margin,
                                    changed=int(changed.sum()), **_score(y, p.argmax(1))))
        else:
            p = val[path].argmax(1) if path == 'direct' else val[path]
            records.append(dict(path=path, rank=rank, margin=0.0,
                                changed=int(np.sum(p != val['direct'].argmax(1))), **_score(y, p)))
    winner = max(records, key=lambda r: (r['macro_f1_fixed37'], r['accuracy'],
                                       -r['changed'], -r['rank'], -r['margin']))
    path = winner['path']
    if path in ('group', 'sequence', 'direct_sequence_gated'):
        prediction = apply_margin(target['direct'], target[path], winner['margin'])[0].argmax(1)
    else:
        prediction = target[path].argmax(1) if path == 'direct' else target[path]
    directory = RUNS / f'{key}_{value:g}'
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / 'predictions.npz',
                        interval_id=splits['target']['ids'], target_label=prediction,
                        direct_label=target['direct'].argmax(1))
    policy = dict(transition_lambda=config['transition_lambda'],
                  transition_smoothing=config['transition_smoothing'],
                  selected_path=path, selected_margin=winner['margin'],
                  direct_sequence_margin=gate['margin'])
    (directory / 'selection.json').write_text(json.dumps(policy, ensure_ascii=False, indent=2),
                                               encoding='utf-8')
    print(f'已冻结 {directory.name}，路径 {DISPLAY_PATH[path]}，阈值 {winner["margin"]}', flush=True)
    return directory


def read_evaluation_labels(ids, indices):
    manifest = json.loads(TRUTH.read_text(encoding='utf-8'))
    mapping = json.loads(CLASSES.read_text(encoding='utf-8'))['label_to_id']
    rows = [manifest['intervals'][int(i)] for i in indices]
    actual = np.asarray([f'{r["annotation_file"]}:{r["line_number"]}' for r in rows])
    if not np.array_equal(actual, ids):
        raise ValueError('F0 label identities differ from frozen predictions')
    return np.asarray([mapping[r['label']] for r in rows], dtype=int)


def main():
    reproduction_started = time.perf_counter()
    splits = load_inputs()
    baseline = construct(splits, 'transition_lambda', 0.45)
    with np.load(baseline / 'predictions.npz', allow_pickle=False) as z:
        labels = read_evaluation_labels(z['interval_id'], splits['target']['indices'])
        direct_score = 100 * _score(labels, z['direct_label'])['macro_f1_fixed37']
        prism_score = 100 * _score(labels, z['target_label'])['macro_f1_fixed37']
    del labels
    policy = json.loads((baseline / 'selection.json').read_text(encoding='utf-8'))
    print(f'复现：Direct={direct_score:.4f}，PRISM={prism_score:.4f}，路径={DISPLAY_PATH[policy["selected_path"]]}', flush=True)
    if (round(direct_score, 1) != 88.5 or round(prism_score, 1) != 99.9
            or policy['selected_path'] != 'direct_sequence_gated'):
        raise SystemExit('复现未通过，按提示词停止。')
    elapsed = time.perf_counter() - reproduction_started
    print(f'预计总耗时约 {elapsed * 8 / 60:.2f} 分钟', flush=True)
    grid = [('transition_lambda', v) for v in (0.15, 0.30, 0.45, 0.60, 0.75)]
    grid += [('transition_smoothing', v) for v in (0.25, 0.5, 2.0, 4.0)]
    directories = {}
    for key, value in grid:
        directories[(key, value)] = baseline if (key, value) == ('transition_lambda', 0.45) else construct(splits, key, value)
    print('全部 9 个配置预测、路径和阈值已写盘；开始读取 F0 标签评测。', flush=True)
    labels = read_evaluation_labels(splits['target']['ids'], splits['target']['indices'])
    with np.load(baseline / 'predictions.npz', allow_pickle=False) as z:
        reference = z['target_label']
    rows = []
    predictions = {}
    for key, value in grid:
        directory = directories[(key, value)]
        with np.load(directory / 'predictions.npz', allow_pickle=False) as z:
            prediction = z['target_label']
        predictions[(key, value)] = prediction
        policy = json.loads((directory / 'selection.json').read_text(encoding='utf-8'))
        score = 100 * _score(labels, prediction)['macro_f1_fixed37']
        rows.append(dict(parameter=key, value=value, macro_f1=score, delta_vs_direct=score - direct_score,
                         path=DISPLAY_PATH[policy['selected_path']],
                         changed_intervals=int(np.sum(prediction != reference))))
    with (ROOT / 'results_mmact.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# MMAct 解码超参数敏感性（F0）', '',
             '37 类 macro-F1，缺席类计 0；F1 重新选择门控和路径。分数与差值单位为百分点。最终配置仅运行一次。', '',
             f'最终配置复现：Direct {direct_score:.4f}（88.5），PRISM {prism_score:.4f}（99.9），路径 D→S。', '',
             '| 常数 | 取值 | macro-F1 | Δ vs Direct | 路径 | 变化的区间数 |',
             '|---|---:|---:|---:|---|---:|']
    for row in rows:
        marker = '*' if row['parameter'] == 'transition_lambda' and row['value'] == 0.45 else ''
        lines.append(f'| {row["parameter"]} | {row["value"]:g}{marker} | {row["macro_f1"]:.4f} | {row["delta_vs_direct"]:+.4f} | {row["path"]} | {row["changed_intervals"]} |')
    lines += ['', '* 为最终配置（λ=0.45，smoothing=1.0）；smoothing=1.0 的结果复用该行。', '']
    for key in ('transition_lambda', 'transition_smoothing'):
        selected = [r['macro_f1'] for r in rows if r['parameter'] == key]
        group_predictions = [predictions[(k, v)] for k, v in grid if k == key]
        if key == 'transition_smoothing':
            selected.append(prism_score)
            group_predictions.append(reference)
        lines.append(f'- `{key}`：{min(selected):.4f}–{max(selected):.4f}，变化范围（最大值减最小值）{max(selected) - min(selected):.4f}。')
        if all(np.array_equal(group_predictions[0], p) for p in group_predictions[1:]):
            lines.append(f'- `{key}`：全部取值预测完全相同，该键可能未生效。')
    total = time.perf_counter() - STARTED
    lines += ['', f'数据实际路径：`{BANK.as_posix()}`。',
              f'冻结 group 输出：`{FROZEN.as_posix()}`；元信息：`{METADATA.as_posix()}`。',
              f'评测标签：`{TRUTH.as_posix()}`；37 类映射：`{CLASSES.as_posix()}`。',
              '实际使用的键名：`transition_lambda` → `_decode(weight=λ)`；`transition_smoothing` → `fit_source_transition(smoothing=值)`。',
              f'总耗时：{total:.2f} 秒（{total / 60:.2f} 分钟，含复现与 8 配置预测及评测）。']
    (ROOT / 'RESULTS_mmact.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'完成：{len(rows)} 个配置，总耗时 {total:.2f} 秒。', flush=True)


if __name__ == '__main__':
    main()
