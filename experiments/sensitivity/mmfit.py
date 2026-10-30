import argparse
import csv
import json
import os
import sys
from pathlib import Path
from time import perf_counter

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.special import logsumexp

parser = argparse.ArgumentParser()
parser.add_argument('--cache', type=Path, required=True)
parser.add_argument('--v1', type=Path, required=True)
parser.add_argument('--v2', type=Path, required=True)
parser.add_argument('--teachers', type=Path, required=True)
parser.add_argument('--reference', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
ROOT = args.output
CACHE = args.cache
V1 = args.v1
V2 = args.v2
TEACHERS = args.teachers
REFERENCE = args.reference
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prism.core.workout import SequenceConfig, _ordered_bouts, decode_sequence, fit_source_transition

GRID = (0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95)
PATHS = ("direct", "group", "sequence", "direct_sequence", "direct_sequence_nogate", "online_forward")
EXPERTS = ("rgb", "pose", "imu", "videomaev2_large", "vjepa2", "joint")
DISPLAY = {"direct": "D", "group": "D→G", "sequence": "D→G→S", "direct_sequence": "D→S", "direct_sequence_nogate": "D→S（无门控）", "online_forward": "在线前向"}


def score(y, pred):
    matrix = np.zeros((11, 11), dtype=np.int64)
    np.add.at(matrix, (y, pred), 1)
    denominator = matrix.sum(0) + matrix.sum(1)
    f1 = np.divide(2 * np.diag(matrix), denominator, out=np.zeros(11), where=denominator > 0)
    return float(f1.mean()), float(np.mean(y == pred))


def gate(direct, proposal, threshold):
    base, proposed = direct.argmax(1), proposal.argmax(1)
    rows = np.arange(len(base))
    accepted = (base != proposed) & (proposal[rows, proposed] - proposal[rows, base] >= threshold)
    return np.where(accepted, proposed, base)


def forward(direct, work, times, transition, gap):
    output = np.empty(len(direct), dtype=np.int64)
    for rows in _ordered_bouts(work, times, gap)[0]:
        alpha = 0.2 * np.log(np.maximum(direct[rows[0]], 1e-300)) + transition.start_log_probability
        alpha -= logsumexp(alpha)
        output[rows[0]] = alpha.argmax()
        for row in rows[1:]:
            alpha = 0.2 * np.log(np.maximum(direct[row], 1e-300)) + logsumexp(alpha[:, None] + transition.transition_log_probability, axis=0)
            alpha -= logsumexp(alpha)
            output[row] = alpha.argmax()
    return output


def construct(data, train, validation_labels, weight, smoothing, gap):
    transition = fit_source_transition(train["work"], train["time"], train["labels"], num_classes=11, smoothing=smoothing, gap_seconds=gap)
    config = SequenceConfig(gap_seconds=gap, transition_weight=weight, emission_scale=0.2, path_mix=0.51)
    candidates = {}
    for split, item in data.items():
        kwargs = dict(sample_ids=item["ids"], config=config)
        ds = decode_sequence(item["work"], item["time"], item["direct"], transition, **kwargs).probabilities
        gs = decode_sequence(item["work"], item["time"], 0.65 * item["group"] + 0.35 * item["direct"], transition, **kwargs).probabilities.astype(np.float32)
        candidates[split] = {"direct": item["direct"].argmax(1), "group": item["group"], "sequence": gs, "direct_sequence": ds, "direct_sequence_nogate": ds.argmax(1), "online_forward": forward(item["direct"], item["work"], item["time"], transition, gap)}
    records = []
    base = candidates["validation"]["direct"]
    for path in PATHS:
        for threshold in GRID if path in ("group", "sequence", "direct_sequence") else (None,):
            pred = gate(data["validation"]["direct"], candidates["validation"][path], threshold) if threshold is not None else candidates["validation"][path]
            f1, accuracy = score(validation_labels, pred)
            records.append(dict(path=path, threshold=threshold, macro_f1=f1, accuracy=accuracy, changed=int(np.count_nonzero(pred != base))))
    winner = max(records, key=lambda record: (record["macro_f1"], record["accuracy"], -record["changed"]))
    path, threshold = winner["path"], winner["threshold"]
    predictions = {}
    for split, item in data.items():
        predictions[split] = gate(item["direct"], candidates[split][path], threshold) if threshold is not None else candidates[split][path].copy()
    return predictions, winner


def freeze(name, group, value, weight, smoothing, gap, predictions, winner, data):
    folder = ROOT / "runs_mmfit" / name
    folder.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(folder / "predictions.npz", target_labels=predictions["test"], target_ids=data["test"]["ids"], target_indices=data["test"]["indices"], selection_predictions=predictions["validation"])
    policy = dict(config=name, group=group, value=value, transition_lambda=weight, transition_smoothing=smoothing, gap_seconds=gap, selected_path=winner["path"], selected_threshold=winner["threshold"], selection=winner, gate_acceptance_fraction=float(np.mean(predictions["test"] != data["test"]["direct"].argmax(1))))
    (folder / "selection.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return policy


def main():
    started = perf_counter()
    rows = pd.read_csv(V1 / "inputs.csv", usecols=["segment_id", "split", "workout", "center_frame"])
    source_labels = np.load(V1 / "source_labels_test_minus1.npy", allow_pickle=False)
    train_indices = np.flatnonzero(rows["split"].to_numpy() == "train")
    train = dict(work=rows.iloc[train_indices].workout.to_numpy(str), time=rows.iloc[train_indices].center_frame.to_numpy(float) / 30, labels=source_labels[train_indices])
    data = {}
    with np.load(TEACHERS / "frozen_targets.npz", allow_pickle=False) as frozen:
        for split, filename in (("validation", "teacher_bank_source_validation.npz"), ("test", "teacher_bank_unlabeled_test.npz")):
            indices = frozen[f"{split}_indices"]
            with np.load(V2 / filename, allow_pickle=False) as bank:
                direct = np.einsum("k,nkc->nc", np.array([0, .25, 0, .25, 0, .5], np.float32), np.stack([bank[key] for key in EXPERTS], 1))
            data[split] = dict(indices=indices, ids=frozen[f"{split}_ids"], work=rows.iloc[indices].workout.to_numpy(str), time=rows.iloc[indices].center_frame.to_numpy(float) / 30, direct=direct, group=frozen[f"{split}_group"])
    validation_labels = source_labels[data["validation"]["indices"]]
    if len(data["test"]["ids"]) != 11668 or len(np.unique(data["test"]["work"])) != 5:
        raise SystemExit("复现停止：测试窗口数或 workout 数与提示词不同。")

    baseline, winner = construct(data, train, validation_labels, .1, 1.0, 1.5)
    # Exact replay establishes the published scores without opening TEST truth before the sweep is frozen.
    with np.load(REFERENCE / "frozen_predictions.npz", allow_pickle=False) as prior:
        direct_match = np.array_equal(data["test"]["direct"].argmax(1), prior["evaluation_prediction__direct"])
        selected_match = np.array_equal(baseline["test"], prior["evaluation_prediction__direct_sequence"])
        order_match = np.array_equal(data["test"]["ids"], prior["evaluation_ids"])
    metrics = json.loads((REFERENCE / "results.json").read_text(encoding="utf-8"))["splits"]["evaluation_all"]["methods"]
    direct_f1 = metrics["direct"]["macro_f1_fixed11"] * 100
    selected_f1 = metrics["direct_sequence"]["macro_f1_fixed11"] * 100
    if not (direct_match and selected_match and order_match and winner["path"] == "direct_sequence" and round(direct_f1, 1) == 94.6 and round(selected_f1, 1) == 94.0):
        raise SystemExit(f"复现停止：Direct预测一致={direct_match}，PRISM预测一致={selected_match}，顺序一致={order_match}，路径={winner['path']}，历史分数={direct_f1:.6f}/{selected_f1:.6f}。未执行敏感性网格。")
    policies = [freeze("lambda_0.1_final", "lambda", .1, .1, 1.0, 1.5, baseline, winner, data)]
    reproduction_seconds = perf_counter() - started
    print(f"复现通过：Direct {direct_f1:.6f}，PRISM {selected_f1:.6f}，D→S，阈值 {winner['threshold']}；与论文冻结预测逐窗一致，未读取测试标签。", flush=True)
    print(f"预计总耗时约 {reproduction_seconds * 11 / 60:.2f} 分钟", flush=True)
    configs = [(f"lambda_{value:g}", "lambda", value, value, 1.0, 1.5) for value in (.05, .2, .3, .45)]
    configs += [(f"smoothing_{value:g}", "transition_smoothing", value, .1, value, 1.5) for value in (.25, .5, 2.0)]
    configs += [(f"gap_{value:g}", "gap_seconds", value, .1, 1.0, value) for value in (1.0, 3.0, 5.0)]
    for name, group, value, weight, smoothing, gap in configs:
        predictions, selected = construct(data, train, validation_labels, weight, smoothing, gap)
        policies.append(freeze(name, group, value, weight, smoothing, gap, predictions, selected, data))
        print(f"预测已冻结 {len(policies)}/11：{name}，路径 {DISPLAY[selected['path']]}，阈值 {selected['threshold']}", flush=True)

    print("11 个配置预测、路径和阈值全部写盘；现在读取测试标签并评分。", flush=True)
    with np.load(CACHE / "mmfit_fullroute_v3/evaluation/frozen_evaluation_arrays.npz", allow_pickle=False) as truth:
        if not (np.array_equal(truth["ids"].astype(str), data["test"]["ids"].astype(str)) and np.array_equal(truth["indices"], data["test"]["indices"])):
            raise SystemExit("评分停止：正式评测标签与冻结预测的窗口顺序不同。")
        test_labels = truth["truth"].astype(int)
    predictions = {}
    for policy in policies:
        with np.load(ROOT / "runs_mmfit" / policy["config"] / "predictions.npz", allow_pickle=False) as frozen:
            predictions[policy["config"]] = frozen["target_labels"]
    computed_direct = score(test_labels, data["test"]["direct"].argmax(1))[0] * 100
    computed_final = score(test_labels, predictions["lambda_0.1_final"])[0] * 100
    if round(computed_direct, 1) != 94.6 or round(computed_final, 1) != 94.0:
        raise SystemExit(f"评分停止：Direct={computed_direct:.6f}，PRISM={computed_final:.6f}，与论文不一致；不重跑不修复。")
    policies.sort(key=lambda p: ({"lambda": 0, "transition_smoothing": 1, "gap_seconds": 2}[p["group"]], p["value"]))
    records = []
    for policy in policies:
        pred = predictions[policy["config"]]
        f1 = score(test_labels, pred)[0] * 100
        records.append(dict(config=policy["config"], macro_f1_pct=f1, delta_vs_direct_pp=f1 - 94.6, path=DISPLAY[policy["selected_path"]], gate_acceptance_pct=policy["gate_acceptance_fraction"] * 100, changed_windows=int(np.count_nonzero(pred != predictions["lambda_0.1_final"]))))
    with (ROOT / "results_mmfit.csv").open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = ["# MM-Fit 解码超参数敏感性（S3）", "", "测试集：5 个 workout，11,668 个窗口，含非运动类的 11 类 macro-F1。单次只改变一个常数，其余固定为 λ=0.1、smoothing=1.0、gap=1.5 s；emission_scale=0.2、path_mix=0.51。", "", f"最终配置复现：Direct {computed_direct:.6f}%，PRISM {computed_final:.6f}%，D→S，门控阈值 0.4。第 0 步通过与论文冻结预测逐窗一致确认复现及对应已存分数；11 个配置的预测、路径和阈值全部写盘后才读取测试标签重新评分。", "", "每个配置均在选择集重选门控阈值及六条候选路径，按 11 类 macro-F1、accuracy、较少变化窗口数依次排序，同分保持原路径及阈值顺序。使用冻结 group 概率；无训练、无神经网络推理、未用 GPU。门控接受比例为最终目标相对 Direct 改变的窗口数 / 11,668。Δ 按提示词以 Direct 94.6% 为基准，单位为百分点。", "", "| 配置（* 为最终值） | macro-F1 (%) | Δ vs Direct (pp) | 路径 | 门控接受比例 (%) | 变化的窗口数 |", "|---|---:|---:|---|---:|---:|"]
    for record, policy in zip(records, policies):
        label = {"lambda": "λ", "transition_smoothing": "smoothing", "gap_seconds": "gap (s)"}[policy["group"]] + f"={policy['value']:g}" + ("*" if policy["config"] == "lambda_0.1_final" else "")
        lines.append(f"| {label} | {record['macro_f1_pct']:.6f} | {record['delta_vs_direct_pp']:+.6f} | {record['path']} | {record['gate_acceptance_pct']:.6f} | {record['changed_windows']} |")
    lines += ["", "最终值仅列一行；smoothing 和 gap 的范围统计均包含共享的最终配置。"]
    for group, label in (("lambda", "λ"), ("transition_smoothing", "transition_smoothing"), ("gap_seconds", "gap_seconds")):
        members = [r for r, p in zip(records, policies) if p["group"] == group or p["config"] == "lambda_0.1_final"]
        values = [r["macro_f1_pct"] for r in members]
        lines.append(f"- {label}：macro-F1 {min(values):.6f}%–{max(values):.6f}%，变化范围（最大减最小）{max(values) - min(values):.6f} pp。")
        if all(np.array_equal(predictions[members[0]["config"]], predictions[r["config"]]) for r in members[1:]):
            lines.append(f"  该键可能未生效（{label} 所有取值的最终预测完全相同）。")
    elapsed = perf_counter() - started
    lines += ["", f"数据实际路径：六专家输出 `{V2.as_posix()}/teacher_bank_{{source_validation,unlabeled_test}}.npz`；冻结 group 概率 `{TEACHERS.as_posix()}/frozen_targets.npz`。", f"窗口元信息及选择集标签：`{V1.as_posix()}/inputs.csv`、`source_labels_test_minus1.npy`；最终评测标签：`{(CACHE / 'mmfit_fullroute_v3/evaluation/frozen_evaluation_arrays.npz').as_posix()}` 的 `truth`。", "实际使用键：`SequenceConfig.transition_weight`（0.1），`fit_source_transition.smoothing`（1.0），`fit_source_transition.gap_seconds` / `SequenceConfig.gap_seconds`（1.5）；在线前向路径使用同一会话间隔，先验权重保持原来的 1。发布入口硬编码这些值，YAML 顶级键未被读取；`sequence_kernel_defaults.transition_weight=0.45` 未用于最终解码。本入口将网格接入上述实际参数。", f"总耗时：{elapsed:.2f} 秒（{elapsed / 60:.2f} 分钟，含第 0 步）；复现用时 {reproduction_seconds:.2f} 秒，按其 ×11 估计 {reproduction_seconds * 11 / 60:.2f} 分钟。"]
    (ROOT / "RESULTS_mmfit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"完成：11 个配置；总耗时 {elapsed:.2f} 秒。", flush=True)


if __name__ == "__main__":
    main()
