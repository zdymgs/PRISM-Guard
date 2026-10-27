"""Fit one of the seven source-pure CUHK-X expert families."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("foundation", "skeleton", "thermal", "token", "irthermal", "lavila", "physical"))
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--pretrained-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, default=Path("external"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), action="append",
                        help="Repeat for selected outer folds; default: all three.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--thermal-manifest", type=Path)
    parser.add_argument("--thermal-fold-manifest", type=Path)
    parser.add_argument("--extraction-manifest", type=Path)
    parser.add_argument("--probe-dir", type=Path)
    args = parser.parse_args()
    os.environ["PRISM_TEACHER_WORK_ROOT"] = str(args.work_root.resolve())
    os.environ["PRISM_PRETRAINED_ROOT"] = str(args.pretrained_root.resolve())
    os.environ["PRISM_EXTERNAL_ROOT"] = str(args.external_root.resolve())
    native = Path(__file__).resolve().parent / "modules"
    sys.path.insert(0, str(native))
    from _release_paths import configure_local_resources
    configure_local_resources()
    import numpy as np
    import torch
    from sklearn.model_selection import GroupKFold
    from threadpoolctl import threadpool_limits
    from aligned_multimodal.teacher_common import load_protocol
    from aligned_multimodal.nested_frozen_family_router import EXCLUDED_USERS

    if args.family not in ("foundation", "irthermal"):
        torch.set_num_threads(4)
        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for the paper teacher training configuration")
            budget = (8 if args.family == "thermal" else 4) * 1024 ** 3
            torch.cuda.set_per_process_memory_fraction(min(1., budget / torch.cuda.get_device_properties(0).total_memory))

    protocol = load_protocol()
    keep = ~np.isin(protocol.users.astype(str), list(EXCLUDED_USERS))
    ids, labels, users, folds = (v[keep] for v in (protocol.sample_ids, protocol.labels, protocol.users, protocol.fold_id))
    root = args.work_root.resolve()
    here = root / "aligned_multimodal"
    family = args.family
    population = None
    if family == "foundation":
        from aligned_multimodal.rebuild_foundation_bank import load_raw
        from aligned_multimodal.foundation_provider import FoundationProvider, EXPERT_NAMES
        raw, kinetics, _ = load_raw(ids)
        provider = FoundationProvider(raw, kinetics, ids)
        names = list(EXPERT_NAMES)
    elif family == "skeleton":
        from aligned_multimodal.skeleton_provider import SkeletonProvider
        provider = SkeletonProvider(here / "cache/aligned_192x144", ids)
        names = ["skeleton"]
    elif family == "thermal":
        if not args.thermal_manifest or not args.thermal_fold_manifest:
            parser.error("thermal requires --thermal-manifest and --thermal-fold-manifest")
        from aligned_multimodal.thermal_population import ThermalPopulation
        from aligned_multimodal.thermal_provider import ThermalProvider
        from aligned_multimodal.thermal_data import load_path_map
        population = ThermalPopulation(args.thermal_manifest, ids, labels, users)
        paths = load_path_map(args.thermal_fold_manifest, population.ids)
        provider = ThermalProvider(paths, population.ids)
        names = ["thermal"]
    elif family == "token":
        if not args.extraction_manifest:
            parser.error("token requires --extraction-manifest")
        from aligned_multimodal.token_cache import TokenCache
        from aligned_multimodal.token_provider import TokenProvider, EXPERTS
        cache = TokenCache(root / "runs/vjepa2_vitl_ssv2_dense24_fold0", here / "data/manifest.csv", args.extraction_manifest)
        provider = TokenProvider(cache.select(ids), ids, users, cache_provenance=cache.provenance)
        names = list(EXPERTS)
    elif family == "irthermal":
        from aligned_multimodal.irthermal_provider import Provider, load_raw
        raw, mask, provenance = load_raw(protocol.sample_ids)
        provider = Provider(raw[keep], ids, users, mask[keep], provenance)
        names = ["ir_thermal"]
    elif family == "lavila":
        if not args.probe_dir:
            parser.error("lavila requires --probe-dir from cuhkx-lavila-check")
        from aligned_multimodal.lavila_cache import LaViLaCache
        from aligned_multimodal.lavila_provider import ClassProvider
        from aligned_multimodal.probe_contract import validate_spotcheck
        cache = LaViLaCache(here / "runs/lavila_frame_token_cache", here / "runs/visual_pixel_cache/rows.csv", here / "data/manifest.csv")
        evidence = validate_spotcheck(cache, args.probe_dir, dict(zip(ids, users)))
        provider = ClassProvider(cache, ids, users, evidence)
        names = ["lavila_frame_token"]
    else:
        from aligned_multimodal.physical_cache import PhysicalTokenCache
        from aligned_multimodal.physical_provider import PhysicalProvider
        cache = PhysicalTokenCache(manifest=here / "data/manifest.csv")
        provider = PhysicalProvider(cache, ids, users)
        names = ["physical_token"]

    args.output.mkdir(parents=True, exist_ok=False)
    for fold in args.fold or (0, 1, 2):
        train = np.flatnonzero(folds != fold)
        held = np.flatnonzero(folds == fold)
        requests = [("outer", train, held, None)]
        requests.extend((f"inner{k}", train[tr], train[va], va)
                        for k, (tr, va) in enumerate(GroupKFold(3).split(train, groups=users[train])))
        inner = np.empty((len(train), len(names), 40), np.float32)
        coverage = np.zeros(len(train), int)
        inner_fold = np.full(len(train), -1, int)
        for context_name, source, target, local in requests:
            context = f"fold{fold}.{context_name}"
            folder = args.output / f"fold{fold}" / context_name
            folder.mkdir(parents=True)

            def callback(*values):
                if family == "token":
                    expert, seed, logits, state, record = values
                    member = folder / "members" / expert / f"seed{seed}"
                elif family == "thermal":
                    member_name, logits, state, diagnostics, fit_ids, predict_ids = values
                    record = {"diagnostics": diagnostics, "fit_ids": fit_ids.tolist(), "prediction_ids": predict_ids.tolist()}
                    member = folder / "members" / member_name
                else:
                    seed, logits, state, record = values
                    member = folder / "members" / f"seed{seed}"
                member.mkdir(parents=True, exist_ok=False)
                torch.save(state, member / "checkpoint.pt")
                np.savez_compressed(member / "outputs.npz", logits=logits)
                (member / "receipt.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

            resource_context = threadpool_limits(4 if family == "foundation" else 1) if family in ("foundation", "irthermal") else nullcontext()
            with resource_context:
                if family in ("foundation", "skeleton"):
                    result = provider.fit_predict(source, labels[source], users[source], target, users[target], context=context,
                                                  **({} if family == "foundation" else {"device": args.device}))
                elif family == "thermal":
                    tx, ty, tu, tids, tusers = population.context(source, target)
                    result = provider.fit_predict(tx, ty, tu, tids, tusers, context=context, device=args.device, fit_callback=callback)
                elif family == "irthermal":
                    result = provider.fit_predict(source, labels[source], target, context=context)
                else:
                    result = provider.fit_predict(source, labels[source], target, outer_fold=fold, context=context,
                                                  device=args.device, fit_callback=callback)
            probability, receipt, extra = result[:3]
            if len(result) == 4:
                torch.save(result[3], folder / "checkpoint.pt")
            np.savez_compressed(folder / "bank.npz", sample_ids=ids[target], users=users[target],
                                expert_names=np.asarray(names), probabilities=probability.astype(np.float32), **extra)
            (folder / "provenance.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
            if local is None:
                outer = probability
            else:
                inner[local] = probability
                coverage[local] += 1
                inner_fold[local] = int(context_name.removeprefix("inner"))
        if not np.all(coverage == 1):
            raise ValueError("inner bank must cover each source row exactly once")
        np.savez_compressed(args.output / f"fold{fold}_banks.npz", inner_sample_ids=ids[train], outer_sample_ids=ids[held],
                            inner_users=users[train], outer_users=users[held], expert_names=np.asarray(names),
                            inner_probability_bank=inner, outer_probability_bank=outer.astype(np.float32), inner_fold=inner_fold)


if __name__ == "__main__":
    main()
