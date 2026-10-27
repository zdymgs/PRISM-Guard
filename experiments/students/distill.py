"""CUHK-X distillation dispatch retaining the original training function."""

from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
from .runtime import (Context, SEEDS, sha, write, seed_all, require_cuda, load_targets)
from .adapt_cuhkx import train_label_free


def run_cuhkx(args):
    require_cuda(args)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("output directory must be empty")
    probability = load_targets(args)
    seed = args.seed if args.seed is not None else SEEDS[args.dataset]
    seed_all(seed)
    ctx = Context(args, probability)
    from experiments.baselines.training import infer
    c0_logits = infer(ctx, amp=ctx.device.type == "cuda")
    # Each arm starts from the same C0 and resets its original adaptation RNG.
    seed_all(seed)
    recipe = SimpleNamespace(epochs=args.epochs, fusion_learning_rate=1e-4,
        visual_head_learning_rate=5e-5, minimum_learning_rate=5e-6,
        weight_decay=0.02, temperature=1.0, confidence_power=0.0,
        adaptation_scope="heads_motion_encoder", emission_warmup_epochs=0,
        target="structured", max_train_batches=0)
    history = train_label_free(ctx.model, ctx.make_loader(ctx.train, True), recipe, ctx.device)
    logits = infer(ctx, amp=ctx.device.type == "cuda")
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": ctx.model.state_dict(), "dataset": args.dataset,
                "fold": args.fold, "epochs": args.epochs}, out/"final.pt")
    np.savez_compressed(out/"predictions.npz", record_id=np.asarray(ctx.ids), logits=logits)
    np.savez_compressed(out/"c0_predictions.npz", record_id=np.asarray(ctx.ids), logits=c0_logits)
    write(out/"history.json", history)
    write(out/"config.json", dict(dataset=args.dataset, fold=args.fold, seed=seed,
        recipe=vars(recipe), c0_sha256=sha(args.c0), targets_sha256=sha(args.targets),
        inputs_sha256=sha(args.inputs)))
    summary = dict(status="complete", dataset=args.dataset, fold=args.fold,
        records=len(ctx.ids), epochs=args.epochs,
        target_agreement=float(np.mean(logits.argmax(1)==probability.argmax(1))),
        source_labels_read=False)
    write(out/"summary.json", summary)
    return summary
