"""The original G2 training loop with explicit release input/output adapters."""
from __future__ import annotations
import csv, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from prism.baselines.objectives import kd_soft_loss, pl_conf_loss
from experiments.students.runtime import Context, sha, write, seed_all, CLASSES, SEEDS, require_cuda
from experiments.students.cache import aligned_values, read_json

class IndexedDataset(Dataset):
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, index): return {**self.base[index], '_extra_row_index': index}

def memory_allocated():
    return torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
def memory_reserved():
    return torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0

def mean_entropy(logits):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(-1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
    return float(-(np.exp(logp)*logp).sum(-1).mean())

def positions(ctx, batch, cursor=None):
    if '_extra_row_index' in batch:
        return np.asarray(batch.pop('_extra_row_index'), dtype=int)
    return ctx.row_positions(batch)

def data_loader(ctx, train, epoch=0):
    ds = ctx.train if train else ctx.base
    if ctx.name == 'mmfit':
        if not hasattr(ctx, '_mmfit_train_loader'):
            ctx._mmfit_train_loader = ctx.make_loader(IndexedDataset(ctx.train), True)
        return ctx._mmfit_train_loader if train else ctx.make_loader(IndexedDataset(ctx.base), False)
    return ctx.make_loader(ds, train, epoch)

@torch.no_grad()
def infer(ctx, amp=False):
    ctx.model.eval()
    logits = np.empty((ctx.info['rows'], ctx.info['classes']), np.float32)
    covered = np.zeros(ctx.info['rows'], int)
    for batch in data_loader(ctx, False):
        row = positions(ctx, batch)
        assert not {'label', 'labels', 'class_id', 'truth'} & set(batch)
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            values = ctx.forward(ctx.move(batch))
        logits[row] = values.float().cpu().numpy()
        covered[row] += 1
    assert np.all(covered == 1) and np.isfinite(logits).all()
    return logits

def run(args, *, distill=False):
    name = args.dataset
    arm = {'kd-soft': 'kd_soft', 'confidence-pl': 'pl_conf'}.get(args.method, 'kd_soft')
    require_cuda(args)
    cfg = read_json(args.inputs)
    key = 'targets' if distill else 'direct_probability'
    probability = aligned_values(args.targets, [str(s) for s in cfg['record_id']], key=key, classes=CLASSES[name])
    seed = args.seed if args.seed is not None else SEEDS[name]
    info = dict(rows=len(probability), classes=CLASSES[name], seed=seed)
    arrays = dict(probabilities=probability, ids=np.asarray(cfg['record_id']),
                  indices=np.arange(len(probability)), direct_target=probability.argmax(1))
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('output directory must be empty')
    out.mkdir(parents=True, exist_ok=True)
    contract = dict(dataset=name, method='prism-d' if distill else args.method,
        epochs=args.epochs, batch_size=64, seed=seed,
        precision='AMP float16' if name=='cuhkx' else 'float32',
        c0_sha256=sha(args.c0), input_sha256=sha(args.inputs), target_sha256=sha(args.targets),
        optimizer='AdamW',
        motion_fusion_lr=1e-4,
        visual_head_lr=5e-5, minimum_lr=5e-6,
        confidence_threshold=0.9 if arm=='pl_conf' else None)
    write(out/'config.json', contract)
    torch.set_num_threads(4 if name=='mmfit' else 2)
    seed_all(seed)
    began=time.perf_counter()
    device=torch.device(args.device)
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    ctx=Context(args, probability)
    model=ctx.model
    resource_probe=False
    # CUHK's existing trainer resets the RNG immediately before adaptation.
    if name == 'cuhkx':
        seed_all(info['seed'])
    fusion, visual = ctx.configure()
    parameters = fusion + visual
    parameter_names = [n for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{'params': fusion, 'lr': 1e-4},
                                   {'params': visual, 'lr': 5e-5}], weight_decay=.02)
    write(out / 'parameter_groups.json', dict(trainable=parameter_names,
          trainable_count=sum(p.numel() for p in parameters),
          frozen_count=sum(p.numel() for p in model.parameters() if not p.requires_grad)))
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=5e-6)
                 if name == 'mmfit' else None)
    amp = contract['precision'] == 'AMP float16'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    history = []
    updates = 0
    # MMAct's original trainer performs one deterministic C0 inference pass
    # before adaptation. This loader also advances exactly the original RNG.
    if name == 'mmact':
        c0_logits = infer(ctx)
    for epoch in range(contract['epochs']):
        model.train()
        ctx.keep_cnn_eval()
        if scheduler is None:
            cosine = .5 * (1 + math.cos(math.pi * epoch / 39))
            for group, initial in zip(optimizer.param_groups, [1e-4, 5e-5]):
                group['lr'] = 5e-6 + (initial - 5e-6) * cosine
        epoch_start = time.perf_counter()
        loss_sum = 0.0
        participants = seen = skipped = steps = attempted_steps = amp_skips = nonfinite_norms = 0
        lrs = [g['lr'] for g in optimizer.param_groups]
        for batch_index, batch in enumerate(data_loader(ctx, True, epoch)):
            row = positions(ctx, batch)
            assert not {'label', 'labels', 'class_id', 'truth'} & set(batch)
            b = ctx.move(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                logits = ctx.forward(b).float()
                p = torch.as_tensor(arrays['probabilities'][row], device=device, dtype=torch.float32)
                if arm == 'kd_soft':
                    loss, kept = kd_soft_loss(logits, p), len(row)
                elif arm == 'pl_conf':
                    loss, kept = pl_conf_loss(logits, p)
            seen += len(row)
            if loss is None:
                skipped += 1
            else:
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite objective')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
                if not torch.isfinite(norm) and not amp:
                    raise FloatingPointError('nonfinite gradients')
                nonfinite_norms += int(not torch.isfinite(norm))
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                amp_skipped = amp and scaler.get_scale() < scale_before
                amp_skips += int(amp_skipped)
                attempted_steps += 1
                loss_sum += float(loss.detach()) * kept
                participants += kept
                steps += int(not amp_skipped)
                updates += int(not amp_skipped)
            if resource_probe:
                torch.cuda.synchronize()
                write(out / 'probe.json', dict(status='one_full_batch_passed', batch_size=len(row),
                    peak_allocated_bytes=memory_allocated(),
                    seconds=time.perf_counter()-began, evaluation_labels_read=False))
                return
        if scheduler:
            scheduler.step()
        record = dict(epoch=epoch+1, loss=loss_sum / participants if participants else None,
            loss_samples=participants, input_samples=seen, skipped_batches=skipped,
            updates_this_epoch=steps, total_updates=updates, learning_rates=lrs,
            optimizer_step_attempts=attempted_steps, amp_skipped_updates=amp_skips,
            nonfinite_gradient_norms=nonfinite_norms,
            epoch_seconds=time.perf_counter()-epoch_start,
            cumulative_seconds=time.perf_counter()-began,
            peak_allocated_bytes=memory_allocated(),
            seed=info['seed'], evaluation_labels_read=False)
        assert seen == info['rows']
        history.append(record)
        with (out / 'train.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record) + '\n')
        print(json.dumps(dict(dataset=name, baseline=arm, **record)), flush=True)
        # Epoch persistence stays entirely in the new directory.
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch+1,
            scheduler=scheduler.state_dict() if scheduler else None,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(), python_rng=random.getstate()), out / 'latest.pt')
    logits = infer(ctx, amp=amp)
    pred = logits.argmax(-1)
    np.savez_compressed(out / 'predictions.npz', ids=arrays['ids'], indices=arrays['indices'], logits=logits)
    with (out / 'predictions.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['record_id', 'pred'])
        writer.writerows(zip(arrays['ids'], pred.tolist()))
    torch.save(dict(model=model.state_dict(), config=contract), out / 'final.pt')
    summary = dict(status='complete', dataset=name, method=contract['method'],
        epochs=len(history), updates=updates, seconds=time.perf_counter()-began,
        peak_allocated_bytes=memory_allocated(), peak_reserved_bytes=memory_reserved(),
        target_agreement=float(np.mean(pred==arrays['direct_target'])),
        source_labels_read=False)
    write(out/'summary.json', summary)
    return summary
