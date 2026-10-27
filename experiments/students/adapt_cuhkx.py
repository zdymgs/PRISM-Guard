"""Original 40-epoch CUHK-X label-free distillation loop."""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from .source_fusion import visual_head_parameters, model_forward, to_device
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def configure_adaptation_parameters(
    model, adaptation_scope: str = "heads"
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    if adaptation_scope not in {"heads", "heads_motion_encoder"}:
        raise ValueError(f"unknown adaptation scope: {adaptation_scope}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    visual_parameters = visual_head_parameters(model.visual)
    for parameter in visual_parameters:
        parameter.requires_grad_(True)
    fusion_parameters = []
    for name, parameter in model.motion_residual.named_parameters():
        if adaptation_scope == "heads_motion_encoder" or not name.startswith("encoder."):
            parameter.requires_grad_(True)
            fusion_parameters.append(parameter)
    if not visual_parameters or not fusion_parameters:
        raise RuntimeError("challenge adaptation parameter groups are unexpectedly empty")
    return fusion_parameters, visual_parameters

def tempered_probability(probability: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if temperature == 1.0:
        return probability
    scaled = probability.clamp_min(1e-12).pow(1.0 / temperature)
    return scaled / scaled.sum(dim=1, keepdim=True)

def train_label_free(
    model,
    data: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    fusion_parameters, visual_parameters = configure_adaptation_parameters(
        model, args.adaptation_scope
    )
    groups = [
        {
            "params": fusion_parameters,
            "lr": args.fusion_learning_rate,
            "initial_lr": args.fusion_learning_rate,
        },
        {
            "params": visual_parameters,
            "lr": args.visual_head_learning_rate,
            "initial_lr": args.visual_head_learning_rate,
        },
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        cosine = 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / max(args.epochs - 1, 1)))
        for group in optimizer.param_groups:
            group["lr"] = args.minimum_learning_rate + (
                float(group["initial_lr"]) - args.minimum_learning_rate
            ) * cosine
        loss_sum = entropy_sum = confidence_sum = 0.0
        agreement = samples = 0
        started = time.perf_counter()
        for batch_index, batch in enumerate(data):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            if "label" in batch:
                raise RuntimeError("Ground-truth label entered label-free challenge adaptation")
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model_forward(model, batch)
                target_kind = (
                    "emission"
                    if epoch <= args.emission_warmup_epochs
                    else args.target
                )
                target_tensor = (
                    batch["pseudo_emission_probability"]
                    if target_kind == "emission"
                    and "pseudo_emission_probability" in batch
                    else batch["pseudo_probability"]
                )
                target = tempered_probability(target_tensor.float(), args.temperature)
                per_sample = F.kl_div(
                    F.log_softmax(output["logits"].float() / args.temperature, dim=1),
                    target,
                    reduction="none",
                ).sum(dim=1) * args.temperature**2
                if args.confidence_power > 0:
                    sample_weight = batch["pseudo_confidence"].float().clamp_min(1e-3).pow(
                        args.confidence_power
                    )
                    loss = (per_sample * sample_weight).sum() / sample_weight.sum()
                else:
                    loss = per_sample.mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            count = len(batch["sample_id"])
            samples += count
            loss_sum += float(loss.detach()) * count
            entropy_sum += float(
                (-(target * target.clamp_min(1e-12).log()).sum(dim=1)).sum().detach()
            )
            confidence_sum += float(batch["pseudo_confidence"].sum().detach())
            agreement += int(
                output["logits"].argmax(dim=1).eq(target.argmax(dim=1)).sum().detach()
            )
        record = {
            "epoch": epoch,
            "target": (
                "emission"
                if epoch <= args.emission_warmup_epochs
                else args.target
            ),
            "fusion_learning_rate": optimizer.param_groups[0]["lr"],
            "visual_head_learning_rate": optimizer.param_groups[1]["lr"],
            "train_kl": loss_sum / max(samples, 1),
            "target_entropy": entropy_sum / max(samples, 1),
            "mean_target_confidence": confidence_sum / max(samples, 1),
            "student_target_agreement": agreement / max(samples, 1),
            "train_samples": samples,
            "ground_truth_fields_seen": 0,
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    return history
