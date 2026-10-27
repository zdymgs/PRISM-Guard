from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from cached_motion_data import (
    MOTION_FIELDS,
    CachedSequenceMotionDataset,
    collate_cached_motion,
)
from mobind_lite_model import (
    JointMotionEncoder,
    MoBindLite,
    MoBindMotionResidual,
    SeparateMotionEncoder,
    SkeletonPartEncoder,
    UnifiedMoBindStudent,
)
from temporal_mobind_model import (
    TemporalMoBindStudent,
    TemporalRefinedMoBindStudent,
    TemporalCrossAttentionPoolStudent,
)
from spatial_mobind_model import SpatialCrossAttentionPoolStudent
from train_cached_motion_proxy import load_npz, load_visual
from train_visual_student_oof import (
    class_weights,
    metric_dict,
    relation_loss,
    split_universe,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PIXELS = PROJECT_DIR / "runs/visual_pixel_cache_t16_r160_v12"
DEFAULT_MOTION = PROJECT_DIR / "runs/motion_window_cache_t16"
DEFAULT_TEACHER_FEATURES = (
    PROJECT_DIR / "runs/videomae_large_multiclip_full40/complete_features.npz"
)
DEFAULT_TEACHER_LOGITS = (
    PROJECT_DIR
    / "runs/videomae_large_multiclip_full40_head/candidate_oof_logits.npz"
)
DEFAULT_IMU_TEACHER = (
    PROJECT_DIR
    / "runs/p3_imu_oof/stat_random_forest_device_dropout_aligned_oof.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-modality student MoBind-lite feature-fusion screen."
    )
    parser.add_argument("--visual-checkpoint", type=Path, required=True)
    parser.add_argument("--sequence-cache", type=Path, required=True)
    parser.add_argument(
        "--compact-sequence-cache",
        type=Path,
        help="Exact compact student cache paired with a v4 spatial cache.",
    )
    parser.add_argument("--pretrain-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--initial-fusion-checkpoint",
        type=Path,
        help=(
            "Initialize a joint model from a trained Skeleton-fusion checkpoint. "
            "The Skeleton encoder is remapped into the joint encoder and all "
            "shared Visual/fusion weights are preserved."
        ),
    )
    parser.add_argument(
        "--temporal-anchor-checkpoint",
        type=Path,
        help=(
            "v2/v3 only: load an exact trained student separate/clip checkpoint, "
            "keep its complete local/global path frozen, and train only the new "
            "zero-initialized temporal mechanism."
        ),
    )
    parser.add_argument("--motion-cache", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--pixel-cache", type=Path, default=DEFAULT_PIXELS)
    parser.add_argument("--teacher-features", type=Path, default=DEFAULT_TEACHER_FEATURES)
    parser.add_argument("--teacher-logits", type=Path, default=DEFAULT_TEACHER_LOGITS)
    parser.add_argument("--imu-teacher-logits", type=Path)
    parser.add_argument("--imu-event-features", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--modality",
        choices=("skeleton", "imu", "separate", "joint"),
        required=True,
    )
    parser.add_argument(
        "--fusion-position",
        choices=("clip", "temporal", "temporal_refined", "temporal_v3", "spatial_v4"),
        default="clip",
        help=(
            "clip keeps proven student post-pooling retrieval; temporal is the legacy "
            "replacement candidate; temporal_refined preserves student and adds a bounded "
            "modality-private pre-pooling residual; temporal_v3 preserves student and "
            "uses local cross-attention only to reweight visual temporal pooling; "
            "spatial_v4 conditions layer4 spatial pooling before visual temporal encoding."
        ),
    )
    parser.add_argument(
        "--temporal-radius",
        type=int,
        default=1,
        help="Local motion radius for --fusion-position temporal.",
    )
    parser.add_argument(
        "--temporal-residual-budget",
        type=float,
        default=0.10,
        help="Fixed total temporal residual budget for v2.",
    )
    parser.add_argument(
        "--temporal-attention-logit-limit",
        type=float,
        default=1.0,
        help="Fixed absolute pooling-logit limit for v3 cross-attention.",
    )
    parser.add_argument("--spatial-grid", type=int, default=5)
    parser.add_argument(
        "--spatial-attention-logit-limit",
        type=float,
        default=1.0,
        help="Fixed absolute region-logit limit for v4.",
    )
    parser.add_argument("--stage-a-epochs", type=int, default=4)
    parser.add_argument("--stage-b-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--fusion-learning-rate", type=float, default=4e-4)
    parser.add_argument(
        "--stage-b-fusion-learning-rate",
        type=float,
        help="Optional lower fusion LR after adapter-only Stage A.",
    )
    parser.add_argument("--initial-residual-strength", type=float, default=0.25)
    parser.add_argument("--skeleton-time-position", action="store_true")
    parser.add_argument("--skeleton-multistream", action="store_true")
    parser.add_argument("--skeleton-multistream-strength", type=float, default=0.10)
    parser.add_argument("--skeleton-adaptive-graph", action="store_true")
    parser.add_argument("--joint-initial-imu-gate", type=float, default=0.35)
    parser.add_argument("--joint-max-imu-residual", type=float, default=1.0)
    parser.add_argument("--joint-freeze-pretrained-encoders", action="store_true")
    parser.add_argument(
        "--joint-train-alignment",
        action="store_true",
        help="Keep S/I projection layers trainable while modality encoders stay frozen.",
    )
    parser.add_argument(
        "--joint-zero-initialize-imu-residual",
        action="store_true",
        help="Start joint fusion as an exact Skeleton-model function.",
    )
    parser.add_argument(
        "--joint-stage-a-adapter-only",
        action="store_true",
        help="During Stage A update only new joint adapters and optional alignment.",
    )
    parser.add_argument(
        "--separate-modality-dropout",
        type=float,
        default=0.0,
        help="Training-only probability of dropping Skeleton or IMU token groups.",
    )
    parser.add_argument("--joint-gate-weight", type=float, default=0.0)
    parser.add_argument("--joint-gate-temperature", type=float, default=0.5)
    parser.add_argument(
        "--global-fusion-mode",
        choices=("additive", "conditional"),
        default="additive",
    )
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-4)
    parser.add_argument("--visual-learning-rate", type=float, default=2e-5)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--class-weight-power", type=float, default=0.35)
    parser.add_argument("--worst-group-weight", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.08)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--distillation-weight", type=float, default=0.5)
    parser.add_argument("--relation-weight", type=float, default=0.1)
    parser.add_argument("--motion-aux-weight", type=float, default=0.35)
    parser.add_argument("--selective-anchor-weight", type=float, default=0.15)
    parser.add_argument("--reliability-weight", type=float, default=1.0)
    parser.add_argument("--reliability-positive-weight", type=float, default=1.0)
    parser.add_argument("--reliability-groups", type=int, default=1)
    parser.add_argument(
        "--reliability-target",
        choices=(
            "corrupted",
            "loss_advantage",
            "oof_complementarity",
            "oof_visual_error",
        ),
        default="corrupted",
    )
    parser.add_argument("--reliability-advantage-temperature", type=float, default=0.5)
    parser.add_argument("--imu-teacher-weight", type=float, default=0.0)
    parser.add_argument("--imu-teacher-temperature", type=float, default=1.0)
    parser.add_argument("--visual-corruption-probability", type=float, default=0.65)
    parser.add_argument("--visual-feature-dropout", type=float, default=0.25)
    parser.add_argument("--visual-view-dropout", type=float, default=0.35)
    parser.add_argument(
        "--live-visual-anchor",
        action="store_true",
        help=(
            "Use the current subject-safe visual model output as the anchor and "
            "reported visual baseline instead of sequence-cache logits."
        ),
    )
    parser.add_argument("--accuracy-gate-pp", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument(
        "--final-refit",
        action="store_true",
        help=(
            "Train all 2470 non-permanent samples and evaluate fused and pure "
            "Visual predictions together on the fixed 444 validation samples."
        ),
    )
    parser.add_argument(
        "--all-label-refit",
        action="store_true",
        help=(
            "Terminal refit on all 2914 true-labeled Train rows with fixed stage "
            "budgets. Saves a checkpoint without evaluating any training row."
        ),
    )
    parser.add_argument(
        "--subject-holdout-users",
        nargs="+",
        help=(
            "Leakage-safe challenge protocol: train fusion on all other subjects and "
            "evaluate this explicit pseudo-Test once after fixed stages."
        ),
    )
    parser.add_argument(
        "--train-users",
        nargs="+",
        help=(
            "Optional explicit training-user allowlist. Requires "
            "--subject-holdout-users; every other subject remains untouched."
        ),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(
    args: argparse.Namespace,
    full: CachedSequenceMotionDataset,
    sample_ids: np.ndarray,
    temporal_augment: bool,
) -> CachedSequenceMotionDataset:
    lookup = full.index_lookup
    missing = set(sample_ids.tolist()) - set(lookup)
    if missing:
        raise RuntimeError(f"cached motion is missing samples: {sorted(missing)[:3]}")
    return CachedSequenceMotionDataset(
        args.sequence_cache,
        args.motion_cache,
        args.pixel_cache,
        args.teacher_features,
        args.teacher_logits,
        indices=np.asarray([lookup[value] for value in sample_ids], dtype=np.int64),
        temporal_augment=temporal_augment,
        imu_teacher_logits=args.imu_teacher_logits,
        imu_event_features=args.imu_event_features,
        compact_sequence_cache=args.compact_sequence_cache,
    )


def loader(
    dataset: CachedSequenceMotionDataset,
    args: argparse.Namespace,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=True,
        collate_fn=collate_cached_motion,
        # Keep proxy batching bit-for-bit comparable, but consume every labeled
        # row in either terminal refit protocol.
        drop_last=(
            shuffle
            and not args.final_refit
            and not args.all_label_refit
            and not args.subject_holdout_users
            and len(dataset) >= args.batch_size
        ),
    )


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def corrupt_visual_sequence(
    sequence: torch.Tensor, args: argparse.Namespace
) -> torch.Tensor:
    batch = sequence.shape[0]
    selected = torch.rand(batch, 1, 1, 1, 1, device=sequence.device)
    selected = selected < args.visual_corruption_probability
    feature_keep = torch.rand(
        batch, 1, 1, 1, sequence.shape[-1], device=sequence.device
    ) >= args.visual_feature_dropout
    view_keep = torch.rand(
        batch, 2, 3, 1, 1, device=sequence.device
    ) >= args.visual_view_dropout
    # Always preserve at least one camera in each early/late window.
    empty = ~view_keep.any(dim=2, keepdim=True)
    fallback = torch.zeros_like(view_keep)
    fallback[:, :, 0] = True
    view_keep = torch.where(empty, fallback, view_keep)
    corrupted = sequence * feature_keep.to(sequence.dtype) * view_keep.to(sequence.dtype)
    return torch.where(selected, corrupted, sequence)


def corrupt_visual_pair(
    compact: torch.Tensor,
    regions: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    if compact.ndim != 5 or regions.ndim != 6:
        raise ValueError("compact/spatial visual tensors must be rank 5/6")
    if regions.shape[:4] != compact.shape[:4] or regions.shape[-1] != compact.shape[-1]:
        raise ValueError("compact and spatial visual grids differ")
    batch = compact.shape[0]
    selected = (
        torch.rand(batch, 1, 1, 1, 1, device=compact.device)
        < args.visual_corruption_probability
    )
    feature_keep = torch.rand(
        batch, 1, 1, 1, compact.shape[-1], device=compact.device
    ) >= args.visual_feature_dropout
    view_keep = torch.rand(
        batch, 2, 3, 1, 1, device=compact.device
    ) >= args.visual_view_dropout
    empty = ~view_keep.any(dim=2, keepdim=True)
    fallback = torch.zeros_like(view_keep)
    fallback[:, :, 0] = True
    view_keep = torch.where(empty, fallback, view_keep)
    compact_mask = feature_keep & view_keep
    region_mask = feature_keep.unsqueeze(-2) & view_keep.unsqueeze(-2)
    corrupted_compact = compact * compact_mask.to(compact.dtype)
    corrupted_regions = regions * region_mask.to(regions.dtype)
    return (
        torch.where(selected, corrupted_compact, compact),
        torch.where(selected.unsqueeze(-2), corrupted_regions, regions),
    )


def model_forward(
    model: nn.Module,
    batch: dict[str, Any],
    args: argparse.Namespace | None = None,
    augment_visual: bool = False,
) -> dict[str, torch.Tensor]:
    motion = {field: batch[field] for field in MOTION_FIELDS}
    for field in ("imu_event_features", "imu_event_valid"):
        if field in batch:
            motion[field] = batch[field]
    sequence = batch["backbone_sequence"]
    region_sequence = batch.get("backbone_region_sequence")
    if augment_visual:
        if args is None:
            raise ValueError("visual corruption requires training arguments")
        if region_sequence is None:
            sequence = corrupt_visual_sequence(sequence, args)
        else:
            sequence, region_sequence = corrupt_visual_pair(
                sequence, region_sequence, args
            )
    if isinstance(model, SpatialCrossAttentionPoolStudent):
        if region_sequence is None:
            raise RuntimeError("v4 requires a spatial region sequence")
        return model.forward_from_backbone_region_sequence(
            sequence,
            region_sequence,
            batch["view_valid"],
            batch["view_quality"],
            batch["global_time_position"],
            motion,
        )
    return model.forward_from_backbone_sequence(
        sequence,
        batch["view_valid"],
        batch["view_quality"],
        batch["global_time_position"],
        motion,
    )


def visual_head_parameters(visual: nn.Module) -> list[nn.Parameter]:
    modules = [
        visual.temporal_encoder,
        visual.temporal_fusion,
        visual.token_encoder,
        visual.quality_gate,
        visual.stage_fusion,
        visual.classifier,
    ]
    result = [visual.time_position, visual.view_embedding, visual.window_embedding]
    result.extend(parameter for module in modules for parameter in module.parameters())
    return [parameter for parameter in result if parameter is not None]


def build_model(
    args: argparse.Namespace,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    visual, visual_config = load_visual(args.visual_checkpoint)
    checkpoint = torch.load(
        args.pretrain_checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    config = dict(checkpoint["model_config"])
    pretrained = MoBindLite(**config)
    incompatible = pretrained.load_state_dict(checkpoint["model_state"], strict=False)
    allowed_missing_prefixes = (
        (
            "imu_encoder.device_encoder.",
            "imu_encoder.statistics_projection.",
        )
        if args.modality == "skeleton"
        else ()
    )
    unexpected = list(incompatible.unexpected_keys)
    missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(allowed_missing_prefixes)
    ]
    if unexpected or missing:
        raise RuntimeError(
            "unexpected MoBind pretrain state mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if args.modality == "skeleton":
        desired_time = (
            args.skeleton_time_position
            or pretrained.skeleton_encoder.explicit_time_position
        )
        desired_multistream = (
            args.skeleton_multistream
            or pretrained.skeleton_encoder.multistream_input
        )
        desired_adaptive_graph = (
            args.skeleton_adaptive_graph
            or pretrained.skeleton_encoder.adaptive_graph
        )
        needs_upgrade = (
            desired_time != pretrained.skeleton_encoder.explicit_time_position
            or desired_multistream != pretrained.skeleton_encoder.multistream_input
            or desired_adaptive_graph != pretrained.skeleton_encoder.adaptive_graph
        )
        if needs_upgrade:
            # Preserve the initialization stream for all common fusion layers so a
            # paired architecture experiment differs only by the new encoder path.
            cpu_rng_state = torch.random.get_rng_state()
            encoder = SkeletonPartEncoder(
                width=int(config["width"]),
                dropout=float(config["dropout"]),
                explicit_time_position=desired_time,
                multistream_input=desired_multistream,
                multistream_residual_strength=args.skeleton_multistream_strength,
                adaptive_graph=desired_adaptive_graph,
            )
            incompatible = encoder.load_state_dict(
                pretrained.skeleton_encoder.state_dict(), strict=False
            )
            unexpected = list(incompatible.unexpected_keys)
            allowed_upgrade_prefixes = []
            if desired_time and not pretrained.skeleton_encoder.explicit_time_position:
                allowed_upgrade_prefixes.append("time_projection.")
            if desired_multistream and not pretrained.skeleton_encoder.multistream_input:
                allowed_upgrade_prefixes.extend(
                    (
                        "stream_stems.",
                        "stream_fusion.",
                        "multistream_residual_logit",
                    )
                )
            if desired_adaptive_graph and not pretrained.skeleton_encoder.adaptive_graph:
                allowed_upgrade_prefixes.extend(
                    (
                        "blocks.0.adaptive_adjacency",
                        "blocks.1.adaptive_adjacency",
                    )
                )
            missing = [
                key
                for key in incompatible.missing_keys
                if not key.startswith(tuple(allowed_upgrade_prefixes))
            ]
            if unexpected or missing:
                raise RuntimeError(
                    "unexpected Skeleton encoder-upgrade state mismatch: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            torch.random.set_rng_state(cpu_rng_state)
            config["skeleton_time_position"] = desired_time
            config["skeleton_multistream"] = desired_multistream
            config["skeleton_multistream_strength"] = (
                args.skeleton_multistream_strength
            )
            config["skeleton_adaptive_graph"] = desired_adaptive_graph
        else:
            encoder = pretrained.skeleton_encoder
        semantic_head = pretrained.skeleton_head
        teacher_projection = pretrained.skeleton_teacher_projection
    elif args.modality == "imu":
        encoder = pretrained.imu_encoder
        semantic_head = pretrained.imu_head
        teacher_projection = pretrained.imu_teacher_projection
    elif args.modality == "separate":
        cpu_rng_state = torch.random.get_rng_state()
        encoder = SeparateMotionEncoder(
            pretrained.skeleton_encoder,
            pretrained.imu_encoder,
            width=int(config["width"]),
            modality_dropout=args.separate_modality_dropout,
        )
        torch.random.set_rng_state(cpu_rng_state)
        semantic_head = pretrained.skeleton_head
        teacher_projection = pretrained.skeleton_teacher_projection
    else:
        cpu_rng_state = torch.random.get_rng_state()
        encoder = JointMotionEncoder(
            pretrained.skeleton_encoder,
            pretrained.imu_encoder,
            pretrained.skeleton_projection,
            pretrained.imu_projection,
            pretrained.imu_head,
            width=int(config["width"]),
            alignment_width=int(config["alignment_width"]),
            dropout=float(config["dropout"]),
            initial_imu_gate=args.joint_initial_imu_gate,
            maximum_imu_residual=args.joint_max_imu_residual,
        )
        # Keep all downstream residual/fusion initialization paired with V+S.
        torch.random.set_rng_state(cpu_rng_state)
        # Skeleton remains the semantic anchor; IMU enters before this one head.
        semantic_head = pretrained.skeleton_head
        teacher_projection = pretrained.skeleton_teacher_projection
    if args.imu_event_features and int(config.get("imu_event_feature_width", 0)) == 0:
        with np.load(args.imu_event_features.resolve(), allow_pickle=False) as data:
            reliability_event_feature_width = int(data["features"].shape[1])
    else:
        reliability_event_feature_width = 0
    residual = MoBindMotionResidual(
        args.modality,
        encoder,
        semantic_head,
        teacher_projection,
        visual_width=512,
        motion_width=int(config["width"]),
        dropout=float(config["dropout"]),
        initial_residual_strength=args.initial_residual_strength,
        reliability_event_feature_width=reliability_event_feature_width,
        global_fusion_mode=args.global_fusion_mode,
        reliability_groups=args.reliability_groups,
    )
    if args.fusion_position == "temporal":
        model = TemporalMoBindStudent(
            visual, residual, temporal_radius=args.temporal_radius
        )
    elif args.fusion_position == "temporal_refined":
        model = TemporalRefinedMoBindStudent(
            visual,
            residual,
            temporal_radius=args.temporal_radius,
            temporal_budget=args.temporal_residual_budget,
            dropout=float(config["dropout"]),
        )
    elif args.fusion_position == "temporal_v3":
        model = TemporalCrossAttentionPoolStudent(
            visual,
            residual,
            temporal_radius=args.temporal_radius,
            attention_heads=4,
            maximum_pooling_logit=args.temporal_attention_logit_limit,
            dropout=float(config["dropout"]),
        )
    elif args.fusion_position == "spatial_v4":
        model = SpatialCrossAttentionPoolStudent(
            visual,
            residual,
            spatial_grid=args.spatial_grid,
            attention_heads=4,
            maximum_spatial_logit=args.spatial_attention_logit_limit,
            dropout=float(config["dropout"]),
        )
    else:
        model = UnifiedMoBindStudent(visual, residual)
    return model, visual_config, config


def initialize_temporal_from_visual(
    model: (
        TemporalRefinedMoBindStudent
        | TemporalCrossAttentionPoolStudent
        | SpatialCrossAttentionPoolStudent
    ),
    checkpoint_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load every student tensor while leaving only new temporal tensors fresh."""
    checkpoint = torch.load(
        checkpoint_path.resolve(), map_location="cpu", weights_only=False
    )
    if checkpoint.get("modality") != "separate":
        raise ValueError("fusion temporal anchor must use separate Skeleton/IMU tokens")
    if checkpoint.get("fusion_position", "clip") != "clip":
        raise ValueError("fusion temporal anchor must be a student clip-fusion checkpoint")
    source = checkpoint["model_state"]
    incompatible = model.load_state_dict(source, strict=False)
    if isinstance(model, TemporalRefinedMoBindStudent):
        allowed_missing = (
            "skeleton_temporal_query.",
            "imu_temporal_query.",
            "skeleton_temporal_projection.",
            "imu_temporal_projection.",
        )
        candidate_position = "temporal_refined"
    elif isinstance(model, TemporalCrossAttentionPoolStudent):
        allowed_missing = (
            "skeleton_cross_attention.",
            "imu_cross_attention.",
        )
        candidate_position = "temporal_v3"
    else:
        allowed_missing = (
            "skeleton_spatial_scorer.",
            "imu_spatial_scorer.",
        )
        candidate_position = "spatial_v4"
    unexpected = list(incompatible.unexpected_keys)
    invalid_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(allowed_missing)
    ]
    if unexpected or invalid_missing:
        raise RuntimeError(
            "student temporal anchor mismatch: "
            f"missing={invalid_missing}, unexpected={unexpected}"
        )
    loaded = set(source)
    target = model.state_dict()
    shape_mismatch = [
        key
        for key in loaded & set(target)
        if source[key].shape != target[key].shape
    ]
    if shape_mismatch:
        raise RuntimeError(f"student temporal anchor shape mismatch: {shape_mismatch}")
    audit = {
        "checkpoint": str(checkpoint_path.resolve()),
        "source_modality": checkpoint["modality"],
        "source_fusion_position": checkpoint.get("fusion_position", "clip"),
        "candidate_fusion_position": candidate_position,
        "loaded_tensors": len(source),
        "new_candidate_tensors": len(incompatible.missing_keys),
        "exact_function_required": True,
    }
    return audit, checkpoint


def initialize_joint_from_skeleton_fusion(
    model: UnifiedMoBindStudent,
    checkpoint_path: Path,
    zero_imu_residual: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Preserve a trained V+S function while adding the joint IMU adapter."""
    joint = model.motion_residual.encoder
    if not isinstance(joint, JointMotionEncoder):
        raise ValueError("a Skeleton fusion anchor can initialize only joint modality")
    checkpoint = torch.load(
        checkpoint_path.resolve(), map_location="cpu", weights_only=False
    )
    if checkpoint.get("modality") != "skeleton":
        raise ValueError("initial fusion checkpoint must be a Skeleton run")
    source = checkpoint["model_state"]
    target = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    missing_source_targets = []
    shape_mismatch = []
    source_encoder_prefix = "motion_residual.encoder."
    target_encoder_prefix = "motion_residual.encoder.skeleton_encoder."
    for key, value in source.items():
        target_key = (
            target_encoder_prefix + key.removeprefix(source_encoder_prefix)
            if key.startswith(source_encoder_prefix)
            else key
        )
        if target_key not in target:
            missing_source_targets.append((key, target_key))
        elif target[target_key].shape != value.shape:
            shape_mismatch.append(
                (key, tuple(value.shape), tuple(target[target_key].shape))
            )
        else:
            mapped[target_key] = value
    if missing_source_targets or shape_mismatch:
        raise RuntimeError(
            "Skeleton fusion anchor is incompatible with the joint model: "
            f"missing_targets={missing_source_targets}, "
            f"shape_mismatch={shape_mismatch}"
        )
    incompatible = model.load_state_dict(mapped, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected anchor keys after remapping: {incompatible.unexpected_keys}"
        )
    if zero_imu_residual:
        nn.init.zeros_(joint.imu_residual[4].weight)
        nn.init.zeros_(joint.imu_residual[4].bias)
    audit = {
        "checkpoint": str(checkpoint_path.resolve()),
        "source_modality": checkpoint["modality"],
        "loaded_tensors": len(mapped),
        "source_tensors": len(source),
        "zero_initialized_imu_residual": bool(zero_imu_residual),
        "uninitialized_joint_tensors": len(incompatible.missing_keys),
    }
    return audit, checkpoint


ANCHORED_TYPES = (
    TemporalRefinedMoBindStudent,
    TemporalCrossAttentionPoolStudent,
    SpatialCrossAttentionPoolStudent,
)


def candidate_parameters(model: nn.Module) -> list[nn.Parameter]:
    if isinstance(model, SpatialCrossAttentionPoolStudent):
        return model.spatial_parameters()
    if isinstance(model, (TemporalRefinedMoBindStudent, TemporalCrossAttentionPoolStudent)):
        return model.temporal_parameters()
    raise TypeError("model is not an anchored fusion candidate")


def set_stage(model: nn.Module, stage: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if isinstance(model, ANCHORED_TYPES):
        if stage not in {"A", "B"}:
            raise ValueError(stage)
        for parameter in candidate_parameters(model):
            parameter.requires_grad_(True)
        return
    for parameter in model.motion_residual.parameters():
        parameter.requires_grad_(True)
    if stage == "B":
        for parameter in visual_head_parameters(model.visual):
            parameter.requires_grad_(True)
    elif stage != "A":
        raise ValueError(stage)


def selective_anchor_loss(
    logits: torch.Tensor,
    anchor_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    anchor_correct = anchor_logits.argmax(dim=1).eq(labels)
    if not anchor_correct.any():
        return logits.sum() * 0.0
    per_sample = F.kl_div(
        F.log_softmax(logits / temperature, dim=1),
        F.softmax(anchor_logits / temperature, dim=1),
        reduction="none",
    ).sum(dim=1) * temperature**2
    return per_sample[anchor_correct].mean()


def subject_robust_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    users: list[str],
    weights: torch.Tensor,
    label_smoothing: float,
    worst_group_weight: float,
) -> torch.Tensor:
    mean_loss = F.cross_entropy(
        logits,
        labels,
        weight=weights,
        label_smoothing=label_smoothing,
    )
    if worst_group_weight <= 0:
        return mean_loss
    per_sample = F.cross_entropy(
        logits,
        labels,
        weight=weights,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    group_losses = []
    for user in sorted(set(users)):
        selected = torch.tensor(
            [value == user for value in users],
            device=logits.device,
            dtype=torch.bool,
        )
        if selected.any():
            group_losses.append(per_sample[selected].mean())
    worst = torch.stack(group_losses).amax() if group_losses else mean_loss
    amount = float(worst_group_weight)
    return (1.0 - amount) * mean_loss + amount * worst


def losses(
    output: dict[str, torch.Tensor],
    batch: dict[str, Any],
    weights: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    temperature = args.distillation_temperature
    classification = subject_robust_cross_entropy(
        output["logits"],
        batch["label"],
        batch["user_id"],
        weights,
        args.label_smoothing,
        args.worst_group_weight,
    )
    distillation = F.kl_div(
        F.log_softmax(output["logits"] / temperature, dim=1),
        F.softmax(batch["teacher_logits"] / temperature, dim=1),
        reduction="batchmean",
    ) * temperature**2
    relation = relation_loss(
        output["clip_embeddings"], batch["teacher_features"], output["clip_mask"]
    )
    motion_aux = subject_robust_cross_entropy(
        output["motion_logits"],
        batch["label"],
        batch["user_id"],
        weights,
        args.label_smoothing,
        args.worst_group_weight,
    )
    imu_teacher_distillation = output["motion_logits"].sum() * 0.0
    if args.imu_teacher_weight > 0 and "imu_teacher_logits" in batch:
        teacher_valid = batch["imu_teacher_valid"]
        if teacher_valid.any():
            imu_temperature = args.imu_teacher_temperature
            imu_teacher_distillation = F.kl_div(
                F.log_softmax(
                    output["motion_logits"][teacher_valid] / imu_temperature,
                    dim=1,
                ),
                F.softmax(
                    batch["imu_teacher_logits"][teacher_valid] / imu_temperature,
                    dim=1,
                ),
                reduction="batchmean",
            ) * imu_temperature**2
    # v2/v3 protect the complete student fusion anchor, not merely the original
    # pure-Visual cache.  Legacy student/fusion runs retain their historical target.
    anchor_logits = output.get(
        "anchor_fusion_logits",
        (
            output["unfused_visual_logits"].detach()
            if args.live_visual_anchor
            else batch["anchor_logits"]
        ),
    )
    anchor = selective_anchor_loss(
        output["logits"],
        anchor_logits,
        batch["label"],
        temperature,
    )
    joint_gate = output["logits"].sum() * 0.0
    if args.joint_gate_weight > 0 and "joint_raw_imu_gate" in output:
        with torch.no_grad():
            skeleton_loss = F.cross_entropy(
                output["joint_skeleton_logits"], batch["label"], reduction="none"
            )
            imu_loss = F.cross_entropy(
                output["joint_imu_logits"], batch["label"], reduction="none"
            )
            gate_temperature = max(float(args.joint_gate_temperature), 1e-3)
            joint_gate_target = torch.sigmoid(
                (skeleton_loss - imu_loss) / gate_temperature
            )
        predicted_gate = output["joint_raw_imu_gate"].flatten(1).mean(dim=1)
        predicted_gate = predicted_gate.float().clamp(1e-6, 1.0 - 1e-6)
        joint_gate_target = joint_gate_target.float()
        joint_gate = -(
            joint_gate_target * predicted_gate.log()
            + (1.0 - joint_gate_target) * torch.log1p(-predicted_gate)
        ).mean()
    with torch.no_grad():
        if args.reliability_target == "loss_advantage":
            visual_loss = F.cross_entropy(
                output["unfused_visual_logits"], batch["label"], reduction="none"
            )
            motion_loss = F.cross_entropy(
                output["motion_logits"], batch["label"], reduction="none"
            )
            advantage_temperature = max(
                float(args.reliability_advantage_temperature), 1e-3
            )
            reliability_target = torch.sigmoid(
                (visual_loss - motion_loss) / advantage_temperature
            )
            reliability_valid = output["motion_available"].any(dim=1)
        elif args.reliability_target == "oof_complementarity":
            if "imu_teacher_logits" not in batch:
                raise RuntimeError(
                    "OOF complementarity requires --imu-teacher-logits"
                )
            visual_correct = batch["teacher_logits"].argmax(dim=1).eq(
                batch["label"]
            )
            motion_correct = batch["imu_teacher_logits"].argmax(dim=1).eq(
                batch["label"]
            )
            reliability_valid = (
                batch["imu_teacher_valid"] & (visual_correct ^ motion_correct)
            )
        elif args.reliability_target == "oof_visual_error":
            # Historical v18 name retained for reproducibility.  These are
            # foundation VideoMAE Large OOF logits, not MC3-small OOF logits, so this
            # target must not be interpreted as a deployable MC3 error router.
            visual_correct = batch["teacher_logits"].argmax(dim=1).eq(
                batch["label"]
            )
            motion_correct = ~visual_correct
            reliability_valid = output["motion_available"].any(dim=1)
        else:
            visual_correct = output["unfused_visual_logits"].argmax(dim=1).eq(
                batch["label"]
            )
            motion_correct = output["motion_logits"].argmax(dim=1).eq(
                batch["label"]
            )
            reliability_valid = visual_correct ^ motion_correct
        if args.reliability_target != "loss_advantage":
            reliability_target = motion_correct.to(output["logits"].dtype)
    global_reliability = output["motion_global_reliability"].mean(dim=-1)
    local_reliability = output["motion_reliability"].flatten(1).mean(dim=1)
    if reliability_valid.any():
        target = reliability_target[reliability_valid].float()

        def probability_bce(probability: torch.Tensor) -> torch.Tensor:
            probability = probability[reliability_valid].float().clamp(1e-6, 1.0 - 1e-6)
            sample_weight = torch.where(
                target > 0.5,
                torch.full_like(target, args.reliability_positive_weight),
                torch.ones_like(target),
            )
            per_sample = -(
                target * probability.log()
                + (1.0 - target) * torch.log1p(-probability)
            )
            return (per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)

        reliability = 0.5 * (
            probability_bce(global_reliability)
            + probability_bce(local_reliability)
        )
    else:
        reliability = output["logits"].sum() * 0.0
    total = (
        classification
        + args.distillation_weight * distillation
        + args.relation_weight * relation
        + args.motion_aux_weight * motion_aux
        + args.selective_anchor_weight * anchor
        + args.reliability_weight * reliability
        + args.imu_teacher_weight * imu_teacher_distillation
        + args.joint_gate_weight * joint_gate
    )
    return {
        "loss": total,
        "classification": classification,
        "distillation": distillation,
        "relation": relation,
        "motion_aux": motion_aux,
        "anchor": anchor,
        "reliability": reliability,
        "imu_teacher_distillation": imu_teacher_distillation,
        "joint_gate": joint_gate,
    }


def train_stage(
    model: nn.Module,
    data: DataLoader,
    labels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    stage: str,
    epochs: int,
) -> list[dict[str, Any]]:
    if epochs <= 0:
        return []
    set_stage(model, stage)
    is_temporal_candidate = isinstance(model, ANCHORED_TYPES)
    fresh_encoder_parameters: list[nn.Parameter] = []
    encoder = model.motion_residual.encoder
    if (
        isinstance(encoder, SeparateMotionEncoder)
        and args.joint_freeze_pretrained_encoders
    ):
        for module in (encoder.skeleton_encoder, encoder.imu_encoder):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    if (
        isinstance(encoder, JointMotionEncoder)
        and args.joint_freeze_pretrained_encoders
    ):
        frozen_modules: tuple[nn.Module, ...] = (
            encoder.skeleton_encoder,
            encoder.imu_encoder,
            encoder.imu_semantic_head,
        )
        if not args.joint_train_alignment:
            frozen_modules += (
                encoder.skeleton_alignment,
                encoder.imu_alignment,
            )
        for module in frozen_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    if isinstance(encoder, SkeletonPartEncoder) and encoder.multistream_input:
        assert encoder.stream_stems is not None
        assert encoder.stream_fusion is not None
        fresh_encoder_parameters.extend(encoder.stream_stems.parameters())
        fresh_encoder_parameters.extend(encoder.stream_fusion.parameters())
        assert encoder.multistream_residual_logit is not None
        fresh_encoder_parameters.append(encoder.multistream_residual_logit)
    if isinstance(encoder, SkeletonPartEncoder) and encoder.adaptive_graph:
        fresh_encoder_parameters.extend(
            block.adaptive_adjacency
            for block in encoder.blocks
            if block.adaptive_adjacency is not None
        )
    if isinstance(encoder, (SeparateMotionEncoder, JointMotionEncoder)):
        fresh_encoder_parameters.extend(encoder.fresh_parameters())
        if stage == "A" and args.joint_stage_a_adapter_only:
            for parameter in model.motion_residual.parameters():
                parameter.requires_grad_(False)
            for parameter in fresh_encoder_parameters:
                parameter.requires_grad_(True)
            if args.joint_train_alignment:
                for module in (encoder.skeleton_alignment, encoder.imu_alignment):
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)
    fusion_learning_rate = (
        args.stage_b_fusion_learning_rate
        if stage == "B" and args.stage_b_fusion_learning_rate is not None
        else args.fusion_learning_rate
    )
    if is_temporal_candidate:
        groups = [
            {
                "params": candidate_parameters(model),
                "lr": fusion_learning_rate,
                "initial_lr": fusion_learning_rate,
            }
        ]
    else:
        fresh_encoder_ids = {id(parameter) for parameter in fresh_encoder_parameters}
        core_parameters = [
            parameter
            for parameter in encoder.parameters()
            if id(parameter) not in fresh_encoder_ids and parameter.requires_grad
        ] + [
            parameter
            for parameter in model.motion_residual.semantic_head.parameters()
            if parameter.requires_grad
        ]
        core_ids = {id(parameter) for parameter in core_parameters}
        fusion_parameters = [
            parameter
            for parameter in model.motion_residual.parameters()
            if id(parameter) not in core_ids and parameter.requires_grad
        ]
        groups = [
            {
                "params": core_parameters,
                "lr": args.encoder_learning_rate,
                "initial_lr": args.encoder_learning_rate,
            },
            {
                "params": fusion_parameters,
                "lr": fusion_learning_rate,
                "initial_lr": fusion_learning_rate,
            },
        ]
        if stage == "B":
            groups.append(
                {
                    "params": visual_head_parameters(model.visual),
                    "lr": args.visual_learning_rate,
                    "initial_lr": args.visual_learning_rate,
                }
            )
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    weights = class_weights(labels, args.class_weight_power, device)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        if is_temporal_candidate:
            # The student anchor remains deterministic and immutable.  model.train()
            # above still enables dropout in the new temporal projections.
            model.visual.eval()
            model.motion_residual.eval()
        elif stage == "A":
            model.visual.eval()
            model.motion_residual.train()
        cosine = 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / max(epochs - 1, 1)))
        for group in optimizer.param_groups:
            group["lr"] = args.minimum_learning_rate + (
                float(group["initial_lr"]) - args.minimum_learning_rate
            ) * cosine
        sums = {
            key: 0.0
            for key in (
                "loss",
                "classification",
                "distillation",
                "relation",
                "motion_aux",
                "anchor",
                "reliability",
                "imu_teacher_distillation",
                "joint_gate",
            )
        }
        samples = 0
        reliability_sum = reliability_count = 0.0
        temporal_ratio_sum = temporal_ratio_count = 0.0
        spatial_ratio_sum = spatial_ratio_count = 0.0
        started = time.perf_counter()
        for batch_index, batch in enumerate(data):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model_forward(
                    model, batch, args=args, augment_visual=True
                )
                values = losses(output, batch, weights, args)
            scaler.scale(values["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                2.0,
            )
            scaler.step(optimizer)
            scaler.update()
            count = len(batch["label"])
            samples += count
            for key in sums:
                sums[key] += float(values[key].detach()) * count
            reliability_sum += float(output["motion_reliability"].detach().sum())
            reliability_count += output["motion_reliability"].numel()
            temporal_effect = output.get(
                "temporal_effect_rms_ratio",
                output.get("temporal_residual_rms_ratio"),
            )
            if temporal_effect is not None:
                temporal_ratio_sum += float(
                    temporal_effect.detach().sum()
                )
                temporal_ratio_count += temporal_effect.numel()
            spatial_effect = output.get("spatial_clip_effect_rms_ratio")
            if spatial_effect is not None:
                spatial_ratio_sum += float(spatial_effect.detach().sum())
                spatial_ratio_count += spatial_effect.numel()
        if args.all_label_refit and not args.max_train_batches and samples != len(data.dataset):
            raise RuntimeError(
                "all-label fusion refit did not consume all 2914 rows in this epoch: "
                f"observed={samples}, expected={len(data.dataset)}"
            )
        record = {
            "stage": stage,
            "epoch": epoch,
            "encoder_learning_rate": (
                0.0 if is_temporal_candidate else optimizer.param_groups[0]["lr"]
            ),
            "fusion_learning_rate": (
                optimizer.param_groups[0]["lr"]
                if is_temporal_candidate
                else optimizer.param_groups[1]["lr"]
            ),
            "visual_learning_rate": (
                optimizer.param_groups[2]["lr"]
                if not is_temporal_candidate and len(optimizer.param_groups) > 2
                else 0.0
            ),
            **{f"train_{key}": value / max(samples, 1) for key, value in sums.items()},
            "mean_motion_reliability": reliability_sum / max(reliability_count, 1.0),
            "residual_strength": float(
                model.motion_residual.strength().detach()
            ),
            "mean_temporal_effect_rms_ratio": (
                temporal_ratio_sum / max(temporal_ratio_count, 1.0)
            ),
            "train_samples": samples,
            "seconds": time.perf_counter() - started,
        }
        if spatial_ratio_count:
            record["mean_spatial_clip_effect_rms_ratio"] = (
                spatial_ratio_sum / spatial_ratio_count
            )
        if (
            isinstance(encoder, SkeletonPartEncoder)
            and encoder.multistream_residual_logit is not None
        ):
            record["skeleton_multistream_strength"] = float(
                torch.sigmoid(encoder.multistream_residual_logit).detach()
            )
        if (
            isinstance(encoder, JointMotionEncoder)
            and encoder.last_token_reliability is not None
        ):
            record["joint_mean_imu_gate"] = float(
                encoder.last_token_reliability
            )
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    return history


def temporal_sample_audit(
    output: dict[str, torch.Tensor], index: int
) -> dict[str, float]:
    if "spatial_pool_attention" in output:
        pool = output["spatial_pool_attention"][index].float().clamp_min(1e-8)
        pool_entropy = -(pool * pool.log()).sum(dim=-1)
        modality_weight = output["spatial_modality_weight"][index].float()
        available = output["spatial_motion_available"][index].float()

        def part_entropy(key: str) -> float:
            probability = output[key][index].float().clamp_min(1e-8)
            entropy = -(probability * probability.log()).sum(dim=-1)
            return float(entropy.mean().detach().cpu())

        return {
            "mean_spatial_clip_effect_rms_ratio": float(
                output["spatial_clip_effect_rms_ratio"][index]
                .float()
                .mean()
                .detach()
                .cpu()
            ),
            "mean_spatial_correction_rms_ratio": float(
                output["spatial_correction_rms_ratio"][index]
                .float()
                .mean()
                .detach()
                .cpu()
            ),
            "mean_spatial_skeleton_weight": float(
                modality_weight[..., 0].mean().detach().cpu()
            ),
            "mean_spatial_imu_weight": float(
                modality_weight[..., 1].mean().detach().cpu()
            ),
            "spatial_skeleton_availability": float(
                available[..., 0].mean().detach().cpu()
            ),
            "spatial_imu_availability": float(
                available[..., 1].mean().detach().cpu()
            ),
            "spatial_skeleton_part_attention_entropy": part_entropy(
                "spatial_skeleton_part_attention"
            ),
            "spatial_imu_part_attention_entropy": part_entropy(
                "spatial_imu_part_attention"
            ),
            "spatial_pool_attention_entropy": float(
                pool_entropy.mean().detach().cpu()
            ),
            "spatial_pool_weight_l1_from_uniform": float(
                output["spatial_pool_weight_l1_from_uniform"][index]
                .float()
                .mean()
                .detach()
                .cpu()
            ),
            "mean_spatial_pool_logit_abs": float(
                output["spatial_pool_logit"][index]
                .float()
                .abs()
                .mean()
                .detach()
                .cpu()
            ),
        }
    if "temporal_effect_rms_ratio" in output:
        modality_weight = output["temporal_modality_weight"][index].float()
        available = output["temporal_motion_available"][index].float()

        def cross_entropy(key: str) -> float:
            probability = output[key][index].float().flatten(-2).clamp_min(1e-8)
            value = -(probability * probability.log()).sum(dim=-1)
            return float(value.mean().detach().cpu())

        pool_probability = output["temporal_pool_attention"][index].float()
        pool_probability = pool_probability.clamp_min(1e-8)
        pool_entropy = -(pool_probability * pool_probability.log()).sum(dim=-1)
        return {
            "mean_temporal_effect_rms_ratio": float(
                output["temporal_effect_rms_ratio"][index]
                .float()
                .mean()
                .detach()
                .cpu()
            ),
            "mean_temporal_skeleton_weight": float(
                modality_weight[..., 0].mean().detach().cpu()
            ),
            "mean_temporal_imu_weight": float(
                modality_weight[..., 1].mean().detach().cpu()
            ),
            "temporal_skeleton_availability": float(
                available[..., 0].mean().detach().cpu()
            ),
            "temporal_imu_availability": float(
                available[..., 1].mean().detach().cpu()
            ),
            "temporal_skeleton_cross_attention_entropy": cross_entropy(
                "temporal_skeleton_cross_attention"
            ),
            "temporal_imu_cross_attention_entropy": cross_entropy(
                "temporal_imu_cross_attention"
            ),
            "temporal_skeleton_mean_abs_offset": float(
                output["temporal_skeleton_mean_abs_offset"][index]
                .float()
                .mean()
                .detach()
                .cpu()
            ),
            "temporal_imu_mean_abs_offset": float(
                output["temporal_imu_mean_abs_offset"][index]
                .float()
                .mean()
                .detach()
                .cpu()
            ),
            "temporal_pool_attention_entropy": float(
                pool_entropy.mean().detach().cpu()
            ),
            "temporal_pool_weight_l1_from_uniform": float(
                output["temporal_pool_weight_l1_from_uniform"][index]
                .float()
                .mean()
                .detach()
                .cpu()
            ),
            "mean_temporal_pool_logit_abs": float(
                output["temporal_pool_logit"][index]
                .float()
                .abs()
                .mean()
                .detach()
                .cpu()
            ),
        }
    if "temporal_residual_rms_ratio" in output:

        def entropy(key: str) -> float:
            probability = output[key][index].float().clamp_min(1e-8)
            value = -(probability * probability.log()).sum(dim=-1)
            return float(value.mean().detach().cpu())

        modality_weight = output["temporal_modality_weight"][index].float()
        available = output["temporal_motion_available"][index].float()
        return {
            "mean_temporal_residual_rms_ratio": float(
                output["temporal_residual_rms_ratio"][index]
                .float()
                .mean()
                .detach()
                .cpu()
            ),
            "mean_temporal_residual_rms": float(
                output["temporal_residual_rms"][index]
                .float()
                .mean()
                .detach()
                .cpu()
            ),
            "mean_temporal_skeleton_weight": float(
                modality_weight[..., 0].mean().detach().cpu()
            ),
            "mean_temporal_imu_weight": float(
                modality_weight[..., 1].mean().detach().cpu()
            ),
            "temporal_skeleton_availability": float(
                available[..., 0].mean().detach().cpu()
            ),
            "temporal_imu_availability": float(
                available[..., 1].mean().detach().cpu()
            ),
            "temporal_skeleton_part_attention_entropy": entropy(
                "temporal_skeleton_part_attention"
            ),
            "temporal_imu_part_attention_entropy": entropy(
                "temporal_imu_part_attention"
            ),
            "temporal_skeleton_time_attention_entropy": entropy(
                "temporal_skeleton_time_attention"
            ),
            "temporal_imu_time_attention_entropy": entropy(
                "temporal_imu_time_attention"
            ),
        }
    return {}


def evaluate(
    model: UnifiedMoBindStudent,
    data: DataLoader,
    device: torch.device,
    max_batches: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    model.eval()
    rows = []
    logits_rows = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(data):
            if max_batches and batch_index >= max_batches:
                break
            batch = to_device(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model_forward(model, batch)
            logits = output["logits"].float().cpu()
            probability = torch.softmax(logits, dim=1)
            motion_probability = torch.softmax(
                output["motion_logits"].float(), dim=1
            ).cpu()
            reliability = output["motion_reliability"].float().flatten(1).mean(
                dim=1
            ).cpu()
            attention = output["motion_part_attention"].float().clamp_min(1e-8)
            attention_entropy = -(attention * attention.log()).sum(dim=-1)
            attention_entropy = attention_entropy.mean(dim=(1, 2, 3)).cpu()
            labels = batch["label"].cpu()
            logits_rows.append(logits.numpy())
            for index, sample_id in enumerate(batch["sample_id"]):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "user_id": batch["user_id"][index],
                        "label": int(labels[index]),
                        "prediction": int(probability[index].argmax()),
                        "confidence": float(probability[index].max()),
                        "motion_prediction": int(motion_probability[index].argmax()),
                        "motion_confidence": float(motion_probability[index].max()),
                        "mean_motion_reliability": float(reliability[index]),
                        "mean_part_attention_entropy": float(attention_entropy[index]),
                        **temporal_sample_audit(output, index),
                    }
                )
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    predictions = np.asarray([row["prediction"] for row in rows], dtype=np.int64)
    return (
        metric_dict(labels, predictions, [row["user_id"] for row in rows]),
        rows,
        np.concatenate(logits_rows, axis=0),
    )


def evaluate_fused_and_visual(
    model: UnifiedMoBindStudent,
    data: DataLoader,
    device: torch.device,
    max_batches: int,
    live_visual_anchor: bool = False,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
]:
    """Evaluate fused logits and the immutable Visual-checkpoint anchor together."""
    model.eval()
    fused_rows: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []
    fused_logits_rows = []
    visual_logits_rows = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(data):
            if max_batches and batch_index >= max_batches:
                break
            batch = to_device(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model_forward(model, batch)
            fused_logits = output["logits"].float().cpu()
            # Stage B may update the Visual temporal head.  The cached anchor was
            # produced before multimodal training by the freshly refit Visual
            # checkpoint, so it is the strict pure-Visual baseline.
            visual_logits = (
                output["unfused_visual_logits"]
                if live_visual_anchor
                else batch["anchor_logits"]
            ).float().cpu()
            fused_probability = torch.softmax(fused_logits, dim=1)
            visual_probability = torch.softmax(visual_logits, dim=1)
            labels = batch["label"].cpu()
            fused_logits_rows.append(fused_logits.numpy())
            visual_logits_rows.append(visual_logits.numpy())
            for index, sample_id in enumerate(batch["sample_id"]):
                common = {
                    "sample_id": sample_id,
                    "user_id": batch["user_id"][index],
                    "label": int(labels[index]),
                }
                fused_rows.append(
                    {
                        **common,
                        "prediction": int(fused_probability[index].argmax()),
                        "confidence": float(fused_probability[index].max()),
                        **temporal_sample_audit(output, index),
                    }
                )
                visual_rows.append(
                    {
                        **common,
                        "prediction": int(visual_probability[index].argmax()),
                        "confidence": float(visual_probability[index].max()),
                    }
                )
    labels = np.asarray([row["label"] for row in fused_rows], dtype=np.int64)
    users = [row["user_id"] for row in fused_rows]
    fused_predictions = np.asarray(
        [row["prediction"] for row in fused_rows], dtype=np.int64
    )
    visual_predictions = np.asarray(
        [row["prediction"] for row in visual_rows], dtype=np.int64
    )
    return (
        metric_dict(labels, fused_predictions, users),
        metric_dict(labels, visual_predictions, users),
        fused_rows,
        visual_rows,
        np.concatenate(fused_logits_rows, axis=0),
        np.concatenate(visual_logits_rows, axis=0),
    )


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    subject_holdout_mode = bool(args.subject_holdout_users)
    all_label_mode = bool(args.all_label_refit)
    joint_only_options = (
        args.initial_fusion_checkpoint,
        args.joint_train_alignment,
        args.joint_zero_initialize_imu_residual,
        args.joint_stage_a_adapter_only,
    )
    if args.modality != "joint" and any(joint_only_options):
        raise ValueError("joint initialization/training options require --modality joint")
    if bool(args.train_users) and not subject_holdout_mode:
        raise ValueError("--train-users requires --subject-holdout-users")
    if args.temporal_anchor_checkpoint and args.fusion_position not in {
        "temporal_refined",
        "temporal_v3",
        "spatial_v4",
    }:
        raise ValueError(
            "--temporal-anchor-checkpoint requires temporal_refined, temporal_v3 or spatial_v4"
        )
    if args.fusion_position in {"temporal_refined", "temporal_v3", "spatial_v4"}:
        version = args.fusion_position.split("_")[-1]
        if args.modality != "separate":
            raise ValueError(f"{version} requires --modality separate")
        if not args.temporal_anchor_checkpoint:
            raise ValueError(
                f"{version} requires --temporal-anchor-checkpoint"
            )
        if args.initial_fusion_checkpoint:
            raise ValueError(
                f"{version} cannot also use joint anchor initialization"
            )
    if args.modality != "separate" and args.separate_modality_dropout > 0.0:
        raise ValueError("--separate-modality-dropout requires --modality separate")
    if args.final_refit and all_label_mode:
        raise ValueError("--final-refit and --all-label-refit are mutually exclusive")
    if (args.final_refit or all_label_mode) and args.initial_fusion_checkpoint:
        raise ValueError("terminal refit must not load a proxy fusion checkpoint")
    if subject_holdout_mode and (args.final_refit or all_label_mode):
        raise ValueError("subject holdout mode cannot be combined with --final-refit")
    if args.smoke:
        args.stage_a_epochs = min(args.stage_a_epochs, 1)
        args.stage_b_epochs = min(args.stage_b_epochs, 1)
        args.max_train_batches = args.max_train_batches or 2
        args.max_eval_batches = args.max_eval_batches or 2
    seed_all(args.seed)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    teacher = load_npz(args.teacher_logits)
    split = split_universe(teacher, outer_fold=0, seed=args.seed)
    proxy_train_indices = np.asarray(split["inner_train"], dtype=np.int64)
    proxy_indices = np.asarray(split["outer_held"], dtype=np.int64)
    forbidden_indices = np.asarray(split["inner_dev"], dtype=np.int64)
    if (len(proxy_train_indices), len(proxy_indices), len(forbidden_indices)) != (
        1497,
        973,
        444,
    ):
        raise RuntimeError("student fixed proxy counts changed")
    if subject_holdout_mode:
        requested_users = set(map(str, args.subject_holdout_users))
        all_users = np.asarray(split["users"]).astype(str)
        observed_users = set(
            all_users[np.isin(all_users, sorted(requested_users))].tolist()
        )
        if observed_users != requested_users:
            raise ValueError(
                f"Requested holdout users {sorted(requested_users)}, observed "
                f"{sorted(observed_users)}"
            )
        if args.train_users:
            requested_train_users = set(map(str, args.train_users))
            if requested_train_users & requested_users:
                raise ValueError("training and holdout user allowlists overlap")
            observed_train_users = set(
                all_users[np.isin(all_users, sorted(requested_train_users))].tolist()
            )
            if observed_train_users != requested_train_users:
                raise ValueError(
                    f"Requested train users {sorted(requested_train_users)}, observed "
                    f"{sorted(observed_train_users)}"
                )
            train_indices = np.flatnonzero(
                np.isin(all_users, sorted(requested_train_users))
            )
        else:
            train_indices = np.flatnonzero(
                ~np.isin(all_users, sorted(requested_users))
            )
        evaluation_indices = np.flatnonzero(
            np.isin(all_users, sorted(requested_users))
        )
        if set(all_users[train_indices].tolist()) & requested_users:
            raise RuntimeError("subject holdout isolation failed")
    elif all_label_mode:
        train_indices = np.arange(len(split["sample_ids"]), dtype=np.int64)
        evaluation_indices = np.empty(0, dtype=np.int64)
        if len(train_indices) != 2914:
            raise RuntimeError("terminal fusion refit requires exactly 2914 labels")
    elif args.final_refit:
        train_indices = np.sort(
            np.concatenate((proxy_train_indices, proxy_indices))
        )
        evaluation_indices = forbidden_indices
    else:
        train_indices = proxy_train_indices
        evaluation_indices = proxy_indices
    full = CachedSequenceMotionDataset(
        args.sequence_cache,
        args.motion_cache,
        args.pixel_cache,
        args.teacher_features,
        args.teacher_logits,
        imu_teacher_logits=args.imu_teacher_logits,
        imu_event_features=args.imu_event_features,
        compact_sequence_cache=args.compact_sequence_cache,
    )
    if args.fusion_position == "spatial_v4":
        if not full.uses_spatial_regions or full.compact_backbone_sequence is None:
            raise ValueError(
                "v4 requires --sequence-cache with spatial regions and "
                "--compact-sequence-cache with the exact paired student sequence"
            )
    elif full.uses_spatial_regions:
        raise ValueError("this controlled screen requires the proven compact sequence")
    training = make_dataset(
        args, full, split["sample_ids"][train_indices], temporal_augment=True
    )
    evaluation = (
        None
        if all_label_mode
        else make_dataset(
            args,
            full,
            split["sample_ids"][evaluation_indices],
            temporal_augment=False,
        )
    )
    model, visual_config, pretrain_config = build_model(args)
    anchor_initialization = None
    anchor_checkpoint = None
    if args.initial_fusion_checkpoint:
        anchor_initialization, anchor_checkpoint = (
            initialize_joint_from_skeleton_fusion(
                model,
                args.initial_fusion_checkpoint,
                args.joint_zero_initialize_imu_residual,
            )
        )
    if args.temporal_anchor_checkpoint:
        assert isinstance(
            model,
            ANCHORED_TYPES,
        )
        anchor_initialization, anchor_checkpoint = initialize_temporal_from_visual(
            model, args.temporal_anchor_checkpoint
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    initial_metrics = initial_predictions = initial_logits = None
    baseline_metrics = baseline_predictions = baseline_logits = None
    temporal_anchor_mode = isinstance(model, ANCHORED_TYPES)
    should_measure_initial = (
        not args.final_refit
        and not all_label_mode
        and (not subject_holdout_mode or temporal_anchor_mode)
    )
    if should_measure_initial:
        assert evaluation is not None
        initial_metrics, initial_predictions, initial_logits = evaluate(
            model, loader(evaluation, args, False), device, args.max_eval_batches
        )
        if temporal_anchor_mode and anchor_checkpoint is not None and not args.max_eval_batches:
            anchor_metrics = (
                anchor_checkpoint["subject_holdout_metrics"]
                if subject_holdout_mode
                else anchor_checkpoint["proxy_metrics"]
            )
            if initial_metrics["correct"] != anchor_metrics["correct"]:
                raise RuntimeError(
                    "zero-residual temporal initialization did not preserve student: "
                    f"anchor: initial={initial_metrics['correct']}, "
                    f"anchor={anchor_metrics['correct']}"
                )
            prediction_stem = (
                "subject_holdout" if subject_holdout_mode else "proxy_validation"
            )
            anchor_logits = np.load(
                args.temporal_anchor_checkpoint.resolve().parent
                / f"{prediction_stem}_logits.npy"
            )
            anchor_initialization["maximum_initial_logit_error"] = float(
                np.max(np.abs(initial_logits - anchor_logits))
            )
            if anchor_initialization["maximum_initial_logit_error"] > 1e-4:
                raise RuntimeError(
                    "temporal initialization is not an exact student anchor: "
                    f"max_logit_error={anchor_initialization['maximum_initial_logit_error']}"
                )
        elif anchor_checkpoint is not None and not args.max_eval_batches:
            anchor_metrics = anchor_checkpoint["proxy_metrics"]
            if initial_metrics["correct"] != anchor_metrics["correct"]:
                raise RuntimeError(
                    "zero-residual joint initialization did not preserve the Skeleton "
                    f"anchor: initial={initial_metrics['correct']}, "
                    f"anchor={anchor_metrics['correct']}"
                )
            anchor_logits = np.load(
                args.initial_fusion_checkpoint.resolve().parent
                / "proxy_validation_logits.npy"
            )
            anchor_initialization["maximum_initial_logit_error"] = float(
                np.max(np.abs(initial_logits - anchor_logits))
            )
            if anchor_initialization["maximum_initial_logit_error"] > 1e-4:
                raise RuntimeError(
                    "joint initialization is not an exact Skeleton anchor: "
                    f"max_logit_error={anchor_initialization['maximum_initial_logit_error']}"
                )
        if anchor_checkpoint is not None and not args.max_eval_batches:
            checkpoint_path = (
                args.temporal_anchor_checkpoint
                if temporal_anchor_mode
                else args.initial_fusion_checkpoint
            )
            assert checkpoint_path is not None
            anchor_dir = checkpoint_path.resolve().parent
            baseline_metrics = anchor_checkpoint["cached_visual_baseline_metrics"]
            baseline_predictions = read_rows(
                anchor_dir / "cached_visual_baseline_predictions.csv"
            )
            baseline_logits = np.load(anchor_dir / "cached_visual_baseline_logits.npy")
        else:
            baseline_metrics = initial_metrics
            baseline_predictions = initial_predictions
            baseline_logits = initial_logits
    history = []
    history.extend(
        train_stage(
            model,
            loader(training, args, True),
            split["labels"][train_indices],
            args,
            device,
            "A",
            args.stage_a_epochs,
        )
    )
    history.extend(
        train_stage(
            model,
            loader(training, args, True),
            split["labels"][train_indices],
            args,
            device,
            "B",
            args.stage_b_epochs,
        )
    )
    if all_label_mode:
        metrics = None
        predictions: list[dict[str, Any]] = []
        logits = np.empty((0, 40), dtype=np.float32)
    elif args.final_refit or subject_holdout_mode:
        assert evaluation is not None
        (
            metrics,
            baseline_metrics,
            predictions,
            baseline_predictions,
            logits,
            baseline_logits,
        ) = evaluate_fused_and_visual(
            model,
            loader(evaluation, args, False),
            device,
            args.max_eval_batches,
            live_visual_anchor=args.live_visual_anchor,
        )
        if not temporal_anchor_mode:
            initial_metrics = baseline_metrics
            initial_predictions = baseline_predictions
            initial_logits = baseline_logits
    else:
        metrics, predictions, logits = evaluate(
            model, loader(evaluation, args, False), device, args.max_eval_batches
        )
    counterfactual_audit: dict[str, Any] | None = None
    if temporal_anchor_mode and not all_label_mode:
        assert evaluation is not None
        assert initial_metrics is not None and initial_logits is not None
        counterfactual_audit = {}
        for ablation in ("zero", "reverse_time", "sample_roll"):
            if isinstance(model, SpatialCrossAttentionPoolStudent):
                model.set_spatial_ablation(ablation)
            else:
                model.set_temporal_ablation(ablation)
            ablation_metrics, ablation_rows, ablation_logits = evaluate(
                model,
                loader(evaluation, args, False),
                device,
                args.max_eval_batches,
            )
            maximum_anchor_error = (
                float(np.max(np.abs(ablation_logits - initial_logits)))
                if ablation == "zero"
                else None
            )
            if (
                ablation == "zero"
                and not args.max_eval_batches
                and maximum_anchor_error > 1e-4
            ):
                raise RuntimeError(
                    "trained fusion temporal zero ablation does not recover its student anchor: "
                    f"max_logit_error={maximum_anchor_error}"
                )
            counterfactual_audit[ablation] = {
                "metrics": ablation_metrics,
                "delta_correct_vs_p86_anchor": int(
                    ablation_metrics["correct"] - initial_metrics["correct"]
                ),
                "maximum_logit_error_vs_p86_anchor": maximum_anchor_error,
            }
            write_rows(
                output / f"counterfactual_{ablation}_predictions.csv",
                ablation_rows,
            )
            np.save(
                output / f"counterfactual_{ablation}_logits.npy",
                ablation_logits,
            )
        if isinstance(model, SpatialCrossAttentionPoolStudent):
            model.set_spatial_ablation("none")
        else:
            model.set_temporal_ablation("none")
    write_rows(output / "training_history.csv", history)
    if all_label_mode:
        baseline_metrics = initial_metrics = delta = delta_vs_initial = None
    else:
        assert metrics is not None
        assert baseline_metrics is not None
        assert baseline_predictions is not None
        assert baseline_logits is not None
        assert initial_metrics is not None
        assert initial_predictions is not None
        assert initial_logits is not None
        write_rows(output / "initial_fusion_predictions.csv", initial_predictions)
        write_rows(output / "cached_visual_baseline_predictions.csv", baseline_predictions)
        prediction_stem = (
            "subject_holdout"
            if subject_holdout_mode
            else ("final_validation" if args.final_refit else "proxy_validation")
        )
        write_rows(output / f"{prediction_stem}_predictions.csv", predictions)
        np.save(output / "initial_fusion_logits.npy", initial_logits)
        np.save(output / "cached_visual_baseline_logits.npy", baseline_logits)
        np.save(output / f"{prediction_stem}_logits.npy", logits)
        delta = {
            "correct": int(metrics["correct"] - baseline_metrics["correct"]),
            "accuracy_pp": 100.0 * (metrics["accuracy"] - baseline_metrics["accuracy"]),
            "macro_f1_pp": 100.0 * (metrics["macro_f1"] - baseline_metrics["macro_f1"]),
            "worst_subject_accuracy_pp": 100.0
            * (metrics["worst_subject_accuracy"] - baseline_metrics["worst_subject_accuracy"]),
        }
        delta_vs_initial = {
            "correct": int(metrics["correct"] - initial_metrics["correct"]),
            "accuracy_pp": 100.0 * (metrics["accuracy"] - initial_metrics["accuracy"]),
            "macro_f1_pp": 100.0 * (metrics["macro_f1"] - initial_metrics["macro_f1"]),
            "worst_subject_accuracy_pp": 100.0
            * (
                metrics["worst_subject_accuracy"]
                - initial_metrics["worst_subject_accuracy"]
            ),
        }
    parameters = sum(parameter.numel() for parameter in model.parameters())
    temporal_stage = (
        "spatial_cross_attention_pool_v4"
        if isinstance(model, SpatialCrossAttentionPoolStudent)
        else (
            "temporal_cross_attention_pool_v3"
            if isinstance(model, TemporalCrossAttentionPoolStudent)
            else "temporal_mobind_refined"
        )
    )
    stage_name = (
        f"{temporal_stage}_subject_holdout"
        if temporal_anchor_mode and subject_holdout_mode
        else (
        f"{temporal_stage}_proxy"
        if temporal_anchor_mode
        else (
        "S_mobind_fusion_subject_holdout"
        if subject_holdout_mode
        else (
            "S_mobind_fusion_all2914_refit"
            if all_label_mode
            else (
                "mobind_lite_fusion_final2470"
                if args.final_refit
                else "mobind_lite_fusion_proxy"
            )
        )
        )
        )
    )
    all_subjects = set(np.asarray(split["users"]).astype(str).tolist())
    training_subjects = sorted(set(split["users"][train_indices].tolist()))
    holdout_subjects = sorted(set(split["users"][evaluation_indices].tolist()))
    excluded_subjects = sorted(
        all_subjects - set(training_subjects) - set(holdout_subjects)
    )
    checkpoint = {
        "stage": stage_name,
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "visual_config": visual_config,
        "pretrain_config": pretrain_config,
        "modality": args.modality,
        "fusion_position": args.fusion_position,
        "temporal_radius": args.temporal_radius,
        "temporal_residual_budget": args.temporal_residual_budget,
        "temporal_attention_logit_limit": args.temporal_attention_logit_limit,
        "spatial_grid": args.spatial_grid,
        "spatial_attention_logit_limit": args.spatial_attention_logit_limit,
        "proxy_metrics": (
            metrics
            if not args.final_refit and not subject_holdout_mode and not all_label_mode
            else None
        ),
        "final_validation_metrics": metrics if args.final_refit else None,
        "subject_holdout_metrics": metrics if subject_holdout_mode else None,
        "training_subjects": training_subjects,
        "holdout_subjects": holdout_subjects,
        "excluded_subjects": excluded_subjects,
        "cached_visual_baseline_metrics": baseline_metrics,
        "delta_vs_cached_visual": delta,
        "initial_fusion_metrics": initial_metrics,
        "delta_vs_initial_fusion": delta_vs_initial,
        "anchor_initialization": anchor_initialization,
        "counterfactual_audit": counterfactual_audit,
    }
    torch.save(checkpoint, output / "unified_student.pt")
    summary = {
        "stage": stage_name,
        "status": (
            "smoke"
            if args.smoke
            else (
                "formal_subject_holdout"
                if subject_holdout_mode
                else (
                    "formal_terminal_refit"
                    if all_label_mode
                    else ("formal_final" if args.final_refit else "formal_proxy")
                )
            )
        ),
        "modality": args.modality,
        "fusion_position": args.fusion_position,
        "temporal_radius": args.temporal_radius,
        "protocol": (
            f"{args.fusion_position} loads the exact paired student separate/clip "
            "anchor trained on the same explicit user allowlist. The complete student "
            "path is frozen; only zero-initialized candidate parameters are updated. "
            "Evaluation users and excluded H3 users never enter training or epoch "
            "selection."
            if temporal_anchor_mode
            else (
            "Train fusion from the freshly subject-disjoint Visual/MoBind checkpoints "
            "on all non-holdout labels, then evaluate the explicit S pseudo-Test "
            "once after fixed stages. No holdout label participates in initialization, "
            "losses or epoch selection."
            if subject_holdout_mode
            else (
            "Terminal refit on all 2914 true-labeled Train rows using frozen stage "
            "budgets. No training row is reused as validation; no early stop, Test input "
            "or checkpoint selection is used."
            if all_label_mode
            else
            "Train all 15 non-permanent subjects with the frozen proxy recipe, then "
            "evaluate fused and pure Visual logits together on fixed user1/user2/user21 "
            "exactly once. No validation-driven selection is performed."
            if args.final_refit
            else "Train nine candidate-train subjects and evaluate six disjoint proxy "
            "subjects once after fixed-stage training. Permanent user1/user2/user21 "
            "validation remains untouched."
            )
            )
        ),
        "counts": {
            "train": len(training),
            (
                "subject_holdout"
                if subject_holdout_mode
                else (
                    "validation"
                    if all_label_mode
                    else ("final_validation" if args.final_refit else "proxy")
                )
            ): 0 if evaluation is None else len(evaluation),
            "permanent_untouched": (
                0 if args.final_refit or subject_holdout_mode or all_label_mode else 444
            ),
        },
        "training_subjects": training_subjects,
        "holdout_subjects": holdout_subjects,
        "excluded_subjects": excluded_subjects,
        "proxy_metrics": (
            metrics
            if not args.final_refit and not subject_holdout_mode and not all_label_mode
            else None
        ),
        "final_validation_metrics": metrics if args.final_refit else None,
        "subject_holdout_metrics": metrics if subject_holdout_mode else None,
        "cached_visual_baseline_metrics": baseline_metrics,
        "delta_vs_cached_visual": delta,
        "initial_fusion_metrics": initial_metrics,
        "delta_vs_initial_fusion": delta_vs_initial,
        "anchor_initialization": anchor_initialization,
        "counterfactual_audit": counterfactual_audit,
        "accuracy_gate_pp": args.accuracy_gate_pp,
        "passes_accuracy_gate": (
            None
            if (delta_vs_initial if temporal_anchor_mode else delta) is None
            else (delta_vs_initial if temporal_anchor_mode else delta)["accuracy_pp"]
            >= args.accuracy_gate_pp
        ),
        "parameters": parameters,
        "fp32_mib": parameters * 4 / 1024**2,
        "residual_strength": float(
            model.motion_residual.strength().detach()
        ),
        "large_videomae_required_at_inference": False,
        "cached_sequence_is_final_model": False,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    encoder = model.motion_residual.encoder
    if (
        isinstance(encoder, SkeletonPartEncoder)
        and encoder.multistream_residual_logit is not None
    ):
        summary["skeleton_multistream_strength"] = float(
            torch.sigmoid(encoder.multistream_residual_logit).detach()
        )
    if (
        isinstance(encoder, JointMotionEncoder)
        and encoder.last_token_reliability is not None
    ):
        summary["joint_mean_imu_gate"] = float(encoder.last_token_reliability)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
