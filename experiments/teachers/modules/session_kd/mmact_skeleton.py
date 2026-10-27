from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import numpy as np
from session_kd.mmact_backbones.constants import H36M_PARENTS

def coco_to_projected_h36m(keypoints, threshold=0.25):
    p = np.asarray(keypoints, np.float32)
    if p.ndim != 3 or p.shape[1:] != (17, 3):
        raise ValueError('expected [T,17,(x,y,confidence)] in image coordinates')
    out = np.zeros_like(p)
    direct = {1: 12, 2: 14, 3: 16, 4: 11, 5: 13, 6: 15, 10: 0, 11: 5, 12: 7, 13: 9, 14: 6, 15: 8, 16: 10}
    for target, source in direct.items():
        out[:, target] = p[:, source]
    for target, a, b in ((0, 11, 12), (8, 5, 6)):
        out[:, target, :2] = (p[:, a, :2] + p[:, b, :2]) / 2
        out[:, target, 2] = np.minimum(p[:, a, 2], p[:, b, 2])
    for target, a, b in ((7, 0, 8), (9, 8, 10)):
        out[:, target, :2] = (out[:, a, :2] + out[:, b, :2]) / 2
        out[:, target, 2] = np.minimum(out[:, a, 2], out[:, b, 2])
    valid = np.isfinite(out).all(-1) & (out[..., 2] >= threshold)
    out = np.nan_to_num(out)
    out[~valid] = 0
    return (out, valid)

def _derivative(values, valid, times):
    result = np.zeros_like(values)
    mask = np.zeros_like(valid)
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        if dt <= 0:
            continue
        mask[i] = valid[i] & valid[i - 1]
        result[i] = np.where(mask[i, :, None], (values[i] - values[i - 1]) / dt, 0)
    return (result, mask)

def projected_skeleton_features(keypoints, timestamps_seconds, threshold=0.25):
    """Encode one window; call separately for early/late to avoid false derivatives."""
    p, valid = coco_to_projected_h36m(keypoints, threshold)
    times = np.asarray(timestamps_seconds, np.float64)
    if times.shape != (len(p),) or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError('timestamps must be finite nondecreasing seconds')
    xy = p[..., :2].copy()
    xy[..., 1] *= -1
    scale = np.linalg.norm(xy[:, 8] - xy[:, 0], axis=-1)
    root_ok = valid[:, 0] & valid[:, 8] & (scale > 1e-06)
    valid &= root_ok[:, None]
    xy = (xy - xy[:, 0:1]) / np.maximum(scale[:, None, None], 1e-06)
    xy[~valid] = 0
    parents = np.asarray(H36M_PARENTS)
    bone = xy - xy[:, parents]
    bone_valid = valid & valid[:, parents]
    velocity, velocity_valid = _derivative(xy, valid, times)
    acceleration, acceleration_valid = _derivative(velocity, velocity_valid, times)
    features = np.zeros((len(p), 17, 13), np.float32)
    masks = np.zeros_like(features, dtype=bool)
    for offset, values, present in ((0, xy, valid), (3, bone, bone_valid), (6, velocity, velocity_valid), (9, acceleration, acceleration_valid)):
        features[..., offset:offset + 2] = np.where(present[..., None], values, 0)
        masks[..., offset:offset + 2] = present[..., None]
    features[..., 12] = np.where(valid, np.clip(p[..., 2], 0, 1), 0)
    masks[..., 12] = valid
    relations = np.zeros((len(p), 18), np.float32)
    relation_mask = np.zeros_like(relations, dtype=bool)
    for slot, a, b in ((0, 13, 10), (1, 16, 10), (2, 13, 16), (3, 13, 0), (4, 16, 0), (9, 11, 14), (10, 4, 1), (11, 0, 10)):
        relation_mask[:, slot] = valid[:, a] & valid[:, b]
        relations[:, slot] = np.linalg.norm(xy[:, a] - xy[:, b], axis=-1)
    for slot, a, b, c in ((5, 11, 12, 13), (6, 14, 15, 16), (7, 4, 5, 6), (8, 1, 2, 3)):
        u, v = (xy[:, a] - xy[:, b], xy[:, c] - xy[:, b])
        norm = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)
        relation_mask[:, slot] = valid[:, a] & valid[:, b] & valid[:, c] & (norm > 1e-06)
        relations[:, slot] = np.clip((u * v).sum(-1) / np.maximum(norm, 1e-06), -1, 1)
    torso = xy[:, 10] - xy[:, 0]
    norm = np.linalg.norm(torso, axis=-1)
    relations[:, 12:14] = torso / np.maximum(norm[:, None], 1e-06)
    relation_mask[:, 12:14] = (valid[:, 10] & valid[:, 0] & (norm > 1e-06))[:, None]
    for slot, joint in ((15, 13), (16, 16)):
        relations[:, slot] = np.linalg.norm(velocity[:, joint], axis=-1)
        relation_mask[:, slot] = velocity_valid[:, joint]
    relations[:, 17] = np.linalg.norm(velocity, axis=-1).sum(1) / np.maximum(velocity_valid.sum(1), 1)
    relation_mask[:, 17] = velocity_valid.any(1)
    relations[~relation_mask] = 0
    return {'skeleton_features': features, 'skeleton_feature_mask': masks, 'skeleton_joint_mask': valid, 'skeleton_relations': relations, 'skeleton_relation_mask': relation_mask, 'skeleton_frame_quality': features[..., 12].sum(1) / np.maximum(valid.sum(1), 1)}
