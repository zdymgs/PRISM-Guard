from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
from collections import OrderedDict
from pathlib import Path
import cv2
import numpy as np
from torch import nn
from session_kd.mmact_backbones.constants import H36M_PARENTS
from session_kd.mmact_backbones.mobind import MoBindLite, SkeletonPartEncoder, SeparateMotionEncoder, MotionSemanticHead, MoBindMotionResidual, UnifiedMoBindStudent
from session_kd.mmact_visual import MMACTVisualStudent
from session_kd.mmact_imu import MMACTIMUEncoder, NativeIMUSchema
EARLY_LATE = np.stack((np.rint(np.linspace(0, 0.7 * 149, 16)), np.rint(np.linspace(0.3 * 149, 149, 16)))).astype(int)
REGIONS = ('scene', 'person', 'body_motion')
DEVICE_MAPPING = {'left_watch': 1, 'right_watch': 2, 'right_pocket_phone': 4, 'left_earbud': 0}

class MMFitIMUEncoder(MMACTIMUEncoder):
    """Native raw 6-channel stems, channel masks, temporal/device/statistic paths.

    No compensated acceleration/quaternions are invented. Ear/head evidence is
    put in the contextual torso slot, explicitly not a measured torso sensor.
    Left leg remains absent; eight streams remain four actual devices.
    """

    def __init__(self):
        super().__init__(NativeIMUSchema.create(), width=96, dropout=0.12)

def build_student():
    v = MMACTVisualStudent(classes=11, kinetics_pretrained=False, region_names=REGIONS)
    s = SkeletonPartEncoder(width=96, dropout=0.12)
    i = MMFitIMUEncoder()
    encoder = SeparateMotionEncoder(s, i, width=96, modality_dropout=0.0)
    head = MotionSemanticHead(width=96, classes=11, dropout=0.12)
    projection = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, 1024))
    residual = MoBindMotionResidual('separate', encoder, head, projection, visual_width=512, motion_width=96, dropout=0.12)
    return UnifiedMoBindStudent(v, residual)

def skeleton_3d_features(xyz, times):
    """Measured H36M-17 xyz, bone, velocity, acceleration, quality: all 13 slots.

    Window-relative root subtraction and measured hip/spine scale have no fitted
    label ancestry. Missingness is finite observations and nondegenerate torso;
    zero root alone is valid. Derivatives restart at each early/late window.
    """
    x = np.asarray(xyz, np.float32).copy()
    times = np.asarray(times, float)
    if x.shape != (16, 17, 3) or times.shape != (16,) or np.any(np.diff(times) <= 0):
        raise ValueError('3D/time layout')
    valid = np.isfinite(x).all(-1)
    scale = np.linalg.norm(x[:, 8] - x[:, 0], axis=-1)
    valid &= (valid[:, 0] & valid[:, 8] & (scale > 1e-06))[:, None]
    x = (np.nan_to_num(x) - np.nan_to_num(x[:, 0:1])) / np.maximum(scale[:, None, None], 1e-06)
    x[~valid] = 0
    parent = np.asarray(H36M_PARENTS)
    bone = x - x[:, parent]
    bm = valid & valid[:, parent]

    def derivative(value, mask):
        z = np.zeros_like(value)
        m = np.zeros_like(mask)
        m[1:] = mask[1:] & mask[:-1]
        z[1:] = (value[1:] - value[:-1]) / np.diff(times)[:, None, None]
        z[~m] = 0
        return (z, m)
    vel, vm = derivative(x, valid)
    acc, am = derivative(vel, vm)
    f = np.zeros((16, 17, 13), np.float32)
    fm = np.zeros_like(f, bool)
    for off, val, mask in ((0, x, valid), (3, bone, bm), (6, vel, vm), (9, acc, am)):
        f[..., off:off + 3] = np.where(mask[..., None], val, 0)
        fm[..., off:off + 3] = mask[..., None]
    f[..., 12] = valid
    fm[..., 12] = valid
    rel = np.zeros((16, 18), np.float32)
    rm = np.zeros_like(rel, bool)
    for slot, u, v in ((0, 13, 10), (1, 16, 10), (2, 13, 16), (3, 13, 0), (4, 16, 0), (9, 11, 14), (10, 4, 1), (11, 0, 10)):
        rm[:, slot] = valid[:, u] & valid[:, v]
        rel[:, slot] = np.linalg.norm(x[:, u] - x[:, v], axis=-1)
    for slot, u, v, w in ((5, 11, 12, 13), (6, 14, 15, 16), (7, 4, 5, 6), (8, 1, 2, 3)):
        a, b = (x[:, u] - x[:, v], x[:, w] - x[:, v])
        den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
        rm[:, slot] = valid[:, u] & valid[:, v] & valid[:, w] & (den > 1e-06)
        rel[:, slot] = np.clip((a * b).sum(-1) / np.maximum(den, 1e-06), -1, 1)
    torso = x[:, 10] - x[:, 0]
    den = np.linalg.norm(torso, axis=-1)
    rel[:, 12:15] = torso / np.maximum(den[:, None], 1e-06)
    rm[:, 12:15] = (valid[:, 10] & valid[:, 0] & (den > 1e-06))[:, None]
    for slot, j in ((15, 13), (16, 16)):
        rel[:, slot] = np.linalg.norm(vel[:, j], axis=-1)
        rm[:, slot] = vm[:, j]
    rel[:, 17] = (np.linalg.norm(vel, axis=-1) * vm).sum(1) / np.maximum(vm.sum(1), 1)
    rm[:, 17] = vm.any(1)
    rel[~rm] = 0
    return {'skeleton_features': f, 'skeleton_feature_mask': fm, 'skeleton_joint_mask': valid, 'skeleton_relations': rel, 'skeleton_relation_mask': rm, 'skeleton_frame_quality': valid.mean(1).astype(np.float32)}

def imumodules_grid(raw, presence, mean, std):
    """Existing audited 24x250 streams -> four devices in five anatomical slots."""
    normalized = (np.asarray(raw, np.float32) - mean) / std
    seq = np.zeros((2, 16, 5, 4, 6), np.float32)
    mask = np.zeros_like(seq, bool)
    centers = EARLY_LATE / 30.0 * 50.0
    points = np.clip(np.rint(centers[..., None] + np.array([-2, -1, 1, 2])), 0, 249).astype(int)
    for device, slot in enumerate((1, 2, 4, 0)):
        channels = np.arange(device * 6, device * 6 + 6)
        x = normalized[channels][:, points].transpose(1, 2, 3, 0)
        m = np.repeat(np.asarray(presence[device * 2:device * 2 + 2], bool), 3)[None, None, None, :] & np.isfinite(x)
        seq[:, :, slot] = np.where(m, x, 0)
        mask[:, :, slot] = np.broadcast_to(m, x.shape)
    return {'imu_sequences': seq, 'imu_sequence_mask': mask, 'imu_bin_statistics': np.zeros((2, 16, 5, 30), np.float32), 'imu_bin_mask': mask.any(-1), 'imu_global_statistics': np.zeros((2, 5, 30), np.float32), 'imu_global_mask': mask.any(axis=(1, 3, 4))}

def bout_repeat_peers(workouts, times, features, probabilities, min_gap=10.0, max_gap=600.0, threshold=0.85, max_bouts=8):
    """Label-free repeat sets across pseudo-bouts, aligned by relative phase.

    Sustained same-class windows define candidate bouts only; uncertain tiny
    bouts remain valid. Self-bout and temporally overlapping candidates are
    excluded. There is no unique-class constraint or maximum group size three.
    """
    workouts = np.asarray(workouts)
    times = np.asarray(times)
    x = np.asarray(features, np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-08)
    pred = probabilities.argmax(1)
    peers = [[] for _ in times]
    bout_count = 0
    edge_count = 0
    for workout in np.unique(workouts):
        order = np.flatnonzero(workouts == workout)
        order = order[np.argsort(times[order], kind='stable')]
        cut = np.r_[0, np.flatnonzero((np.diff(pred[order]) != 0) | (np.diff(times[order]) > 1.5)) + 1, len(order)]
        bouts = [order[a:b] for a, b in zip(cut[:-1], cut[1:])]
        bout_count += len(bouts)
        desc = np.stack([x[b].mean(0) for b in bouts])
        desc /= np.maximum(np.linalg.norm(desc, axis=1, keepdims=True), 1e-08)
        sim = desc @ desc.T
        for j, rows in enumerate(bouts):
            candidates = []
            for k, other in enumerate(bouts):
                gap = max(times[other[0]] - times[rows[-1]], times[rows[0]] - times[other[-1]])
                if j != k and min_gap <= gap <= max_gap and (sim[j, k] >= threshold):
                    candidates.append(k)
            selected = sorted(candidates, key=lambda k: (-sim[j, k], k))[:max_bouts]
            for n, row in enumerate(rows):
                phase = n / max(len(rows) - 1, 1)
                peers[row] = [int(bouts[k][round(phase * (len(bouts[k]) - 1))]) for k in selected]
                edge_count += len(peers[row])
    return (tuple((tuple(p) for p in peers)), {'pseudo_bouts': bout_count, 'directed_phase_edges': edge_count, 'rows_with_peers': sum((bool(p) for p in peers)), 'test_labels_received': False, 'label_free_segmentation': True})

class SharedRGBReader:
    """Bounded in-memory unique-frame ROI LRU; no per-window pixel disk writes."""

    def __init__(self, path, pose2d, origin, max_frames=512):
        self.path = Path(path)
        self.pose = pose2d
        self.origin = origin
        self.max_frames = max_frames
        self.cap = cv2.VideoCapture(str(path))
        self.cache = OrderedDict()
        self.decoded = 0
        self.hits = 0
        if not self.cap.isOpened():
            raise ValueError(f'video unreadable {path}')

    def close(self):
        self.cap.release()

    def frame(self, frame):
        frame = self.origin + int(round((frame - self.origin) / 5)) * 5
        if frame in self.cache:
            self.hits += 1
            self.cache.move_to_end(frame)
            return self.cache[frame]
        pos = int(round(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
        if frame < pos or frame - pos > 150:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            pos = frame
        while pos < frame:
            if not self.cap.grab():
                raise RuntimeError('RGB grab failed')
            pos += 1
        ok, bgr = self.cap.read()
        if not ok or bgr is None or int(round(self.cap.get(cv2.CAP_PROP_POS_FRAMES))) != frame + 1:
            raise RuntimeError(f'exact RGB decode failed {frame}')
        self.decoded += 1
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        clock = self.pose[0, :, 0]
        j = int(np.searchsorted(clock, frame))
        valid = np.zeros(3, bool)
        valid[0] = True
        points = np.empty((0, 2)) if j >= len(clock) or clock[j] != frame else self.pose[:, j, 1:].T

        def crop(indices, scale):
            q = points[indices] if len(points) else points
            q = q[np.isfinite(q).all(1) & (q[:, 0] > 0) & (q[:, 1] > 0)]
            if len(q) < 2:
                return (rgb, False)
            lo = q.min(0)
            hi = q.max(0)
            center = (lo + hi) / 2
            side = max(hi - lo) * scale
            x0 = max(0, int(center[0] - side / 2))
            x1 = min(w, int(center[0] + side / 2))
            y0 = max(0, int(center[1] - side / 2))
            y1 = min(h, int(center[1] + side / 2))
            if x1 <= x0 or y1 <= y0:
                return (rgb, False)
            return (rgb[y0:y1, x0:x1], True)
        person, valid[1] = crop(slice(None), 1.15)
        motion, valid[2] = crop([4, 7, 10, 13], 1.4)
        crops = np.stack([cv2.resize(z, (160, 160), interpolation=cv2.INTER_AREA).transpose(2, 0, 1) for z in (rgb, person, motion)])
        self.cache[frame] = (crops, valid)
        if len(self.cache) > self.max_frames:
            self.cache.popitem(last=False)
        return (crops, valid)

    def window(self, start):
        for frame in sorted(set((start + EARLY_LATE.reshape(-1)).tolist())):
            self.frame(int(frame))
        frames = [self.frame(start + int(k)) for k in EARLY_LATE.reshape(-1)]
        pixels = np.stack([x[0] for x in frames]).reshape(2, 16, 3, 3, 160, 160)
        valid = np.stack([x[1] for x in frames]).reshape(2, 16, 3).transpose(0, 2, 1)
        return (pixels, valid)
