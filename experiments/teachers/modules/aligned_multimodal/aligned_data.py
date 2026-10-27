from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from aligned_multimodal.depth_encoding import decode_jet_rgb, resize_decoded_depth
LEFT_RIGHT_PAIRS = ((1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16))
SKELETON_STRATEGIES = ('first', 'temporal_nearest')
SKELETON_FEATURE_DIMS = {'frame_joint': 4, 'clip_joint': 4, 'clip_joint_velocity': 7, 'clip_joint_bone': 7, 'clip_joint_bone_velocity': 10, 'clip_joint_root_z': 5}
H36M_PARENTS = np.asarray((0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15))
PROJECT_DIR = WORK_ROOT / 'aligned_multimodal'

def resolve_project_path(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_DIR / value).resolve()
DEPTH_REPRESENTATIONS = ('jet_rgb', 'jet_rgb_mask', 'decoded_replicated', 'decoded_mask')
VISUAL_NORMALIZATIONS = ('legacy', 'imagenet', 'imagenet_ir_clip_robust')
IR_MOTION_MODES = ('none', 'segment_peak_absdiff')
TEMPORAL_SAMPLING_MODES = ('uniform', 'uniform_plus_ir_motion_peak', 'uniform_plus_ir_motion_pairs')
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENET_GRAY_MEAN = float(sum(IMAGENET_MEAN) / len(IMAGENET_MEAN))
IMAGENET_GRAY_STD = float(sum(IMAGENET_STD) / len(IMAGENET_STD))

@dataclass(frozen=True)
class AlignedSample:
    sample_id: str
    class_id: int
    user_id: str
    trial_id: str
    depth_dir: Path
    ir_dir: Path
    skeleton_dir: Path

def read_manifest(path: Path, split: str) -> list[AlignedSample]:
    samples: list[AlignedSample] = []
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['split'] != split:
                continue
            samples.append(AlignedSample(sample_id=row['sample_id'], class_id=int(row['class_id']), user_id=row['user_id'], trial_id=row['trial_id'], depth_dir=Path(row['depth_dir']), ir_dir=Path(row['ir_dir']), skeleton_dir=Path(row['skeleton_dir'])))
    return samples

def canonical_frame_id(path: Path, modality: str) -> str:
    stem = path.stem
    if modality == 'depth':
        return stem[len('Depth_'):-len('_Color')]
    if modality == 'ir':
        return stem[len('IR_'):]
    if modality == 'skeleton':
        return stem[len('Color_'):]
    raise ValueError(modality)

def frame_map(trial_dir: Path, modality: str) -> dict[str, Path]:
    if modality == 'skeleton':
        files = (trial_dir / 'predictions').glob('*.json')
    else:
        files = (path for path in trial_dir.iterdir() if path.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    return {canonical_frame_id(path, modality): path for path in files if path.is_file()}

def _flip_skeleton(skeleton: torch.Tensor) -> torch.Tensor:
    skeleton = skeleton.clone()
    skeleton[:, 0] *= -1.0
    for left, right in LEFT_RIGHT_PAIRS:
        skeleton[[left, right]] = skeleton[[right, left]]
    return skeleton

def _normalise_person(person: dict, flip: bool=False) -> torch.Tensor | None:
    keypoints = np.asarray(person.get('keypoints', []), dtype=np.float32)
    scores = np.asarray(person.get('keypoint_scores', []), dtype=np.float32)
    if keypoints.shape != (17, 3):
        return None
    if scores.shape != (17,):
        scores = np.ones(17, dtype=np.float32)
    keypoints = keypoints - keypoints[0:1]
    scale = float(np.linalg.norm(keypoints, axis=1).max())
    if not np.isfinite(scale) or scale < 1e-06:
        scale = 1.0
    keypoints = keypoints / scale
    keypoints = np.nan_to_num(keypoints, nan=0.0, posinf=0.0, neginf=0.0)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    skeleton = torch.from_numpy(np.concatenate([keypoints, scores[:, None]], axis=1)).float()
    return _flip_skeleton(skeleton) if flip else skeleton

def load_skeleton_people(path: Path, flip: bool=False) -> list[torch.Tensor]:
    try:
        people = json.loads(path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        people = []
    if not isinstance(people, list):
        return []
    skeletons = [_normalise_person(person, flip) for person in people if isinstance(person, dict)]
    return [skeleton for skeleton in skeletons if skeleton is not None]

def load_skeleton(path: Path, flip: bool) -> torch.Tensor:
    people = load_skeleton_people(path, flip)
    return people[0] if people else torch.zeros(17, 4, dtype=torch.float32)

def skeleton_distance(previous: torch.Tensor, candidate: torch.Tensor) -> float:
    joint_distance = torch.linalg.vector_norm(previous[:, :3] - candidate[:, :3], dim=1)
    confidence = torch.minimum(previous[:, 3], candidate[:, 3]).clamp_min(0.0)
    if float(confidence.sum()) < 1e-06:
        return float(joint_distance.mean())
    return float((joint_distance * confidence).sum() / confidence.sum())

def load_skeleton_sequence(paths: list[Path], flip: bool, strategy: str='first') -> torch.Tensor:
    if strategy not in SKELETON_STRATEGIES:
        raise ValueError(f'未知 Skeleton 选人策略：{strategy}')
    if strategy == 'first':
        return torch.stack([load_skeleton(path, flip) for path in paths])
    selected: list[torch.Tensor] = []
    previous: torch.Tensor | None = None
    for path in paths:
        people = load_skeleton_people(path, flip=False)
        if not people:
            selected.append(torch.zeros(17, 4, dtype=torch.float32))
            continue
        if previous is None:
            current = people[0]
        else:
            current = min(people, key=lambda candidate: skeleton_distance(previous, candidate))
        selected.append(_flip_skeleton(current) if flip else current)
        previous = current
    return torch.stack(selected)

class AlignedMultimodalDataset(Dataset):

    def __init__(self, manifest_path: str | Path, split: str, modalities: list[str], num_frames: int, image_height: int, image_width: int, augment: bool, cache_dir: str | Path | None=None, skeleton_strategy: str='first', depth_representation: str='jet_rgb', visual_normalization: str='legacy', skeleton_representation: str='frame_joint', skeleton_raw_cache_dir: str | Path | None=None, ir_gain_jitter: float=0.0, ir_offset_jitter: float=0.0, ir_motion_mode: str='none', ir_roi_csv: str | Path | None=None, ir_roi_context: float=0.3, temporal_view: float=0.5, force_horizontal_flip: bool=False, ir_random_resized_crop_min_scale: float=1.0, temporal_sampling: str='uniform', ir_event_sampling: str | None=None) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.split = split
        self.modalities = tuple(modalities)
        self.num_frames = num_frames
        self.image_height = image_height
        self.image_width = image_width
        self.augment = augment
        self.temporal_view = float(temporal_view)
        self.force_horizontal_flip = bool(force_horizontal_flip)
        if not 0.0 <= self.temporal_view <= 1.0:
            raise ValueError('temporal_view must be in [0, 1]')
        if temporal_sampling not in TEMPORAL_SAMPLING_MODES:
            raise ValueError(f'unknown temporal_sampling: {temporal_sampling}')
        if temporal_sampling != 'uniform' and 'ir' not in self.modalities:
            raise ValueError('IR event sampling requires the IR modality')
        if temporal_sampling != 'uniform' and self.num_frames % 2:
            raise ValueError('IR event sampling requires an even num_frames')
        self.temporal_sampling = temporal_sampling
        if ir_event_sampling is not None:
            if ir_event_sampling not in TEMPORAL_SAMPLING_MODES[1:]:
                raise ValueError('ir_event_sampling must be a non-uniform IR sampling mode')
            if 'ir' not in self.modalities or cache_dir is None:
                raise ValueError('IR event view requires cached IR input')
        self.ir_event_sampling = ir_event_sampling
        self._ir_difference_cache: dict[tuple[int, int], np.ndarray] = {}
        if depth_representation not in DEPTH_REPRESENTATIONS:
            raise ValueError(f'未知Depth表示：{depth_representation}')
        self.depth_representation = depth_representation
        if visual_normalization not in VISUAL_NORMALIZATIONS:
            raise ValueError(f'未知视觉归一化：{visual_normalization}')
        if visual_normalization.startswith('imagenet') and depth_representation != 'jet_rgb':
            raise ValueError('ImageNet 归一化当前只支持 jet_rgb Depth')
        self.visual_normalization = visual_normalization
        self.ir_gain_jitter = float(ir_gain_jitter)
        self.ir_offset_jitter = float(ir_offset_jitter)
        self.ir_random_resized_crop_min_scale = float(ir_random_resized_crop_min_scale)
        if not 0.0 < self.ir_random_resized_crop_min_scale <= 1.0:
            raise ValueError('ir_random_resized_crop_min_scale must be in (0, 1]')
        if ir_motion_mode not in IR_MOTION_MODES:
            raise ValueError(f'Unknown IR motion mode: {ir_motion_mode}')
        if ir_motion_mode != 'none' and 'ir' not in self.modalities:
            raise ValueError('IR motion requires the IR modality')
        self.ir_motion_mode = ir_motion_mode
        self.ir_roi_context = float(ir_roi_context)
        if self.ir_roi_context < 0:
            raise ValueError('IR ROI context must be non-negative')
        if self.ir_gain_jitter < 0 or self.ir_offset_jitter < 0:
            raise ValueError('IR photometric jitter magnitudes must be non-negative')
        if skeleton_strategy not in SKELETON_STRATEGIES:
            raise ValueError(f'未知 Skeleton 选人策略：{skeleton_strategy}')
        self.skeleton_strategy = skeleton_strategy
        if skeleton_representation not in SKELETON_FEATURE_DIMS:
            raise ValueError(f'未知 Skeleton 表示：{skeleton_representation}')
        self.skeleton_representation = skeleton_representation
        self.samples = read_manifest(self.manifest_path, split)
        self.cache_dir = resolve_project_path(cache_dir) if cache_dir else None
        if self.temporal_sampling != 'uniform' and self.cache_dir is None:
            raise ValueError('IR event sampling currently requires aligned cache')
        self.ir_rois: dict[str, tuple[float, float, float, float, float]] = {}
        if ir_roi_csv is not None:
            if 'ir' not in self.modalities:
                raise ValueError('IR ROI requires the IR modality')
            if self.cache_dir is None:
                raise ValueError('IR ROI currently requires the aligned cache')
            roi_path = resolve_project_path(ir_roi_csv)
            with roi_path.open('r', encoding='utf-8-sig', newline='') as handle:
                for row in csv.DictReader(handle):
                    self.ir_rois[row['sample_id']] = (float(row['x0']), float(row['y0']), float(row['x1']), float(row['y1']), float(row.get('roi_quality', 1.0)))
            missing_rois = [sample.sample_id for sample in self.samples if sample.sample_id not in self.ir_rois]
            if missing_rois:
                raise KeyError(f'IR ROI file misses {len(missing_rois)} samples; first={missing_rois[0]}')
        self.skeleton_raw_cache_dir = resolve_project_path(skeleton_raw_cache_dir) if skeleton_raw_cache_dir else None
        self._cache_arrays: dict[str, np.ndarray] = {}
        self._raw_skeleton_arrays: dict[str, np.ndarray] = {}
        if not self.samples:
            raise RuntimeError(f'清单中没有 split={split} 的样本')
        self.cache_locations: dict[str, tuple[int, int]] = {}
        if self.cache_dir is not None:
            metadata_path = self.cache_dir / 'metadata.json'
            if not metadata_path.is_file():
                raise FileNotFoundError(f'缓存元数据不存在：{metadata_path}')
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            if (int(metadata['image_height']), int(metadata['image_width'])) != (self.image_height, self.image_width):
                raise ValueError('缓存图像尺寸与配置不一致')
            cached_strategy = str(metadata.get('skeleton_strategy', 'first'))
            if 'skeleton' in self.modalities and cached_strategy != self.skeleton_strategy:
                raise ValueError(f'缓存 Skeleton 策略为 {cached_strategy}，配置要求 {self.skeleton_strategy}：{self.cache_dir}')
            self.cache_locations = {sample_id: (int(offset), int(length)) for sample_id, (offset, length) in zip(metadata['sample_ids'], metadata['offsets'])}
            missing = [sample.sample_id for sample in self.samples if sample.sample_id not in self.cache_locations]
            if missing:
                raise KeyError(f'缓存缺少 {len(missing)} 个 manifest 样本，第一个是 {missing[0]}')
        if 'skeleton' in self.modalities and self.skeleton_representation != 'frame_joint':
            if self.cache_dir is None or self.skeleton_raw_cache_dir is None:
                raise ValueError('clip-level Skeleton 表示需要 aligned cache 和 skeleton_raw_cache_dir')
            raw_metadata_path = self.skeleton_raw_cache_dir / 'metadata.json'
            if not raw_metadata_path.is_file():
                raise FileNotFoundError(f'raw Skeleton 元数据不存在：{raw_metadata_path}')
            raw_metadata = json.loads(raw_metadata_path.read_text(encoding='utf-8'))
            raw_locations = {sample_id: (int(offset), int(length)) for sample_id, (offset, length) in zip(raw_metadata['sample_ids'], raw_metadata['offsets'])}
            if raw_locations != self.cache_locations:
                raise ValueError('raw Skeleton 缓存与 aligned cache 的样本/offset 不一致')

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_cache_arrays'] = {}
        state['_raw_skeleton_arrays'] = {}
        return state

    def _cache_array(self, modality: str) -> np.ndarray:
        if modality not in self._cache_arrays:
            filenames = {'depth': 'depth_uint8.npy', 'ir': 'ir_uint8.npy', 'skeleton': 'skeleton_float32.npy', 'depth_scalar': 'depth_scalar_uint8.npy', 'depth_valid': 'depth_valid_uint8.npy'}
            assert self.cache_dir is not None
            self._cache_arrays[modality] = np.load(self.cache_dir / filenames[modality], mmap_mode='r', allow_pickle=False)
        return self._cache_arrays[modality]

    def _raw_skeleton_array(self, name: str) -> np.ndarray:
        if name not in self._raw_skeleton_arrays:
            filenames = {'skeleton': 'skeleton_raw_float32.npy', 'time': 'frame_time_float32.npy'}
            assert self.skeleton_raw_cache_dir is not None
            self._raw_skeleton_arrays[name] = np.load(self.skeleton_raw_cache_dir / filenames[name], mmap_mode='r', allow_pickle=False)
        return self._raw_skeleton_arrays[name]

    def _clip_skeleton_features(self, raw: np.ndarray, times: np.ndarray, flip: bool) -> torch.Tensor:
        skeleton = np.asarray(raw, dtype=np.float32).copy()
        if flip:
            skeleton[:, :, 0] *= -1.0
            for left, right in LEFT_RIGHT_PAIRS:
                skeleton[:, [left, right]] = skeleton[:, [right, left]]
        coordinates = skeleton[:, :, :3]
        scores = skeleton[:, :, 3:4]
        roots = coordinates[:, 0].copy()
        centered = coordinates - roots[:, None]
        frame_scales = np.linalg.norm(centered, axis=2).max(axis=1)
        valid_scales = frame_scales[np.isfinite(frame_scales) & (frame_scales > 1e-06)]
        clip_scale = float(np.median(valid_scales)) if len(valid_scales) else 1.0
        joint = np.nan_to_num(centered / max(clip_scale, 1e-06)).astype(np.float32)
        features = [joint, np.nan_to_num(scores).astype(np.float32)]
        if 'bone' in self.skeleton_representation:
            bone = joint - joint[:, H36M_PARENTS]
            bone[:, 0] = 0.0
            features.append(bone)
        if 'velocity' in self.skeleton_representation:
            velocity = np.zeros_like(joint)
            delta_time = np.diff(np.asarray(times, dtype=np.float32))
            valid_delta = np.isfinite(delta_time) & (delta_time > 1e-06)
            safe_delta = np.where(valid_delta, delta_time, 1.0)
            velocity[1:] = np.where(valid_delta[:, None, None], np.diff(joint, axis=0) / safe_delta[:, None, None], 0.0)
            features.append(velocity)
        if self.skeleton_representation.endswith('root_z'):
            root_z = np.nan_to_num(roots[:, 2:3] / max(clip_scale, 1e-06)).astype(np.float32)
            features.append(np.repeat(root_z[:, None], 17, axis=1))
        return torch.from_numpy(np.concatenate(features, axis=2))

    def _format_decoded_depth(self, depth: np.ndarray, valid: np.ndarray) -> torch.Tensor:
        depth_tensor = torch.from_numpy(np.ascontiguousarray(depth)).float().div_(127.5).sub_(1.0)
        valid_tensor = torch.from_numpy(np.ascontiguousarray(valid)).float()
        depth_tensor = depth_tensor * valid_tensor
        if self.depth_representation == 'decoded_replicated':
            return depth_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
        signed_valid = valid_tensor.mul(2.0).sub(1.0)
        return torch.stack([depth_tensor, depth_tensor, signed_valid], dim=1)

    def _normalise_depth_rgb(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.visual_normalization == 'legacy':
            return tensor.div(127.5).sub(1.0)
        tensor = tensor.div(255.0)
        mean = tensor.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = tensor.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        return tensor.sub(mean).div(std)

    def _normalise_ir(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.visual_normalization == 'legacy':
            return tensor.div(127.5).sub(1.0)
        if self.visual_normalization == 'imagenet_ir_clip_robust':
            flattened = tensor[..., ::8, ::8].reshape(-1)
            lower = torch.quantile(flattened, 0.02)
            upper = torch.quantile(flattened, 0.98)
            tensor = tensor.sub(lower).div((upper - lower).clamp_min(8.0))
            return tensor.clamp_(0.0, 1.0).sub(IMAGENET_GRAY_MEAN).div(IMAGENET_GRAY_STD)
        return tensor.div(255.0).sub(IMAGENET_GRAY_MEAN).div(IMAGENET_GRAY_STD)

    def _augment_ir_uint8_scale(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.augment:
            return tensor
        if self.ir_gain_jitter > 0:
            gain = 1.0 + float(torch.empty(1).uniform_(-self.ir_gain_jitter, self.ir_gain_jitter))
            tensor = tensor.mul(gain)
        if self.ir_offset_jitter > 0:
            offset = float(torch.empty(1).uniform_(-self.ir_offset_jitter, self.ir_offset_jitter))
            tensor = tensor.add(offset)
        tensor = tensor.clamp_(0.0, 255.0)
        if self.ir_random_resized_crop_min_scale < 1.0:
            height, width = tensor.shape[-2:]
            scale = float(torch.empty(1).uniform_(self.ir_random_resized_crop_min_scale, 1.0))
            crop_height = max(2, min(height, int(round(height * scale))))
            crop_width = max(2, min(width, int(round(width * scale))))
            top = int(torch.randint(0, height - crop_height + 1, (1,)).item())
            left = int(torch.randint(0, width - crop_width + 1, (1,)).item())
            tensor = F.interpolate(tensor[..., top:top + crop_height, left:left + crop_width], size=(height, width), mode='bilinear', align_corners=False)
        return tensor

    def _segment_peak_ir_motion(self, frames: np.ndarray) -> torch.Tensor:
        """Return one label-free peak absolute-difference map per time segment."""
        frames_float = np.asarray(frames, dtype=np.float32)
        if len(frames_float) < 2:
            shape = (self.num_frames, *frames_float.shape[-2:])
            return torch.zeros(shape, dtype=torch.float32)
        differences = np.abs(np.diff(frames_float, axis=0))
        height, width = differences.shape[-2:]
        y0, y1 = (height // 10, max(height // 10 + 1, height - height // 10))
        x0, x1 = (width // 10, max(width // 10 + 1, width - width // 10))
        energy = differences[:, y0:y1, x0:x1].mean(axis=(1, 2))
        boundaries = np.linspace(0, len(differences), self.num_frames + 1).astype(np.int64)
        selected: list[np.ndarray] = []
        for segment in range(self.num_frames):
            start = min(len(differences) - 1, int(boundaries[segment]))
            end = max(start + 1, int(boundaries[segment + 1]))
            end = min(len(differences), end)
            local_index = int(np.argmax(energy[start:end]))
            selected.append(differences[start + local_index])
        tensor = torch.from_numpy(np.ascontiguousarray(np.stack(selected))).float()
        sparse = tensor[..., ::8, ::8].reshape(-1)
        nonzero = sparse[sparse > 0]
        scale = torch.quantile(nonzero, 0.95) if nonzero.numel() else tensor.new_tensor(8.0)
        tensor = tensor.div(scale.clamp_min(8.0)).clamp_(0.0, 1.0)
        return tensor.sub_(0.2).div_(0.25)

    def _crop_ir_context(self, frames: torch.Tensor, sample_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Crop a broad, fold-pure IR context box and resize it to full input size."""
        x0, y0, x1, y1, quality = self.ir_rois[sample_id]
        raw_width, raw_height = (640.0, 480.0)
        left, right = sorted((x0 / raw_width, x1 / raw_width))
        top, bottom = sorted((y0 / raw_height, y1 / raw_height))
        width = max(right - left, 0.04)
        height = max(bottom - top, 0.04)
        left = max(0.0, left - self.ir_roi_context * width)
        right = min(1.0, right + self.ir_roi_context * width)
        top = max(0.0, top - self.ir_roi_context * height)
        bottom = min(1.0, bottom + self.ir_roi_context * height)
        image_height, image_width = frames.shape[-2:]
        px0 = max(0, min(image_width - 2, int(np.floor(left * image_width))))
        px1 = max(px0 + 2, min(image_width, int(np.ceil(right * image_width))))
        py0 = max(0, min(image_height - 2, int(np.floor(top * image_height))))
        py1 = max(py0 + 2, min(image_height, int(np.ceil(bottom * image_height))))
        cropped = frames[..., py0:py1, px0:px1]
        resized = F.interpolate(cropped, size=(image_height, image_width), mode='bilinear', align_corners=False)
        return (resized, frames.new_tensor(float(np.clip(quality, 0.0, 1.0))))

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_positions(self, length: int) -> list[int]:
        if length <= self.num_frames:
            return torch.linspace(0, length - 1, self.num_frames).round().long().tolist()
        boundaries = torch.linspace(0, length, self.num_frames + 1).floor().long()
        positions: list[int] = []
        for i in range(self.num_frames):
            start = int(boundaries[i])
            end = max(start + 1, int(boundaries[i + 1]))
            if self.augment:
                positions.append(int(torch.randint(start, end, (1,)).item()))
            else:
                if self.temporal_view == 0.5:
                    position = (start + end - 1) // 2
                else:
                    position = start + int(round((end - start - 1) * self.temporal_view))
                positions.append(min(length - 1, position))
        return positions

    def _uniform_plus_ir_motion_peak_positions(self, ir_cache: np.ndarray, offset: int, length: int) -> list[int]:
        """Interleave phase context with label-free IR motion-peak frames."""
        if length < 2:
            return self._sample_positions(length)
        segments = self.num_frames // 2
        boundaries = torch.linspace(0, length, segments + 1).floor().long()
        differences = self._ir_frame_differences(ir_cache, offset, length)
        positions: list[int] = []
        for segment in range(segments):
            start = min(length - 1, int(boundaries[segment]))
            end = max(start + 1, int(boundaries[segment + 1]))
            end = min(length, end)
            if self.augment:
                context = int(torch.randint(start, end, (1,)).item())
            elif self.temporal_view == 0.5:
                context = (start + end - 1) // 2
            else:
                context = start + int(round((end - start - 1) * self.temporal_view))
            diff_start = min(len(differences) - 1, start)
            diff_end = min(len(differences), max(diff_start + 1, end - 1))
            ranked = np.argsort(differences[diff_start:diff_end], kind='stable')[::-1]
            peak_frame = context
            for local_index in ranked:
                candidate = min(length - 1, diff_start + int(local_index) + 1)
                if candidate != context:
                    peak_frame = candidate
                    break
            if peak_frame == context and end - start > 1:
                peak_frame = start if context != start else end - 1
            positions.extend([context, peak_frame])
        return sorted(positions)

    def _uniform_plus_ir_motion_pair_positions(self, ir_cache: np.ndarray, offset: int, length: int) -> list[int]:
        """Keep global phase anchors and explicit peak-transition frame pairs."""
        if length <= self.num_frames:
            return self._sample_positions(length)
        anchor_count = 4
        pair_count = (self.num_frames - anchor_count) // 2
        if anchor_count + 2 * pair_count != self.num_frames:
            raise ValueError('uniform_plus_ir_motion_pairs requires num_frames = 4 + 2k')
        differences = self._ir_frame_differences(ir_cache, offset, length)
        anchors: list[int] = []
        anchor_boundaries = torch.linspace(0, length, anchor_count + 1).floor().long()
        for segment in range(anchor_count):
            start = min(length - 1, int(anchor_boundaries[segment]))
            end = min(length, max(start + 1, int(anchor_boundaries[segment + 1])))
            if self.augment:
                anchor = int(torch.randint(start, end, (1,)).item())
            elif self.temporal_view == 0.5:
                anchor = (start + end - 1) // 2
            else:
                anchor = start + int(round((end - start - 1) * self.temporal_view))
            anchors.append(anchor)
        used = set(anchors)
        pairs: list[tuple[int, int]] = []
        pair_boundaries = torch.linspace(0, length, pair_count + 1).floor().long()
        for segment in range(pair_count):
            start = min(length - 2, int(pair_boundaries[segment]))
            end = min(length, max(start + 2, int(pair_boundaries[segment + 1])))
            diff_end = min(len(differences), end - 1)
            ranked = np.argsort(differences[start:diff_end], kind='stable')[::-1]
            selected: tuple[int, int] | None = None
            for local_index in ranked:
                before = start + int(local_index)
                candidate = (before, before + 1)
                if candidate[0] not in used and candidate[1] not in used:
                    selected = candidate
                    break
            if selected is None:
                for before in range(start, max(start + 1, end - 1)):
                    candidate = (before, before + 1)
                    if candidate[0] not in used and candidate[1] not in used:
                        selected = candidate
                        break
            if selected is not None:
                pairs.append(selected)
                used.update(selected)
        positions = anchors + [position for pair in pairs for position in pair]
        if len(positions) < self.num_frames:
            fallback = torch.linspace(0, length - 1, self.num_frames).round().long()
            for position in fallback.tolist() + list(range(length)):
                if position not in used:
                    positions.append(int(position))
                    used.add(int(position))
                if len(positions) == self.num_frames:
                    break
        if len(positions) != self.num_frames:
            raise RuntimeError(f'failed to construct {self.num_frames} unique temporal positions')
        return sorted(positions)

    def _ir_frame_differences(self, ir_cache: np.ndarray, offset: int, length: int) -> np.ndarray:
        key = (int(offset), int(length))
        cached = self._ir_difference_cache.get(key)
        if cached is None:
            sparse = np.asarray(ir_cache[offset:offset + length, ::8, ::8], dtype=np.float32)
            cached = np.abs(np.diff(sparse, axis=0)).mean(axis=(1, 2))
            self._ir_difference_cache[key] = cached
        return cached

    def _relative_positions_for_mode(self, mode: str, ir_cache: np.ndarray, offset: int, length: int) -> list[int]:
        if mode == 'uniform_plus_ir_motion_peak':
            return self._uniform_plus_ir_motion_peak_positions(ir_cache, offset, length)
        if mode == 'uniform_plus_ir_motion_pairs':
            return self._uniform_plus_ir_motion_pair_positions(ir_cache, offset, length)
        if mode == 'uniform':
            return self._sample_positions(length)
        raise ValueError(f'unknown temporal sampling mode: {mode}')

    def _getitem_from_cache(self, sample: AlignedSample) -> dict[str, torch.Tensor | int | str]:
        offset, length = self.cache_locations[sample.sample_id]
        if self.temporal_sampling == 'uniform':
            relative_positions = self._sample_positions(length)
        else:
            sampling_ir_cache = self._cache_array('ir')
            relative_positions = self._relative_positions_for_mode(self.temporal_sampling, sampling_ir_cache, offset, length)
        positions = np.asarray(relative_positions, dtype=np.int64) + offset
        flip = bool(torch.rand(1).item() < 0.5 if self.augment else self.force_horizontal_flip)
        result: dict[str, torch.Tensor | int | str] = {'label': sample.class_id, 'sample_id': sample.sample_id, 'user_id': sample.user_id}
        if 'depth' in self.modalities:
            if self.depth_representation in {'jet_rgb', 'jet_rgb_mask'}:
                depth = np.asarray(self._cache_array('depth')[positions])
                if flip:
                    depth = np.flip(depth, axis=2)
                depth_tensor = torch.from_numpy(np.ascontiguousarray(depth)).permute(0, 3, 1, 2).float()
                depth_tensor = self._normalise_depth_rgb(depth_tensor)
                if self.depth_representation == 'jet_rgb_mask':
                    valid = np.asarray(self._cache_array('depth_valid')[positions])
                    if flip:
                        valid = np.flip(valid, axis=2)
                    valid_tensor = torch.from_numpy(np.ascontiguousarray(valid)).float()
                    depth_tensor = torch.cat([depth_tensor, valid_tensor.mul(2.0).sub(1.0).unsqueeze(1)], dim=1)
                result['depth'] = depth_tensor
            else:
                depth = np.asarray(self._cache_array('depth_scalar')[positions])
                valid = np.asarray(self._cache_array('depth_valid')[positions])
                if flip:
                    depth = np.flip(depth, axis=2)
                    valid = np.flip(valid, axis=2)
                result['depth'] = self._format_decoded_depth(depth, valid)
        if 'ir' in self.modalities:
            ir_cache = self._cache_array('ir')
            ir = np.asarray(ir_cache[positions])
            ir_tensor = torch.from_numpy(np.ascontiguousarray(ir)).unsqueeze(1).float()
            ir_tensor = self._augment_ir_uint8_scale(ir_tensor)
            if self.ir_rois:
                ir_local, ir_local_quality = self._crop_ir_context(ir_tensor, sample.sample_id)
                if flip:
                    ir_local = torch.flip(ir_local, dims=(-1,))
                result['ir_local'] = self._normalise_ir(ir_local)
                result['ir_local_quality'] = ir_local_quality
            if flip:
                ir_tensor = torch.flip(ir_tensor, dims=(-1,))
            result['ir'] = self._normalise_ir(ir_tensor)
            if self.ir_event_sampling is not None:
                event_relative_positions = self._relative_positions_for_mode(self.ir_event_sampling, ir_cache, offset, length)
                event_positions = np.asarray(event_relative_positions, dtype=np.int64) + offset
                event_ir = np.asarray(ir_cache[event_positions])
                event_tensor = torch.from_numpy(np.ascontiguousarray(event_ir)).unsqueeze(1).float()
                event_tensor = self._augment_ir_uint8_scale(event_tensor)
                if flip:
                    event_tensor = torch.flip(event_tensor, dims=(-1,))
                result['ir_event'] = self._normalise_ir(event_tensor)
            if self.ir_motion_mode == 'segment_peak_absdiff':
                full_ir = np.asarray(ir_cache[offset:offset + length])
                motion = self._segment_peak_ir_motion(full_ir)
                if flip:
                    motion = torch.flip(motion, dims=(-1,))
                result['ir_motion'] = motion.unsqueeze(1)
        if 'skeleton' in self.modalities:
            if self.skeleton_representation == 'frame_joint':
                skeleton = np.array(self._cache_array('skeleton')[positions], dtype=np.float32, copy=True)
                if flip:
                    skeleton[:, :, 0] *= -1.0
                    for left, right in LEFT_RIGHT_PAIRS:
                        skeleton[:, [left, right]] = skeleton[:, [right, left]]
                result['skeleton'] = torch.from_numpy(skeleton)
            else:
                raw = self._raw_skeleton_array('skeleton')[offset:offset + length]
                times = self._raw_skeleton_array('time')[offset:offset + length]
                full_features = self._clip_skeleton_features(raw, times, flip)
                result['skeleton'] = full_features[torch.from_numpy(positions - offset)]
        return result

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        sample = self.samples[index]
        if self.cache_dir is not None:
            return self._getitem_from_cache(sample)
        maps = {'depth': frame_map(sample.depth_dir, 'depth'), 'ir': frame_map(sample.ir_dir, 'ir'), 'skeleton': frame_map(sample.skeleton_dir, 'skeleton')}
        common_ids = sorted(set.intersection(*(set(mapping) for mapping in maps.values())))
        if not common_ids:
            raise RuntimeError(f'三个模态没有共同帧：{sample.sample_id}')
        selected_ids = [common_ids[i] for i in self._sample_positions(len(common_ids))]
        flip = bool(torch.rand(1).item() < 0.5 if self.augment else self.force_horizontal_flip)
        result: dict[str, torch.Tensor | int | str] = {'label': sample.class_id, 'sample_id': sample.sample_id, 'user_id': sample.user_id}
        if 'depth' in self.modalities:
            depth_frames: list[torch.Tensor] = []
            for frame_id in selected_ids:
                try:
                    with Image.open(maps['depth'][frame_id]) as image:
                        image = image.convert('RGB')
                        if self.depth_representation in {'jet_rgb', 'jet_rgb_mask'}:
                            original_rgb = np.asarray(image, dtype=np.uint8)
                            if flip:
                                image = TF.hflip(image)
                            image = TF.resize(image, [self.image_height, self.image_width], interpolation=InterpolationMode.BILINEAR, antialias=True)
                            tensor = TF.to_tensor(image)
                            if self.visual_normalization == 'legacy':
                                tensor = TF.normalize(tensor, [0.5] * 3, [0.5] * 3)
                            else:
                                tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
                            if self.depth_representation == 'jet_rgb_mask':
                                decoded, valid, _ = decode_jet_rgb(original_rgb)
                                _, valid = resize_decoded_depth(decoded, valid, self.image_height, self.image_width)
                                if flip:
                                    valid = np.flip(valid, axis=1)
                                valid_tensor = torch.from_numpy(np.ascontiguousarray(valid)).float().mul(2.0).sub(1.0)
                                tensor = torch.cat([tensor, valid_tensor.unsqueeze(0)], dim=0)
                        else:
                            rgb = np.asarray(image, dtype=np.uint8)
                            depth, valid, _ = decode_jet_rgb(rgb)
                            depth, valid = resize_decoded_depth(depth, valid, self.image_height, self.image_width)
                            if flip:
                                depth = np.flip(depth, axis=1)
                                valid = np.flip(valid, axis=1)
                            tensor = self._format_decoded_depth(depth[None], valid[None])[0]
                except (UnidentifiedImageError, OSError):
                    channels = 4 if self.depth_representation == 'jet_rgb_mask' else 3
                    tensor = torch.zeros((channels, self.image_height, self.image_width))
                depth_frames.append(tensor)
            result['depth'] = torch.stack(depth_frames)
        if 'ir' in self.modalities:
            ir_frames: list[torch.Tensor] = []
            for frame_id in selected_ids:
                try:
                    with Image.open(maps['ir'][frame_id]) as image:
                        image = image.convert('L')
                        if flip:
                            image = TF.hflip(image)
                        image = TF.resize(image, [self.image_height, self.image_width], interpolation=InterpolationMode.BILINEAR, antialias=True)
                        tensor = TF.pil_to_tensor(image).float()
                        tensor = self._augment_ir_uint8_scale(tensor)
                        tensor = self._normalise_ir(tensor.unsqueeze(0))[0]
                except (UnidentifiedImageError, OSError):
                    tensor = torch.full((1, self.image_height, self.image_width), -1.0)
                ir_frames.append(tensor)
            result['ir'] = torch.stack(ir_frames)
            if self.ir_motion_mode == 'segment_peak_absdiff':
                full_ir_frames: list[np.ndarray] = []
                for frame_id in common_ids:
                    try:
                        with Image.open(maps['ir'][frame_id]) as image:
                            image = image.convert('L')
                            image = TF.resize(image, [self.image_height, self.image_width], interpolation=InterpolationMode.BILINEAR, antialias=True)
                            full_ir_frames.append(np.asarray(image, dtype=np.uint8))
                    except (UnidentifiedImageError, OSError):
                        full_ir_frames.append(np.zeros((self.image_height, self.image_width), dtype=np.uint8))
                motion = self._segment_peak_ir_motion(np.stack(full_ir_frames))
                if flip:
                    motion = torch.flip(motion, dims=(-1,))
                result['ir_motion'] = motion.unsqueeze(1)
        if 'skeleton' in self.modalities:
            full_skeleton = load_skeleton_sequence([maps['skeleton'][frame_id] for frame_id in common_ids], flip, self.skeleton_strategy)
            selected_positions = [common_ids.index(frame_id) for frame_id in selected_ids]
            result['skeleton'] = full_skeleton[selected_positions]
        return result
