from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import csv
import re
from dataclasses import dataclass
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
NUMBER_PATTERN = re.compile('(\\d+)')

def natural_key(path: Path) -> tuple[object, ...]:
    parts = NUMBER_PATTERN.split(path.name)
    return tuple((int(part) if part.isdigit() else part.lower() for part in parts))

@dataclass(frozen=True)
class ThermalSample:
    sample_id: str
    class_id: int
    user_id: str
    trial_id: str
    trial_dir: Path

def read_manifest(path: Path, split: str) -> list[ThermalSample]:
    samples: list[ThermalSample] = []
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['split'] != split:
                continue
            trial_dir = Path(row['trial_dir'])
            if not trial_dir.is_dir():
                continue
            samples.append(ThermalSample(sample_id=row['sample_id'], class_id=int(row['class_id']), user_id=row['user_id'], trial_id=row['trial_id'], trial_dir=trial_dir))
    return samples

class ThermalOOFDataset(Dataset):
    """Thermal clips sampled at normalized trial-progress positions."""

    def __init__(self, manifest_path: str | Path, split: str, num_frames: int, image_height: int, image_width: int, augment: bool, normalization: str='legacy') -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.split = split
        self.num_frames = int(num_frames)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.augment = bool(augment)
        if normalization not in {'legacy', 'imagenet'}:
            raise ValueError(f'Unknown normalization: {normalization}')
        self.normalization = normalization
        self.samples = read_manifest(self.manifest_path, split)
        if not self.samples:
            raise RuntimeError(f'No valid split={split} samples in {self.manifest_path}')

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_indices(self, frame_count: int) -> list[int]:
        if frame_count <= self.num_frames:
            return torch.linspace(0, frame_count - 1, self.num_frames).round().long().tolist()
        boundaries = torch.linspace(0, frame_count, self.num_frames + 1).floor().long()
        indices: list[int] = []
        for index in range(self.num_frames):
            start = int(boundaries[index])
            end = max(start + 1, int(boundaries[index + 1]))
            if self.augment:
                selected = int(torch.randint(start, end, (1,)).item())
            else:
                selected = min(frame_count - 1, (start + end - 1) // 2)
            indices.append(selected)
        return indices

    def _augmentation(self, width: int, height: int) -> tuple[bool, float, list[int], float]:
        if not self.augment:
            return (False, 0.0, [0, 0], 1.0)
        flip = bool(torch.rand(1).item() < 0.5)
        angle = float(torch.empty(1).uniform_(-5.0, 5.0).item())
        max_dx = int(round(width * 0.04))
        max_dy = int(round(height * 0.04))
        translate = [int(torch.randint(-max_dx, max_dx + 1, (1,)).item()), int(torch.randint(-max_dy, max_dy + 1, (1,)).item())]
        scale = float(torch.empty(1).uniform_(0.95, 1.05).item())
        return (flip, angle, translate, scale)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        frame_paths = tuple(sorted((path for path in sample.trial_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS), key=natural_key))
        if not frame_paths:
            raise RuntimeError(f'No Thermal frames: {sample.trial_dir}')
        selected = self._sample_indices(len(frame_paths))
        images: list[Image.Image] = []
        for frame_index in selected:
            with Image.open(frame_paths[frame_index]) as image:
                images.append(image.convert('RGB'))
        flip, angle, translate, scale = self._augmentation(*images[0].size)
        tensors: list[torch.Tensor] = []
        for image in images:
            if self.augment:
                image = TF.affine(image, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR, fill=0)
                if flip:
                    image = TF.hflip(image)
            image = TF.resize(image, [self.image_height, self.image_width], interpolation=InterpolationMode.BILINEAR, antialias=True)
            tensor = TF.to_tensor(image)
            if self.normalization == 'imagenet':
                tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            else:
                tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            tensors.append(tensor)
        return {'clip': torch.stack(tensors), 'label': sample.class_id, 'sample_id': sample.sample_id}
