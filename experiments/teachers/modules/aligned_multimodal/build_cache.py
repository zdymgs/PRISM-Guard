from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from aligned_multimodal.aligned_data import AlignedSample, frame_map, load_skeleton
PROJECT_DIR = WORK_ROOT / 'aligned_multimodal'
DEFAULT_MANIFEST = PROJECT_DIR / 'data' / 'manifest.csv'
DEFAULT_CACHE_DIR = PROJECT_DIR / 'cache' / 'aligned_192x144'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='将对齐后的三模态数据预处理为 NumPy 内存映射缓存')
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument('--image-height', type=int, default=144)
    parser.add_argument('--image-width', type=int, default=192)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--chunk-size', type=int, default=32)
    parser.add_argument('--finalize-only', action='store_true', help='仅封装已经构建完成的 .building 文件')
    return parser.parse_args()

def read_all_samples(path: Path) -> tuple[list[AlignedSample], list[int]]:
    samples: list[AlignedSample] = []
    expected_lengths: list[int] = []
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            samples.append(AlignedSample(sample_id=row['sample_id'], class_id=int(row['class_id']), user_id=row['user_id'], trial_id=row['trial_id'], depth_dir=Path(row['depth_dir']), ir_dir=Path(row['ir_dir']), skeleton_dir=Path(row['skeleton_dir'])))
            expected_lengths.append(int(row['num_aligned_frames']))
    return (samples, expected_lengths)

def resize_image(path: Path, mode: str, height: int, width: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert(mode)
        image = image.resize((width, height), resample=Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)

def process_sample(args: tuple[AlignedSample, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample, height, width = args
    maps = {'depth': frame_map(sample.depth_dir, 'depth'), 'ir': frame_map(sample.ir_dir, 'ir'), 'skeleton': frame_map(sample.skeleton_dir, 'skeleton')}
    common_ids = sorted(set.intersection(*(set(mapping) for mapping in maps.values())))
    depth = np.stack([resize_image(maps['depth'][key], 'RGB', height, width) for key in common_ids])
    ir = np.stack([resize_image(maps['ir'][key], 'L', height, width) for key in common_ids])
    skeleton = np.stack([load_skeleton(maps['skeleton'][key], False).numpy() for key in common_ids])
    return (depth, ir, skeleton)

def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    samples, expected_lengths = read_all_samples(manifest_path)
    total_frames = sum(expected_lengths)
    if not samples or total_frames <= 0:
        raise RuntimeError('manifest 中没有可缓存的样本')
    shapes = {'depth_uint8.npy': (total_frames, args.image_height, args.image_width, 3), 'ir_uint8.npy': (total_frames, args.image_height, args.image_width), 'skeleton_float32.npy': (total_frames, 17, 4)}
    temp_paths = {name: cache_dir / f'{name}.building' for name in shapes}
    if args.finalize_only:
        offsets: list[list[int]] = []
        cursor = 0
        for length in expected_lengths:
            offsets.append([cursor, length])
            cursor += length
        for final_name, expected_shape in shapes.items():
            final_path = cache_dir / final_name
            temp_path = temp_paths[final_name]
            source = final_path if final_path.is_file() else temp_path
            if not source.is_file():
                raise FileNotFoundError(f'待封装缓存不存在：{source}')
            array = np.load(source, mmap_mode='r', allow_pickle=False)
            if tuple(array.shape) != expected_shape:
                raise ValueError(f'{source.name} 形状错误：{array.shape} != {expected_shape}')
            mmap_handle = getattr(array, '_mmap', None)
            if mmap_handle is not None:
                mmap_handle.close()
            del array
            if source == temp_path:
                os.replace(temp_path, final_path)
        metadata = {'version': 1, 'manifest': str(manifest_path), 'image_height': args.image_height, 'image_width': args.image_width, 'num_samples': len(samples), 'total_frames': total_frames, 'sample_ids': [sample.sample_id for sample in samples], 'offsets': offsets, 'files': list(shapes), 'build_seconds': None}
        (cache_dir / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False), encoding='utf-8')
        print(f'缓存封装完成：{cache_dir}', flush=True)
        return
    for path in temp_paths.values():
        path.unlink(missing_ok=True)
    arrays = {'depth': np.lib.format.open_memmap(temp_paths['depth_uint8.npy'], mode='w+', dtype=np.uint8, shape=shapes['depth_uint8.npy']), 'ir': np.lib.format.open_memmap(temp_paths['ir_uint8.npy'], mode='w+', dtype=np.uint8, shape=shapes['ir_uint8.npy']), 'skeleton': np.lib.format.open_memmap(temp_paths['skeleton_float32.npy'], mode='w+', dtype=np.float32, shape=shapes['skeleton_float32.npy'])}
    offsets: list[list[int]] = []
    cursor = 0
    started = time.time()
    workers = max(1, int(args.workers))
    chunk_size = max(workers, int(args.chunk_size))
    print(f'准备缓存 {len(samples)} 个 trial、{total_frames:,} 个对齐帧，线程数={workers}', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for chunk_start in range(0, len(samples), chunk_size):
            chunk = samples[chunk_start:chunk_start + chunk_size]
            work = ((sample, args.image_height, args.image_width) for sample in chunk)
            for local_index, (depth, ir, skeleton) in enumerate(executor.map(process_sample, work)):
                sample_index = chunk_start + local_index
                length = len(depth)
                expected = expected_lengths[sample_index]
                if length != expected or len(ir) != expected or len(skeleton) != expected:
                    raise RuntimeError(f'{samples[sample_index].sample_id} 帧数变化：manifest={expected}，实际={length}/{len(ir)}/{len(skeleton)}')
                end = cursor + length
                arrays['depth'][cursor:end] = depth
                arrays['ir'][cursor:end] = ir
                arrays['skeleton'][cursor:end] = skeleton
                offsets.append([cursor, length])
                cursor = end
            done = min(len(samples), chunk_start + len(chunk))
            elapsed = time.time() - started
            rate = done / max(elapsed, 1e-06)
            eta = (len(samples) - done) / max(rate, 1e-06)
            print(f'已完成 {done:4d}/{len(samples)} trial（{cursor:,}/{total_frames:,} 帧），耗时 {elapsed / 60:.1f} 分，预计剩余 {eta / 60:.1f} 分', flush=True)
    if cursor != total_frames:
        raise RuntimeError(f'最终缓存帧数不一致：预计 {total_frames}，实际 {cursor}')
    for array in arrays.values():
        array.flush()
        mmap_handle = getattr(array, '_mmap', None)
        if mmap_handle is not None:
            mmap_handle.close()
    del array
    del arrays
    for final_name, temp_path in temp_paths.items():
        os.replace(temp_path, cache_dir / final_name)
    metadata = {'version': 1, 'manifest': str(manifest_path), 'image_height': args.image_height, 'image_width': args.image_width, 'num_samples': len(samples), 'total_frames': total_frames, 'sample_ids': [sample.sample_id for sample in samples], 'offsets': offsets, 'files': list(shapes), 'build_seconds': round(time.time() - started, 2)}
    (cache_dir / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False), encoding='utf-8')
    print(f'缓存完成：{cache_dir}，总耗时 {metadata['build_seconds'] / 60:.1f} 分', flush=True)
