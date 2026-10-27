"""Single-checkpoint, offline loading. Only load the supplied trusted artifact."""
from pathlib import Path
import hashlib
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_bundle(path):
    path = Path(path)
    if not 0 < path.stat().st_size < 100_000_000:
        raise ValueError('Entire inference checkpoint must be <100,000,000 bytes')
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data.get('format') not in {'p246-single-checkpoint-v1', 'prism-challenge-checkpoint-v1'}:
        raise ValueError('Unsupported checkpoint format')
    student = data['student']
    student['model_state'] = {
        key: value.float() if value.is_floating_point() else value
        for key, value in student['model_state'].items()
    }
    yolo = data['yolo_checkpoint_bytes']
    if hashlib.sha256(yolo).hexdigest() != data['yolo_sha256']:
        raise ValueError('Embedded pose checkpoint hash mismatch')
    return student, yolo
