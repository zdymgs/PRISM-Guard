from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Iterable
from zoneinfo import ZoneInfo
ANNOTATION_RE = re.compile('^(?P<start>\\d{4}/\\d{2}/\\d{2} \\d{2}:\\d{2}:\\d{2}(?:\\.\\d+)?)\\s*-\\s*(?P<end>\\d{4}/\\d{2}/\\d{2} \\d{2}:\\d{2}:\\d{2}(?:\\.\\d+)?)\\s*-\\s*(?P<label>.+?)\\s*$')
TOKEN_PATTERNS = {'subject': re.compile('subject\\d+', re.IGNORECASE), 'scene': re.compile('scene\\d+', re.IGNORECASE), 'session': re.compile('session\\d+', re.IGNORECASE)}

@dataclass(frozen=True)
class MMActInterval:
    subject: str
    scene: str
    session: str
    start_ms: int
    end_ms: int
    label: str
    annotation_file: str
    line_number: int
    removed_prefix_del_count: int

def _natural_id(value: str) -> tuple[str, int]:
    match = re.match('([^\\d]+)(\\d+)$', value.lower())
    return (match.group(1), int(match.group(2))) if match else (value.lower(), -1)

def _path_token(path: Path, kind: str) -> str:
    matches = TOKEN_PATTERNS[kind].findall(path.as_posix())
    unique = sorted({value.lower() for value in matches}, key=_natural_id)
    if len(unique) != 1:
        raise ValueError(f'expected one {kind} token in {path}, found {unique}')
    return unique[0]

def _timestamp_ms(value: str) -> int:
    time_format = '%Y/%m/%d %H:%M:%S.%f' if '.' in value else '%Y/%m/%d %H:%M:%S'
    parsed = datetime.strptime(value, time_format).replace(tzinfo=ZoneInfo('Asia/Tokyo'))
    return int(round(parsed.timestamp() * 1000.0))

def parse_annotation_file(path: Path, root: Path | None=None) -> list[MMActInterval]:
    subject = _path_token(path, 'subject')
    scene = _path_token(path, 'scene')
    session = _path_token(path, 'session')
    intervals: list[MMActInterval] = []
    for line_number, raw_line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        cleaned = line.lstrip('\x7f')
        removed_prefix_del_count = len(line) - len(cleaned)
        line = cleaned
        match = ANNOTATION_RE.match(line)
        if match is None:
            raise ValueError(f'invalid MMAct annotation at {path}:{line_number}: {line!r}')
        start_ms = _timestamp_ms(match.group('start'))
        end_ms = _timestamp_ms(match.group('end'))
        if end_ms <= start_ms:
            raise ValueError(f'non-positive interval at {path}:{line_number}')
        display_path = path.relative_to(root).as_posix() if root is not None else path.as_posix()
        intervals.append(MMActInterval(subject=subject, scene=scene, session=session, start_ms=start_ms, end_ms=end_ms, label=match.group('label').strip().lower(), annotation_file=display_path, line_number=line_number, removed_prefix_del_count=removed_prefix_del_count))
    return intervals

def discover_annotation_files(root: Path) -> list[Path]:
    """Support both documented challenge layouts without guessing data contents."""
    candidates: set[Path] = set()
    for relative in (Path('untrimmed/annotation/trainval'), Path('untrimmed/annotations/trainval'), Path('untrimmed/sessions')):
        directory = root / relative
        if directory.is_dir():
            candidates.update(directory.rglob('*.txt'))
    return sorted(candidates, key=lambda item: item.as_posix().lower())

def subject_folds(subjects: Iterable[str], n_folds: int=5) -> dict[str, int]:
    """Assign complete subjects by numeric round robin, independent of labels."""
    unique = sorted({subject.lower() for subject in subjects}, key=_natural_id)
    if n_folds < 2 or len(unique) < n_folds:
        raise ValueError('subject-disjoint evaluation requires at least one subject per fold')
    return {subject: index % n_folds for index, subject in enumerate(unique)}

def audit_intervals(intervals: list[MMActInterval], n_folds: int=5) -> dict[str, object]:
    if not intervals:
        raise ValueError('no MMAct annotation intervals discovered')
    grouped: dict[tuple[str, str, str], list[MMActInterval]] = defaultdict(list)
    for interval in intervals:
        grouped[interval.subject, interval.scene, interval.session].append(interval)
    overlaps: list[dict[str, object]] = []
    for key, group in grouped.items():
        ordered = sorted(group, key=lambda item: (item.start_ms, item.end_ms))
        for previous, current in zip(ordered, ordered[1:]):
            if current.start_ms < previous.end_ms:
                overlaps.append({'subject': key[0], 'scene': key[1], 'session': key[2], 'previous_line': previous.line_number, 'current_line': current.line_number})
    subjects = sorted({item.subject for item in intervals}, key=_natural_id)
    folds = subject_folds(subjects, n_folds=n_folds)
    return {'instances': len(intervals), 'subjects': subjects, 'scenes': sorted({item.scene for item in intervals}, key=_natural_id), 'sessions': sorted({item.session for item in intervals}, key=_natural_id), 'class_counts': dict(sorted(Counter((item.label for item in intervals)).items())), 'subject_folds': folds, 'outer_fold_subjects': {f'F{fold}': [subject for subject in subjects if folds[subject] == fold] for fold in range(n_folds)}, 'overlap_count': len(overlaps), 'overlaps': overlaps, 'removed_prefix_del_count': sum((item.removed_prefix_del_count for item in intervals)), 'affected_prefix_del_files': sorted({item.annotation_file for item in intervals if item.removed_prefix_del_count > 0}), 'all_checks_pass': len(overlaps) == 0}
