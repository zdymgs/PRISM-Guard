from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
'Small, side-effect-free numerical kernels for the multimodal/foundation/student heads.\n\nThis module deliberately contains no paths, artifact loading, or training at\nimport time.  The shapes and class alignment here are part of the frozen\nfull-40 protocol.\n'
from typing import Any
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize_scalar
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
FEATURE_RECIPES: dict[str, dict[str, float]] = {'early': {'alpha': 1000.0, 'power': 0.75}, 'late': {'alpha': 1000.0, 'power': 0.75}, 'window_mean': {'alpha': 1000.0, 'power': 0.75}, 'early_late': {'alpha': 3000.0, 'power': 0.75}, 'temporal_delta': {'alpha': 3000.0, 'power': 0.75}, 'kinetics': {'alpha': 1000.0, 'power': 0.5}}

def _finite_array(value: Any, name: str, dtype: Any=np.float32) -> np.ndarray:
    out = np.asarray(value, dtype=dtype)
    if not np.isfinite(out).all():
        raise ValueError(f'{name} contains non-finite values')
    return out

def normalize_tokens(values: np.ndarray) -> np.ndarray:
    """L2-normalize the final token axis, with a 1e-8 norm floor."""
    values = _finite_array(values, 'values')
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norm, 1e-08)

def _row_standardize(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=-1, keepdims=True)
    return centered / np.maximum(centered.std(axis=-1, keepdims=True), 1e-06)

def feature_sets(raw: np.ndarray, kinetics: np.ndarray) -> dict[str, np.ndarray]:
    """Build the six frozen foundation representations from VideoMAE/Kinetics inputs."""
    raw = _finite_array(raw, 'raw')
    kinetics = _finite_array(kinetics, 'kinetics')
    if raw.ndim != 4 or raw.shape[1:] != (2, 3, 1024):
        raise ValueError(f'Expected raw shape (n,2,3,1024), got {raw.shape}')
    if kinetics.shape != (len(raw), 2, 3, 400):
        raise ValueError(f'Expected kinetics shape {(len(raw), 2, 3, 400)}, got {kinetics.shape}')
    values = normalize_tokens(raw)
    early, late = (values[:, 0], values[:, 1])
    mean = normalize_tokens(values.mean(axis=1))
    difference = late - early
    standardized_kinetics = _row_standardize(kinetics.reshape(len(kinetics), -1))
    return {'early': early.reshape(len(values), -1), 'late': late.reshape(len(values), -1), 'window_mean': mean.reshape(len(values), -1), 'early_late': values.reshape(len(values), -1), 'temporal_delta': np.concatenate((mean, difference), axis=1).reshape(len(values), -1), 'kinetics': standardized_kinetics}

def mean_impute_blocks(values: np.ndarray, training_mean: np.ndarray, *, windows: tuple[int, ...]=(), views: tuple[int, ...]=()) -> np.ndarray:
    values = _finite_array(values, 'values')
    training_mean = _finite_array(training_mean, 'training_mean')
    if values.ndim != 4 or values.shape[1:3] != (2, 3):
        raise ValueError(f'Expected values shape (n,2,3,d), got {values.shape}')
    if training_mean.shape != values.shape[1:]:
        raise ValueError(f'Expected training_mean shape {values.shape[1:]}, got {training_mean.shape}')
    if any((i not in (0, 1) for i in windows)) or any((i not in (0, 1, 2) for i in views)):
        raise ValueError('windows/views contain an invalid index')
    output = values.copy()
    for window in windows or (0, 1):
        for view in views or (0, 1, 2):
            output[:, window, view] = training_mean[window, view]
    return output

def fixed_perturbations(normalized_target: np.ndarray, source_training_mean: np.ndarray) -> dict[str, np.ndarray]:
    """Return the exact ten old student intervention conditions."""
    source = _finite_array(normalized_target, 'normalized_target')
    if source.ndim != 4 or source.shape[1:3] != (2, 3):
        raise ValueError(f'Expected normalized_target shape (n,2,3,d), got {source.shape}')
    source_training_mean = _finite_array(source_training_mean, 'source_training_mean')
    if source_training_mean.shape != source.shape[1:]:
        raise ValueError('source_training_mean shape must match target non-sample axes')
    window_mean = source.mean(axis=1, keepdims=True)
    view_mean = source.mean(axis=2, keepdims=True)
    swapped_views = source.copy()
    swapped_views[:, :, [1, 2]] = swapped_views[:, :, [2, 1]]
    return {'baseline': source, 'drop_scene': mean_impute_blocks(source, source_training_mean, views=(0,)), 'drop_person': mean_impute_blocks(source, source_training_mean, views=(1,)), 'drop_workspace': mean_impute_blocks(source, source_training_mean, views=(2,)), 'drop_early': mean_impute_blocks(source, source_training_mean, windows=(0,)), 'drop_late': mean_impute_blocks(source, source_training_mean, windows=(1,)), 'swap_early_late': source[:, ::-1].copy(), 'collapse_early_late': np.repeat(window_mean, 2, axis=1), 'swap_person_workspace': swapped_views, 'collapse_view_identity': np.repeat(view_mean, 3, axis=2)}

def _sample_weights(labels: np.ndarray, power: float) -> np.ndarray:
    counts = np.bincount(labels, minlength=40).astype(float)
    present = counts > 0
    reference = counts[present].mean()
    weights = np.ones(40, dtype=float)
    if power:
        weights[present] = (reference / counts[present]) ** power
    result = weights[labels]
    return result / result.mean()

def fit_ridge(x_train: np.ndarray, y_train: np.ndarray, alpha: float, power: float) -> Pipeline:
    x_train = _finite_array(x_train, 'x_train', np.float32)
    y_train = np.asarray(y_train)
    if y_train.ndim != 1 or not np.issubdtype(y_train.dtype, np.number) or (not np.isfinite(y_train).all()) or np.any(y_train != np.floor(y_train)):
        raise ValueError('y_train must contain finite integer class IDs')
    y_train = y_train.astype(np.int64)
    if x_train.ndim != 2 or len(x_train) != len(y_train) or len(y_train) == 0:
        raise ValueError('x_train must be nonempty 2-D and align with y_train')
    if not np.isfinite(alpha) or alpha <= 0 or (not np.isfinite(power)) or (power < 0):
        raise ValueError('alpha must be positive and power nonnegative')
    if np.any((y_train < 0) | (y_train >= 40)):
        raise ValueError('y_train labels must lie in [0, 39]')
    model = Pipeline([('scale', StandardScaler()), ('ridge', RidgeClassifier(alpha=float(alpha), class_weight=None, solver='lsqr', tol=1e-05, max_iter=5000))])
    model.fit(x_train, y_train, ridge__sample_weight=_sample_weights(y_train, float(power)))
    return model

def aligned_scores(model: Any, x: np.ndarray) -> np.ndarray:
    x = _finite_array(x, 'x', np.float32)
    if x.ndim != 2:
        raise ValueError('x must be 2-D')
    scores = np.asarray(model.decision_function(x), dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError('model produced nonfinite scores')
    ridge = model.named_steps.get('ridge', model) if hasattr(model, 'named_steps') else model
    classes = np.asarray(getattr(ridge, 'classes_', []))
    if classes.ndim != 1 or len(classes) == 0 or (not np.issubdtype(classes.dtype, np.number)) or (not np.isfinite(classes).all()) or np.any(classes != np.floor(classes)) or (len(np.unique(classes)) != len(classes)) or np.any((classes < 0) | (classes >= 40)):
        raise ValueError('model classes must be nonempty integers in [0,39]')
    classes = classes.astype(np.int64)
    if scores.ndim == 1:
        if len(classes) == 2:
            scores = np.column_stack((-scores, scores))
        elif len(classes) == 1:
            scores = scores[:, None]
        else:
            raise ValueError('1-D decision scores require one or two classes')
    if scores.ndim != 2 or scores.shape[0] != len(x) or scores.shape[1] != len(classes):
        raise ValueError('decision scores do not align with model classes')
    margin = np.maximum(np.ptp(scores, axis=1, keepdims=True), 1.0)
    floor = np.min(scores, axis=1, keepdims=True) - margin
    output = np.repeat(floor, 40, axis=1)
    output[:, classes] = scores
    return output

def softmax(scores: np.ndarray) -> np.ndarray:
    scores = _finite_array(scores, 'scores', np.float64)
    if scores.ndim != 2:
        raise ValueError('scores must be 2-D')
    return np.exp(scores - logsumexp(scores, axis=1, keepdims=True))

def fit_temperature(scores: np.ndarray, source_labels: np.ndarray) -> float:
    scores = _finite_array(scores, 'scores', np.float64)
    source_labels = np.asarray(source_labels)
    if source_labels.ndim != 1 or not np.issubdtype(source_labels.dtype, np.number) or (not np.isfinite(source_labels).all()) or np.any(source_labels != np.floor(source_labels)):
        raise ValueError('source_labels must contain finite integer class IDs')
    source_labels = source_labels.astype(np.int64)
    if scores.ndim != 2 or len(scores) != len(source_labels) or len(scores) == 0:
        raise ValueError('scores and source_labels must be nonempty and aligned')
    if np.any((source_labels < 0) | (source_labels >= scores.shape[1])):
        raise ValueError('source_labels outside score columns')

    def objective(log_temperature: float) -> float:
        scaled = scores / np.exp(log_temperature)
        return float(-np.mean(scaled[np.arange(len(source_labels)), source_labels] - logsumexp(scaled, axis=1)))
    result = minimize_scalar(objective, bounds=(-2.302585, 2.302585), method='bounded')
    if not result.success or not np.isfinite(result.x):
        raise RuntimeError(f'temperature calibration failed: {result.message}')
    return float(np.exp(result.x))
