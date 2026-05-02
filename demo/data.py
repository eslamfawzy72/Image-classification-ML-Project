"""MNIST loader + feature extraction pipelines for the demo.

Wraps the project's existing preprocessing helpers. Phase 1 produces a
binary label (0 vs not-0); Phase 2 keeps the original 0-9 labels.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from urllib.request import urlretrieve

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "Models")
if _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)

# Reuse the project's own PCA + HOG implementations, but pull them in
# without dragging in the heavy `keras` import that lives at the top of
# Models/preprocessing.py. We reach into the module dict directly.
import importlib.util as _ilu

_pp_spec = _ilu.spec_from_file_location(
    "_project_preprocessing", os.path.join(_MODELS_DIR, "preprocessing.py")
)


def _load_project_preprocessing():
    # Lazy: only imports at first use, and tolerates a missing keras
    # because the symbols we need (CustomPCA, CustomHOG, get_class_weights)
    # don't depend on it.
    try:
        mod = _ilu.module_from_spec(_pp_spec)
        _pp_spec.loader.exec_module(mod)
        return mod.CustomPCA, mod.CustomHOG, mod.get_class_weights
    except Exception:
        # Fall back to local re-implementation (same algorithms).
        return _LocalPCA, _LocalHOG, _local_class_weights


class _LocalPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        Xc = X - self.mean
        cov = (Xc.T @ Xc) / (X.shape[0] - 1)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        self.components = vecs[:, order][:, : self.n_components]

    def transform(self, X):
        return (X - self.mean) @ self.components

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class _LocalHOG:
    def __init__(self, pixels_per_cell=(7, 7), cells_per_block=(2, 2), num_bins=9):
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.num_bins = num_bins

    def _gradients(self, image):
        p = np.pad(image, 1, mode="constant")
        gx = p[1:-1, 2:] - p[1:-1, :-2]
        gy = p[2:, 1:-1] - p[:-2, 1:-1]
        mag = np.sqrt(gx ** 2 + gy ** 2)
        ang = np.rad2deg(np.arctan2(gy, gx)) % 180
        return mag, ang

    def _cell_hists(self, mag, ang):
        h, w = mag.shape
        ch, cw = self.pixels_per_cell
        cy, cx = h // ch, w // cw
        out = np.zeros((cy, cx, self.num_bins))
        for y in range(cy):
            for x in range(cx):
                ys, xs = y * ch, x * cw
                m = mag[ys : ys + ch, xs : xs + cw].flatten()
                a = ang[ys : ys + ch, xs : xs + cw].flatten()
                out[y, x] = np.histogram(a, bins=self.num_bins, range=(0, 180), weights=m)[0]
        return out

    def _normalize(self, hists):
        cy, cx, _ = hists.shape
        by, bx = self.cells_per_block
        blocks = []
        for y in range(cy - by + 1):
            for x in range(cx - bx + 1):
                v = hists[y : y + by, x : x + bx, :].flatten()
                blocks.append(v / np.sqrt((v ** 2).sum() + 1e-5))
        return np.concatenate(blocks)

    def extract_single(self, image):
        m, a = self._gradients(image)
        return self._normalize(self._cell_hists(m, a))

    def transform(self, X):
        return np.array([self.extract_single(x) for x in X])


def _local_class_weights(y):
    counts = np.bincount(y)
    n = len(y)
    n_classes = len(counts)
    w = n / (n_classes * counts)
    return {0: w[0], 1: w[1]}


CustomPCA, CustomHOG, get_class_weights = _load_project_preprocessing()


_MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def _mnist_npz_path() -> str:
    cache_dir = os.path.join(_THIS_DIR, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "mnist.npz")


@lru_cache(maxsize=1)
def _load_mnist_raw():
    """Load MNIST without requiring tensorflow/keras.

    Tries (in order): keras → local cached .npz → download from the public
    Keras MNIST URL. The .npz layout matches what keras.datasets.mnist
    expects, so all three paths give identical arrays.
    """
    try:
        import keras  # type: ignore

        (Xtr, ytr), (Xte, yte) = keras.datasets.mnist.load_data(path="mnist.npz")
    except Exception:
        path = _mnist_npz_path()
        if not os.path.exists(path):
            urlretrieve(_MNIST_URL, path)
        with np.load(path, allow_pickle=True) as f:
            Xtr, ytr = f["x_train"], f["y_train"]
            Xte, yte = f["x_test"], f["y_test"]

    Xtr = Xtr.astype("float32") / 255.0
    Xte = Xte.astype("float32") / 255.0
    return Xtr, ytr, Xte, yte


def get_test_images(n: int = 200):
    """Return the first `n` raw MNIST test images (28x28) and labels.

    The UI lets the user pick a sample by index from this slice.
    """
    _, _, X_test, y_test = _load_mnist_raw()
    return X_test[:n], y_test[:n]


def _flatten(X):
    return X.reshape(-1, 784)


def _hog_transform(X):
    return CustomHOG().transform(X)


@lru_cache(maxsize=8)
def _build_dataset(phase: str, feature_method: str, n_pca: int = 50, train_subset: int = 4000):
    """Build a (small) train set + matching transform for fast demo training.

    Returns: dict with X_train, y_train, weights, transform_image (callable
    mapping a raw 28x28 image -> feature vector matching X_train rows).
    """
    X_train_full, y_train_full, X_test, y_test = _load_mnist_raw()

    if phase == "phase1":
        # Phase 1: roughly balance digit-0 vs not-0 to mirror the notebooks.
        zero_idx = np.where(y_train_full == 0)[0]
        n_zeros = len(zero_idx)
        per_class = max(1, n_zeros // 9)
        not_zero_idx = []
        for d in range(1, 10):
            di = np.where(y_train_full == d)[0]
            not_zero_idx.extend(di[:per_class])
        not_zero_idx = np.array(not_zero_idx)
        balanced = np.concatenate([zero_idx, not_zero_idx])
        rng = np.random.default_rng(42)
        rng.shuffle(balanced)
        X = X_train_full[balanced]
        y = np.where(y_train_full[balanced] == 0, 0, 1)
    else:
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(X_train_full))
        X = X_train_full[idx]
        y = y_train_full[idx]

    if train_subset and train_subset < len(X):
        X = X[:train_subset]
        y = y[:train_subset]

    if feature_method == "flatten":
        X_feat = _flatten(X)

        def transform_image(img: np.ndarray) -> np.ndarray:
            return img.reshape(1, -1)

    elif feature_method == "pca":
        X_flat = _flatten(X)
        pca = CustomPCA(n_components=n_pca)
        X_feat = pca.fit_transform(X_flat)

        def transform_image(img: np.ndarray) -> np.ndarray:
            return pca.transform(img.reshape(1, -1))

    elif feature_method == "hog":
        hog = CustomHOG()
        X_feat = hog.transform(X)

        def transform_image(img: np.ndarray) -> np.ndarray:
            return hog.transform(img[np.newaxis, ...])

    else:
        raise ValueError(f"Unknown feature_method: {feature_method}")

    weights = get_class_weights(y) if phase == "phase1" else None

    return {
        "X_train": X_feat,
        "y_train": y,
        "weights": weights,
        "transform_image": transform_image,
        "feature_dim": X_feat.shape[1],
    }


def build_dataset(phase: str, feature_method: str, train_subset: int = 4000, n_pca: int = 50):
    return _build_dataset(phase, feature_method, n_pca, train_subset)
