"""Phase 2 models — 10-class digit classification.

Mirrors the implementations from `phase_2/*.ipynb` so the demo runs the
multi-class versions of the project's own models.
"""

import numpy as np

from demo.models_phase1 import _DTNode, DecisionTree as _BaseDT


class DecisionTreeMulti(_BaseDT):
    """The Phase 1 DecisionTree class is class-agnostic (uses np.unique on y),
    so it works directly for multi-class. We keep a thin alias for clarity."""

    def _traverse_proba(self, x, node):
        if node.label is not None:
            p = np.zeros(10)
            p[int(node.label)] = 1.0
            return p
        if x[node.feature] < node.threshold:
            return self._traverse_proba(x, node.left)
        return self._traverse_proba(x, node.right)


class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples=5, n_features=None, class_weights=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.n_features = n_features
        self.class_weights = class_weights
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        for _ in range(self.n_trees):
            idx = np.random.choice(n_samples, n_samples, replace=True)
            tree = DecisionTreeMulti(
                max_depth=self.max_depth,
                min_samples=self.min_samples,
                n_features=self.n_features,
                class_weights=self.class_weights,
            )
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)

    def predict(self, X):
        all_preds = np.array([t.predict(X) for t in self.trees])
        out = []
        for i in range(X.shape[0]):
            classes, counts = np.unique(all_preds[:, i], return_counts=True)
            out.append(classes[np.argmax(counts)])
        return np.array(out)


class GaussianNaiveBayesMulti:
    def __init__(self, var_smoothing=1e-2):
        self.var_smoothing = var_smoothing
        self.means = None
        self.variances = None
        self.log_priors = None
        self.classes = None

    def fit(self, X, y, class_weights=None):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        self.log_priors = np.zeros(n_classes)
        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))
        for i, c in enumerate(self.classes):
            prior = np.sum(y == c) / n_samples
            if class_weights is not None:
                prior *= class_weights.get(int(c), 1.0)
            self.log_priors[i] = np.log(prior + 1e-12)
            X_c = X[y == c]
            self.means[i] = X_c.mean(axis=0)
            self.variances[i] = X_c.var(axis=0)
        self.variances += self.var_smoothing * np.max(self.variances)

    def _ll(self, X, idx):
        mean, var = self.means[idx], self.variances[idx]
        log_norm = -0.5 * np.log(2 * np.pi * var)
        log_gauss = -0.5 * ((X - mean) ** 2) / var
        return (log_norm + log_gauss).sum(axis=1)

    def predict(self, X):
        log_post = np.array([self.log_priors[i] + self._ll(X, i) for i in range(len(self.classes))])
        return self.classes[np.argmax(log_post, axis=0)]

    def predict_proba(self, X):
        log_post = np.array([self.log_priors[i] + self._ll(X, i) for i in range(len(self.classes))]).T
        log_post -= log_post.max(axis=1, keepdims=True)
        p = np.exp(log_post)
        return p / p.sum(axis=1, keepdims=True)


class _BinarySVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=50):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        y_signed = np.where(y <= 0, -1, 1)
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                y_i = y_signed[idx]
                f_i = np.dot(self.w, x_i) + self.b
                if f_i * y_i < 1:
                    self.w = self.w - self.lr * (self.lambda_param * self.w - y_i * x_i)
                    self.b = self.b + self.lr * y_i
                else:
                    self.w = self.w - self.lr * self.lambda_param * self.w


class OneVsRestSVM:
    def __init__(self, n_classes=10, learning_rate=0.001, lambda_param=0.01, n_iters=30):
        self.n_classes = n_classes
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.models = []

    def fit(self, X, y):
        self.models = []
        for i in range(self.n_classes):
            y_bin = np.where(y == i, 1, 0)
            m = _BinarySVM(self.lr, self.lambda_param, self.n_iters)
            m.fit(X, y_bin)
            self.models.append(m)

    def decision_function(self, X):
        scores = np.zeros((X.shape[0], self.n_classes))
        for i, m in enumerate(self.models):
            scores[:, i] = X @ m.w + m.b
        return scores

    def predict(self, X):
        return np.argmax(self.decision_function(X), axis=1)


class KNNMulti:
    """Multi-class KNN — mirrors the Phase 2 notebook implementation."""

    def __init__(self, k=3, metric="euclidean", weights="distance"):
        self.k = k
        self.metric = metric
        self.weights = weights

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

    def _compute_distances(self, X):
        if self.metric == "euclidean":
            a2 = np.sum(X ** 2, axis=1, keepdims=True)
            b2 = np.sum(self.X_train ** 2, axis=1, keepdims=True)
            return np.sqrt(np.maximum(a2 + b2.T - 2 * X @ self.X_train.T, 0))
        elif self.metric == "manhattan":
            return np.array([np.sum(np.abs(self.X_train - x), axis=1) for x in X])
        else:
            raise ValueError(f"Metric '{self.metric}' not supported")

    def predict(self, X):
        X = np.asarray(X)
        dists = self._compute_distances(X)
        k_idx = np.argsort(dists, axis=1)[:, : self.k]
        predictions = []
        for i, neighbors in enumerate(k_idx):
            labels = self.y_train[neighbors]
            w = np.ones(self.k) if self.weights == "uniform" else 1 / (dists[i, neighbors] + 1e-9)
            votes = {}
            for label, wi in zip(labels, w):
                votes[label] = votes.get(label, 0) + wi
            predictions.append(max(votes, key=votes.get))
        return np.array(predictions)

    def predict_proba(self, X):
        X = np.asarray(X)
        dists = self._compute_distances(X)
        k_idx = np.argsort(dists, axis=1)[:, : self.k]
        n_classes = 10
        probs = []
        for i, neighbors in enumerate(k_idx):
            labels = self.y_train[neighbors]
            w = np.ones(self.k) if self.weights == "uniform" else 1 / (dists[i, neighbors] + 1e-9)
            votes = np.zeros(n_classes)
            for label, wi in zip(labels, w):
                votes[int(label)] += wi
            probs.append(votes / votes.sum())
        return np.array(probs)


class MultinomialRegression:
    """Softmax regression for 10-class digit classification — mirrors the Phase 2 notebook."""

    def __init__(self, lr=0.5, iterations=1000):
        self.lr = lr
        self.iterations = iterations
        self.W = None
        self.b = None

    @staticmethod
    def _softmax(z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    @staticmethod
    def _one_hot(y, n_classes):
        return np.eye(n_classes)[np.asarray(y, dtype=int)]

    def fit(self, X, y, class_weights=None):
        n_samples, n_features = X.shape
        n_classes = 10
        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros(n_classes)
        y_oh = self._one_hot(y, n_classes)
        for _ in range(self.iterations):
            y_hat = self._softmax(X @ self.W + self.b)
            dz = y_hat - y_oh
            self.W -= self.lr * (X.T @ dz) / n_samples
            self.b -= self.lr * np.sum(dz, axis=0) / n_samples

    def predict(self, X):
        return np.argmax(self._softmax(X @ self.W + self.b), axis=1)

    def predict_proba(self, X):
        return self._softmax(X @ self.W + self.b)
