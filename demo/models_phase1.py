"""Phase 1 models — binary classification (digit 0 vs not 0).

Each class mirrors the implementation used in the Phase 1 notebooks under
`Models/` so the demo runs the students' own code, not a sklearn substitute.
"""

import numpy as np


class KNN:
    def __init__(self, weights="uniform", k=3, metric="euclidean"):
        self.k = k
        self.weights = weights
        self.metric = metric

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

    def _compute_distances(self, X):
        if self.metric == "euclidean":
            a2 = np.sum(X ** 2, axis=1, keepdims=True)
            b2 = np.sum(self.X_train ** 2, axis=1, keepdims=True)
            dists = np.sqrt(np.maximum(a2 + b2.T - 2 * X @ self.X_train.T, 0))
        elif self.metric == "manhattan":
            dists = np.array([np.sum(np.abs(self.X_train - x), axis=1) for x in X])
        elif self.metric == "cosine":
            X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
            T_norm = self.X_train / (np.linalg.norm(self.X_train, axis=1, keepdims=True) + 1e-9)
            dists = 1 - X_norm @ T_norm.T
        else:
            raise ValueError(f"Metric '{self.metric}' not supported")
        return dists

    def predict(self, X):
        X = np.asarray(X)
        dists = self._compute_distances(X)
        k_idx = np.argsort(dists, axis=1)[:, : self.k]
        predictions = []
        for i, neighbors in enumerate(k_idx):
            labels = self.y_train[neighbors]
            if self.weights == "uniform":
                w = np.ones(self.k)
            else:
                w = 1 / (dists[i, neighbors] + 1e-9)
            votes = {}
            for label, wi in zip(labels, w):
                votes[label] = votes.get(label, 0) + wi
            predictions.append(max(votes, key=votes.get))
        return np.array(predictions)


class LogisticRegression:
    def __init__(self, lr=0.1, iterations=400):
        self.lr = lr
        self.iterations = iterations
        self.w = None
        self.b = 0.0

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y, class_weights=None):
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0.0
        weights = class_weights or {0: 1.0, 1: 1.0}
        for _ in range(self.iterations):
            z = X @ self.w + self.b
            y_hat = self._sigmoid(z)
            sample_weights = np.where(y == 1, weights[1], weights[0])
            error = (y_hat - y) * sample_weights
            dw = (X.T @ error) / len(y)
            db = np.sum(error) / len(y)
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        z = X @ self.w + self.b
        return (self._sigmoid(z) >= 0.5).astype(int)

    def predict_proba(self, X):
        z = X @ self.w + self.b
        p = self._sigmoid(z)
        return np.stack([1 - p, p], axis=1)


class GaussianNaiveBayesBinary:
    def __init__(self):
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
                prior *= class_weights[c]
            self.log_priors[i] = np.log(prior + 1e-12)
            X_c = X[y == c]
            if class_weights is not None:
                w = np.full(len(X_c), class_weights[c])
                w = w / w.sum()
                self.means[i] = (w[:, None] * X_c).sum(axis=0)
                self.variances[i] = (w[:, None] * (X_c - self.means[i]) ** 2).sum(axis=0)
            else:
                self.means[i] = X_c.mean(axis=0)
                self.variances[i] = X_c.var(axis=0)
        self.variances += 1e-9

    def _log_likelihood(self, X, idx):
        mean = self.means[idx]
        var = self.variances[idx]
        log_norm = -0.5 * np.log(2 * np.pi * var)
        log_gauss = -0.5 * ((X - mean) ** 2) / var
        return (log_norm + log_gauss).sum(axis=1)

    def predict(self, X):
        log_post = np.array([
            self.log_priors[i] + self._log_likelihood(X, i)
            for i in range(len(self.classes))
        ])
        return self.classes[np.argmax(log_post, axis=0)]

    def predict_proba(self, X):
        log_post = np.array([
            self.log_priors[i] + self._log_likelihood(X, i)
            for i in range(len(self.classes))
        ]).T
        log_post -= log_post.max(axis=1, keepdims=True)
        post = np.exp(log_post)
        return post / post.sum(axis=1, keepdims=True)


class _DTNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label


class DecisionTree:
    def __init__(self, max_depth=10, min_samples=5, n_features=None, class_weights=None):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.n_features = n_features
        self.class_weights = class_weights
        self.root = None

    def fit(self, X, y):
        self.n_features_total = X.shape[1]
        self.n_features = self.n_features or self.n_features_total
        self.root = self._build(X, y, depth=0)

    def _build(self, X, y, depth):
        if depth >= self.max_depth or X.shape[0] < self.min_samples or len(np.unique(y)) == 1:
            return _DTNode(label=self._majority(y))
        feats = np.random.choice(self.n_features_total, self.n_features, replace=False)
        best_f, best_t = self._best_split(X, y, feats)
        if best_f is None:
            return _DTNode(label=self._majority(y))
        left_mask = X[:, best_f] < best_t
        right_mask = ~left_mask
        return _DTNode(
            best_f,
            best_t,
            self._build(X[left_mask], y[left_mask], depth + 1),
            self._build(X[right_mask], y[right_mask], depth + 1),
        )

    def _best_split(self, X, y, feats):
        best_gini, best_f, best_t = float("inf"), None, None
        for f in feats:
            col = X[:, f]
            ts = np.unique(col)
            step = max(1, len(ts) // 10)
            for t in ts[::step]:
                gini = self._gini(y, col, t)
                if gini < best_gini:
                    best_gini, best_f, best_t = gini, f, t
        return best_f, best_t

    def _gini(self, y, col, t):
        left, right = y[col < t], y[col >= t]
        if len(left) == 0 or len(right) == 0:
            return float("inf")

        def wg(group):
            classes, counts = np.unique(group, return_counts=True)
            if self.class_weights is not None:
                counts = np.array([counts[i] * self.class_weights.get(classes[i], 1) for i in range(len(classes))])
            probs = counts / counts.sum()
            return 1 - np.sum(probs ** 2)

        n = len(y)
        return (len(left) / n) * wg(left) + (len(right) / n) * wg(right)

    def _majority(self, y):
        classes, counts = np.unique(y, return_counts=True)
        if self.class_weights is not None:
            counts = np.array([counts[i] * self.class_weights.get(classes[i], 1) for i in range(len(classes))])
        return classes[np.argmax(counts)]

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.label is not None:
            return node.label
        if x[node.feature] < node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)


class LinearSVMBinary:
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

    def predict(self, X):
        return np.where(np.dot(X, self.w) + self.b >= 0, 1, 0)

    def decision_function(self, X):
        return np.dot(X, self.w) + self.b
