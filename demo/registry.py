"""Declarative registry of all demo-able models per phase."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

from demo import models_phase1 as p1
from demo import models_phase2 as p2


@dataclass
class ModelSpec:
    key: str
    label: str
    description: str
    builder: Callable[[], object]
    feature_methods: List[str]
    default_feature: str
    icon: str = "◆"
    train_subset: int = 4000
    needs_class_weights: bool = False


PHASE_LABELS = {
    "phase1": "Phase 1 — Binary (Is the digit a 0?)",
    "phase2": "Phase 2 — Multi-class (Digit 0-9)",
}

PHASE1_MODELS: Dict[str, ModelSpec] = {
    "knn": ModelSpec(
        key="knn",
        label="K-Nearest Neighbors",
        description="Distance-based vote across the K closest training samples.",
        builder=lambda: p1.KNN(k=3, weights="uniform", metric="euclidean"),
        feature_methods=["pca", "hog", "flatten"],
        default_feature="pca",
        icon="◉",
        train_subset=2000,
    ),
    "logistic": ModelSpec(
        key="logistic",
        label="Logistic Regression",
        description="Linear model with sigmoid output, trained via gradient descent.",
        builder=lambda: p1.LogisticRegression(lr=0.1, iterations=400),
        feature_methods=["flatten", "pca", "hog"],
        default_feature="flatten",
        icon="∫",
        needs_class_weights=True,
    ),
    "naive_bayes": ModelSpec(
        key="naive_bayes",
        label="Gaussian Naive Bayes",
        description="Per-class Gaussian likelihoods with weighted priors.",
        builder=lambda: p1.GaussianNaiveBayesBinary(),
        feature_methods=["pca", "hog", "flatten"],
        default_feature="pca",
        icon="∿",
        needs_class_weights=True,
    ),
    "tree": ModelSpec(
        key="tree",
        label="Decision Tree",
        description="Greedy Gini splits with class-weighted majority labels.",
        builder=lambda: p1.DecisionTree(max_depth=10, min_samples=10),
        feature_methods=["pca", "hog", "flatten"],
        default_feature="pca",
        icon="⎇",
        needs_class_weights=True,
        train_subset=2500,
    ),
    "svm": ModelSpec(
        key="svm",
        label="Linear SVM",
        description="Hinge-loss linear SVM trained with sub-gradient descent.",
        builder=lambda: p1.LinearSVMBinary(learning_rate=0.001, lambda_param=0.01, n_iters=15),
        feature_methods=["pca", "hog", "flatten"],
        default_feature="pca",
        icon="╱",
        train_subset=2500,
    ),
}

PHASE2_MODELS: Dict[str, ModelSpec] = {
    "tree": ModelSpec(
        key="tree",
        label="Decision Tree",
        description="Multi-class Gini decision tree.",
        builder=lambda: p2.DecisionTreeMulti(max_depth=12, min_samples=10),
        feature_methods=["pca", "hog", "flatten", "cnn"],
        default_feature="hog",
        icon="⎇",
        train_subset=3000,
    ),
    "random_forest": ModelSpec(
        key="random_forest",
        label="Random Forest",
        description="Bagged ensemble of decision trees with majority vote.",
        builder=lambda: p2.RandomForest(n_trees=10, max_depth=12, min_samples=10),
        feature_methods=["pca", "hog", "flatten", "cnn"],
        default_feature="hog",
        icon="🌲",
        train_subset=2500,
    ),
    "naive_bayes": ModelSpec(
        key="naive_bayes",
        label="Gaussian Naive Bayes",
        description="10-class Gaussian NB with variance smoothing.",
        builder=lambda: p2.GaussianNaiveBayesMulti(var_smoothing=1e-2),
        feature_methods=["hog", "pca", "flatten", "cnn"],
        default_feature="hog",
        icon="∿",
        train_subset=4000,
    ),
    "svm": ModelSpec(
        key="svm",
        label="One-vs-Rest Linear SVM",
        description="Ten binary SVMs combined via highest-score voting.",
        builder=lambda: p2.OneVsRestSVM(n_classes=10, learning_rate=0.001, lambda_param=0.01, n_iters=15),
        feature_methods=["hog", "pca", "flatten", "cnn"],
        default_feature="hog",
        icon="╱",
        train_subset=2500,
    ),
    "knn": ModelSpec(
        key="knn",
        label="K-Nearest Neighbors",
        description="Distance-weighted vote across the K closest training samples (multi-class).",
        builder=lambda: p2.KNNMulti(k=3, metric="euclidean", weights="distance"),
        feature_methods=["pca", "hog", "flatten", "cnn"],
        default_feature="pca",
        icon="◉",
        train_subset=2000,
    ),
    "multinomial": ModelSpec(
        key="multinomial",
        label="Multinomial Regression",
        description="Softmax regression with gradient descent over 10 digit classes.",
        builder=lambda: p2.MultinomialRegression(lr=0.5, iterations=500),
        feature_methods=["hog", "pca", "flatten", "cnn"],
        default_feature="hog",
        icon="∫",
    ),
}


def models_for(phase: str) -> Dict[str, ModelSpec]:
    return PHASE1_MODELS if phase == "phase1" else PHASE2_MODELS


def get_spec(phase: str, model_key: str) -> ModelSpec:
    return models_for(phase)[model_key]


FEATURE_LABELS = {
    "flatten": "Raw pixels (flatten)",
    "pca": "PCA (50 components)",
    "hog": "HOG features",
    "cnn": "CNN embeddings (MobileNetV2)",
}
