import keras
import numpy as np


class CustomPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None 
        self.mean = None       

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        N = X.shape[0]
        covariance_matrix = np.dot(X_centered.T, X_centered) / (N - 1)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
    
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        X_centered = X - self.mean
        X_reduced = np.dot(X_centered, self.components)
        return X_reduced

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def get_class_weights(y):
    counts = np.bincount(y) 
    n_samples = len(y)
    n_classes = len(counts)
    
    weights = n_samples / (n_classes * counts)
    
    return {0: weights[0], 1: weights[1]}


def preprocess(feature_method="flatten", n_pca=50):
    print("Loading MNIST dataset...")
    # Load the raw dataset
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz")

    # Label Conversion (0 or Not 0)
    y_train_full = np.where(y_train_full == 0, 0, 1)
    y_test = np.where(y_test == 0, 0, 1)

    # Normalization 
    X_train_full = X_train_full.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Train/Validation Split 10%
    split_idx = 54000 
    X_train = X_train_full[:split_idx]
    y_train = y_train_full[:split_idx]
    
    X_val = X_train_full[split_idx:]
    y_val = y_train_full[split_idx:]
    
    print(f"Split completed: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    weights = get_class_weights(y_train)

    # Feature Extraction 
    if feature_method == "pca":
        X_train_flat = X_train.reshape(-1, 784)
        X_val_flat = X_val.reshape(-1, 784)
        X_test_flat = X_test.reshape(-1, 784)
        
        pca = CustomPCA(n_components=n_pca)
        
        X_train_final = pca.fit_transform(X_train_flat)
        
        X_val_final = pca.transform(X_val_flat)
        X_test_final = pca.transform(X_test_flat)
        
    elif feature_method == "flatten":
        X_train_final = X_train.reshape(-1, 784)
        X_val_final = X_val.reshape(-1, 784)
        X_test_final = X_test.reshape(-1, 784)

    return X_train_final, y_train, X_val_final, y_val, X_test_final, y_test, weights
