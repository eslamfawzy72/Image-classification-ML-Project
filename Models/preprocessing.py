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

class CustomHOG:
    
    def __init__(self, pixels_per_cell=(7, 7), cells_per_block=(2, 2), num_bins=9):
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.num_bins = num_bins

    def compute_gradients(self,image):
    
        padded_img = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    
        Gx = padded_img[1:-1, 2:] - padded_img[1:-1, :-2]
        
    
        Gy = padded_img[2:, 1:-1] - padded_img[:-2, 1:-1]
        
        magnitude = np.sqrt(Gx**2 + Gy**2)
        
        
        angle_radians = np.arctan2(Gy, Gx)
        angle_degrees = np.rad2deg(angle_radians)
    
        angle_unsigned = angle_degrees % 180
        
        return magnitude, angle_unsigned


    def compute_cell_histograms(self,magnitude, angle):
        
        h, w = magnitude.shape
        c_h, c_w = self.pixels_per_cell
        cells_y = h // c_h
        cells_x = w // c_w
        
        histograms = np.zeros((cells_y, cells_x,self. num_bins))
        
        for y in range(cells_y):
            for x in range(cells_x):
                y_start = y * c_h
                y_end = y_start + c_h
                x_start = x * c_w
                x_end = x_start + c_w
                
                cell_magnitude = magnitude[y_start:y_end, x_start:x_end]
                cell_angle = angle[y_start:y_end, x_start:x_end]
            
                hist, _ = np.histogram(
                    cell_angle.flatten(), 
                    bins=self.num_bins, 
                    range=(0, 180), 
                    weights=cell_magnitude.flatten()
                )
                
                histograms[y, x, :] = hist
                
        return histograms
    

    def normalize_blocks(self, histograms):
        cells_y, cells_x, _ = histograms.shape
        b_y, b_x = self.cells_per_block
        
        blocks_y = cells_y - b_y + 1
        blocks_x = cells_x - b_x + 1
        normalized_blocks = []
        
        for y in range(blocks_y):
            for x in range(blocks_x):
                block = histograms[y:y+b_y, x:x+b_x, :]
                block_vector = block.flatten()
                
                epsilon = 1e-5 
                norm = np.sqrt(np.sum(block_vector**2) + epsilon)
                
                normalized_blocks.append(block_vector / norm)
                
        return np.concatenate(normalized_blocks)

    def extract_single(self, image):
        mag, ang = self.compute_gradients(image)
        hists = self.compute_cell_histograms(mag, ang)
        feature_vector = self.normalize_blocks(hists)
        return feature_vector

    def transform(self, X_images):        
        features = [self.extract_single(img) for img in X_images]
        return np.array(features)


def preprocess(feature_method="flatten", n_pca=50, balance=True):
    print("Loading MNIST dataset...")
    # Load the raw dataset
    (X_train_full, y_train_full_raw), (X_test, y_test_raw) = keras.datasets.mnist.load_data(path="mnist.npz")

    # Normalization 
    X_train_full = X_train_full.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    if balance==True:
        zero_indices = np.where(y_train_full_raw == 0)[0]
        n_zeros = len(zero_indices) 
        
        samples_per_class = n_zeros // 9  
        
        not_zero_indices = []
        for digit in range(1, 10):
            digit_idx = np.where(y_train_full_raw == digit)[0]
            not_zero_indices.extend(digit_idx[:samples_per_class])
            
        not_zero_indices = np.array(not_zero_indices)
        
        balanced_indices = np.concatenate([zero_indices, not_zero_indices])
        
        np.random.seed(42)
        np.random.shuffle(balanced_indices)
        
        X_balanced = X_train_full[balanced_indices]
        y_balanced_raw = y_train_full_raw[balanced_indices]
        
        y_balanced_binary = np.where(y_balanced_raw == 0, 0, 1)
        
        split_idx = int(len(X_balanced) * 0.9)
        X_train = X_balanced[:split_idx]
        y_train = y_balanced_binary[:split_idx]
        X_val = X_balanced[split_idx:]
        y_val = y_balanced_binary[split_idx:]



    else:
    
        y_train_full_binary = np.where(y_train_full_raw == 0, 0, 1)
                
        # Train/Validation Split 10%
        split_idx = 54000 
        X_train = X_train_full[:split_idx]
        y_train = y_train_full_binary[:split_idx]
        
        X_val = X_train_full[split_idx:]
        y_val = y_train_full_binary[split_idx:]
        
    y_test = np.where(y_test_raw == 0, 0, 1)

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
    
    elif feature_method == "hog":
        hog = CustomHOG()
        X_train_final = hog.transform(X_train)
        X_val_final = hog.transform(X_val)
        X_test_final = hog.transform(X_test)

    elif feature_method == "hog_pca":
        hog = CustomHOG()
        X_train_hog = hog.transform(X_train)
        X_val_hog = hog.transform(X_val)
        X_test_hog = hog.transform(X_test)

        pca = CustomPCA(n_components=n_pca)
        X_train_final = pca.fit_transform(X_train_hog)
        X_val_final = pca.transform(X_val_hog)
        X_test_final = pca.transform(X_test_hog)

    else:
        raise ValueError("Invalid feature_method. Choose 'flatten', 'pca', 'hog', or 'hog_pca'.")

    return X_train_final, y_train, X_val_final, y_val, X_test_final, y_test, weights



def k_fold_indices(X, k=3):
    n_samples = len(X)
    indices = np.arange(n_samples)

    np.random.seed(42)
    np.random.shuffle(indices)

    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1

    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_idx, val_idx))
        current = stop

    return folds


def custom_confusion_matrix(y_true, y_pred, n_classes=None):
    
    if n_classes is None:
        n_classes = len(np.unique(y_true))
        
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for true_label, pred_label in zip(y_true, y_pred):
        t = int(true_label)
        p = int(pred_label)
        matrix[t, p] += 1
        
    return matrix

def custom_classification_report(y_true, y_pred, target_names=None):
    
    n_classes = len(np.unique(y_true))
    cm = custom_confusion_matrix(y_true, y_pred, n_classes)
    
    if target_names is None:
        target_names = [f"Class {i}" for i in range(n_classes)]
        
    report = f"{'':<15} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}\n\n"
    
    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0
    total_support = 0
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        support = np.sum(cm[i, :])
        
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        
        report += f"{target_names[i]:<15} {precision:>10.2f} {recall:>10.2f} {f1:>10.2f} {support:>10}\n"
        
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        total_support += support

    total_tp = np.trace(cm) 
    total_samples = np.sum(cm)
    accuracy = total_tp / total_samples

    # Calculate final averages
    macro_precision /= n_classes
    macro_recall /= n_classes
    macro_f1 /= n_classes

    # Append bottom summary to report
    report += f"\n{'accuracy':<15} {'':>10} {'':>10} {accuracy:>10.2f} {total_support:>10}\n"
    report += f"{'macro avg':<15} {macro_precision:>10.2f} {macro_recall:>10.2f} {macro_f1:>10.2f} {total_support:>10}\n"
    
    return report

def custom_accuracy_score(y_true, y_pred):
    correct = np.sum(np.array(y_true) == np.array(y_pred))
    total = len(y_true)
    return correct / total if total > 0 else 0.0