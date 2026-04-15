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
    
    elif feature_method == "hog":
        hog = CustomHOG()
        X_train_final = hog.transform(X_train)
        X_val_final = hog.transform(X_val)
        X_test_final = hog.transform(X_test)
        
    else:
        raise ValueError("Invalid feature_method. Choose 'flatten', 'pca', or 'hog'.")

    return X_train_final, y_train, X_val_final, y_val, X_test_final, y_test, weights
