"""
t-SNE Explorer - Desktop Application
A transparent t-SNE implementation with synthetic data generation and file upload support
"""

import os
import sys
import json
import base64
import threading
import time
import warnings
import numpy as np
import pandas as pd
import inspect
from io import BytesIO
from PIL import Image
import webview
from pathlib import Path
import urllib.request
import gzip
import shutil

try:
    from sklearn.manifold import TSNE as SklearnTSNE
except Exception:
    SklearnTSNE = None

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Global state
datasets = {}
current_operation = None
stop_flag = False

# Setup storage directory
def get_storage_dir():
    """Get or create the user-specific storage directory"""
    home = Path.home()
    storage_dir = home / '.t-sne'
    storage_dir.mkdir(exist_ok=True)

    # Create subdirectories
    (storage_dir / 'uploads').mkdir(exist_ok=True)
    (storage_dir / 'datasets').mkdir(exist_ok=True)
    (storage_dir / 'exports').mkdir(exist_ok=True)

    return storage_dir


def clean_for_json(obj):
    """Replace NaN and Inf values with None for JSON serialization"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, (np.ndarray)):
        # Replace NaN and Inf with 0
        arr = np.array(obj, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr.tolist()
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return obj
    else:
        return obj


class TSNEExplorer:
    """Backend API for pywebview"""

    def __init__(self):
        self.window = None
        self.current_dataset_id = None

    def __dir__(self):
        """Explicitly define which methods are exposed to JavaScript to prevent recursion errors"""
        return [
            'generate_simplex_points',
            'save_synthetic_dataset',
            'upload_csv',
            'upload_images',
            'compute_embeddings',
            'list_datasets',
            'prepare_csv_dataset',
            'run_tsne',
            'stop_tsne',
            'run_clustering',
            'export_results',
            'get_image_at_index',
            'set_window',
            'load_mnist'
        ]

    def set_window(self, window):
        self.window = window

    # ==================== Synthetic Data Generation ====================

    def generate_simplex_points(self, n, d, k, seed=42):
        """
        Generate n points in d dimensions with k distinct distance types.

        k = number of unique distance values between all pairs

        Optimal configurations (exact k distances):
        - k=1: Regular simplex with all equal distances (max n = d+1)
        - k=2: Regular pentagon (n=5 optimal)
        - k=3: Regular heptagon or hexagon+center (n=7 optimal)

        Note: For n larger than optimal configurations, approximate solutions
        are generated using lattice constructions. These may produce more
        than k distinct distances (actual_k > k).

        Examples:
        - k=1, n≤d+1: All distances exactly equal (PERFECT)
        - k=2, n=5: Pentagon with exactly 2 distances
        - k=2, n>5: Square/triangular lattice (approximate, actual_k≥2)
        """
        np.random.seed(seed)

        # Validate inputs
        max_distances = (n * (n - 1)) // 2
        if k > max_distances:
            return {
                'success': False,
                'error': f'Cannot create {k} distinct distances with only {n} points. '
                        f'Maximum possible is {max_distances} distinct distances.'
            }

        if k < 1:
            return {
                'success': False,
                'error': f'k must be at least 1 (you specified k={k}).'
            }

        # Special case: k=1 (all distances equal - regular simplex)
        if k == 1:
            if n > d + 1:
                return {
                    'success': False,
                    'error': f'For k=1 (equidistant points), maximum n is {d+1} in {d}D. '
                            f'Please reduce n to {d+1} or increase d to at least {n-1}.'
                }
            X = self._generate_regular_simplex(n, d)

        # General case: k > 1 (multiple distance types)
        else:
            X = self._generate_k_distance_set(n, d, k, seed)

        # Compute pairwise distances and count unique values
        distances = self._compute_pairwise_distances(X)
        unique_distances = np.unique(np.round(distances[distances > 0], decimals=6))

        return {
            'success': True,
            'points': X.tolist(),
            'n': n,
            'd': d,
            'k': k,
            'actual_k': len(unique_distances),
            'unique_distances': unique_distances.tolist(),
            'distances_min': float(np.min(distances[distances > 0])) if n > 1 else 0,
            'distances_mean': float(np.mean(distances[distances > 0])) if n > 1 else 0,
            'distances_max': float(np.max(distances)),
        }

    def _generate_regular_simplex(self, n, d):
        """
        Generate regular n-simplex with ALL equal pairwise distances = 1.

        Standard construction: start with (n-1)-dimensional unit cube vertices,
        center them, and scale.
        """
        if n == 1:
            return np.zeros((1, d))

        if n == 2:
            # Special case: two points on a line
            X = np.zeros((2, d))
            X[0, 0] = -0.5
            X[1, 0] = 0.5
            return X

        # General case: use standard simplex construction in n dimensions
        # Then project to (d) dimensions
        # This guarantees ALL pairwise distances are equal

        # Create vertices as unit vectors in R^n
        vertices = np.eye(n)

        # Center at origin
        vertices = vertices - np.mean(vertices, axis=0)

        # After centering, all pairwise distances are sqrt(2)
        # Scale to make all distances = 1
        vertices = vertices / np.sqrt(2)

        # Now project to d dimensions (keeping first d coordinates)
        if d >= n - 1:
            # We have enough dimensions - use first (n-1) which preserves distances
            X = vertices[:, :min(d, n)]
            if d > n:
                # Pad with zeros
                X = np.pad(X, ((0, 0), (0, d - n)), 'constant')
        else:
            # Project to lower dimensions (distances will not all be equal)
            X = vertices[:, :d]

        return X

    def _generate_k_distance_set(self, n, d, k, seed):
        """
        Generate points aiming for exactly k distinct pairwise distances.

        Exact constructions are used when possible (then `actual_k == k`);
        otherwise we fall back to a numerical approximation.

        Constructions used:
        - k=2: cross-polytope vertices (±e_i) in R^d (up to 2d points) or a
               regular pentagon (n=5) in 2D.
        - k>=3: k-dimensional hypercube vertices in {0,1}^k embedded into R^d
                (up to 2^k points).
        """
        np.random.seed(seed)

        if n <= 0 or d <= 0:
            return np.zeros((0, max(d, 0)))

        if n == 1:
            return np.zeros((1, d))

        # ---- Exact k=2 constructions ----
        if k == 2:
            # 2D exact: regular pentagon
            if d >= 2 and n == 5:
                return self._regular_ngon(n=5, d=d)

            # Cross-polytope in R^d gives exactly 2 distances for any n <= 2d
            if n <= 2 * d:
                return self._cross_polytope(n=n, d=d)

            return self._optimize_k_distance_set(n=n, d=d, k=k, seed=seed)

        # ---- Exact k>=3 constructions via k-cube ----
        if k >= 3 and d >= k and k <= 12 and n <= (2 ** k):
            return self._k_cube_k_distance_set(n=n, d=d, k=k)

        # Special small planar exact 3-distance sets
        if k == 3 and d >= 2 and n in (6, 7):
            return self._regular_ngon(n=n, d=d)

        return self._optimize_k_distance_set(n=n, d=d, k=k, seed=seed)

    def _regular_ngon(self, n, d):
        """Regular n-gon embedded in first 2 dimensions (remaining dims are 0)."""
        X = np.zeros((n, d))
        if d < 2:
            return X

        angles = np.linspace(0, 2 * np.pi, n + 1)[:-1]
        X[:, 0] = np.cos(angles)
        X[:, 1] = np.sin(angles)
        return X

    def _cross_polytope(self, n, d):
        """
        Cross polytope vertices (±e_i) in R^d.

        For any n <= 2d and n >= 2, this produces exactly two distances:
        - 2 between opposite points (e_i, -e_i)
        - sqrt(2) between all other pairs
        """
        X = np.zeros((n, d))
        if n == 1:
            return X

        point_idx = 0
        for i in range(d):
            if point_idx >= n:
                break
            X[point_idx, i] = 1.0
            point_idx += 1
            if point_idx >= n:
                break
            X[point_idx, i] = -1.0
            point_idx += 1

        return X

    def _k_cube_k_distance_set(self, n, d, k):
        """
        Construct a k-distance set using vertices of the k-dimensional hypercube in {0,1}^k.

        Pairwise Euclidean distances are exactly sqrt(h) for h in {1..k}, so the cube
        has at most k distinct distances. This selection guarantees all k distances
        appear via origin-v pairs when n >= k+1.
        """
        vertices = []
        seen = set()

        origin = tuple([0] * k)
        vertices.append(origin)
        seen.add(origin)

        # Include one vector of each Hamming weight 1..k (if we have room).
        for weight in range(1, k + 1):
            if len(vertices) >= n:
                break
            v = tuple([1] * weight + [0] * (k - weight))
            if v not in seen:
                vertices.append(v)
                seen.add(v)

        # Fill remaining points with additional cube vertices (lexicographic masks).
        for mask in range(1, 2 ** k):
            if len(vertices) >= n:
                break
            v = tuple((mask >> bit) & 1 for bit in range(k))
            if v in seen:
                continue
            vertices.append(v)
            seen.add(v)

        Xk = np.array(vertices[:n], dtype=float)

        # Embed into R^d and center for nicer display (centering preserves distances).
        X = np.zeros((n, d), dtype=float)
        X[:, :k] = Xk
        X = X - X.mean(axis=0, keepdims=True)
        return X

    def _optimize_k_distance_set(self, n, d, k, seed, n_iter=2000, lr=0.02):
        """
        Heuristic optimization to make pairwise distances cluster into k values.

        Used when an exact construction is not available for the given (n, d, k).
        """
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, d)) * 0.1

        if n < 2:
            return X

        D0 = self._compute_pairwise_distances(X)
        upper = D0[np.triu_indices(n, k=1)]
        if upper.size == 0:
            return X

        r_min = float(np.percentile(upper, 10))
        r_max = float(np.percentile(upper, 90))
        if r_max <= 1e-8:
            r_max = 1.0
        radii = np.linspace(max(r_min, 1e-3), max(r_max, 1e-3), k)

        # Full-matrix optimization becomes expensive for large n; switch to minibatches.
        use_minibatch = n > 150
        batch_size = min(5000, (n * (n - 1)) // 2) if use_minibatch else 0
        ema = 0.15

        for _ in range(n_iter):
            if use_minibatch:
                ii = rng.integers(0, n, size=batch_size)
                jj = rng.integers(0, n, size=batch_size)
                mask = ii != jj
                if not np.any(mask):
                    continue
                ii = ii[mask]
                jj = jj[mask]

                diff = X[ii] - X[jj]
                dist = np.sqrt(np.sum(diff * diff, axis=1))
                dist_safe = np.maximum(dist, 1e-12)

                assign = np.argmin(np.abs(dist[:, np.newaxis] - radii[np.newaxis, :]), axis=1)
                target = radii[assign]

                # EMA update keeps radii stable under noisy minibatches
                for m in range(k):
                    m_mask = assign == m
                    if np.any(m_mask):
                        radii[m] = (1 - ema) * radii[m] + ema * float(np.mean(dist[m_mask]))

                err = dist_safe - target
                coef = (2.0 * err / dist_safe)[:, np.newaxis]
                grad_pairs = coef * diff

                grad = np.zeros_like(X)
                np.add.at(grad, ii, grad_pairs)
                np.add.at(grad, jj, -grad_pairs)
            else:
                D = self._compute_pairwise_distances(X)
                iu, ju = np.triu_indices(n, k=1)
                dist = D[iu, ju]
                dist_safe = np.maximum(dist, 1e-12)

                assign = np.argmin(np.abs(dist[:, np.newaxis] - radii[np.newaxis, :]), axis=1)
                target = radii[assign]

                for m in range(k):
                    m_mask = assign == m
                    if np.any(m_mask):
                        radii[m] = float(np.mean(dist[m_mask]))

                err = dist_safe - target
                coef = (2.0 * err / dist_safe)[:, np.newaxis]
                diff = X[iu] - X[ju]
                grad_pairs = coef * diff

                grad = np.zeros_like(X)
                np.add.at(grad, iu, grad_pairs)
                np.add.at(grad, ju, -grad_pairs)

            grad += 1e-3 * X
            X = X - lr * grad
            X = X - X.mean(axis=0, keepdims=True)

        return X

    def _triangular_lattice(self, n, d):
        """
        Generate points from triangular lattice.
        Coordinates: (a + b/2, √3*b/2) for integers a, b
        Distances: √(a² + ab + b²)

        Uses hexagonal array H_s containing 3s²-3s+1 points
        determining at most s²-1 distinct distances.
        """
        X = np.zeros((n, d))

        # Determine hexagon size s to accommodate n points
        # Solve 3s² - 3s + 1 >= n for s
        s = int(np.ceil((3 + np.sqrt(9 + 12*(n-1))) / 6)) + 1

        # Generate hexagonal array centered at origin
        point_idx = 0
        for radius in range(s):
            if radius == 0:
                # Center point at origin
                X[point_idx, 0] = 0
                X[point_idx, 1] = 0
                point_idx += 1
                if point_idx >= n:
                    break
            else:
                # Hexagonal ring at distance 'radius'
                # Six directions from center
                for direction in range(6):
                    for step in range(radius):
                        if point_idx >= n:
                            break

                        # Compute lattice coordinates (a, b)
                        angle = direction * np.pi / 3
                        a = int(radius * np.cos(angle) - step * np.sin(angle))
                        b = int(radius * np.sin(angle) + step * np.cos(angle))

                        # Convert to Cartesian
                        X[point_idx, 0] = a + b / 2.0
                        if d > 1:
                            X[point_idx, 1] = b * np.sqrt(3) / 2.0
                        point_idx += 1

                    if point_idx >= n:
                        break

        return X

    def _compute_pairwise_distances(self, X):
        """Compute pairwise Euclidean distances"""
        n = X.shape[0]
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(X[i] - X[j])
                distances[i, j] = dist
                distances[j, i] = dist
        return distances

    # ==================== MNIST Dataset ====================

    def load_mnist(self, max_samples=1000, subset='train'):
        """
        Load MNIST handwritten digits dataset using sklearn.
        Returns dataset_id for use in t-SNE

        max_samples: number of samples to load (default 1000, max 60000 for train)
        subset: 'train' or 'test'
        """
        try:
            from sklearn.datasets import fetch_openml

            # Load MNIST from sklearn
            print(f"Loading MNIST {subset} set from sklearn...")
            mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

            # Convert data to proper format
            all_images = np.array(mnist.data, dtype=np.float32)

            # Handle labels - they might be strings, so convert properly
            if isinstance(mnist.target[0], str):
                all_labels = np.array([int(label) for label in mnist.target], dtype=np.int32)
            else:
                all_labels = np.array(mnist.target, dtype=np.int32)

            # Get images and labels
            if subset == 'train':
                # First 60000 samples are training
                images_flat = all_images[:60000]
                labels = all_labels[:60000]
            else:
                # Last 10000 samples are test
                images_flat = all_images[60000:]
                labels = all_labels[60000:]

            # Limit samples
            if max_samples > 0 and max_samples < len(images_flat):
                images_flat = images_flat[:max_samples]
                labels = labels[:max_samples]

            # Reshape to 28x28 for display
            images = images_flat.reshape(-1, 28, 28)

            # Normalize to 0-1 range for t-SNE
            X = images_flat / 255.0

            # Convert images to PIL format for thumbnail display
            image_list = []
            for i, img_array in enumerate(images):
                # Convert to uint8 for PIL (0-255 range)
                img_uint8 = (img_array * 255).astype(np.uint8) if img_array.max() <= 1.0 else img_array.astype(np.uint8)
                img = Image.fromarray(img_uint8, mode='L')
                image_list.append({
                    'name': f'digit_{labels[i]}_{i}.png',
                    'image': img,
                    'thumbnail': self._create_thumbnail(img),
                    'label': int(labels[i])
                })

            # Store dataset
            dataset_id = f"mnist_{subset}_{len(datasets)}"
            datasets[dataset_id] = {
                'type': 'mnist',
                'name': f'MNIST {subset.capitalize()} ({len(images)} samples)',
                'X': X,
                'labels': labels,
                'images': image_list,
                'count': len(images)
            }

            return {
                'success': True,
                'dataset_id': dataset_id,
                'count': len(images),
                'shape': X.shape,
                'message': f'Loaded {len(images)} MNIST {subset} samples'
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ==================== Upload Handling ====================

    def upload_csv(self, name, content, delimiter=','):
        """Upload and parse CSV file"""
        try:
            # Save to storage directory
            storage_dir = get_storage_dir()
            upload_path = storage_dir / 'uploads' / name
            with open(upload_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # Parse CSV
            from io import StringIO
            df = pd.read_csv(StringIO(content), delimiter=delimiter)

            # Store dataset
            dataset_id = f"csv_{len(datasets)}"
            datasets[dataset_id] = {
                'type': 'csv',
                'name': name,
                'dataframe': df,
                'columns': df.columns.tolist(),
                'shape': df.shape,
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                'file_path': str(upload_path)
            }

            return {
                'success': True,
                'dataset_id': dataset_id,
                'columns': df.columns.tolist(),
                'numeric_columns': datasets[dataset_id]['numeric_columns'],
                'shape': df.shape,
                'preview': df.head(5).to_dict('records'),
                'saved_to': str(upload_path)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def upload_images(self, files):
        """Upload images and prepare for embedding"""
        try:
            storage_dir = get_storage_dir()
            images_dir = storage_dir / 'uploads' / 'images'
            images_dir.mkdir(exist_ok=True)

            images = []
            saved_paths = []

            for file_info in files:
                # Decode base64 image
                img_data = base64.b64decode(file_info['content'].split(',')[1])
                img = Image.open(BytesIO(img_data))

                # Save to storage directory
                img_path = images_dir / file_info['name']
                img.save(img_path)
                saved_paths.append(str(img_path))

                images.append({
                    'name': file_info['name'],
                    'image': img,
                    'thumbnail': self._create_thumbnail(img),
                    'file_path': str(img_path)
                })

            # Store dataset
            dataset_id = f"images_{len(datasets)}"
            datasets[dataset_id] = {
                'type': 'images',
                'images': images,
                'count': len(images),
                'saved_paths': saved_paths
            }

            return {
                'success': True,
                'dataset_id': dataset_id,
                'count': len(images),
                'names': [img['name'] for img in images],
                'saved_to': str(images_dir)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _create_thumbnail(self, img, size=(64, 64)):
        """Create thumbnail for display"""
        thumb = img.copy()
        thumb.thumbnail(size, Image.Resampling.LANCZOS)
        buffer = BytesIO()
        thumb.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()

    def compute_embeddings(self, dataset_id, method='hist'):
        """Compute embeddings for images"""
        try:
            if dataset_id not in datasets or datasets[dataset_id]['type'] != 'images':
                return {'success': False, 'error': 'Invalid dataset'}

            images_data = datasets[dataset_id]['images']

            if method == 'clip':
                try:
                    import torch
                    import clip

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model, preprocess = clip.load("ViT-B/32", device=device)

                    embeddings = []
                    for img_info in images_data:
                        img_tensor = preprocess(img_info['image']).unsqueeze(0).to(device)
                        with torch.no_grad():
                            embedding = model.encode_image(img_tensor)
                        embeddings.append(embedding.cpu().numpy().flatten())

                    X = np.array(embeddings)
                    datasets[dataset_id]['embeddings'] = X
                    datasets[dataset_id]['embedding_method'] = 'CLIP'

                    return {
                        'success': True,
                        'method': 'CLIP',
                        'shape': X.shape
                    }
                except ImportError:
                    # Fall back to histogram if CLIP not available
                    method = 'hist'

            if method == 'hist':
                # Simple color histogram embedding
                embeddings = []
                for img_info in images_data:
                    img = img_info['image'].convert('RGB')
                    img = img.resize((64, 64))

                    # Compute color histogram
                    hist_r = np.histogram(np.array(img)[:,:,0], bins=32, range=(0, 256))[0]
                    hist_g = np.histogram(np.array(img)[:,:,1], bins=32, range=(0, 256))[0]
                    hist_b = np.histogram(np.array(img)[:,:,2], bins=32, range=(0, 256))[0]

                    # Concatenate and normalize
                    hist = np.concatenate([hist_r, hist_g, hist_b])
                    hist = hist / (hist.sum() + 1e-10)

                    embeddings.append(hist)

                X = np.array(embeddings)
                datasets[dataset_id]['embeddings'] = X
                datasets[dataset_id]['embedding_method'] = 'Color Histogram'

                return {
                    'success': True,
                    'method': 'Color Histogram',
                    'shape': X.shape
                }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def list_datasets(self):
        """List all uploaded datasets"""
        result = []
        for dataset_id, data in datasets.items():
            if data['type'] == 'csv':
                result.append({
                    'id': dataset_id,
                    'type': 'csv',
                    'name': data['name'],
                    'shape': data['shape']
                })
            elif data['type'] == 'images':
                result.append({
                    'id': dataset_id,
                    'type': 'images',
                    'count': data['count']
                })
            elif data['type'] == 'synthetic':
                result.append({
                    'id': dataset_id,
                    'type': 'synthetic',
                    'shape': data['X'].shape
                })
            elif data['type'] == 'mnist':
                result.append({
                    'id': dataset_id,
                    'type': 'mnist',
                    'name': data['name'],
                    'count': data['count'],
                    'shape': data['X'].shape
                })
        return result

    def save_synthetic_dataset(self, points):
        """Save synthetic points as a dataset"""
        dataset_id = f"synthetic_{len(datasets)}"
        X = np.array(points)
        datasets[dataset_id] = {
            'type': 'synthetic',
            'X': X,
            'shape': X.shape
        }
        return {'success': True, 'dataset_id': dataset_id}

    def prepare_csv_dataset(self, dataset_id, selected_columns, handle_missing='drop'):
        """Prepare CSV dataset for t-SNE"""
        try:
            if dataset_id not in datasets or datasets[dataset_id]['type'] != 'csv':
                return {'success': False, 'error': 'Invalid dataset'}

            df = datasets[dataset_id]['dataframe']

            # Select columns
            df_subset = df[selected_columns]

            # Handle missing values
            if handle_missing == 'drop':
                df_subset = df_subset.dropna()
            elif handle_missing == 'mean':
                df_subset = df_subset.fillna(df_subset.mean())
            elif handle_missing == 'zero':
                df_subset = df_subset.fillna(0)

            X = df_subset.values
            datasets[dataset_id]['X'] = X
            datasets[dataset_id]['X_columns'] = selected_columns

            return {
                'success': True,
                'shape': X.shape
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ==================== t-SNE Implementation ====================

    def run_tsne(self, dataset_id, perplexity=30, learning_rate=200, n_iter=1000,
                 early_exaggeration=12, momentum=0.8, init_method='random',
                 init_data=None, seed=42):
        """
        Run t-SNE with transparent internals
        Returns: Y (n×2), P (n×n), Q (n×n), C_history
        """
        global stop_flag
        stop_flag = False

        try:
            # Get data
            if dataset_id not in datasets:
                return {'success': False, 'error': 'Dataset not found'}

            dataset = datasets[dataset_id]

            if dataset['type'] == 'synthetic':
                X = dataset['X']
            elif dataset['type'] == 'csv':
                if 'X' not in dataset:
                    return {'success': False, 'error': 'CSV dataset not prepared. Select columns first.'}
                X = dataset['X']
            elif dataset['type'] == 'images':
                if 'embeddings' not in dataset:
                    return {'success': False, 'error': 'Images not embedded. Compute embeddings first.'}
                X = dataset['embeddings']
            elif dataset['type'] == 'mnist':
                X = dataset['X']
            else:
                return {'success': False, 'error': 'Unknown dataset type'}

            n, d = X.shape

            if n > 1000:
                return {'success': False, 'error': f'Dataset too large ({n} points). Please use n <= 1000 for performance.'}

            # Run t-SNE in background thread
            result = {'success': False}

            def run_in_background():
                nonlocal result
                try:
                    # Initialize Y
                    np.random.seed(seed)
                    if init_method == 'custom' and init_data is not None:
                        Y = np.array(init_data)
                        if Y.shape != (n, 2):
                            raise ValueError(f'Custom initialization must be {n}×2, got {Y.shape}')
                    else:
                        # Random initialization (default)
                        Y = np.random.randn(n, 2) * 0.0001

                    # Compute P (high-dimensional probabilities)
                    self._send_progress(0, n_iter, 'Computing P matrix...')
                    P = self._compute_P(X, perplexity)

                    # Use custom t-SNE implementation for full transparency
                    # (scikit-learn doesn't provide iteration-by-iteration cost tracking)
                    self._send_progress(0, n_iter, 'Starting t-SNE optimization...')
                    Y, Q, C_history = self._optimize_tsne(
                        P, Y, learning_rate, n_iter, early_exaggeration, momentum
                    )

                    # Store results
                    datasets[dataset_id]['tsne_result'] = {
                        'Y': Y,
                        'P': P,
                        'Q': Q,
                        'C_history': C_history
                    }

                    result = clean_for_json({
                        'success': True,
                        'Y': Y.tolist(),
                        'P': P.tolist(),
                        'Q': Q.tolist(),
                        'C_history': C_history,
                        'n': n
                    })

                    # Add true labels for MNIST visualization
                    if dataset['type'] == 'mnist' and 'labels' in dataset:
                        result['labels'] = dataset['labels'].tolist()
                        result['has_labels'] = True

                    self._send_progress(n_iter, n_iter, 'Complete!')

                except Exception as e:
                    result = {'success': False, 'error': str(e)}

            thread = threading.Thread(target=run_in_background)
            thread.start()
            thread.join()

            return result

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _compute_P(self, X, perplexity):
        """
        Compute pairwise affinities P_ij using perplexity
        """
        n = X.shape[0]

        # Compute pairwise squared distances
        sum_X = np.sum(X**2, axis=1)
        D = sum_X[:, np.newaxis] + sum_X[np.newaxis, :] - 2 * X @ X.T
        D = np.maximum(D, 0)  # Numerical stability

        # Compute P_j|i for each point
        P = np.zeros((n, n))

        target_entropy = np.log2(perplexity)

        for i in range(n):
            # Binary search for sigma_i
            beta_min = -np.inf
            beta_max = np.inf
            beta = 1.0

            for _ in range(50):  # Max 50 iterations
                # Compute P_j|i
                Di = D[i].copy()
                Di[i] = 0

                P_i = np.exp(-Di * beta)
                P_i[i] = 0
                sum_P_i = np.sum(P_i)

                if sum_P_i == 0:
                    P_i = np.ones(n) / n
                    sum_P_i = 1.0

                P_i = P_i / sum_P_i

                # Compute entropy
                P_i_nonzero = P_i[P_i > 1e-12]
                H = -np.sum(P_i_nonzero * np.log2(P_i_nonzero))

                # Check convergence
                H_diff = H - target_entropy
                if np.abs(H_diff) < 1e-5:
                    break

                # Update beta
                if H_diff > 0:
                    beta_min = beta
                    if beta_max == np.inf:
                        beta = beta * 2
                    else:
                        beta = (beta + beta_max) / 2
                else:
                    beta_max = beta
                    if beta_min == -np.inf:
                        beta = beta / 2
                    else:
                        beta = (beta + beta_min) / 2

            P[i] = P_i

        # Symmetrize
        P = (P + P.T) / (2 * n)
        P = np.maximum(P, 1e-12)

        return P

    def _compute_Q(self, Y):
        """Compute low-dimensional affinities Q_ij from embedding Y"""
        sum_Y = np.sum(Y**2, axis=1)
        D_low = sum_Y[:, np.newaxis] + sum_Y[np.newaxis, :] - 2 * Y @ Y.T
        D_low = np.maximum(D_low, 0)
        Q = (1 + D_low) ** (-1)
        np.fill_diagonal(Q, 0)
        sum_Q = np.sum(Q)
        if sum_Q < 1e-12:
            sum_Q = 1e-12
        Q = Q / sum_Q
        return np.maximum(Q, 1e-12)

    def _optimize_tsne(self, P, Y, learning_rate, n_iter, early_exaggeration, momentum):
        """
        Optimize t-SNE using gradient descent
        """
        global stop_flag

        n = Y.shape[0]
        Y_velocity = np.zeros_like(Y)
        C_history = []

        # Early exaggeration
        P_exag = P * early_exaggeration

        for iteration in range(n_iter):
            if stop_flag:
                break

            # Use early exaggeration for first 250 iterations
            P_current = P_exag if iteration < 250 else P

            # Compute Q (low-dimensional affinities using Student t-distribution)
            sum_Y = np.sum(Y**2, axis=1)
            D_low = sum_Y[:, np.newaxis] + sum_Y[np.newaxis, :] - 2 * Y @ Y.T
            D_low = np.maximum(D_low, 0)

            # Student t-distribution with df=1
            Q = (1 + D_low) ** (-1)
            np.fill_diagonal(Q, 0)
            sum_Q = np.sum(Q)
            if sum_Q < 1e-12:
                print(f"Warning: Q sum very small: {sum_Q}, using 1e-12")
                sum_Q = 1e-12
            Q = Q / sum_Q
            Q = np.maximum(Q, 1e-12)

            # Compute cost (KL divergence)
            C = np.sum(P_current * np.log((P_current + 1e-12) / (Q + 1e-12)))
            C_history.append(float(C))

            # Compute gradient (t-SNE gradient with Student-t kernel)
            PQ_diff = P_current - Q
            # For each pair (i,j), gradient contribution is (P_ij - Q_ij) * (1 + ||y_i - y_j||^2)^(-1) * (y_i - y_j)
            repulsion = (1 + D_low) ** (-1)
            attraction_repulsion = (PQ_diff * repulsion)[:, :, np.newaxis]
            Y_diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]  # (n, n, 2)
            gradient = 4 * (attraction_repulsion * Y_diff).sum(axis=1)

            # Update Y with momentum
            Y_velocity = momentum * Y_velocity - learning_rate * gradient
            Y = Y + Y_velocity

            # Center Y
            Y = Y - Y.mean(axis=0)

            # Send progress update every 10 iterations
            if iteration % 10 == 0:
                self._send_progress(iteration, n_iter, f'Iteration {iteration}/{n_iter}, Cost: {C:.4f}')

        # Final Q computation with Student t-distribution
        sum_Y = np.sum(Y**2, axis=1)
        D_low = sum_Y[:, np.newaxis] + sum_Y[np.newaxis, :] - 2 * Y @ Y.T
        D_low = np.maximum(D_low, 0)
        Q = (1 + D_low) ** (-1)
        np.fill_diagonal(Q, 0)
        sum_Q = np.sum(Q)
        if sum_Q < 1e-12:
            sum_Q = 1e-12
        Q = Q / sum_Q
        Q = np.maximum(Q, 1e-12)

        return Y, Q, C_history

    def _send_progress(self, current, total, message):
        """Send progress update to frontend"""
        if self.window:
            self.window.evaluate_js(f'window.updateProgress({current}, {total}, "{message}")')

    def stop_tsne(self):
        """Stop running t-SNE"""
        global stop_flag
        stop_flag = True
        return {'success': True}

    # ==================== Clustering ====================

    def run_clustering(self, dataset_id, method='kmeans', k=3, eps=0.5, min_samples=5):
        """Run clustering on t-SNE results"""
        try:
            if dataset_id not in datasets:
                return {'success': False, 'error': 'Dataset not found'}

            dataset = datasets[dataset_id]

            if 'tsne_result' not in dataset:
                return {'success': False, 'error': 'Run t-SNE first'}

            Y = dataset['tsne_result']['Y']

            if method == 'kmeans':
                labels = self._kmeans(Y, k)
            elif method == 'dbscan':
                labels = self._dbscan(Y, eps, min_samples)
            else:
                return {'success': False, 'error': 'Unknown clustering method'}

            # Compute cluster summary
            unique_labels = np.unique(labels)
            summary = []
            for label in unique_labels:
                count = np.sum(labels == label)
                summary.append({
                    'label': int(label),
                    'count': int(count)
                })

            dataset['tsne_result']['cluster_labels'] = labels

            return {
                'success': True,
                'labels': labels.tolist(),
                'summary': summary
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _kmeans(self, X, k, max_iter=100):
        """K-means clustering"""
        n = X.shape[0]

        # Initialize centroids randomly
        indices = np.random.choice(n, k, replace=False)
        centroids = X[indices].copy()

        labels = np.zeros(n, dtype=int)

        for _ in range(max_iter):
            # Assign points to nearest centroid
            distances = np.zeros((n, k))
            for i in range(k):
                distances[:, i] = np.sum((X - centroids[i])**2, axis=1)

            new_labels = np.argmin(distances, axis=1)

            # Check convergence
            if np.all(labels == new_labels):
                break

            labels = new_labels

            # Update centroids
            for i in range(k):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    centroids[i] = cluster_points.mean(axis=0)

        return labels

    def _dbscan(self, X, eps, min_samples):
        """DBSCAN clustering"""
        n = X.shape[0]
        labels = -np.ones(n, dtype=int)  # -1 = noise
        cluster_id = 0

        for i in range(n):
            if labels[i] != -1:
                continue

            # Find neighbors
            neighbors = self._find_neighbors(X, i, eps)

            if len(neighbors) < min_samples:
                labels[i] = -1  # Mark as noise
            else:
                # Start new cluster
                self._expand_cluster(X, labels, i, neighbors, cluster_id, eps, min_samples)
                cluster_id += 1

        return labels

    def _find_neighbors(self, X, point_idx, eps):
        """Find all neighbors within eps distance"""
        distances = np.sum((X - X[point_idx])**2, axis=1)
        return np.where(distances <= eps**2)[0]

    def _expand_cluster(self, X, labels, point_idx, neighbors, cluster_id, eps, min_samples):
        """Expand cluster from seed point"""
        labels[point_idx] = cluster_id

        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id

            if labels[neighbor_idx] != -1:
                i += 1
                continue

            labels[neighbor_idx] = cluster_id

            new_neighbors = self._find_neighbors(X, neighbor_idx, eps)
            if len(new_neighbors) >= min_samples:
                neighbors = np.concatenate([neighbors, new_neighbors])

            i += 1

    # ==================== Export ====================

    def export_results(self, dataset_id):
        """Export t-SNE results as CSV"""
        try:
            if dataset_id not in datasets:
                return {'success': False, 'error': 'Dataset not found'}

            dataset = datasets[dataset_id]

            if 'tsne_result' not in dataset:
                return {'success': False, 'error': 'No t-SNE results to export'}

            Y = dataset['tsne_result']['Y']

            df = pd.DataFrame({
                'index': range(len(Y)),
                'dim_1': Y[:, 0],
                'dim_2': Y[:, 1]
            })

            if 'cluster_labels' in dataset['tsne_result']:
                df['cluster'] = dataset['tsne_result']['cluster_labels']

            # Add true labels for MNIST
            if dataset['type'] == 'mnist' and 'labels' in dataset:
                df['true_label'] = dataset['labels']

            csv_content = df.to_csv(index=False)

            # Save to exports directory
            storage_dir = get_storage_dir()
            export_path = storage_dir / 'exports' / f'{dataset_id}_tsne_results.csv'
            df.to_csv(export_path, index=False)

            return {
                'success': True,
                'csv': csv_content,
                'saved_to': str(export_path)
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_image_at_index(self, dataset_id, index):
        """Get image data for a specific index"""
        try:
            if dataset_id not in datasets:
                return {'success': False, 'error': 'Dataset not found'}

            dataset = datasets[dataset_id]

            if dataset['type'] != 'images':
                return {'success': False, 'error': 'Not an image dataset'}

            if index < 0 or index >= len(dataset['images']):
                return {'success': False, 'error': 'Index out of range'}

            img_info = dataset['images'][index]

            # Convert image to base64
            buffer = BytesIO()
            img_info['image'].save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()

            return {
                'success': True,
                'name': img_info['name'],
                'image': f'data:image/png;base64,{img_base64}'
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}


def main():
    """Launch the application"""

    # Suppress warnings
    import warnings
    warnings.filterwarnings('ignore')

    # Suppress stderr temporarily during pywebview initialization
    import sys
    import io

    print("=" * 60)
    print("Starting t-SNE Explorer...")
    print("=" * 60)

    # Create API instance
    api = TSNEExplorer()

    # Get path to web directory
    web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')
    index_path = os.path.join(web_dir, 'index.html')

    print(f"Web directory: {web_dir}")
    print(f"Index file: {index_path}")

    if not os.path.exists(index_path):
        print(f"\nError: {index_path} not found!")
        print("Please ensure the 'web' directory exists with index.html")
        input("Press Enter to exit...")
        sys.exit(1)

    print("Files found!")
    print("\nOpening window...")
    print("\nNote: Initialization may take a few seconds...")
    print("The app window should appear shortly.\n")

    try:
        # Temporarily suppress stderr during window creation
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()

        # Create window
        window = webview.create_window(
            't-SNE Explorer',
            index_path,
            js_api=api,
            width=1400,
            height=900,
            resizable=True,
            background_color='#667eea'
        )

        api.set_window(window)

        # Restore stderr
        sys.stderr = old_stderr

        print("Window created! Starting application...\n")

        # Start webview
        webview.start(debug=False)

        print("\nApplication closed. Goodbye!")

    except Exception as e:
        print(f"\nError starting application: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure WebView2 is installed (should be automatic on Windows 10/11)")
        print("2. Try: pip install --upgrade pywebview")
        print("3. Check if web/ folder contains index.html, app.js, style.css")
        input("\nPress Enter to exit...")
        sys.exit(1)


if __name__ == '__main__':
    main()
