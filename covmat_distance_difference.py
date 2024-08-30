import os
import pandas as pd
import numpy as np
from scipy.linalg import norm

# Path to the dataset folder
folder_path = 'data/synthetic_data_diff/'

# List all files in the folder
files = os.listdir(folder_path)

# Separate files by shape
triangle_files = [f for f in files if 'triangle' in f]
square_files = [f for f in files if 'square' in f]


def compute_covariance_distance(original_data, variant_data, method='frobenius'):
    # Convert to numpy arrays
    original_data = original_data.values
    variant_data = variant_data.values

    # Compute covariance matrices
    cov_original = np.cov(original_data, rowvar=False)
    cov_variant = np.cov(variant_data, rowvar=False)

    # Compute the distance between covariance matrices
    if method == 'frobenius':
        distance_value = norm(cov_original - cov_variant, 'fro')
    elif method == 'log_euclidean':
        log_diff = np.logm(cov_original) - np.logm(cov_variant)
        distance_value = norm(log_diff, 'fro')
    else:
        raise ValueError("Unsupported distance method")

    return distance_value


# Function to process files by shape
def process_shape_files(files, method='frobenius'):
    distance_values = []

    for i in range(100):
        # Find corresponding original and variant files
        original_file = [f for f in files if f'_{i}_original' in f][0]
        variant_file = [f for f in files if f'_{i}_variant' in f][0]

        # Load datasets
        original_data = pd.read_csv(os.path.join(folder_path, original_file))
        variant_data = pd.read_csv(os.path.join(folder_path, variant_file))

        # Compute covariance distance
        distance = compute_covariance_distance(original_data, variant_data, method)
        distance_values.append(distance)

    return distance_values


# Process triangle and square datasets using the Frobenius norm
triangle_distances = process_shape_files(triangle_files, method='frobenius')
square_distances = process_shape_files(square_files, method='frobenius')

import matplotlib.pyplot as plt

# Plot covariance distance distributions
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(triangle_distances, bins=20, color='blue', alpha=0.7)
plt.title('Covariance Distance Distribution for Triangle Datasets')
plt.xlabel('Covariance Distance')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(square_distances, bins=20, color='green', alpha=0.7)
plt.title('Covariance Distance Distribution for Square Datasets')
plt.xlabel('Covariance Distance')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
