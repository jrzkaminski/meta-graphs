import os
import pandas as pd
import numpy as np
import ot

# Path to the dataset folder
folder_path = 'data/synthetic_data_diff/'

# List all files in the folder
files = os.listdir(folder_path)

# Separate files by shape
triangle_files = [f for f in files if 'triangle' in f]
square_files = [f for f in files if 'square' in f]


def compute_emd(original_data, variant_data):
    # Convert to numpy arrays (ensure same size)
    original_data = original_data.values
    variant_data = variant_data.values

    # Normalize data if needed
    original_data = original_data / np.sum(original_data)
    variant_data = variant_data / np.sum(variant_data)

    # Calculate EMD
    emd_value = ot.emd2([], [], original_data, variant_data)
    return emd_value


# Function to process files by shape
def process_shape_files(files):
    emd_values = []

    for i in range(100):
        # Find corresponding original and variant files
        original_file = [f for f in files if f'_{i}_original' in f][0]
        variant_file = [f for f in files if f'_{i}_variant' in f][0]

        # Load datasets
        original_data = pd.read_csv(os.path.join(folder_path, original_file))
        variant_data = pd.read_csv(os.path.join(folder_path, variant_file))

        # Compute EMD
        emd = compute_emd(original_data, variant_data)
        emd_values.append(emd)

    return emd_values


# Process triangle and square datasets
triangle_emd = process_shape_files(triangle_files)
square_emd = process_shape_files(square_files)

import matplotlib.pyplot as plt

# Plot EMD distributions
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(triangle_emd, bins=20, color='blue', alpha=0.7)
plt.title('EMD Distribution for Triangle Datasets')
plt.xlabel('EMD')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(square_emd, bins=20, color='green', alpha=0.7)
plt.title('EMD Distribution for Square Datasets')
plt.xlabel('EMD')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
