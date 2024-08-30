import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from scipy.linalg import norm, sqrtm
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import KernelDensity

class DatasetProcessor:
    def __init__(self, folder_path, target_dim, exclude_files=None):
        if exclude_files is None:
            exclude_files = []
        self.folder_path = folder_path
        self.target_dim = target_dim
        self.exclude_files = exclude_files
        self.dataset_files = self.load_datasets()
        self.datasets = self.read_datasets()
        self.reduced_datasets = self.reduce_datasets()

    def load_datasets(self):
        data_files = []
        for root, dirs, files in os.walk(self.folder_path):
            for file in files:
                if file.endswith(".csv") and file not in self.exclude_files:
                    data_files.append(os.path.join(root, file))
        return data_files

    def read_datasets(self):
        datasets = []
        for file in self.dataset_files:
            df = pd.read_csv(file)
            df = self.encode_categorical_features(df)
            datasets.append(df.values)
        return datasets

    def encode_categorical_features(self, df):
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) > 0:
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            encoded_data = encoder.fit_transform(df[categorical_columns])
            df = df.drop(categorical_columns, axis=1)
            df = pd.concat([df, pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))], axis=1)
        return df

    def reduce_datasets(self):
        pca = PCA(n_components=self.target_dim)
        reduced_datasets = [pca.fit_transform(dataset) for dataset in self.datasets]
        return reduced_datasets

    @staticmethod
    def compute_covariance_distance(cov1, cov2, method='frobenius'):
        if method == 'frobenius':
            return norm(cov1 - cov2, 'fro')
        elif method == 'riemannian':
            return norm(sqrtm(cov1) @ sqrtm(cov2))
        elif method == 'log_euclidean':
            log_diff = np.logm(cov1) - np.logm(cov2)
            return norm(log_diff, 'fro')
        else:
            raise ValueError("Unsupported distance method")

    def calculate_covariance_distance_matrix(self, method='frobenius'):
        num_datasets = len(self.reduced_datasets)
        dist_matrix = np.zeros((num_datasets, num_datasets))
        for i in tqdm(range(num_datasets), "Computing Covariance Distance Matrix"):
            cov1 = np.cov(self.reduced_datasets[i], rowvar=False)
            for j in range(i, num_datasets):
                cov2 = np.cov(self.reduced_datasets[j], rowvar=False)
                dist_value = self.compute_covariance_distance(cov1, cov2, method)
                dist_matrix[i, j] = dist_value
                dist_matrix[j, i] = dist_value  # Distance is symmetric
        return dist_matrix

def plot_heatmap_and_cluster(dist_df, n_clusters=5, save_html=True):
    # Plot heatmap using Plotly
    fig = px.imshow(dist_df, text_auto=True, aspect="auto", color_continuous_scale='YlGnBu')
    fig.update_layout(
        title="Covariance Matrix Distance Heatmap Between Datasets",
        xaxis_title="Datasets",
        yaxis_title="Datasets"
    )

    if save_html:
        # Save heatmap to HTML
        pio.write_html(fig, file='cov_distance_heatmap.html', auto_open=True)

    # Hierarchical clustering
    linked = sch.linkage(dist_df, method='ward')
    clusters = sch.fcluster(linked, n_clusters, criterion='maxclust')
    cluster_df = pd.DataFrame({'Dataset': dist_df.index, 'Cluster': clusters})
    cluster_df.to_csv('dataset_clusters.csv', index=False)

    # Plot dendrogram
    plt.figure(figsize=(10, 7))
    sch.dendrogram(linked, labels=dist_df.index, orientation='top', color_threshold=0)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Datasets')
    plt.ylabel('Distance')
    plt.show()

def main():
    folder_path = 'data/synthetic_data_small_linear'
    target_dim = 10  # Desired dimensionality for PCA
    processor = DatasetProcessor(folder_path, target_dim)
    cov_dist_matrix = processor.calculate_covariance_distance_matrix(method='frobenius')

    # Convert to DataFrame
    dataset_names = [os.path.basename(file).replace('.csv', '') for file in processor.dataset_files]
    cov_dist_df = pd.DataFrame(cov_dist_matrix, index=dataset_names, columns=dataset_names)

    # Normalize the DataFrame
    cov_dist_df = (cov_dist_df - cov_dist_df.min().min()) / (cov_dist_df.max().max() - cov_dist_df.min().min())

    # Save the normalized DataFrame
    cov_dist_df.to_csv('cov_distance_matrix_normalized.csv')

    # Plot heatmap and perform clustering
    plot_heatmap_and_cluster(cov_dist_df)

    print("Normalized Covariance Distance Matrix DataFrame:")
    print(cov_dist_df)

if __name__ == "__main__":
    main()
