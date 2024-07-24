import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.distance import cdist
import ot
import matplotlib.pyplot as plt

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

    def compute_emd(self, data1, data2):
        cost_matrix = cdist(data1, data2, metric='euclidean')
        n, n1 = data1.shape[0], data2.shape[0]
        weights1 = np.ones(n) / n
        weights2 = np.ones(n1) / n1
        emd_value = ot.emd2(weights1, weights2, cost_matrix)
        return emd_value

    def calculate_emd_matrix(self):
        num_datasets = len(self.reduced_datasets)
        emd_matrix = np.zeros((num_datasets, num_datasets))
        for i in range(num_datasets):
            for j in range(i, num_datasets):
                emd_value = self.compute_emd(self.reduced_datasets[i], self.reduced_datasets[j])
                emd_matrix[i, j] = emd_value
                emd_matrix[j, i] = emd_value  # EMD is symmetric
        return emd_matrix

def plot_heatmap_and_cluster(emd_df, n_clusters=5, save_html=True):
    # Plot heatmap using Plotly
    fig = px.imshow(emd_df, text_auto=True, aspect="auto", color_continuous_scale='YlGnBu')
    fig.update_layout(
        title="EMD Heatmap Between Datasets",
        xaxis_title="Datasets",
        yaxis_title="Datasets"
    )

    if save_html:
        # Save heatmap to HTML
        pio.write_html(fig, file='emd_heatmap.html', auto_open=True)

    # Hierarchical clustering
    linked = sch.linkage(emd_df, method='ward')
    clusters = sch.fcluster(linked, n_clusters, criterion='maxclust')
    cluster_df = pd.DataFrame({'Dataset': emd_df.index, 'Cluster': clusters})
    cluster_df.to_csv('dataset_clusters.csv', index=False)

    # Plot dendrogram
    plt.figure(figsize=(10, 7))
    sch.dendrogram(linked, labels=emd_df.index, orientation='top', color_threshold=0)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Datasets')
    plt.ylabel('Distance')
    plt.show()

def main():
    folder_path = 'data/'
    target_dim = 3  # Desired dimensionality for PCA
    exclude_files = ['mehra-complete.csv']  # Files to exclude
    processor = DatasetProcessor(folder_path, target_dim, exclude_files)
    emd_matrix = processor.calculate_emd_matrix()

    # Convert to DataFrame and save
    dataset_names = [os.path.basename(file).replace('.csv', '') for file in processor.dataset_files]
    emd_df = pd.DataFrame(emd_matrix, index=dataset_names, columns=dataset_names)
    emd_df.to_csv('emd_matrix.csv')

    # Plot heatmap and perform clustering
    plot_heatmap_and_cluster(emd_df)

    print("EMD Matrix DataFrame:")
    print(emd_df)

if __name__ == "__main__":
    main()
