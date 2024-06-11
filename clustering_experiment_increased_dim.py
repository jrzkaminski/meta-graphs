import os
import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis, KernelPCA
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
from umap import UMAP
from sklearn.cluster import KMeans
import torch
from torch import nn
from torch.optim import Adam
import networkx as nx
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import from_networkx
from sklearn.metrics import adjusted_rand_score
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, encoding_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(encoding_dim, input_dim), nn.Sigmoid())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# GCN model
class GCN(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# GraphSAGE model
class GraphSAGENet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGENet, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# Function to load graphs from txt files
def load_graphs_from_txt(directory):
    graph_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                graph_files.append(os.path.join(root, file))
    return graph_files


# Function to load datasets from csv files
def load_datasets(directory):
    data_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                data_files.append(os.path.join(root, file))
    return data_files


# Function to preprocess the dataset
def preprocess_dataset(file):
    dataset = pd.read_csv(file)
    if "Unnamed: 0" in dataset.columns:
        dataset = dataset.drop("Unnamed: 0", axis=1)
    dataset = pd.get_dummies(dataset)
    dataset = dataset.fillna(dataset.mean())
    dataset = dataset.apply(pd.to_numeric, errors="coerce")
    dataset = dataset.fillna(0)
    return dataset.astype(np.float32).to_numpy()


# Function to pad embeddings to the same length
def pad_embeddings(embeddings, max_length):
    padded_embeddings = []
    for emb in embeddings:
        if len(emb) < max_length:
            padding = np.zeros(max_length - len(emb))
            padded_emb = np.concatenate((emb, padding))
        else:
            padded_emb = emb
        padded_embeddings.append(padded_emb)
    return np.array(padded_embeddings)


# Function to generate embeddings using various methods
def generate_embeddings(dataset, method, embedding_dim):
    match method:
        case 'tsne':
            model = TSNE(n_components=embedding_dim)
        case 'umap':
            model = UMAP(n_components=embedding_dim)
        case 'se':
            model = SpectralEmbedding(n_components=embedding_dim)
        case 'fa':
            model = FactorAnalysis(n_components=embedding_dim)
        case 'kpca':
            model = KernelPCA(n_components=embedding_dim, kernel='rbf')
        case _:
            raise ValueError(f"Unknown method: {method}")
    return model.fit_transform(dataset).flatten()


# Function to train autoencoder and generate embeddings
def autoencoder_embeddings(dataset, embedding_dim):
    autoencoder = Autoencoder(dataset.shape[1], embedding_dim)
    optimizer = Adam(autoencoder.parameters())
    criterion = nn.MSELoss()
    dataset_torch = torch.tensor(dataset, dtype=torch.float32)
    dataset_torch = (dataset_torch - dataset_torch.min()) / (dataset_torch.max() - dataset_torch.min())

    for epoch in tqdm(range(50), desc="Training autoencoder"):
        autoencoder.train()
        optimizer.zero_grad()
        outputs = autoencoder(dataset_torch)
        loss = criterion(outputs, dataset_torch)
        loss.backward()
        optimizer.step()

    autoencoder.eval()
    with torch.no_grad():
        return autoencoder.encoder(dataset_torch).flatten().numpy()


# Function to load graph from txt file and convert to PyTorch geometric data
def load_graph(graph_file):
    with open(graph_file, "r") as f:
        edges = [tuple(line.strip().split()) for line in f]
    G = nx.DiGraph()
    G.add_edges_from(edges)
    mapping = {node: idx for idx, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    for i in G.nodes:
        G.nodes[i]["feature"] = [1.0] * 10
    data = from_networkx(G)
    data.x = torch.tensor([G.nodes[i]["feature"] for i in G.nodes], dtype=torch.float32)
    return data


def load_graph_vis(graph_file):
    with open(graph_file, "r") as f:
        edges = [tuple(line.strip().split()) for line in f]
    G = nx.DiGraph()
    G.add_edges_from(edges)
    mapping = {node: idx for idx, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    return G


# Function to train GCN model and generate embeddings
def gcn_embeddings(data):
    gcn_model = GCN(num_node_features=10, hidden_dim=8, num_classes=10)
    optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(200), desc="Training GCN"):
        gcn_model.train()
        optimizer.zero_grad()
        out = gcn_model(data)
        loss = criterion(out, torch.tensor([0 for _ in data.x], dtype=torch.long))
        loss.backward()
        optimizer.step()

    gcn_model.eval()
    with torch.no_grad():
        return gcn_model.conv1(data.x, data.edge_index).mean(axis=0).numpy()


# Function to train GraphSAGE model and generate embeddings
def sage_embeddings(data):
    sage_model = GraphSAGENet(in_channels=10, hidden_channels=16, out_channels=10)
    optimizer = torch.optim.Adam(sage_model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(200), desc="Training GraphSAGE"):
        sage_model.train()
        optimizer.zero_grad()
        out = sage_model(data)
        loss = criterion(out, torch.tensor([0 for _ in data.x], dtype=torch.long))
        loss.backward()
        optimizer.step()

    sage_model.eval()
    with torch.no_grad():
        return sage_model.conv1(data.x, data.edge_index).mean(axis=0).numpy()


def visualize_clusters(graph_clusters, graph_files, filename):
    unique_clusters = np.unique(graph_clusters)
    num_clusters = len(unique_clusters)

    # Determine the number of graphs in the largest cluster to set the number of columns
    max_graphs_in_cluster = max(np.bincount(graph_clusters))

    fig, axes = plt.subplots(num_clusters, max_graphs_in_cluster, figsize=(max_graphs_in_cluster * 5, num_clusters * 5))

    if num_clusters == 1:
        axes = [axes]

    colors = plt.get_cmap("tab10")

    for i, cluster in enumerate(unique_clusters):
        cluster_indices = np.where(graph_clusters == cluster)[0]
        for j, idx in enumerate(cluster_indices):
            G = load_graph_vis(graph_files[idx])
            sorted_nodes = list(nx.topological_sort(G))
            pos = nx.spiral_layout(G)  # use spring layout for better visualization
            color_map = [colors(cluster)] * G.number_of_nodes()
            if num_clusters == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]
            nx.draw(G, pos, ax=ax, node_color=color_map, nodelist=sorted_nodes, node_size=50)
            ax.set_title(f"Graph {idx} (Cluster {cluster})")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    logger.info("Loading datasets...")
    data_files = load_datasets("data/")
    datasets = [preprocess_dataset(file) for file in data_files]

    embedding_dim = 5  # Fixed embedding dimension

    data_embeddings_features = {method: [] for method in ['autoencoder']}
    data_embeddings_samples = {method: [] for method in ['autoencoder']}
    dataset_names = [os.path.basename(file).replace(".csv", "") for file in data_files]

    for method in ['autoencoder']:
        logger.info(f"Generating embeddings using {method} method on datasets...")
        for dataset in tqdm(datasets, desc=f"Generating embeddings with {method}"):
            match method:
                case 'autoencoder':
                    embedding = autoencoder_embeddings(dataset, embedding_dim)
                case _:
                    embedding = generate_embeddings(dataset, method, embedding_dim)
            data_embeddings_features[method].append(np.array(embedding))
            data_embeddings_samples[method].append(np.array(embedding))

    logger.info("Loading graphs...")
    graph_files = load_graphs_from_txt("data/")
    graph_data = [load_graph(file) for file in graph_files]

    logger.info("Generating GCN embeddings...")
    gcn_embeddings_list = [gcn_embeddings(data) for data in graph_data]
    logger.info("Generating GraphSAGE embeddings...")
    sage_embeddings_list = [sage_embeddings(data) for data in graph_data]

    graph_embeddings_methods = {
        "gcn": gcn_embeddings_list,
        "sage": sage_embeddings_list,
    }

    for method in ['autoencoder']:
        max_length_features = max(len(emb) for emb in data_embeddings_features[method])
        data_embeddings_features[method] = pad_embeddings(data_embeddings_features[method], max_length_features)

        max_length_samples = max(len(emb) for emb in data_embeddings_samples[method])
        data_embeddings_samples[method] = pad_embeddings(data_embeddings_samples[method], max_length_samples)

    for graph_method, graph_embs in graph_embeddings_methods.items():
        logger.info(f"Clustering with {graph_method} embeddings...")
        graph_clusters = KMeans(n_clusters=3).fit_predict(graph_embs)

        for dataset_method in ['autoencoder']:
            logger.info(f"Clustering dataset embeddings using {dataset_method} method with feature standardization...")
            dataset_embs = np.vstack(data_embeddings_features[dataset_method])
            dataset_clusters = KMeans(n_clusters=3).fit_predict(dataset_embs)

            score = adjusted_rand_score(graph_clusters, dataset_clusters)
            logger.info(
                f"ARI for {dataset_method} embeddings with {graph_method} graph clusters (features): {score:.4f}")

            logger.info(f"Clustering dataset embeddings using {dataset_method} method with sample standardization...")
            dataset_embs = np.vstack(data_embeddings_samples[dataset_method])
            dataset_clusters = KMeans(n_clusters=3).fit_predict(dataset_embs)
            score = adjusted_rand_score(graph_clusters, dataset_clusters)
            logger.info(
                f"ARI for {dataset_method} embeddings with {graph_method} graph clusters (samples): {score:.4f}")

        visualize_clusters(graph_clusters, graph_files, f"{graph_method}_clusters.png")


if __name__ == "__main__":
    main()
