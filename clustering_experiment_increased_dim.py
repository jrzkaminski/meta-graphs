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
    dataset = pd.get_dummies(dataset)
    if 'Unnamed: 0' in dataset.columns:
        dataset = dataset.drop(columns=['Unnamed: 0'])
    dataset = dataset.fillna(dataset.mean())
    dataset = dataset.apply(pd.to_numeric, errors="coerce")
    dataset = dataset.fillna(0)
    return dataset.astype(np.float32)


# Function to standardize dataset features
def standardize_features(datasets):
    max_features = max(dataset.shape[1] for dataset in datasets)
    standardized_datasets = []
    for dataset in datasets:
        if dataset.shape[1] < max_features:
            extra_features = np.zeros((dataset.shape[0], max_features - dataset.shape[1]))
            standardized_dataset = np.hstack((dataset, extra_features))
        else:
            standardized_dataset = dataset
        standardized_datasets.append(standardized_dataset)
    return standardized_datasets


# Function to standardize dataset samples
def standardize_samples(datasets):
    max_samples = max(dataset.shape[0] for dataset in datasets)
    standardized_datasets = []
    for dataset in datasets:
        if dataset.shape[0] < max_samples:
            extra_samples = np.zeros((max_samples - dataset.shape[0], dataset.shape[1]))
            standardized_dataset = np.vstack((dataset, extra_samples))
        else:
            standardized_dataset = dataset
        standardized_datasets.append(standardized_dataset)
    return standardized_datasets


# Function to generate embeddings using various methods
def generate_embeddings(dataset, method):
    if method == 'tsne':
        model = TSNE(n_components=1, n_jobs=-1)
    elif method == 'umap':
        model = UMAP(n_components=1, n_jobs=-1)
    elif method == 'se':
        model = SpectralEmbedding(n_components=1)
    elif method == 'fa':
        model = FactorAnalysis(n_components=1)
    elif method == 'kpca':
        model = KernelPCA(n_components=1, kernel='rbf', n_jobs=-1)
    else:
        raise ValueError(f"Unknown method: {method}")

    return model.fit_transform(dataset).flatten()


# Function to train autoencoder and generate embeddings
def autoencoder_embeddings(dataset):
    if isinstance(dataset, pd.DataFrame):
        dataset = dataset.values
    encoding_dim = 1
    autoencoder = Autoencoder(dataset.shape[1], encoding_dim)
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
        print(edges)
    G = nx.DiGraph()
    G.add_edges_from(edges)
    mapping = {node: idx for idx, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    for i in G.nodes:
        G.nodes[i]["feature"] = [1.0] * 10
    data = from_networkx(G)
    data.x = torch.tensor([G.nodes[i]["feature"] for i in G.nodes], dtype=torch.float32)
    return data


# Function to train GCN model and generate embeddings
def gcn_embeddings(data):
    gcn_model = GCN(num_node_features=10, hidden_dim=16, num_classes=3)
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
    sage_model = GraphSAGENet(in_channels=10, hidden_channels=16, out_channels=3)
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


def main():
    logger.info("Loading datasets...")
    data_files = load_datasets("data/")
    datasets = [preprocess_dataset(file) for file in data_files]

    logger.info("Standardizing dataset features...")
    standardized_features = standardize_features(datasets)
    logger.info("Standardizing dataset samples...")
    standardized_samples = standardize_samples(datasets)

    data_embeddings = []
    dataset_names = [os.path.basename(file).replace(".csv", "") for file in data_files]
    dataset_embedding_methods = [
        # 'tsne',
        #                            'umap',
                                   # 'se',
                                   # 'fa',
                                   # 'kpca',
                                   'autoencoder']
    for method in dataset_embedding_methods:
        logger.info(f"Generating embeddings using {method} method...")
        for dataset in tqdm(standardized_features, desc=f"Feature standardization with {method}"):
            if method == 'autoencoder':
                embedding = autoencoder_embeddings(dataset)
            else:
                embedding = generate_embeddings(dataset, method)
            data_embeddings.append((method, embedding))

        for dataset in tqdm(standardized_samples, desc=f"Sample standardization with {method}"):
            if method == 'autoencoder':
                embedding = autoencoder_embeddings(dataset)
            else:
                embedding = generate_embeddings(dataset, method)
            data_embeddings.append((method, embedding))

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
    for graph_method, graph_embs in graph_embeddings_methods.items():
        logger.info(f"Clustering with {graph_method} embeddings...")
        graph_clusters = KMeans(n_clusters=3).fit_predict(graph_embs)
        for dataset_method in dataset_embedding_methods:
            dataset_embs = np.array([emb for method, emb in data_embeddings if method == dataset_method])
            dataset_clusters = KMeans(n_clusters=3).fit_predict(dataset_embs)
            score = adjusted_rand_score(graph_clusters, dataset_clusters)
            logger.info(f"ARI for {dataset_method} embeddings with {graph_method} graph clusters: {score:.4f}")


if __name__ == "__main__":
    main()
