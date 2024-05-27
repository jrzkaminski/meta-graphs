import os
import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
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
import plotly.express as px


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
class GraphSAGENet(torch.nn.Module):
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
    dataset = dataset.fillna(dataset.mean())
    dataset = dataset.apply(pd.to_numeric, errors="coerce")
    dataset = dataset.fillna(0)
    return dataset.astype(np.float32)


# Function to generate embeddings using various methods
def generate_embeddings(dataset):
    tsne = TSNE(n_components=3)
    tsne_embeddings = tsne.fit_transform(dataset).mean(axis=0)

    umap = UMAP(n_components=3)
    umap_embeddings = umap.fit_transform(dataset).mean(axis=0)

    se = SpectralEmbedding(n_components=3)
    se_embeddings = se.fit_transform(dataset).mean(axis=0)

    fa = FactorAnalysis(n_components=3)
    fa_embeddings = fa.fit_transform(dataset).mean(axis=0)

    isomap = Isomap(n_components=3)
    isomap_embeddings = isomap.fit_transform(dataset).mean(axis=0)

    return (
        tsne_embeddings,
        umap_embeddings,
        se_embeddings,
        fa_embeddings,
        isomap_embeddings,
    )


# Function to train autoencoder and generate embeddings
def autoencoder_embeddings(dataset):
    encoding_dim = 3
    autoencoder = Autoencoder(dataset.shape[1], encoding_dim)
    optimizer = Adam(autoencoder.parameters())
    criterion = nn.MSELoss()
    dataset_torch = torch.tensor(dataset.values, dtype=torch.float32)
    dataset_torch = (dataset_torch - dataset_torch.min()) / (
        dataset_torch.max() - dataset_torch.min()
    )

    for epoch in range(50):
        autoencoder.train()
        optimizer.zero_grad()
        outputs = autoencoder(dataset_torch)
        loss = criterion(outputs, dataset_torch)
        loss.backward()
        optimizer.step()

    autoencoder.eval()
    with torch.no_grad():
        return autoencoder.encoder(dataset_torch).mean(axis=0).numpy()


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


# Function to train GCN model and generate embeddings
def gcn_embeddings(data):
    gcn_model = GCN(num_node_features=10, hidden_dim=16, num_classes=3)
    optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    gcn_model.train()
    for epoch in range(200):
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

    sage_model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = sage_model(data)
        loss = criterion(out, torch.tensor([0 for _ in data.x], dtype=torch.long))
        loss.backward()
        optimizer.step()

    sage_model.eval()
    with torch.no_grad():
        return sage_model.conv1(data.x, data.edge_index).mean(axis=0).numpy()


# Function to plot and save embeddings
def save_3d_scatter(embeddings, clusters, dataset_names, title, filename):
    fig = px.scatter_3d(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        z=embeddings[:, 2],
        color=clusters,
        text=dataset_names,
        title=title,
    )
    fig.write_html(filename)


def main():
    data_files = load_datasets("data/")
    data_embeddings = []
    gcn_embeddings_list = []
    sage_embeddings_list = []
    dataset_names = []

    for file in data_files:
        dataset = preprocess_dataset(file)
        (
            tsne_embeddings,
            umap_embeddings,
            se_embeddings,
            fa_embeddings,
            isomap_embeddings,
        ) = generate_embeddings(dataset)
        autoencoder_emb = autoencoder_embeddings(dataset)
        data_embeddings.append(
            (
                file,
                tsne_embeddings,
                umap_embeddings,
                se_embeddings,
                fa_embeddings,
                isomap_embeddings,
                autoencoder_emb,
            )
        )
        graph_file = file.replace(".csv", ".txt")
        graph_data = load_graph(graph_file)
        gcn_emb = gcn_embeddings(graph_data)
        gcn_embeddings_list.append(gcn_emb)
        sage_emb = sage_embeddings(graph_data)
        sage_embeddings_list.append(sage_emb)
        dataset_names.append(os.path.basename(file).replace(".csv", ""))

    graph_embeddings_methods = {
        "gcn": gcn_embeddings_list,
        "sage": sage_embeddings_list,
    }

    dataset_embeddings_methods = {
        "tsne": [emb[1] for emb in data_embeddings],
        "umap": [emb[2] for emb in data_embeddings],
        "se": [emb[3] for emb in data_embeddings],
        "fa": [emb[4] for emb in data_embeddings],
        "isomap": [emb[5] for emb in data_embeddings],
        "autoencoder": [emb[6] for emb in data_embeddings],
    }

    for graph_method, graph_embs in graph_embeddings_methods.items():
        graph_clusters = KMeans(n_clusters=3).fit_predict(graph_embs)
        for dataset_method, dataset_embs in dataset_embeddings_methods.items():
            dataset_embs_np = np.array(dataset_embs)
            save_3d_scatter(
                dataset_embs_np,
                graph_clusters,
                dataset_names,
                f"{dataset_method.upper()} Embeddings with {graph_method.upper()} Graph Clusters",
                f"{dataset_method}_{graph_method}_embeddings.html",
            )


if __name__ == "__main__":
    main()
