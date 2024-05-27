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

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, encoding_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(encoding_dim, input_dim), nn.Sigmoid())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Define GCN model
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

# Define GraphSAGE model
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

# Load the datasets
data_files = []
for root, dirs, files in os.walk("data/"):
    for file in files:
        if file.endswith(".csv"):
            data_files.append(os.path.join(root, file))

# Initialize the lists
data_embeddings = []
gcn_embeddings = []
sage_embeddings = []
file_names = []

for file in data_files:
    # Load the dataset
    dataset = pd.read_csv(file)

    # One-Hot Encoding for categorical features
    dataset = pd.get_dummies(dataset)

    # Handle missing values by filling them with the mean of each column
    dataset = dataset.fillna(dataset.mean())

    # Ensure all data is numeric
    dataset = dataset.apply(pd.to_numeric, errors="coerce")

    # Convert boolean columns to integers
    for col in dataset.select_dtypes(include="bool").columns:
        dataset[col] = dataset[col].astype(int)

    # Fill any remaining NaN values that could result from the conversion
    dataset = dataset.fillna(0)

    # t-SNE for numerical data
    tsne = TSNE(n_components=3)
    tsne_embeddings = tsne.fit_transform(dataset).mean(axis=0)

    # UMAP for numerical data
    umap = UMAP(n_components=3)
    umap_embeddings = umap.fit_transform(dataset).mean(axis=0)

    # SE for numerical data
    se = SpectralEmbedding(n_components=3)
    se_embeddings = se.fit_transform(dataset).mean(axis=0)

    # Factor Analysis for numerical data
    fa = FactorAnalysis(n_components=3)
    fa_embeddings = fa.fit_transform(dataset).mean(axis=0)

    # Isomap for numerical data
    isomap = Isomap(n_components=3)
    isomap_embeddings = isomap.fit_transform(dataset).mean(axis=0)

    # Define the size of the encoded representations
    encoding_dim = 3

    # Define the autoencoder model
    autoencoder = Autoencoder(dataset.shape[1], encoding_dim)

    # Define the optimizer and loss function
    optimizer = Adam(autoencoder.parameters())
    criterion = nn.MSELoss()

    # Convert the dataset to PyTorch tensors
    dataset_torch = torch.tensor(dataset.values, dtype=torch.float32)

    # Normalize the data to be between 0 and 1
    dataset_torch = (dataset_torch - dataset_torch.min()) / (dataset_torch.max() - dataset_torch.min())

    # Train the autoencoder
    for epoch in range(50):
        autoencoder.train()
        optimizer.zero_grad()
        outputs = autoencoder(dataset_torch)
        loss = criterion(outputs, dataset_torch)
        loss.backward()
        optimizer.step()

    # Switch the model to evaluation mode
    autoencoder.eval()

    # Generate the embeddings
    with torch.no_grad():
        autoencoder_embeddings = autoencoder.encoder(dataset_torch).mean(axis=0).numpy()

    # Append the embeddings to the list
    data_embeddings.append(
        (
            file,
            tsne_embeddings,
            umap_embeddings,
            se_embeddings,
            fa_embeddings,
            isomap_embeddings,
            autoencoder_embeddings,
        )
    )

    # Load the corresponding graph
    graph_file = file.replace(".csv", ".txt")
    with open(graph_file, "r") as f:
        edges = [tuple(line.strip().split()) for line in f]

    # Create a directed graph
    G = nx.DiGraph()
    G.add_edges_from(edges)

    # Create a mapping from node labels to numeric indices
    mapping = {node: idx for idx, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    # Add dummy node features
    for i in G.nodes:
        G.nodes[i]["feature"] = [1.0] * 10

    data = from_networkx(G)

    # Convert node features to tensor
    data.x = torch.tensor([G.nodes[i]["feature"] for i in G.nodes], dtype=torch.float)

    # Initialize GCN model, optimizer, and loss function
    gcn_model = GCN(num_node_features=10, hidden_dim=16, num_classes=3)
    optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop for GCN
    gcn_model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = gcn_model(data)
        loss = criterion(out, torch.tensor([0 for _ in G.nodes], dtype=torch.long))  # Dummy labels
        loss.backward()
        optimizer.step()

    gcn_model.eval()
    with torch.no_grad():
        gcn_emb = gcn_model.conv1(data.x, data.edge_index).mean(axis=0).numpy()
    gcn_embeddings.append(gcn_emb)

    # Initialize GraphSAGE model, optimizer, and loss function
    sage_model = GraphSAGENet(in_channels=10, hidden_channels=16, out_channels=3)
    optimizer = torch.optim.Adam(sage_model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop for GraphSAGE
    sage_model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = sage_model(data)
        loss = criterion(out, torch.tensor([0 for _ in G.nodes], dtype=torch.long))  # Dummy labels
        loss.backward()
        optimizer.step()

    sage_model.eval()
    with torch.no_grad():
        sage_emb = sage_model.conv1(data.x, data.edge_index).mean(axis=0).numpy()
    sage_embeddings.append(sage_emb)

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

# Prepare and save plots for each combination of dataset embedding and graph embedding
graph_embeddings_methods = {
    "gcn": gcn_embeddings,
    "sage": sage_embeddings,
}

dataset_embeddings_methods = {
    "tsne": [emb[1] for emb in data_embeddings],
    "umap": [emb[2] for emb in data_embeddings],
    "se": [emb[3] for emb in data_embeddings],
    "fa": [emb[4] for emb in data_embeddings],
    "isomap": [emb[5] for emb in data_embeddings],
    "autoencoder": [emb[6] for emb in data_embeddings],
}

dataset_names = [
    os.path.basename(emb[0]).replace(".csv", "") for emb in data_embeddings
]

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
