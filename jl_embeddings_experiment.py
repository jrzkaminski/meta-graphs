import os
import logging
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.random_projection import GaussianRandomProjection, johnson_lindenstrauss_min_dim
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from tqdm import tqdm


class JLEmbeddingGenerator:
    def __init__(self, input_folder, eps=0.1):
        self.input_folder = input_folder
        self.eps = eps
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        # Create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Add formatter to ch
        ch.setFormatter(formatter)

        # Add ch to logger
        logger.addHandler(ch)

        return logger

    def _load_files(self):
        csv_files = []
        dag_files = []
        for root, _, files in os.walk(self.input_folder):
            for file in files:
                if file.endswith(".csv"):
                    csv_files.append(os.path.join(root, file))
                    dag_file = os.path.join(root, file.replace(".csv", ".txt"))
                    if os.path.exists(dag_file):
                        dag_files.append(dag_file)
                    else:
                        dag_files.append(None)
        return csv_files, dag_files

    def _preprocess_data(self, df):
        # Identify categorical and numerical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        numerical_cols = df.select_dtypes(include=['number']).columns

        # Define transformers for preprocessing
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        # Combine transformers into a ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        return preprocessor

    def _apply_jl_embedding(self, X, n_components):
        jlp = GaussianRandomProjection(n_components=n_components, eps=self.eps)
        X_embedded = jlp.fit_transform(X)
        return X_embedded

    def process_dataset(self, file):
        df = pd.read_csv(file)

        # Check if the first column is an index column and drop it
        if df.columns[0] == "" or df.columns[0].lower() in ["index", "id"]:
            df = df.iloc[:, 1:]

        preprocessor = self._preprocess_data(df)
        X_processed = preprocessor.fit_transform(df)

        n_components = 1

        X_embedded = self._apply_jl_embedding(X_processed, n_components)

        self.logger.info(f"Processed {file}")
        self.logger.info(f"Initial dimensionality: {X_processed.shape[1]}")
        self.logger.info(f"Reduced dimensionality: {X_embedded.shape[1]}")

        return X_embedded

    def process_dag(self, dag_file):
        # Assuming the DAG is stored in an edge list format in the txt file
        G = nx.read_edgelist(dag_file, create_using=nx.DiGraph)
        data = self._convert_to_pyg_data(G)
        gcn_model = GCN(num_node_features=data.num_node_features, hidden_dim=64, output_dim=32)
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

    def _convert_to_pyg_data(self, G):
        mapping = {node: idx for idx, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        for i in G.nodes:
            G.nodes[i]["feature"] = [1.0] * 10
        data = from_networkx(G)
        data.x = torch.tensor([G.nodes[i]["feature"] for i in G.nodes], dtype=torch.float32)
        edge_index = torch.tensor(list(G.edges)).t().contiguous()
        x = torch.eye(G.number_of_nodes(), dtype=torch.float)
        return Data(x=x, edge_index=edge_index)


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


if __name__ == "__main__":
    input_folder = 'data/'  # Folder containing datasets
    eps = 0.1  # Error tolerance

    embedding_generator = JLEmbeddingGenerator(input_folder, eps)

    # Load CSV and DAG files
    csv_files, dag_files = embedding_generator._load_files()

    jl_embeddings = []
    dag_embeddings = []
    for csv_file, dag_file in zip(csv_files, dag_files):
        jl_emb = embedding_generator.process_dataset(csv_file)
        jl_embeddings.append(jl_emb)

        if dag_file:
            dag_emb = embedding_generator.process_dag(dag_file)
            dag_embeddings.append(dag_emb)

    # Pad JL embeddings
    max_length = max(emb.shape[0] for emb in jl_embeddings)
    print(max_length)
    padded_jl_embeddings = np.array([np.pad(emb, (0, max_length - emb.shape[0]), 'constant') for emb in jl_embeddings])
    #
    # # Cluster embeddings
    # jl_embeddings_matrix = np.array(padded_jl_embeddings)
    dag_embeddings_matrix = np.array(dag_embeddings)
    print(dag_embeddings_matrix.shape)
    # n_clusters = 3
    # kmeans_jl = KMeans(n_clusters=n_clusters)
    # kmeans_dag = KMeans(n_clusters=n_clusters)
    #
    # jl_labels = kmeans_jl.fit_predict(jl_embeddings_matrix)
    # dag_labels = kmeans_dag.fit_predict(dag_embeddings_matrix)
    #
    # ari = adjusted_rand_score(dag_labels, jl_labels)
    # embedding_generator.logger.info(f"ARI between GCN and JL embeddings: {ari}")
