import networkx as nx
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster


def find_edges(graph):
    return len(graph.edges)


def find_triangles(graph):
    triangles = list(nx.triangles(graph).values())
    return sum(triangles) // 3


def find_squares(graph):
    squares = 0
    nodes = list(graph.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            for k in range(j + 1, len(nodes)):
                for l in range(k + 1, len(nodes)):
                    if graph.has_edge(nodes[i], nodes[j]) and graph.has_edge(nodes[j], nodes[k]) and \
                            graph.has_edge(nodes[k], nodes[l]) and graph.has_edge(nodes[l], nodes[i]):
                        squares += 1
    return squares


def find_stars(graph):
    stars = 0
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        if len(neighbors) > 2:
            stars += 1
    return stars


# Add more motifs as needed
def find_pentagons(graph):
    pentagons = 0
    nodes = list(graph.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            for k in range(j + 1, len(nodes)):
                for l in range(k + 1, len(nodes)):
                    for m in range(l + 1, len(nodes)):
                        if graph.has_edge(nodes[i], nodes[j]) and graph.has_edge(nodes[j], nodes[k]) and \
                                graph.has_edge(nodes[k], nodes[l]) and graph.has_edge(nodes[l], nodes[m]) and \
                                graph.has_edge(nodes[m], nodes[i]):
                            pentagons += 1
    return pentagons


def find_cliques(graph, size):
    cliques = list(nx.find_cliques(graph))
    return sum(1 for clique in cliques if len(clique) == size)


def find_wheels(graph):
    wheels = 0
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        if len(neighbors) > 3:
            subgraph = graph.subgraph(neighbors)
            if nx.is_connected(subgraph):
                wheels += 1
    return wheels


def find_diamonds(graph):
    diamonds = 0
    for edge in graph.edges():
        u, v = edge
        neighbors_u = set(graph.neighbors(u))
        neighbors_v = set(graph.neighbors(v))
        common_neighbors = neighbors_u.intersection(neighbors_v)
        if len(common_neighbors) >= 2:
            diamonds += 1
    return diamonds


def find_house(graph):
    houses = 0
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        if len(neighbors) >= 4:
            subgraph = graph.subgraph(neighbors)
            if nx.is_connected(subgraph):
                houses += 1
    return houses


def find_tent(graph):
    tents = 0
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        if len(neighbors) >= 3:
            subgraph = graph.subgraph(neighbors)
            if nx.is_connected(subgraph):
                tents += 1
    return tents


# Add more motif functions if necessary

# Function to read graph from edge list file
def read_graph_from_edgelist(file_path):
    return nx.read_edgelist(file_path)


# Function to count all motifs
def count_all_motifs(graph):
    motifs = {
        'triangles': find_triangles(graph),
        'squares': find_squares(graph),
        'stars': find_stars(graph),
        'pentagons': find_pentagons(graph),
        'cliques_4': find_cliques(graph, 4),
        'wheels': find_wheels(graph),
        'diamonds': find_diamonds(graph),
        'houses': find_house(graph),
        'tents': find_tent(graph)
    }

    max_motif_count = max(motifs.values())
    if max_motif_count == 0:
        max_motif_count = 1  # To avoid division by zero
    for key in motifs:
        motifs[key] = motifs[key] / max_motif_count
    return motifs


# Main function to process files in a directory and store results in a DataFrame
def process_graph_files(directory):
    data = []
    graphs = {}
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, directory)
                graph = read_graph_from_edgelist(file_path)
                graphs[relative_path] = graph
                motif_counts = count_all_motifs(graph)
                motif_counts['file'] = relative_path
                data.append(motif_counts)

    df = pd.DataFrame(data)
    print("DataFrame columns:", df.columns)  # Debug statement to check columns
    print("DataFrame head:\n", df.head())  # Debug statement to check DataFrame content
    return df, graphs


# Function to cluster datasets based on motif counts
# def cluster_datasets(df, n_clusters):
#     df_features = df.drop(columns=['file'])
#     kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(df_features)
#     df['cluster'] = kmeans.labels_
#     return df, kmeans.labels_

def cluster_datasets(df, n_clusters):
    df_features = df.drop(columns=['file'])

    # Perform hierarchical clustering
    linkage_matrix = linkage(df_features, method='ward')

    # Assign clusters
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

    # Add the cluster labels to the DataFrame
    df['cluster'] = cluster_labels

    return df, cluster_labels


# Function to plot graphs in clusters and save the figure
def plot_clusters(df, graphs, n_clusters, output_file):
    cluster_colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
    cluster_counts = df['cluster'].value_counts().sort_index()
    max_cluster_size = cluster_counts.max()

    fig, axes = plt.subplots(n_clusters, max_cluster_size, figsize=(5 * max_cluster_size, 5 * n_clusters))

    for i in range(n_clusters):
        cluster_data = df[df['cluster'] == i]
        for j, (idx, row) in enumerate(cluster_data.iterrows()):
            graph = graphs[row['file']]
            pos = nx.spring_layout(graph)
            ax = axes[i, j] if n_clusters > 1 else axes[j]
            nx.draw(graph, pos, ax=ax, node_color=cluster_colors[i], with_labels=True, node_size=50)
            ax.set_title(f"{row['file']}")
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


# Example usage
if __name__ == "__main__":
    directory = "data/synthetic_data_small_linear"
    output_file = "clustered_graphs_linear.png"  # Output file path
    n_clusters = 5  # Change this to the number of clusters you want

    # Process files and count motifs
    df, graphs = process_graph_files(directory)

    # Cluster datasets
    df, labels = cluster_datasets(df, n_clusters)

    df_cl = df[['file', 'cluster']]

    # remove filepaths from the file column
    df_cl['file'] = df_cl['file'].apply(lambda x: x.split('/')[-1])

    # remove txt extension from the file column
    df_cl['file'] = df_cl['file'].apply(lambda x: x.split('.')[0])

    # Save clustering results to CSV
    df_cl.to_csv("clustering_results_graph_patterns_linear.csv", index=False)

    # Plot clustered graphs and save the figure
    plot_clusters(df, graphs, n_clusters, output_file)
