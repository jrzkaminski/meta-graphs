import networkx as nx
import pandas as pd
import os

def find_edges(graph):
    return len(graph.edges)

def find_triangles(graph):
    triangles = list(nx.triangles(graph).values())
    return sum(triangles) // 3

def find_squares(graph):
    squares = 0
    nodes = list(graph.nodes())
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            for k in range(j+1, len(nodes)):
                for l in range(k+1, len(nodes)):
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
        for j in range(i+1, len(nodes)):
            for k in range(j+1, len(nodes)):
                for l in range(k+1, len(nodes)):
                    for m in range(l+1, len(nodes)):
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
    return {
        'edges': find_edges(graph),
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

# Main function to process files in a directory and store results in a DataFrame
def process_graph_files(directory):
    data = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                graph = read_graph_from_edgelist(file_path)
                motif_counts = count_all_motifs(graph)
                motif_counts['file'] = os.path.relpath(file_path, directory)
                data.append(motif_counts)

    df = pd.DataFrame(data)
    return df

# Example usage
if __name__ == "__main__":
    directory = "data/"
    df = process_graph_files(directory)
    print(df)
