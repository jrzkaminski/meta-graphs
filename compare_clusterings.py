import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score


# Function to read CSV and merge based on Dataset column
def read_and_merge_clusters(ground_truth_file, comparison_file):
    ground_truth_df = pd.read_csv(ground_truth_file)
    comparison_df = pd.read_csv(comparison_file)

    # Merge dataframes on the 'Dataset' column
    merged_df = pd.merge(ground_truth_df, comparison_df, on='Dataset', suffixes=('_ground_truth', '_comparison'))

    return merged_df['Cluster_ground_truth'].values, merged_df['Cluster_comparison'].values


# Function to compare clusterings
def compare_clusterings(ground_truth_file, comparison_file):
    ground_truth_clusters, comparison_clusters = read_and_merge_clusters(ground_truth_file, comparison_file)

    ari = adjusted_rand_score(ground_truth_clusters, comparison_clusters)
    nmi = normalized_mutual_info_score(ground_truth_clusters, comparison_clusters)
    fmi = fowlkes_mallows_score(ground_truth_clusters, comparison_clusters)

    return {
        'Adjusted Rand Index': ari,
        'Normalized Mutual Information': nmi,
        'Fowlkes-Mallows Index': fmi
    }


# Example usage
if __name__ == "__main__":
    ground_truth_file = 'clustering_results_graph_patterns.csv'
    comparison_file = 'dataset_clusters.csv'

    metrics = compare_clusterings(ground_truth_file, comparison_file)

    print("Clustering Comparison Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

# real
# Adjusted Rand Index: -0.007281869238656888
# Normalized Mutual Information: 0.1110798030808294
# Fowlkes-Mallows Index: 0.4610569539421821

# Synthetic
# Adjusted Rand Index: 0.02166603526498552
# Normalized Mutual Information: 0.11051451764115085
# Fowlkes-Mallows Index: 0.5086816433388088