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
    ground_truth_file = 'clustering_results_graph_patterns_linear.csv'
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

# All dependencies are linear, no noise
# Adjusted Rand Index: 0.02503713948378174
# Normalized Mutual Information: 0.08286590358141625
# Fowlkes-Mallows Index: 0.33640748029653916

# All dependencies are linear, no noise, normalized
# Adjusted Rand Index: 0.01640005034855057
# Normalized Mutual Information: 0.06700428828298421
# Fowlkes-Mallows Index: 0.29270139469225775

# All dependencies are linear, no noise, normalized
# Hirarchical clustering for both graphs and datasets
# Adjusted Rand Index: 0.015475266858622157
# Normalized Mutual Information: 0.06798941114662087
# Fowlkes-Mallows Index: 0.2888442473004398

# linear, no noise, normalized, KDE for EMD
# Adjusted Rand Index: 0.04646701692166725
# Normalized Mutual Information: 0.10378519064665899
# Fowlkes-Mallows Index: 0.3551501381607059
