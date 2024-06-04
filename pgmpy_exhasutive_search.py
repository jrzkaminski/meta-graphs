import os
import pandas as pd
from pgmpy.estimators import ExhaustiveSearch, HillClimbSearch, K2Score, BicScore
from sklearn.preprocessing import KBinsDiscretizer


def learn_causal_structure(csv_path):
    # Load the dataset
    data = pd.read_csv(csv_path)
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)

    # Select columns to discretize: float columns and integer columns with more than 10 unique values
    continuous_columns = data.select_dtypes(include='float').columns
    int_columns = [col for col in data.select_dtypes(include='int64').columns if data[col].nunique() > 6]
    columns_to_discretize = continuous_columns.union(int_columns)
    print(data.dtypes)
    if len(columns_to_discretize) > 0:
        discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
        data[columns_to_discretize] = discretizer.fit_transform(data[columns_to_discretize])

    # Choose the search method based on the number of nodes
    if data.shape[1] > 5:
        # Use Hill Climb Search
        hc = HillClimbSearch(data)
        model = hc.estimate(scoring_method=BicScore(data))
    else:
        # Use Exhaustive Search
        es = ExhaustiveSearch(data, scoring_method=K2Score(data))
        model = es.estimate()

    # Extract the edges and return as a list of tuples
    edges = model.edges()
    return edges


def save_edges_to_txt(edges, output_path):
    with open(output_path, 'w') as f:
        for edge in edges:
            f.write(f"{edge[0]} {edge[1]}\n")


def process_datasets_in_folder(folder_path):
    for subdir, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(subdir, file)
                txt_path = os.path.splitext(csv_path)[0] + '.txt'
                print("processing: ", csv_path)
                if os.path.exists(txt_path):
                    print(f"Skipping {csv_path}, corresponding .txt file already exists.")
                else:
                    edges = learn_causal_structure(csv_path)
                    save_edges_to_txt(edges, txt_path)
                    print(f"Processed {csv_path}, edges saved to {txt_path}")


if __name__ == "__main__":
    folder_path = 'data/uci_data/'  # Specify your folder path here
    process_datasets_in_folder(folder_path)
