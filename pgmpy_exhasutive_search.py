import os
import pandas as pd
from pgmpy.estimators import ExhaustiveSearch, BicScore


def learn_causal_structure(csv_path):
    # Load the dataset
    data = pd.read_csv(csv_path)

    # Learn the causal structure using Exhaustive Search
    es = ExhaustiveSearch(data, scoring_method=BicScore(data))
    model = es.estimate()

    # Extract the edges and return as a list of tuples
    edges = model.edges()
    return edges


def save_edges_to_txt(edges, output_path):
    with open(output_path, "w") as f:
        for edge in edges:
            f.write(f"{edge[0]} -> {edge[1]}\n")


def process_datasets_in_folder(folder_path):
    for subdir, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                csv_path = os.path.join(subdir, file)
                edges = learn_causal_structure(csv_path)
                txt_path = os.path.splitext(csv_path)[0] + ".txt"
                save_edges_to_txt(edges, txt_path)
                print(f"Processed {csv_path}, edges saved to {txt_path}")


if __name__ == "__main__":
    folder_path = "data/uci_data/"  # Specify your folder path here
    process_datasets_in_folder(folder_path)
