import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# lightweight embedding model
MODEL_NAME = "all-MiniLM-L6-v2"  


def load_data(path):
    """
    Load the preprocessed dataset from a CSV file.

    Args:
        path (str): Path to the CSV file containing cleaned abstracts.

    Returns:
        pd.DataFrame: DataFrame with at least 'title' and 'abstract' columns.
    """
    return pd.read_csv(path)


def compute_embeddings(texts, model):
    """
    Compute dense vector embeddings for a list of texts.

    Args:
        texts (list[str]): List of input texts (e.g., title + abstract).
        model (SentenceTransformer): Preloaded embedding model.

    Returns:
        np.ndarray: Matrix of embeddings (n_samples x embedding_dim).
    """
    embeddings = model.encode(texts, show_progress_bar=True)

    # Ensure float32 for memory efficiency
    return np.array(embeddings).astype("float32")


if __name__ == "__main__":

    # Step 1: Load cleaned dataset
    print("Loading data...")
    df = load_data("data/processed/pubmed_clean.csv")

    # Combine title and abstract for better semantic representation
    texts = (df["title"] + ". " + df["abstract"]).tolist()

    # Step 2: Load the embedding model
    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)

    # Step 3: Compute the embeddings
    print("Computing embeddings...")
    embeddings = compute_embeddings(texts, model)

    # Step 4: Save results
    print("Saving embeddings...")
    np.save("data/processed/embeddings.npy", embeddings)

    print("Finished !")