import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
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
    Convert a list of texts into vector embeddings using a pre-trained model.

    Args:
        texts (list[str]): List of input texts (e.g., title + abstract).
        model (SentenceTransformer): Preloaded embedding model.

    Returns:
        np.ndarray: Matrix of embeddings (n_samples x embedding_dim).
    """
    return model.encode(texts, show_progress_bar=True)


def build_faiss_index(embeddings):
    """
    Build a FAISS index for fast similarity search.

    Args:
        embeddings (np.ndarray): Matrix of embeddings.

    Returns:
        faiss.Index: FAISS index containing all embeddings.
    """
    dim = embeddings.shape[1]

    # L2 distance (Euclidean)
    index = faiss.IndexFlatL2(dim)

    # add vectors to index
    index.add(embeddings)  

    return index


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

    # Step 4: Build FAISS index
    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    # Step 5: Save results
    print("Saving index and embeddings...")
    faiss.write_index(index, "data/processed/faiss_index.bin")
    np.save("data/processed/embeddings.npy", embeddings)

    print("Finished !")