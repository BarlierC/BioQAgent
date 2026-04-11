import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# lightweight embedding model
MODEL_NAME = "all-MiniLM-L6-v2"

from sklearn.metrics.pairwise import cosine_similarity

class Retriever:
    """
    A semantic search engine for biomedical abstracts using cosine similarity (sklearn-based approach).

    It allows querying the dataset using natural language.
    """

    def __init__(self, embeddings_path, data_path):
        """
        Initialize the Retriever by loading all required components.

        Args:
            embeddings_path (str): Path to saved embeddings (.npy).
            data_path (str): Path to CSV dataset.
        """

        # Load embedding model
        self.model = SentenceTransformer(MODEL_NAME, device="cpu")

        # Load precomputed embeddings
        self.embeddings = np.load(embeddings_path).astype("float32")

        # Load original dataset
        self.df = pd.read_csv(data_path)

    def search(self, query, top_k = 5):
        """
        Search for the most relevant abstracts given a query.

        Args:
            query (str): Natural language query
            top_k (int): Number of results to return, default to 5

        Returns:
            pd.DataFrame: Top-k most relevant rows from the dataset.
        """
        
        # Encode query
        query_vec = self.model.encode(query)

        # Ensure/force correct shape: (1, dim)
        query_vec = np.asarray(query_vec).astype("float32")

        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        # Perform similarity search using cosine
        similarities = cosine_similarity(query_vec, self.embeddings)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Return results
        results = self.df.iloc[top_indices].copy()
        results["score"] = similarities[top_indices]

        return results