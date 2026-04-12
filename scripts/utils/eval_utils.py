import json
import pandas as pd


def load_test_queries(path):
    """
    Load evaluation queries from JSON file.

    Args:
        path (str): Path to JSON file.

    Returns:
        list: List of query dictionaries.
    """
    with open(path, "r") as f:
        return json.load(f)


def keyword_match_score(text, keywords):
    """
    Compute a simple relevance score based on keyword matching.

    Args:
        text (str): Text to evaluate.
        keywords (list[str]): List of expected keywords.

    Returns:
        int: Number of matched keywords.
    """
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


def precision_at_k(results_df, keywords, k = 5, threshold = 2):
    """
    Compute Precision at k based on keyword matching.

    Args:
        results_df (pd.DataFrame): Retrieved results.
        keywords (list[str]): Keywords defining relevance.
        k (int): Number of top results to consider (default to 5).
        thresold (int): number of minimal words that needs to match to be relevant (default to 2)

    Returns:
        float: Precision score.
    """
    relevant = 0

    for _, row in results_df.head(k).iterrows():
        text = row["title"] + " " + row["abstract"]

        if is_relevant(text, keywords, threshold):
            relevant += 1

    return relevant / k

def is_relevant(text, keywords, threshold):
    """
    Detect if text is relevant

    Args:
        text (string): Text result.
        keywords (list[str]): Keywords defining relevance.
        thresold (int): number of minimal words that needs to match to be relevant (default to 2)

    Returns:
        boolean: True if relevant.
    """
    return keyword_match_score(text, keywords) >= threshold

def reciprocal_rank(results, keywords, threshold=2):
    for i, row in enumerate(results.itertuples()):
        text = (row.title + " " + row.abstract).lower()

        if is_relevant(text, keywords, threshold):
            return 1 / (i + 1)

    return 0.0

def ranking_gap(results):
    scores = results["score"].values
    return scores[0] - scores[1]