from sklearn.metrics.pairwise import cosine_similarity
import re

def split_sentences(text):
    """
    Split text into sentences using simple regex.

    Args:
        text (str): Input abstract.

    Returns:
        list[str]: Sentences.
    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    return sentences


def rank_sentences(query_vec, sentences, model):
    """
    Rank sentences based on similarity to query.

    Args:
        query_vec (np.ndarray): Query embedding.
        sentences (list[str]): Candidate sentences.
        model: embedding model.

    Returns:
        list of (sentence, score)
    """

    sent_embeddings = model.encode(sentences).astype("float32")

    scores = cosine_similarity(query_vec, sent_embeddings)[0]

    ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)

    return ranked

class LightweightQAAgent:
    """
    Lightweight QA system using retrieval + sentence extraction.
    No LLM required.
    """

    def __init__(self, retriever, top_k_docs=2, top_k_sentences=3):
        self.retriever = retriever
        self.model = retriever.model
        self.top_k_docs = top_k_docs
        self.top_k_sentences = top_k_sentences

    def answer(self, query):
        """
        Generate answer using extractive approach.

        Args:
            query (str): User query.

        Returns:
            dict: Answer + supporting sentences.
        """

        # 1. Retrieve documents
        docs = self.retriever.search(query, top_k=self.top_k_docs)

        # 2. Encode query once
        query_vec = self.model.encode([query]).astype("float32")

        all_sentences = []

        # 3. Extract sentences from abstracts
        for _, row in docs.iterrows():
            sentences = split_sentences(row["abstract"])

            ranked = rank_sentences(query_vec, sentences, self.model)

            all_sentences.extend([
                (s, score, row["title"])
                for s, score in ranked[:self.top_k_sentences]
            ])

        # 4. Global ranking
        all_sentences = sorted(all_sentences, key=lambda x: x[1], reverse=True)

        top_sentences = all_sentences[:self.top_k_sentences]

        # 5. Build answer
        answer = " ".join([s[0] for s in top_sentences])

        sources = [
            {"title": s[2], "score": float(s[1])}
            for s in top_sentences
        ]

        return {
            "query": query,
            "answer": answer,
            "sources": sources
        }
