import requests

def build_prompt(query, contexts):
    """
    Build a structured prompt for the LLM.

    Args:
        query (str): User question.
        contexts (list[str]): Retrieved documents.

    Returns:
        str: Prompt sent to LLM.
    """

    context_block = "\n\n".join(contexts)

    prompt = f"""
    You are a biomedical assistant.

    Answer briefly using the context.

    Question:
    {query}

    Context:
    {context_block}

    Short answer:
    """
    return prompt


class Generator:
    """
    Local LLM using Ollama Chat API 
    (llama3 model is too heavy for laptop > using phi3)
    """

    def __init__(self, model="phi3"):
        self.model = model
        self.url = "http://localhost:11434/api/chat"

    def generate(self, prompt):
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "num_predict": 200  # limit output length (workaround for memory issues)
                }
            }
        )

        data = response.json()

        return data["message"]["content"]


class QAAgent:
    """
    End-to-end biomedical QA system combining retrieval and generation.
    """

    def __init__(self, retriever, generator, top_k = 5):
        """
        Initialize QA agent.

        Args:
            retriever: Retrieval component
            generator: LLM generator
            top_k (int): Number of documents to retrieve (default to 5)
        """
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def answer(self, query):
        """
        Generate answer to a biomedical query.

        Args:
            query (str): User question

        Returns:
            dict: Answer + sources
        """

        # 1. Retrieve relevant documents
        results = self.retriever.search(query, top_k=self.top_k)

        # 2. Prepare the context
        contexts = [
            f"Title: {row['title']}\nAbstract: {row['abstract']}"
            for _, row in results.iterrows()
        ]

        # 3. Build the prompt
        prompt = build_prompt(query, contexts)

        # 4. Generate the answer with LLM
        answer = self.generator.generate(prompt)

        # 5. Return a structured output
        return {
            "query": query,
            "answer": answer,
            "sources": results[["title", "score"]].to_dict(orient="records")
        }