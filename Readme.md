# BioQAgent

**BioQAgent** is a biomedical question-answering (QA) agent designed to assist researchers and healthcare professionals in **exploring drug discovery literature**.  

The goal of this project is to create an **intelligent agent** that can:
- Retrieve relevant scientific abstracts from public biomedical databases (e.g., PubMed, DrugBank, ChEMBL)
- Summarize and provide clear answers to biomedical questions
- (Future) Suggest potential drug candidates or mechanisms based on retrieved literature

*This project is built with the support of AI-assisted tools as part of an active learning process in biomedical machine learning and software engineering. The goal is to accelerate exploration while maintaining full understanding of the underlying methods and implementations.*

## Project Status

This is a **work-in-progress project**: lightweight QA agent has been implemented. 
**Current stage**: simple UI development.

## Roadmap

1. **Data collection & preprocessing**  
   - Download PubMed abstracts
   - Clean and structure data for retrieval  

2. **Embeddings & retrieval**  
   - Convert abstracts to vector embeddings
   - Implement a simple retrieval system
   - Evaluation: metrics for answer relevance and quality 

3. **QA agent prototype**
   - Initial agent to answer biomedical questions using retrieved documents  
   - Summarization of abstracts for concise responses

4. Interactive interface  
   - Simple Streamlit app for testing questions interactively

## Data collection and preprocessing

```bash
# Set email (NCBI requirement)
export ENTREZ_EMAIL="your_email@example.com"

# Fetch the raw data
python scripts/download_data.py

# Run the preprocessing
python scripts/preprocess_data.py
```

## Generate the embeddings

*I initially used FAISS but replaced it with cosine similarity (sklearn-based approach) due to stability issues on local environments. This alternative approach is sufficient for small-scale datasets and thus the scope of this project.*

```bash
python scripts/build_embeddings.py
```

## QA agent prototype

### LLM-based QA Agent

This project **initially aimed** to implement a full Retrieval-Augmented Generation **(RAG) pipeline**, combining semantic retrieval with **LLM-based** answer generation.

**Implementations** considered or tested:

1. OpenAI API:
- Successfully designed the integration
- Not executed due to lack of API key and billing setup
- **Option discarded** to maintain a **fully free and reproducible project**

2. **Local LLMs with Ollama**

Tested models: `llama3` and `phi3`

While the integration was technically successful, execution revealed **major limitations** including an **extremely slow inference**, **high CPU usage** (100%) and **system instability**.

**Limitation**: due to hardware constraints (CPU-only environment), the LLM-based QA system could not be executed reliably.

As a result: generated answers could not be evaluated and the **RAG pipeline could not be validated end-to-end**.

The **full implementation of the LLM-based QA agent** based on **Ollama phi3 model** is **still included in this repository** in the experimental/LLM_QA/ folder (python module and notebook)
*Note: this code is provided for completeness but was not fully executed due to the limitations described above.*

### Alternative: lightweight QA Agent

Given these previously mentionned constraints, the new design provides a **strong tread-off between performance and interpretability**. A **lightweight extractive QA approach** was implemented with:
- Semantic retrieval (SentenceTransformers)
- Sentence-level similarity scoring
- Extraction of the most relevant evidence

