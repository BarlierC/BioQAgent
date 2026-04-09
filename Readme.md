# BioQAgent

**BioQAgent** is a biomedical question-answering (QA) agent designed to assist researchers and healthcare professionals in **exploring drug discovery literature**.  

The goal of this project is to create an **intelligent agent** that can:
- Retrieve relevant scientific abstracts from public biomedical databases (e.g., PubMed, DrugBank, ChEMBL)
- Summarize and provide clear answers to biomedical questions
- (Future) Suggest potential drug candidates or mechanisms based on retrieved literature

## Project Status

This is a **work-in-progress project**. 
The repository currently contains the initial plan and roadmap for the QA agent *(Readme file)*

## Roadmap

1. **Data collection & preprocessing**  
   - Download PubMed abstracts and DrugBank / ChEMBL datasets  
   - Clean and structure data for retrieval  

2. **Embeddings & retrieval**  
   - Convert abstracts to vector embeddings using BioBERT or PubMedBERT  
   - Implement a simple retrieval system (FAISS / Chroma)

3. **QA agent prototype**  
   - Initial agent to answer biomedical questions using retrieved documents  
   - Summarization of abstracts for concise responses

4. **Evaluation & improvements**  
   - Metrics for answer relevance and quality  
   - Optional scoring of drug candidates

5. **Interactive interface**  
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