from Bio import Entrez
import pandas as pd
from tqdm import tqdm
import time
import os

# Retrieve email (NCBI requirement)
email = os.getenv("ENTREZ_EMAIL")

# Quick check
if email is None:
    raise ValueError("Please set the ENTREZ_EMAIL environment variable")
Entrez.email = email

def fetch_pubmed_ids(query, max_results=5000):
    """
    Query the PubMed database and retrieve a list of article IDs.

    Args:
        query (str): PubMed search query (e.g., "drug discovery AND target protein")
        max_results (int): Maximum number of article IDs to retrieve, default to 5k

    Returns:
        list: List of PubMed IDs (PMIDs)
    """
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results
    )

    record = Entrez.read(handle)

    return record["IdList"]

def fetch_abstracts(id_list):
    """
    Fetch abstracts from PubMed using a list of IDs (retrieval in batches to avoid API overload).

    Args:
        id_list (list): List of PubMed IDs

    Returns:
        list: List of raw MEDLINE text blocks
    """

    records = []

    # Process IDs in batches of 100 (recommended by NCBI)
    for i in tqdm(range(0, len(id_list), 100)):
        batch = id_list[i:i+100]
        handle = Entrez.efetch(
            db="pubmed",
            id=",".join(batch),
            rettype="medline",
            retmode="text"
        )
        data = handle.read()
        
        records.append(data)
        time.sleep(0.3)  # avoid hitting API limits
    
    return records

if __name__ == "__main__":
    # Search query
    query = """
            ("drug discovery"[Title/Abstract] OR "drug development"[Title/Abstract])
            AND ("target protein"[Title/Abstract] OR "therapeutic target"[Title/Abstract])
            AND (inhibitor OR agonist OR antagonist OR "small molecule")
            AND ("cancer" OR "disease" OR "therapy")
            """
    
    print("Fetching PubMed IDs...")
    ids = fetch_pubmed_ids(query, max_results=2000)
    
    print(f"Retrieved {len(ids)} IDs")
    
    print("Fetching abstracts...")
    raw_data = fetch_abstracts(ids)
    
    # Save raw MEDLINE data
    with open("data/raw/pubmed_raw.txt", "w") as f:
        for entry in raw_data:
            f.write(entry + "\n")
    
    print("Data saved to data/raw/pubmed_raw.txt")