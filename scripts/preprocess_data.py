import pandas as pd
import re

def parse_medline(file_path):
    """
    Parse a MEDLINE formatted file and extract titles and abstracts.

    Args:
        file_path (str): Path to raw MEDLINE text file

    Returns:
        pd.DataFrame: DataFrame with columns ['title', 'abstract']
    """
    with open(file_path, "r") as f:
        content = f.read()
    
    entries = content.split("\n\n")
    
    data = []
    
    for entry in entries:
        # Extract title (TI field)
        title = re.search(r"TI  - (.+)", entry)
        # Extract abstract (AB field), allowing multiline content
        abstract = re.search(r"AB  - (.+)", entry, re.DOTALL)
        
        if title and abstract:
            data.append({
                "title": title.group(1).strip(),
                "abstract": abstract.group(1).replace("\n", " ").strip()
            })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Parse raw MEDLINE file
    df = parse_medline("data/raw/pubmed_raw.txt")
    
    # Basic cleaning (remove duplicates and very short abstracts)
    df = df.drop_duplicates()
    df = df[df["abstract"].str.len() > 50]
    
    # Save cleaned text to csv
    df.to_csv("data/processed/pubmed_clean.csv", index=False)
    
    print(f"Saved {len(df)} cleaned abstracts")