# Import libraries
import os
import sys
import streamlit as st
import pandas as pd

# Import the modules
sys.path.append(os.path.abspath("../scripts"))
from utils.retrieval_utils import Retriever
from utils.lightweight_qa_agent import LightweightQAAgent

# Page config
st.set_page_config(
    page_title="Biomedical QA Agent (lightweight version)",
    layout="wide"
)

st.title("Biomedical QA Agent")
st.markdown("Ask questions about biomedical research papers")


# Load system in cache


@st.cache_resource
def load_system():
    """
    Load retriever and lightweighted QA agent only once
    """

    retriever = Retriever(
        embeddings_path="../data/processed/embeddings.npy",
        data_path="../data/processed/pubmed_clean.csv"
    )

    agent = LightweightQAAgent(retriever)

    return agent


agent = load_system()


# User input
query = st.text_input(
    "Enter your question:",
    placeholder="e.g. What are JAK2 inhibitors used for?"
)


# Run QA agent
if query:

    with st.spinner("Searching and analyzing documents..."):

        response = agent.answer(query)

    # Display answer
    st.subheader("Answer")
    st.write(response["answer"])


    # Display sources
    st.subheader("Supporting evidence")
    sources_df = pd.DataFrame(response["sources"])
    st.dataframe(sources_df)


    # Expandable details
    with st.expander("Show detailed sources"):
        for i, src in enumerate(response["sources"]):
            st.markdown(f"**{i+1}. {src['title']}**")
            st.markdown(f"Score: {src['score']:.4f}")
            st.markdown("---")