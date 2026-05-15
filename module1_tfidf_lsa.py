"""
Module 1: Traditional Statistical Models (TF-IDF and LSA)
Implements TF-IDF matrix computation, keyword extraction, and LSA dimensionality reduction.
"""

import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re

# Download NLTK data (only once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


def preprocess_text(text):
    """
    Split text into sentences and clean them.

    Args:
        text: Raw text string

    Returns:
        List of cleaned sentences
    """
    # Split into sentences
    sentences = sent_tokenize(text)

    # Clean each sentence
    cleaned_sentences = []
    for sent in sentences:
        # Remove extra whitespace and convert to lowercase
        cleaned = ' '.join(sent.split()).strip()
        if len(cleaned) > 10:  # Filter out very short sentences
            cleaned_sentences.append(cleaned)

    return cleaned_sentences


def compute_tfidf(documents):
    """
    Compute TF-IDF matrix from documents.

    Args:
        documents: List of document strings

    Returns:
        tuple: (tfidf_matrix, feature_names, vectorizer)
    """
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        max_features=100,
        ngram_range=(1, 1)
    )

    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    return tfidf_matrix, feature_names, vectorizer


def extract_top_keywords(tfidf_matrix, feature_names, top_n=5):
    """
    Extract top N keywords based on TF-IDF scores.

    Args:
        tfidf_matrix: Sparse TF-IDF matrix
        feature_names: Array of feature names
        top_n: Number of top keywords to extract

    Returns:
        List of (keyword, score) tuples
    """
    # Sum TF-IDF scores across all documents
    summed_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()

    # Get top N indices
    top_indices = summed_scores.argsort()[-top_n:][::-1]

    # Get keywords and scores
    keywords = [(feature_names[i], summed_scores[i]) for i in top_indices]

    return keywords


def apply_lsa(tfidf_matrix, n_components=2):
    """
    Apply LSA (TruncatedSVD) for dimensionality reduction.

    Args:
        tfidf_matrix: Sparse TF-IDF matrix
        n_components: Number of dimensions to reduce to

    Returns:
        tuple: (lsa_matrix, svd_model)
    """
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    lsa_matrix = svd.fit_transform(tfidf_matrix)

    return lsa_matrix, svd


def apply_lsa_to_words(documents, n_components=2):
    """
    Apply LSA to word-level document-term matrix for word visualization.

    Args:
        documents: List of document strings
        n_components: Number of dimensions

    Returns:
        tuple: (word_coordinates, vocabulary)
    """
    # Use CountVectorizer for word-level representation
    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words='english',
        max_features=50
    )

    doc_term_matrix = vectorizer.fit_transform(documents)
    vocabulary = vectorizer.get_feature_names_out()

    # Apply LSA
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    # Transpose to get word vectors
    word_matrix = svd.fit_transform(doc_term_matrix.T)

    return word_matrix, vocabulary


def visualize_lsa_words(word_matrix, vocabulary):
    """
    Create a 2D scatter plot of words using LSA coordinates.

    Args:
        word_matrix: Word coordinate matrix from LSA
        vocabulary: Array of vocabulary words

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Add scatter points
    fig.add_trace(go.Scatter(
        x=word_matrix[:, 0],
        y=word_matrix[:, 1],
        mode='markers+text',
        text=vocabulary,
        textposition='top center',
        marker=dict(
            size=10,
            color=np.arange(len(vocabulary)),
            colorscale='Viridis',
            showscale=False
        ),
        textfont=dict(size=10)
    ))

    fig.update_layout(
        title='LSA Word Embeddings (2D Visualization)',
        xaxis_title='Component 1',
        yaxis_title='Component 2',
        height=600,
        hovermode='closest'
    )

    return fig


def display_tfidf_matrix(tfidf_matrix, feature_names, max_docs=10, max_features=10):
    """
    Display a portion of the TF-IDF matrix in a readable format.

    Args:
        tfidf_matrix: Sparse TF-IDF matrix
        feature_names: Array of feature names
        max_docs: Maximum documents to display
        max_features: Maximum features to display

    Returns:
        Plotly figure object
    """
    # Convert to dense array for display (limited subset)
    dense_matrix = tfidf_matrix[:max_docs, :max_features].toarray()

    fig = go.Figure(data=go.Heatmap(
        z=dense_matrix,
        x=feature_names[:max_features],
        y=[f'Doc {i+1}' for i in range(min(max_docs, tfidf_matrix.shape[0]))],
        colorscale='Blues'
    ))

    fig.update_layout(
        title='TF-IDF Matrix (First {} Docs x First {} Terms)'.format(max_docs, max_features),
        xaxis_title='Terms',
        yaxis_title='Documents',
        height=400
    )

    return fig


def render_module():
    """
    Render the TF-IDF and LSA module in Streamlit.
    """
    st.header("Module 1: Traditional Statistical Models (TF-IDF & LSA)")

    st.markdown("""
    This module demonstrates:
    - TF-IDF (Term Frequency-Inverse Document Frequency) for keyword extraction
    - LSA (Latent Semantic Analysis) using TruncatedSVD for dimensionality reduction

    **Goal**: Observe how LSA maps co-occurring words (like subjects and verbs) to similar coordinates.
    """)

    # Text input
    from sample_text import SAMPLE_TEXT

    user_text = st.text_area(
        "Enter English corpus (sentences will be treated as documents):",
        value=SAMPLE_TEXT,
        height=200,
        key="tfidf_text"
    )

    if st.button("Process Text", key="tfidf_process"):
        with st.spinner("Processing..."):
            # Step 1: Preprocess text into sentences
            st.subheader("Step 1: Document Collection")
            documents = preprocess_text(user_text)
            st.write(f"Total documents (sentences): **{len(documents)}**")

            with st.expander("View documents"):
                for i, doc in enumerate(documents[:10], 1):
                    st.write(f"**Doc {i}:** {doc}")
                if len(documents) > 10:
                    st.write(f"... and {len(documents) - 10} more documents")

            # Step 2: Compute TF-IDF
            st.subheader("Step 2: TF-IDF Matrix")
            tfidf_matrix, feature_names, vectorizer = compute_tfidf(documents)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Documents", tfidf_matrix.shape[0])
            with col2:
                st.metric("Vocabulary Size", tfidf_matrix.shape[1])
            with col3:
                st.metric("Matrix Density", f"{tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]) * 100:.2f}%")

            # Display TF-IDF matrix heatmap
            st.plotly_chart(display_tfidf_matrix(tfidf_matrix, feature_names), use_container_width=True)

            # Step 3: Extract top keywords
            st.subheader("Step 3: Top Keywords")
            keywords = extract_top_keywords(tfidf_matrix, feature_names, top_n=5)

            st.write("**Top 5 Keywords by TF-IDF Score:**")
            for i, (word, score) in enumerate(keywords, 1):
                st.write(f"{i}. **{word}** - Score: {score:.4f}")

            # Step 4: LSA Visualization
            st.subheader("Step 4: LSA Word Embeddings (2D)")
            st.markdown("""
            Words are embedded in 2D space using LSA. Words that frequently co-occur
            in similar contexts should appear close to each other in this visualization.
            """)

            word_matrix, vocabulary = apply_lsa_to_words(documents, n_components=2)
            fig = visualize_lsa_words(word_matrix, vocabulary)
            st.plotly_chart(fig, use_container_width=True)

            st.info("""
            **Observation Tip**: Look for semantically related words that cluster together.
            For example, words like 'language', 'processing', 'natural' might appear close to each other.
            """)


if __name__ == "__main__":
    # Test the module
    from sample_text import SAMPLE_TEXT
    documents = preprocess_text(SAMPLE_TEXT)
    print(f"Processed {len(documents)} documents")

    tfidf_matrix, feature_names, _ = compute_tfidf(documents)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    keywords = extract_top_keywords(tfidf_matrix, feature_names)
    print(f"Top keywords: {keywords}")
