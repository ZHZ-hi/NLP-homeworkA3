"""
Module 2: Word2Vec Training and Comparison (CBOW vs Skip-Gram)
Implements real-time Word2Vec training with architecture switching.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from gensim.models import Word2Vec
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


def preprocess_for_word2vec(text):
    """
    Preprocess text for Word2Vec training.

    Args:
        text: Raw text string

    Returns:
        List of tokenized sentences (list of list of words)
    """
    sentences = sent_tokenize(text)
    tokenized = []

    for sent in sentences:
        # Tokenize and clean
        words = word_tokenize(sent.lower())
        # Keep only alphabetic tokens
        words = [w for w in words if w.isalpha() and len(w) > 2]
        if len(words) >= 3:  # Keep sentences with at least 3 words
            tokenized.append(words)

    return tokenized


def train_word2vec(sentences, sg=0, window=5, vector_size=100, min_count=1, workers=4):
    """
    Train a Word2Vec model.

    Args:
        sentences: List of tokenized sentences
        sg: Training algorithm (0=CBOW, 1=Skip-Gram)
        window: Context window size
        vector_size: Dimensionality of word vectors
        min_count: Minimum word frequency
        workers: Number of worker threads

    Returns:
        Trained Word2Vec model
    """
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        workers=workers,
        seed=42
    )

    return model


def get_similar_words(model, word, top_n=5):
    """
    Find similar words using cosine similarity.

    Args:
        model: Trained Word2Vec model
        word: Target word
        top_n: Number of similar words to return

    Returns:
        List of (word, similarity) tuples
    """
    try:
        similar = model.wv.most_similar(positive=[word], topn=top_n)
        return similar
    except KeyError:
        return None


def visualize_word_embeddings(model, words_to_plot=None, max_words=50):
    """
    Visualize word embeddings in 2D using PCA.

    Args:
        model: Trained Word2Vec model
        words_to_plot: Specific words to plot (optional)
        max_words: Maximum number of words to plot

    Returns:
        Plotly figure object
    """
    from sklearn.decomposition import PCA

    # Get all vocabulary words
    vocab = list(model.wv.key_to_index.keys())

    if words_to_plot:
        vocab = [w for w in words_to_plot if w in vocab]
    else:
        vocab = vocab[:max_words]

    if not vocab:
        return None

    # Get vectors
    vectors = np.array([model.wv[w] for w in vocab])

    # Reduce to 2D using PCA
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(vectors)

    # Create scatter plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode='markers+text',
        text=vocab,
        textposition='top center',
        marker=dict(
            size=10,
            color=np.arange(len(vocab)),
            colorscale='Rainbow',
            showscale=False
        ),
        textfont=dict(size=10)
    ))

    fig.update_layout(
        title='Word2Vec Embeddings (2D PCA Projection)',
        xaxis_title='PC1',
        yaxis_title='PC2',
        height=600,
        hovermode='closest'
    )

    return fig


def get_model_info(model):
    """
    Get basic information about the trained model.

    Args:
        model: Trained Word2Vec model

    Returns:
        Dictionary with model info
    """
    return {
        'vocab_size': len(model.wv),
        'vector_size': model.vector_size,
        'window': model.window,
        'algorithm': 'Skip-Gram' if model.sg == 1 else 'CBOW',
        'corpus_count': model.corpus_count
    }


def render_module():
    """
    Render the Word2Vec module in Streamlit.
    """
    st.header("Module 2: Word2Vec Training and Comparison (CBOW vs Skip-Gram)")

    st.markdown("""
    This module demonstrates:
    - Real-time Word2Vec training on your corpus
    - Comparison between **CBOW** (Continuous Bag of Words) and **Skip-Gram** architectures
    - Effect of context window size on word embeddings

    **Goal**: Compare the Top 5 similar words for the same target word under different architectures.
    """)

    # Text input
    from sample_text import SAMPLE_TEXT

    user_text = st.text_area(
        "Enter English corpus for Word2Vec training:",
        value=SAMPLE_TEXT,
        height=150,
        key="w2v_text"
    )

    # Model parameters
    col1, col2 = st.columns(2)

    with col1:
        architecture = st.radio(
            "Training Architecture:",
            options=["CBOW", "Skip-Gram"],
            help="CBOW is faster, Skip-Gram works better with small datasets and rare words"
        )
        sg = 0 if architecture == "CBOW" else 1

    with col2:
        window = st.slider(
            "Context Window Size:",
            min_value=2,
            max_value=10,
            value=5,
            help="Number of words to consider on each side"
        )

    vector_size = st.slider(
        "Vector Size:",
        min_value=50,
        max_value=200,
        value=100,
        help="Dimensionality of word vectors"
    )

    # Training button
    if st.button("Train Word2Vec Model", key="train_w2v"):
        with st.spinner("Training model..."):
            # Preprocess text
            sentences = preprocess_for_word2vec(user_text)

            if len(sentences) < 3:
                st.error("Not enough sentences for training. Please provide more text.")
                return

            st.write(f"**Tokenized sentences:** {len(sentences)}")

            # Train model
            model = train_word2vec(
                sentences=sentences,
                sg=sg,
                window=window,
                vector_size=vector_size
            )

            # Store model in session state
            st.session_state['w2v_model'] = model
            st.session_state['w2v_sentences'] = sentences

            # Display model info
            info = get_model_info(model)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Vocabulary Size", info['vocab_size'])
            with col2:
                st.metric("Vector Size", info['vector_size'])
            with col3:
                st.metric("Algorithm", info['algorithm'])

            st.success("Model trained successfully!")

    # Word similarity section
    if 'w2v_model' in st.session_state:
        st.subheader("Find Similar Words")

        model = st.session_state['w2v_model']

        # Show available vocabulary
        vocab = list(model.wv.key_to_index.keys())
        st.write(f"**Available words:** {', '.join(vocab[:20])}{'...' if len(vocab) > 20 else ''}")

        # Input for target word
        target_word = st.text_input(
            "Enter a word to find similar words:",
            value=vocab[0] if vocab else "",
            key="w2v_target"
        )

        if target_word:
            similar_words = get_similar_words(model, target_word.lower().strip(), top_n=5)

            if similar_words is None:
                st.error(f"Word '{target_word}' not found in vocabulary (OOV)")
            else:
                st.write(f"**Top 5 words similar to '{target_word}':**")

                for i, (word, score) in enumerate(similar_words, 1):
                    st.write(f"{i}. **{word}** - Cosine Similarity: {score:.4f}")

                # Visualization
                st.subheader("Word Embedding Visualization")

                # Get words to visualize (target + similar words)
                words_to_plot = [target_word.lower()] + [w for w, _ in similar_words]
                fig = visualize_word_embeddings(model, words_to_plot=words_to_plot)

                if fig:
                    st.plotly_chart(fig, use_container_width=True)

    # Comparison section
    st.subheader("Architecture Comparison")

    st.markdown("""
    **Key Differences:**
    - **CBOW** (Continuous Bag of Words): Predicts the target word from context words
      - Faster training
      - Better for frequent words
      - Smoother representations

    - **Skip-Gram**: Predicts context words from the target word
      - Slower training
      - Better for rare words
      - Captures more fine-grained semantics

    **Try This**: Train both models and compare the Top 5 similar words for the same target word!
    """)

    # Comparison tool
    if 'w2v_model' in st.session_state:
        sentences = st.session_state['w2v_sentences']

        # Initialize session state for comparison
        if 'compare_trained' not in st.session_state:
            st.session_state.compare_trained = False

        col_btn, col_input = st.columns([1, 2])

        with col_btn:
            if st.button("Compare Both Architectures", key="compare_btn"):
                # Train both models
                with st.spinner("Training CBOW model..."):
                    cbow_model = train_word2vec(sentences, sg=0, window=window, vector_size=vector_size)

                with st.spinner("Training Skip-Gram model..."):
                    sg_model = train_word2vec(sentences, sg=1, window=window, vector_size=vector_size)

                # Store models in session state
                st.session_state.cbow_model = cbow_model
                st.session_state.sg_model = sg_model
                st.session_state.compare_trained = True
                st.rerun()

        # Always show the input box
        vocab = list(st.session_state.w2v_model.wv.key_to_index.keys())

        with col_input:
            compare_word = st.text_input(
                "Enter word to compare:",
                value=st.session_state.get('compare_word', vocab[0] if vocab else ""),
                key="compare_word_input"
            )

            # Update session state
            st.session_state.compare_word = compare_word

        # Show results if models are trained
        if st.session_state.get('compare_trained', False) and compare_word:
            cbow_model = st.session_state.cbow_model
            sg_model = st.session_state.sg_model

            st.success("Models trained! Showing comparison results:")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**CBOW Results**")
                cbow_sim = get_similar_words(cbow_model, compare_word.lower(), top_n=5)
                if cbow_sim:
                    for i, (w, s) in enumerate(cbow_sim, 1):
                        st.write(f"{i}. {w} ({s:.4f})")
                else:
                    st.write("Word not in vocabulary")

            with col2:
                st.markdown("**Skip-Gram Results**")
                sg_sim = get_similar_words(sg_model, compare_word.lower(), top_n=5)
                if sg_sim:
                    for i, (w, s) in enumerate(sg_sim, 1):
                        st.write(f"{i}. {w} ({s:.4f})")
                else:
                    st.write("Word not in vocabulary")


if __name__ == "__main__":
    # Test the module
    from sample_text import SAMPLE_TEXT
    sentences = preprocess_for_word2vec(SAMPLE_TEXT)
    print(f"Tokenized {len(sentences)} sentences")

    model = train_word2vec(sentences, sg=1, window=5)
    print(f"Vocabulary size: {len(model.wv)}")

    similar = get_similar_words(model, 'language', top_n=5)
    print(f"Similar to 'language': {similar}")