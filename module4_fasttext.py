"""
Module 4: FastText and Sent2Vec
Implements FastText for OOV handling and simple Sent2Vec using average pooling.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from gensim.models import FastText, Word2Vec
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
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


def preprocess_for_fasttext(text):
    """
    Preprocess text for FastText training.

    Args:
        text: Raw text string

    Returns:
        List of tokenized sentences (list of list of words)
    """
    sentences = sent_tokenize(text)
    tokenized = []

    for sent in sentences:
        words = word_tokenize(sent.lower())
        words = [w for w in words if w.isalpha() and len(w) > 1]
        if len(words) >= 3:
            tokenized.append(words)

    return tokenized


def train_fasttext(sentences, vector_size=100, window=5, min_count=1):
    """
    Train a FastText model.

    Args:
        sentences: List of tokenized sentences
        vector_size: Dimensionality of word vectors
        window: Context window size
        min_count: Minimum word frequency

    Returns:
        Trained FastText model
    """
    model = FastText(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        seed=42
    )

    return model


def train_word2vec(sentences, vector_size=100, window=5, min_count=1):
    """
    Train a Word2Vec model for comparison.

    Args:
        sentences: List of tokenized sentences
        vector_size: Dimensionality of word vectors
        window: Context window size
        min_count: Minimum word frequency

    Returns:
        Trained Word2Vec model
    """
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        seed=42
    )

    return model


def test_oov(word2vec_model, fasttext_model, word):
    """
    Test OOV (Out-of-Vocabulary) handling between Word2Vec and FastText.

    Args:
        word2vec_model: Trained Word2Vec model
        fasttext_model: Trained FastText model
        word: OOV word to test

    Returns:
        Dictionary with test results
    """
    results = {
        'word': word,
        'word2vec_found': False,
        'word2vec_error': None,
        'word2vec_similar': None,
        'fasttext_found': False,
        'fasttext_similar': None
    }

    # Test Word2Vec
    try:
        similar = word2vec_model.wv.most_similar(word, topn=5)
        results['word2vec_found'] = True
        results['word2vec_similar'] = similar
    except KeyError as e:
        results['word2vec_error'] = str(e)

    # Test FastText
    try:
        similar = fasttext_model.wv.most_similar(word, topn=5)
        results['fasttext_found'] = True
        results['fasttext_similar'] = similar
    except Exception as e:
        results['fasttext_error'] = str(e)

    return results


def sentence_to_vector(model, sentence):
    """
    Convert a sentence to a vector using average pooling.

    Args:
        model: FastText or Word2Vec model
        sentence: Input sentence string

    Returns:
        Numpy array representing the sentence vector
    """
    # Tokenize sentence
    words = word_tokenize(sentence.lower())
    words = [w for w in words if w.isalpha()]

    if not words:
        return None

    # Get word vectors and compute mean
    vectors = []
    for word in words:
        try:
            vec = model.wv[word]
            vectors.append(vec)
        except KeyError:
            # For FastText, this shouldn't happen, but handle anyway
            continue

    if not vectors:
        return None

    # Average pooling
    sentence_vec = np.mean(vectors, axis=0)
    return sentence_vec


def sentence_similarity(vec1, vec2):
    """
    Compute cosine similarity between two sentence vectors.

    Args:
        vec1, vec2: Sentence vectors

    Returns:
        Cosine similarity score
    """
    if vec1 is None or vec2 is None:
        return None

    # Reshape for sklearn
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)

    return cosine_similarity(vec1, vec2)[0][0]


def visualize_oov_comparison(fasttext_model, oov_word, correct_word=None):
    """
    Visualize the OOV word and its similar words in 2D space.

    Args:
        fasttext_model: Trained FastText model
        oov_word: The OOV word
        correct_word: The correct spelling (optional)

    Returns:
        Plotly figure object
    """
    from sklearn.decomposition import PCA

    # Get similar words from FastText
    try:
        similar = fasttext_model.wv.most_similar(oov_word, topn=5)
        words = [oov_word] + [w for w, _ in similar]

        if correct_word:
            words.append(correct_word)

        # Get vectors
        vectors = [fasttext_model.wv[w] for w in words]

        # Apply PCA
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(np.array(vectors))

        # Create visualization
        fig = go.Figure()

        colors = ['red'] + ['blue'] * len(similar)
        if correct_word:
            colors.append('green')

        fig.add_trace(go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode='markers+text',
            text=words,
            textposition='top center',
            marker=dict(size=12, color=colors),
            textfont=dict(size=11)
        ))

        fig.update_layout(
            title=f'FastText OOV Handling: "{oov_word}"',
            xaxis_title='PC1',
            yaxis_title='PC2',
            height=500
        )

        return fig

    except Exception as e:
        return None


def render_module():
    """
    Render the FastText and Sent2Vec module in Streamlit.
    """
    st.header("Module 4: Subword Features & Sentence Vectors (FastText & Sent2Vec)")

    st.markdown("""
    This module demonstrates:
    - **FastText**: Handles OOV words using character n-grams
    - **Sent2Vec**: Sentence representation using average pooling of word vectors

    **Goals**:
    - Verify FastText's robustness with OOV words (e.g., misspellings)
    - Observe how sentence similarity is computed using vector averaging
    """)

    # Text input
    from sample_text import SAMPLE_TEXT

    user_text = st.text_area(
        "Enter English corpus for training:",
        value=SAMPLE_TEXT,
        height=150,
        key="ft_text"
    )

    # Model training
    if st.button("Train FastText Model", key="train_ft"):
        with st.spinner("Preprocessing and training models..."):
            sentences = preprocess_for_fasttext(user_text)

            if len(sentences) < 3:
                st.error("Not enough sentences for training.")
                return

            # Train both models for comparison
            ft_model = train_fasttext(sentences)
            w2v_model = train_word2vec(sentences)

            # Store in session state
            st.session_state['ft_model'] = ft_model
            st.session_state['w2v_model_ft'] = w2v_model
            st.session_state['ft_sentences'] = sentences

            st.success(f"Models trained! Vocabulary size: {len(ft_model.wv)}")
            st.write(f"**Tokenized sentences:** {len(sentences)}")

    # Tabs for different functions
    if 'ft_model' in st.session_state:
        tab1, tab2 = st.tabs(["OOV Test", "Sent2Vec"])

        # Tab 1: OOV Test
        with tab1:
            st.subheader("Out-of-Vocabulary (OOV) Word Test")

            st.markdown("""
            **FastText** uses character n-grams to handle words not seen during training.
            This allows it to find similar words for misspellings or new words!

            **Example**: Try 'computeer' (misspelling of 'computer')
            """)

            ft_model = st.session_state['ft_model']
            w2v_model = st.session_state['w2v_model_ft']

            # OOV word input
            oov_word = st.text_input(
                "Enter a misspelled or OOV word:",
                value="computeer",
                key="oov_input"
            )

            correct_word = st.text_input(
                "Correct spelling (for comparison, optional):",
                value="computer",
                key="correct_input"
            )

            if st.button("Test OOV Handling", key="test_oov"):
                results = test_oov(w2v_model, ft_model, oov_word.lower())

                col1, col2 = st.columns(2)

                # Word2Vec results
                with col1:
                    st.markdown("**Word2Vec Result**")
                    if results['word2vec_found']:
                        st.success("Word found!")
                        for w, s in results['word2vec_similar']:
                            st.write(f"- {w} ({s:.4f})")
                    else:
                        st.error("KeyError: Word not found in vocabulary")
                        st.info("Word2Vec cannot handle OOV words!")

                # FastText results
                with col2:
                    st.markdown("**FastText Result**")
                    if results['fasttext_found']:
                        st.success("Word processed successfully!")
                        for w, s in results['fasttext_similar']:
                            st.write(f"- {w} ({s:.4f})")
                    else:
                        st.error("FastText also failed")

                # Visualization
                if results['fasttext_found']:
                    fig = visualize_oov_comparison(ft_model, oov_word.lower(),
                                                   correct_word.lower() if correct_word else None)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                # Explanation
                st.markdown("---")
                st.markdown("""
                **Why does FastText work?**

                FastText breaks words into character n-grams (e.g., "computer" -> ["com", "omp", "mpu", "put", "ute", "ter"]).
                Even for OOV words like "computeer", it can construct a vector from shared n-grams with similar words!

                **Key insight**: "computeer" shares n-grams like "com", "put", "ute" with "computer", allowing FastText to infer its meaning.
                """)

        # Tab 2: Sent2Vec
        with tab2:
            st.subheader("Sentence to Vector (Sent2Vec)")

            st.markdown("""
            **Simple Sent2Vec implementation**: Average pooling of word vectors

            Process:
            1. Tokenize the sentence into words
            2. Look up each word's vector
            3. Compute the mean of all word vectors
            4. The resulting vector represents the sentence

            This captures the overall semantic meaning of the sentence.
            """)

            ft_model = st.session_state['ft_model']

            # Sentence inputs
            sentence1 = st.text_area(
                "Sentence 1:",
                value="Natural language processing is a machine learning technology.",
                height=80,
                key="sent1"
            )

            sentence2 = st.text_area(
                "Sentence 2:",
                value="Computers can understand human language using artificial intelligence.",
                height=80,
                key="sent2"
            )

            if st.button("Compute Sentence Similarity", key="compute_sent_sim"):
                with st.spinner("Computing sentence vectors..."):
                    # Compute sentence vectors
                    vec1 = sentence_to_vector(ft_model, sentence1)
                    vec2 = sentence_to_vector(ft_model, sentence2)

                    if vec1 is None or vec2 is None:
                        st.error("Could not compute vectors for one or more sentences.")
                        return

                    # Display vectors
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Sentence 1 Vector** (first 10 dimensions)")
                        st.write(f"Shape: {vec1.shape}")
                        st.write(f"[{', '.join([f'{v:.4f}' for v in vec1[:10]])}...]")

                    with col2:
                        st.markdown("**Sentence 2 Vector** (first 10 dimensions)")
                        st.write(f"Shape: {vec2.shape}")
                        st.write(f"[{', '.join([f'{v:.4f}' for v in vec2[:10]])}...]")

                    # Compute similarity
                    similarity = sentence_similarity(vec1, vec2)

                    st.markdown("---")
                    st.markdown("### Similarity Result")

                    # Color-coded similarity display
                    color = "green" if similarity > 0.5 else ("orange" if similarity > 0 else "red")
                    st.markdown(f"**Cosine Similarity:** :{color}[**{similarity:.4f}**]")

                    # Interpretation
                    if similarity > 0.7:
                        interp = "Highly similar sentences"
                    elif similarity > 0.4:
                        interp = "Moderately similar sentences"
                    elif similarity > 0:
                        interp = "Weakly similar sentences"
                    else:
                        interp = "Dissimilar sentences"

                    st.info(f"**Interpretation:** {interp}")

                    # Word breakdown
                    st.markdown("---")
                    st.markdown("### Word Breakdown")

                    words1 = [w for w in word_tokenize(sentence1.lower()) if w.isalpha()]
                    words2 = [w for w in word_tokenize(sentence2.lower()) if w.isalpha()]

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Sentence 1 words:**")
                        st.write(", ".join(words1))

                    with col2:
                        st.write("**Sentence 2 words:**")
                        st.write(", ".join(words2))


if __name__ == "__main__":
    # Test the module
    from sample_text import SAMPLE_TEXT
    sentences = preprocess_for_fasttext(SAMPLE_TEXT)
    print(f"Tokenized {len(sentences)} sentences")

    ft_model = train_fasttext(sentences)
    w2v_model = train_word2vec(sentences)

    print(f"FastText vocab size: {len(ft_model.wv)}")
    print(f"Word2Vec vocab size: {len(w2v_model.wv)}")

    # Test OOV
    results = test_oov(w2v_model, ft_model, 'computeer')
    print(f"OOV test: {results}")

    # Test sentence vectors
    vec1 = sentence_to_vector(ft_model, "Natural language processing is interesting.")
    vec2 = sentence_to_vector(ft_model, "Machine learning helps computers understand text.")

    sim = sentence_similarity(vec1, vec2)
    print(f"Sentence similarity: {sim}")
