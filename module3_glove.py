"""
Module 3: Pre-trained GloVe Model and Word Analogy
Implements word analogy calculations using pre-trained GloVe embeddings.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from gensim import downloader
import logging
import os

# Suppress gensim logging
logging.getLogger('gensim').setLevel(logging.WARNING)


# Streamlit Cloud cold starts are much more reliable with the smaller model.
# Set GLOVE_MODEL_NAME in the environment to try a larger model.
# Available models: glove-twitter-25, glove-wiki-gigaword-50, glove-wiki-gigaword-100
GLOVE_MODEL_NAME = os.getenv('GLOVE_MODEL_NAME', 'glove-twitter-25')


@st.cache_resource(show_spinner=False)
def load_glove_model():
    """
    Load pre-trained GloVe model using gensim downloader.
    Defaults to glove-twitter-25 so Streamlit Cloud can start reliably.

    Returns:
        KeyedVectors object with GloVe embeddings
    """
    try:
        model = downloader.load(GLOVE_MODEL_NAME)
        return model
    except Exception as e:
        # Fallback to smaller model if download fails
        try:
            model = downloader.load('glove-wiki-gigaword-50')
            return model
        except:
            try:
                model = downloader.load('glove-twitter-25')
                return model
            except Exception as e2:
                st.error(f"Error loading GloVe model: {e2}")
                return None


def word_analogy(model, a, b, c, top_n=5):
    """
    Compute word analogy: A is to B as C is to ?
    Formula: result = vector(A) - vector(B) + vector(C)

    Args:
        model: KeyedVectors model
        a, b, c: Words for analogy
        top_n: Number of results to return

    Returns:
        List of (word, similarity) tuples or None if error
    """
    try:
        # Compute: A - B + C
        result = model.most_similar(
            positive=[a, c],
            negative=[b],
            topn=top_n
        )
        return result
    except KeyError as e:
        return None, str(e)


def word_similarity(model, word1, word2):
    """
    Compute cosine similarity between two words.

    Args:
        model: KeyedVectors model
        word1, word2: Words to compare

    Returns:
        Float similarity score or None if error
    """
    try:
        similarity = model.similarity(word1, word2)
        return similarity
    except KeyError as e:
        return None, str(e)


def get_similar_words(model, word, top_n=10):
    """
    Find similar words using the model.

    Args:
        model: KeyedVectors model
        word: Target word
        top_n: Number of similar words

    Returns:
        List of (word, similarity) tuples
    """
    try:
        similar = model.most_similar(word, topn=top_n)
        return similar
    except KeyError:
        return None


def visualize_analogy(model, a, b, c):
    """
    Visualize word analogy in 2D space.

    Args:
        model: KeyedVectors model
        a, b, c: Words for analogy

    Returns:
        Plotly figure object
    """
    from sklearn.decomposition import PCA

    try:
        # Get vectors
        words = [a, b, c]
        vectors = [model[w] for w in words]

        # Compute result vector
        result_vec = model[a] - model[b] + model[c]

        # Find closest word to result
        result = model.most_similar(positive=[a, c], negative=[b], topn=1)[0][0]
        words.append(result)
        vectors.append(model[result])

        # Apply PCA
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(np.array(vectors))

        # Create visualization
        fig = go.Figure()

        # Plot points
        colors = ['red', 'blue', 'green', 'purple']
        for i, (word, coord) in enumerate(zip(words, coords)):
            fig.add_trace(go.Scatter(
                x=[coord[0]],
                y=[coord[1]],
                mode='markers+text',
                text=[word],
                textposition='top center',
                marker=dict(size=15, color=colors[i]),
                name=f'{word} ({["A", "B", "C", "Result"][i]})'
            ))

        # Add arrow from A to B
        fig.add_annotation(
            x=coords[1, 0], y=coords[1, 1],
            ax=coords[0, 0], ay=coords[0, 1],
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='gray'
        )

        # Add arrow from C to Result
        fig.add_annotation(
            x=coords[3, 0], y=coords[3, 1],
            ax=coords[2, 0], ay=coords[2, 1],
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='gray'
        )

        fig.update_layout(
            title=f'Word Analogy: {a} - {b} + {c} = {result}',
            xaxis_title='PC1',
            yaxis_title='PC2',
            height=500,
            showlegend=True
        )

        return fig

    except KeyError as e:
        return None


def render_module():
    """
    Render the GloVe module in Streamlit.
    """
    st.header("Module 3: Pre-trained GloVe Model and Word Analogy")

    st.markdown("""
    This module demonstrates:
    - Pre-trained **GloVe** (Global Vectors for Word Representation) embeddings
    - **Word Analogy** calculations using vector arithmetic
    - Word similarity measurement

    **Goal**: Verify classic examples like "king - man + woman = queen" and observe GloVe's ability to capture linear relationships.
    """)

    # Load model with progress indicator
    model_container = st.empty()

    # Show download size based on model
    model_sizes = {
        'glove-wiki-gigaword-100': '~400MB',
        'glove-wiki-gigaword-50': '~200MB',
        'glove-twitter-25': '~105MB'
    }
    download_size = model_sizes.get(GLOVE_MODEL_NAME, '~400MB')

    model_container.info(f"Loading GloVe model ({GLOVE_MODEL_NAME})... This may take a moment on first run (downloading {download_size}).")

    progress_bar = st.progress(0)

    try:
        model = load_glove_model()
        progress_bar.progress(100)
    except Exception as e:
        st.error(f"Failed to load GloVe model: {e}")
        return
    finally:
        progress_bar.empty()

    if model is None:
        st.error("Failed to load GloVe model. Please check your internet connection.")
        return

    model_container.success(f"GloVe model loaded! Vocabulary size: {len(model):,}")

    # Model information
    vector_dims = {
        'glove-twitter-25': '25-dimensional vectors',
        'glove-wiki-gigaword-50': '50-dimensional vectors',
        'glove-wiki-gigaword-100': '100-dimensional vectors'
    }.get(GLOVE_MODEL_NAME, 'pre-trained vectors')

    st.info(f"""
    **Model Information:**
    - **Model**: {GLOVE_MODEL_NAME} ({vector_dims})
    - **Default for deployment**: the smaller Twitter model keeps Streamlit Cloud cold starts manageable.
    - **Tip**: set environment variable `GLOVE_MODEL_NAME=glove-wiki-gigaword-50` or `glove-wiki-gigaword-100` if you need stronger analogy quality and can tolerate a larger download.
    """)

    # Tab layout for different functions
    tab1, tab2, tab3 = st.tabs(["Word Analogy", "Word Similarity", "Explore Words"])

    # Tab 1: Word Analogy
    with tab1:
        st.subheader("Word Analogy Calculator")
        st.markdown("""
        **Analogy Task**: A is to B as C is to ?

        Formula: `Result = Vector(A) - Vector(B) + Vector(C)`

        **Classic Examples to try:**
        - A=king, B=man, C=woman -> Expected: queen
        - A=paris, B=france, C=italy -> Expected: rome
        - A=big, B=bigger, C=small -> Expected: smaller
        """)

        col1, col2, col3 = st.columns(3)

        with col1:
            word_a = st.text_input("A (e.g., king)", value="king", key="analogy_a")

        with col2:
            word_b = st.text_input("B (e.g., man)", value="man", key="analogy_b")

        with col3:
            word_c = st.text_input("C (e.g., woman)", value="woman", key="analogy_c")

        top_n = st.slider("Number of results to show:", min_value=5, max_value=20, value=10, key="analogy_topn")

        if st.button("Compute Analogy", key="compute_analogy"):
            result = word_analogy(model, word_a.lower(), word_b.lower(), word_c.lower(), top_n=top_n)

            if result is None or (isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], str)):
                st.error(f"One or more words not found in vocabulary. Please try different words.")
            else:
                st.write(f"**{word_a} - {word_b} + {word_c} = ?**")
                st.write("")

                # Highlight expected word if present
                expected_words = {
                    ('king', 'man', 'woman'): 'queen',
                    ('paris', 'france', 'italy'): 'rome',
                    ('big', 'bigger', 'small'): 'smaller',
                    ('walk', 'walking', 'swim'): 'swimming'
                }

                expected = expected_words.get((word_a.lower(), word_b.lower(), word_c.lower()))

                # Find if expected word is in results
                found_expected = False
                expected_rank = None
                for i, (word, score) in enumerate(result):
                    if word.lower() == expected:
                        found_expected = True
                        expected_rank = i + 1
                        break

                if expected and found_expected:
                    st.success(f"Expected word '{expected}' found at rank {expected_rank}!")
                elif expected:
                    st.warning(f"Expected word '{expected}' not in top {top_n} results. This is common with smaller models.")

                st.write("**Top Results:**")

                for i, (word, score) in enumerate(result, 1):
                    if word.lower() not in [word_a.lower(), word_b.lower(), word_c.lower()]:
                        # Highlight expected word
                        if expected and word.lower() == expected:
                            st.markdown(f"**{i}. {word} - Similarity: {score:.4f}** (Expected)")
                        else:
                            st.write(f"{i}. **{word}** - Similarity: {score:.4f}")

                # Visualization
                fig = visualize_analogy(model, word_a.lower(), word_b.lower(), word_c.lower())
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("""
                **Note about results:**
                - This model uses 25-dimensional vectors trained on Twitter data
                - Smaller models capture general semantic relationships but may not always return the "expected" answer first
                - Related words (like 'prince' for royalty) may rank higher than the exact analogy
                """)

    # Tab 2: Word Similarity
    with tab2:
        st.subheader("Word Similarity Calculator")
        st.markdown("""
        Compute the cosine similarity between any two words.

        **Cosine Similarity** measures the angle between word vectors:
        - 1.0 = identical meaning
        - 0.0 = unrelated
        - -1.0 = opposite meaning
        """)

        col1, col2 = st.columns(2)

        with col1:
            sim_word1 = st.text_input("Word 1", value="computer", key="sim_word1")

        with col2:
            sim_word2 = st.text_input("Word 2", value="laptop", key="sim_word2")

        if st.button("Calculate Similarity", key="calc_sim"):
            similarity = word_similarity(model, sim_word1.lower(), sim_word2.lower())

            if isinstance(similarity, tuple):
                st.error(f"Word not found: {similarity[1]}")
            else:
                # Display similarity with color coding
                color = "green" if similarity > 0.5 else ("orange" if similarity > 0 else "red")
                st.markdown(f"**Similarity Score:** :{color}[**{similarity:.4f}**]")

                # Interpret the score
                if similarity > 0.7:
                    interpretation = "Very similar meaning"
                elif similarity > 0.4:
                    interpretation = "Related concepts"
                elif similarity > 0:
                    interpretation = "Weakly related"
                else:
                    interpretation = "Unrelated or opposite"

                st.info(f"**Interpretation:** {interpretation}")

        # Quick similarity examples
        st.markdown("---")
        st.markdown("**Quick Examples:**")

        example_pairs = [
            ("happy", "joyful"),
            ("hot", "cold"),
            ("king", "queen"),
            ("computer", "banana")
        ]

        for w1, w2 in example_pairs:
            sim = word_similarity(model, w1, w2)
            if isinstance(sim, float):
                st.write(f"- **{w1}** vs **{w2}**: {sim:.4f}")

    # Tab 3: Explore Words
    with tab3:
        st.subheader("Explore Word Vectors")
        st.markdown("Find words similar to your query word.")

        explore_word = st.text_input("Enter a word to explore:", value="natural", key="explore_word")

        if explore_word:
            similar = get_similar_words(model, explore_word.lower(), top_n=10)

            if similar is None:
                st.error(f"Word '{explore_word}' not found in vocabulary.")
            else:
                st.write(f"**Words most similar to '{explore_word}':**")

                # Display as a bar chart
                words = [w for w, _ in similar]
                scores = [s for _, s in similar]

                fig = go.Figure(data=go.Bar(
                    x=scores,
                    y=words,
                    orientation='h',
                    marker_color='steelblue'
                ))

                fig.update_layout(
                    title=f"Similar Words to '{explore_word}'",
                    xaxis_title='Cosine Similarity',
                    yaxis_title='',
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    # Test the module
    print("Loading GloVe model...")
    model = load_glove_model()

    if model:
        print(f"Model loaded! Vocabulary size: {len(model)}")

        # Test word analogy
        result = word_analogy(model, 'king', 'man', 'woman', top_n=5)
        print(f"king - man + woman = {result}")

        # Test similarity
        sim = word_similarity(model, 'computer', 'laptop')
        print(f"Similarity(computer, laptop) = {sim}")
