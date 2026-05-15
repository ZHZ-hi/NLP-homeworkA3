"""
Main Application: Interactive Web System for Semantic Representation Models
A Streamlit application with four modules demonstrating various NLP semantic models.
"""

import streamlit as st
import nltk

# Set page configuration FIRST (must be first Streamlit command)
st.set_page_config(
    page_title="Semantic Representation Models",
    page_icon=":brain:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK data silently
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"NLTK download warning: {e}")

# Import modules (will be loaded on-demand to prevent blocking)
import module1_tfidf_lsa
import module2_word2vec
import module4_fasttext


def main():
    """Main application entry point."""

    # Sidebar
    st.sidebar.title("Semantic Representation Models")
    st.sidebar.markdown("---")

    st.sidebar.markdown("""
    ### About This Application

    This interactive web system demonstrates various **semantic representation models** in NLP:

    1. **Module 1**: TF-IDF & LSA
    2. **Module 2**: Word2Vec (CBOW/Skip-Gram)
    3. **Module 3**: GloVe & Word Analogy
    4. **Module 4**: FastText & Sent2Vec

    ---

    ### Quick Start
    1. Navigate to each tab
    2. Use the provided sample text or input your own
    3. Train models and explore results
    """)

    # Main content
    st.title("Interactive Web System for Semantic Representation Models")

    st.markdown("""
    Welcome to this interactive demonstration of semantic representation models in Natural Language Processing.
    Navigate through the tabs below to explore different approaches to representing words and sentences as vectors.

    **Learning Objectives:**
    - Understand TF-IDF and LSA for traditional text representation
    - Compare CBOW and Skip-Gram architectures in Word2Vec
    - Explore word analogies with pre-trained GloVe embeddings
    - Test OOV handling with FastText's subword features
    """)

    # Create tabs for four modules
    tab1, tab2, tab3, tab4 = st.tabs([
        "Module 1: TF-IDF & LSA",
        "Module 2: Word2Vec",
        "Module 3: GloVe & Analogy",
        "Module 4: FastText & Sent2Vec"
    ])

    # Render each module in its tab
    with tab1:
        module1_tfidf_lsa.render_module()

    with tab2:
        module2_word2vec.render_module()

    with tab3:
        # Lazy load Module 3 to prevent blocking the page on first load
        import module3_glove
        module3_glove.render_module()

    with tab4:
        module4_fasttext.render_module()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; padding: 20px;">
        <p>Semantic Representation Models Demo | Built with Streamlit, Gensim, and Scikit-learn</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
