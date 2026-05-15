"""
Flask Backend: Semantic Representation Models API
提供所有 NLP 模型的 API 接口
"""

from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec, FastText
from gensim import downloader
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import logging
import os

# Setup logging
logging.getLogger('gensim').setLevel(logging.WARNING)
logging.getLogger('sklearn').setLevel(logging.WARNING)

app = Flask(__name__, template_folder='templates')

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except:
        pass

# Sample text for default corpus
SAMPLE_TEXT = """Natural language processing is a machine learning technology that gives computers the ability to interpret, manipulate, and comprehend human language. Organizations today have large volumes of voice and text data from various communication channels like emails, text messages, social media newsfeeds, video, audio, and more. They use NLP software to automatically process this data, analyze the intent or sentiment in the message, and respond in real time to human communication.

Natural language processing helps computers communicate with humans in their own language and scales other language-related tasks. For example, NLP makes it possible for computers to read text, hear speech, interpret it, measure sentiment, and determine which parts are important. NLP combines computational linguistics with statistical, machine learning, and deep learning models.

Natural language processing plays a vital part in technology and the way humans interact with it. It is used in many real-world applications in both the business and consumer sectors, including digital assistants, speech recognition, and sentiment analysis. There are many common natural language processing tasks.

Sentiment analysis is a popular NLP task that involves determining the sentiment or emotion expressed in a piece of text. It is widely used in social media monitoring, customer feedback analysis, and market research. Machine translation is another important application of natural language processing. It involves automatically translating text or speech from one language to another.

Word embeddings are a type of word representation that allows words with similar meaning to have a similar representation. They are a key breakthrough in the field of natural language processing and have enabled significant improvements in various NLP tasks. The field of natural language processing has seen tremendous advances in recent years, thanks to the development of deep learning techniques and large language models."""


def preprocess_sentences(text):
    """将文本按句子分割"""
    try:
        sentences = sent_tokenize(text)
    except:
        sentences = text.replace('!', '.').replace('?', '.').split('.')
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences


def preprocess_words(text):
    """将文本分词"""
    try:
        sentences = sent_tokenize(text)
        tokens = []
        for sent in sentences:
            words = word_tokenize(sent.lower())
            words = [w for w in words if w.isalpha() and len(w) > 1]
            if len(words) >= 2:
                tokens.append(words)
        return tokens
    except:
        # Fallback: simple tokenization
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        tokens = []
        for sent in sentences:
            words = [w.lower() for w in sent.split() if w.isalpha() and len(w) > 1]
            if len(words) >= 2:
                tokens.append(words)
        return tokens


# ==================== Module 1: TF-IDF & LSA ====================

@app.route('/api/tfidf-lsa', methods=['POST'])
def api_tfidf_lsa():
    """TF-IDF 和 LSA API"""
    data = request.json
    text = data.get('text', SAMPLE_TEXT)

    try:
        # Preprocess
        documents = preprocess_sentences(text)

        # TF-IDF
        vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', max_features=100)
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()

        # Top keywords
        summed_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
        top_indices = summed_scores.argsort()[-5:][::-1]
        top_keywords = [(feature_names[i], float(summed_scores[i])) for i in top_indices]

        # LSA for words
        count_vec = CountVectorizer(lowercase=True, stop_words='english', max_features=50)
        doc_term = count_vec.fit_transform(documents)
        vocab = count_vec.get_feature_names_out()

        svd = TruncatedSVD(n_components=2, random_state=42)
        word_matrix = svd.fit_transform(doc_term.T)

        # Format word coordinates for visualization
        word_coords = []
        for i, word in enumerate(vocab):
            word_coords.append({
                'word': word,
                'x': float(word_matrix[i, 0]),
                'y': float(word_matrix[i, 1])
            })

        return jsonify({
            'success': True,
            'documents': len(documents),
            'vocab_size': len(feature_names),
            'top_keywords': top_keywords,
            'word_coords': word_coords
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ==================== Module 2: Word2Vec ====================

@app.route('/api/word2vec', methods=['POST'])
def api_word2vec():
    """Word2Vec API"""
    data = request.json
    text = data.get('text', SAMPLE_TEXT)
    word = data.get('word', 'language')
    sg = int(data.get('sg', 1))  # 0=CBOW, 1=Skip-Gram
    window = int(data.get('window', 5))

    try:
        # Preprocess
        tokens = preprocess_words(text)

        # Train model
        model = Word2Vec(sentences=tokens, vector_size=100, window=window,
                         min_count=1, sg=sg, workers=4, seed=42)

        # Get similar words
        try:
            similar = model.wv.most_similar(word.lower(), topn=5)
            similar_words = [{'word': w, 'score': float(s)} for w, s in similar]
            found = True
        except KeyError:
            similar_words = []
            found = False

        # Get vocab for visualization
        vocab = list(model.wv.key_to_index.keys())[:30]
        if word.lower() in vocab:
            vocab.remove(word.lower())
        vocab = vocab[:20]

        return jsonify({
            'success': True,
            'vocab_size': len(model.wv),
            'algorithm': 'Skip-Gram' if sg == 1 else 'CBOW',
            'similar_words': similar_words,
            'found': found,
            'vocab': vocab[:10]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ==================== Module 3: GloVe ====================

# Cache GloVe model
glove_model = None
glove_model_name = 'glove-wiki-gigaword-100'


def get_glove_model():
    """获取 GloVe 模型（带缓存）"""
    global glove_model
    if glove_model is None:
        try:
            glove_model = downloader.load(glove_model_name)
        except:
            try:
                glove_model = downloader.load('glove-wiki-gigaword-50')
            except:
                try:
                    glove_model = downloader.load('glove-twitter-25')
                except:
                    pass
    return glove_model


@app.route('/api/glove-info', methods=['GET'])
def api_glove_info():
    """获取 GloVe 模型信息"""
    try:
        model = get_glove_model()
        if model:
            return jsonify({
                'success': True,
                'model_name': glove_model_name,
                'vocab_size': len(model)
            })
        return jsonify({'success': False, 'error': 'Failed to load model'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/glove-analogy', methods=['POST'])
def api_glove_analogy():
    """词类比 API"""
    data = request.json
    a = data.get('a', 'king')
    b = data.get('b', 'man')
    c = data.get('c', 'woman')

    try:
        model = get_glove_model()
        if model is None:
            return jsonify({'success': False, 'error': 'Model not loaded'})

        result = model.most_similar(positive=[a.lower(), c.lower()], negative=[b.lower()], topn=10)
        analogies = [{'word': w, 'score': float(s)} for w, s in result]

        return jsonify({
            'success': True,
            'result': analogies
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/glove-similarity', methods=['POST'])
def api_glove_similarity():
    """词语相似度 API"""
    data = request.json
    word1 = data.get('word1', 'computer')
    word2 = data.get('word2', 'laptop')

    try:
        model = get_glove_model()
        if model is None:
            return jsonify({'success': False, 'error': 'Model not loaded'})

        sim = model.similarity(word1.lower(), word2.lower())
        return jsonify({
            'success': True,
            'similarity': float(sim)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ==================== Module 4: FastText & Sent2Vec ====================

@app.route('/api/fasttext', methods=['POST'])
def api_fasttext():
    """FastText API"""
    data = request.json
    text = data.get('text', SAMPLE_TEXT)
    oov_word = data.get('oov_word', 'computeer')

    try:
        # Preprocess
        tokens = preprocess_words(text)

        # Train FastText and Word2Vec
        ft_model = FastText(sentences=tokens, vector_size=100, window=5, min_count=1, workers=4, seed=42)
        w2v_model = Word2Vec(sentences=tokens, vector_size=100, window=5, min_count=1, workers=4, seed=42)

        # Test OOV
        result = {
            'oov_word': oov_word,
            'word2vec_found': False,
            'word2vec_error': None,
            'word2vec_similar': [],
            'fasttext_found': False,
            'fasttext_similar': []
        }

        # Word2Vec OOV test
        try:
            sim = w2v_model.wv.most_similar(oov_word, topn=5)
            result['word2vec_found'] = True
            result['word2vec_similar'] = [{'word': w, 'score': float(s)} for w, s in sim]
        except Exception as e:
            result['word2vec_error'] = 'KeyError: Word not in vocabulary'

        # FastText OOV test
        try:
            sim = ft_model.wv.most_similar(oov_word, topn=5)
            result['fasttext_found'] = True
            result['fasttext_similar'] = [{'word': w, 'score': float(s)} for w, s in sim]
        except Exception as e:
            pass

        return jsonify({
            'success': True,
            'result': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/sent2vec', methods=['POST'])
def api_sent2vec():
    """Sent2Vec API"""
    data = request.json
    text = data.get('text', SAMPLE_TEXT)
    sentence1 = data.get('sentence1', 'Natural language processing is a machine learning technology.')
    sentence2 = data.get('sentence2', 'Computers can understand human language using artificial intelligence.')

    try:
        # Preprocess
        tokens = preprocess_words(text)

        # Train FastText
        model = FastText(sentences=tokens, vector_size=100, window=5, min_count=1, workers=4, seed=42)

        # Sentence to vector (average pooling)
        def sentence_to_vec(sent):
            words = word_tokenize(sent.lower()) if 'word_tokenize' in dir() else sent.lower().split()
            words = [w for w in words if w.isalpha()]
            vectors = []
            for word in words:
                try:
                    vectors.append(model.wv[word])
                except:
                    pass
            if vectors:
                return np.mean(vectors, axis=0)
            return None

        vec1 = sentence_to_vec(sentence1)
        vec2 = sentence_to_vec(sentence2)

        if vec1 is not None and vec2 is not None:
            sim = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
            return jsonify({
                'success': True,
                'similarity': float(sim)
            })
        return jsonify({'success': False, 'error': 'Could not compute vectors'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ==================== Frontend ====================

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


if __name__ == '__main__':
    # 0.0.0.0 允许外部网络访问
    # port 可以自定义，默认 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)