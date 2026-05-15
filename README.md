# Semantic Representation Models Streamlit Demo

Interactive Streamlit demo for four NLP representation modules:

- TF-IDF and LSA
- Word2Vec CBOW / Skip-Gram
- GloVe word analogy
- FastText and simple Sent2Vec

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy To Streamlit Cloud

1. Push this folder to a GitHub repository.
2. In Streamlit Cloud, create a new app from that repository.
3. Set the main file path to `app.py`.
4. In Advanced settings, select Python `3.12` if the option is shown.
5. Deploy.

`environment.yml` also pins the deployment environment to Python 3.12. This matters because SciPy/Gensim wheels may be unavailable for the newest Python versions immediately after release.

The GloVe module downloads a pre-trained model on first use. By default it uses `glove-twitter-25` to keep Streamlit Cloud startup reliable. For better analogy quality, add an environment variable in Streamlit Cloud:

```text
GLOVE_MODEL_NAME=glove-wiki-gigaword-50
```

`glove-wiki-gigaword-100` also works in principle, but it is much larger and can make cold starts slow.
