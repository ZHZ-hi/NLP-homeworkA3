# Streamlit Cloud Deployment Checklist

## Required Settings

- Repository: this project pushed to GitHub
- Main file path: `app.py`
- Python version: `3.11` from `runtime.txt`
- Dependencies: installed from `requirements.txt`

## Important Notes

- `scipy` is pinned below `1.13` because the current `gensim` release used here imports `scipy.linalg.triu`.
- The app downloads NLTK tokenizers at runtime if they are missing.
- The GloVe tab downloads a pre-trained model on first use. The default `glove-twitter-25` model is chosen for deployment reliability.
- `app_flask.py`, `templates/`, and the report files are not required by Streamlit Cloud when `app.py` is selected as the entry point.

## Optional Streamlit Environment Variable

Use this only if you want a larger GloVe model:

```text
GLOVE_MODEL_NAME=glove-wiki-gigaword-50
```

Avoid `glove-wiki-gigaword-100` unless you are comfortable with a larger first-run download.
