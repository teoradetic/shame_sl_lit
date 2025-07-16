# WP1-T1: Shame Detection in Slovene Literature

This repository supports **WP1, Task 1** of the _Shapes of Shame in Slovene literature_ project, which explores shame in Slovene literary texts using computational linguistics and NLP techniques.

## Overview

We aim to develop tools for:
- Building a literary corpus with rich metadata
- Extracting and analyzing expressions of shame
- Training and evaluating word/sentence embeddings
- Supporting literary and intersectional analysis

## Repository Structure

- `data/` – input corpus
    - `original_txt_corpus` - original novels (no processing), split into two folders: `paragraph` and `sentence` - they indicate the granularity of each line within a txt doc.
    - `lemma_txt_corpus` - lemmatized novels, split into two folders: `paragraph` and `sentence` - they indicate the granularity of each line within a txt doc.
    - `raw_corpora` - 4 folders with the corpora used to generate this corpus (Wikivir, KDSP, PriLit, ELTeC)
- `helpers/` – scripts to help data processing (data fetching, extracting, cleaning, preparing for NLP tasks, ...)
- `models/` – FastText and other trained embeddings
- `notebooks/` – Jupyter notebooks for analysis
- `streamlit_app/` – Streamlit app for data analysis and presentation
- `output` - useful deliverables (charts, similar words, ...)
- `README.md` – This file
- `requirements.txt` – Python dependencies
- `setup.py` - run this script to download all the data to your local dir with the structure specified in .env

## Progress

✅ Corpus metadata defined  
✅ Extraction helpers implemented  
✅ Initial FastText model trained  

## Tasks in Progress

- [ ] Clean and segment texts (sentence/paragraph level)
- [ ] Extract lemmatized texts
- [ ] Train 50+ variations of embeddings
- [ ] Retrain kontext.io and FastText on lemmas
- [ ] Remove stopwords and retrain
- [ ] Apply community detection to embedding graphs
- [ ] Compare with BabelNet for semantic similarity
- [ ] Temporal/author analysis of embedding shifts
- [ ] Experiment with sentence embeddings (e.g. SBERT)
- [ ] Annotate shame with LLMs
- [ ] Test LemmaGen3
- [ ] Share repository/resources
- [ ] Deploy Streamlit demo

## Running Locally

To set up the project locally with Python 3.11 (required for FastText):

### 1. Install Python 3.11 with pyenv

    pyenv install 3.11.8
    pyenv local 3.11.8

### 2. Create and activate a virtual environment

    python -m venv .venv
    source .venv/bin/activate

### 3. Install dependencies

    pip install --upgrade pip
    pip install -r requirements.txt

### 4. (Optional) Run the Streamlit app

    streamlit run streamlit_app/app.py

> **Note:** Python 3.11 is required due to FastText compatibility.

## Methods & Tools

- FastText (custom + pretrained)
- Sentence-BERT (planned)
- Kontext.io
- LemmaGen3
- BabelNet
- Community detection (e.g., Louvain)

---

**License**: MIT  
