import streamlit as st

st.title("Welcome to the `Shapes of Shame in Slovene literature` Analysis App")
st.markdown("""
Use the sidebar to the left to navigate between pages:
- Start with _Corpus Analysis_ to get a feel for the novels in this corpus.
- _Word Embeddings_ showcases how FastText was trained, evaluated, and used \
to tag paragraphs in the corpus as shame related (or not).
- _Sentence Embeddings_ - TBD
- _LLM few shot annotations - TBD
            
Other important assets:
- All code is open source and can be found on \
[GitHub](https://github.com/teoradetic/shame_sl_lit).
- If you want to rerun the pipelines to generate the data, \
models, and analyses, start by understanding the [corpus \
metadata](https://github.com/teoradetic/shame_sl_lit/blob/main/corpus_metadata.csv),\
 run [_setup.py_](https://github.com/teoradetic/shame_sl_lit/blob/main/setup.py) \
to generate the documents locally, and go through the \
[notebooks](https://github.com/teoradetic/shame_sl_lit/tree/main/notebooks) \
to build up the models and infer with them.
""")