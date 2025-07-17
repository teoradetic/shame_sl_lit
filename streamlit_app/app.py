import streamlit as st

st.title("Welcome to the `Shapes of Shame in Slovene literature` Analysis App")
st.markdown("""
Use the sidebar to the left to navigate between pages:
""")
st.page_link("pages/1_corpus_analysis.py", 
             label=":one: :blue[**Corpus Analysis**] - Start with this page to get a feel for the novels in this corpus.")
st.page_link("pages/2_word_embeddings_evaluations.py", 
             label=":two: :blue[**Word Embeddings Evaluation**] - Showcases how FastText was trained and evaluated.")
st.page_link("pages/3_word_embeddings_annotations.py", 
             label=":three: :blue[**Word Embeddings Annotations**] - How the FastText model was used to annotate the corpus.")
st.page_link("pages/4_LLM_few_shot_annotations.py", 
             label=":four: :blue[**LLM few shot annotations**] - TBD")

st.divider()
st.markdown("""            
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