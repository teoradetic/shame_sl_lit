import streamlit as st

st.title("Word embeddings")
st.markdown("""
**GOAL**
- Use word embeddings models (FastText) to automatically identify shame.
""")

st.header("1. How does it work")
st.header("2. Evaluate word embedding quality")
st.subheader("2.1 Embedding stability")
st.subheader("2.2 Effects of epochs, lemmas, stop words, and granularity")
st.header("3. Find most similar sentences")
st.subheader("3.0 Target shame vector")
st.subheader("3.1 Method: N most similar (cosine) paragraphs")
st.subheader("3.2 Method: kMeans clustering")
st.subheader("3.3 Method: kNN neighbors")
st.subheader("3.4 Method: Community detection")