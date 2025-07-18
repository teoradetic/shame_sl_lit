import json
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.title("Word embeddings annotations")
st.text("Goal: Find shame-related paragraphs in corpus.")
st.text("How: Use the word embeddings model (FastText) to annotate the corpus and evaluate annotation quality.")
st.subheader("Methodological overview")
with st.expander("Show detailed explanation of the methodology (click to expand)"):
    st.markdown("""
    1. **Preprocessing pipeline**: For each text (novel) and for each paragraph: 
        - clean paragraph: remove stop words,remove non-text characters (punctuation, \
        quotes), use gensim simple_preprocessing to tokenize paragraph
        - for each token: compute embedding vector using FastText (trained for 300 epochs \
        on paragraphs of original text (no lemmas) with stop words removed)
        - for all tokens (aka for entire paragraph) compute average embeddings vector
        - save to json as `{doc_id: {paragraph_id: vector}}`
    2. **Prepare shame (target) vectors**: 
        - pick target keywords
        - for each keyword, clean (gensim simple_preprocessing) and vectorize (using \
        FastText model described above)
        - compute average vector from all the target keywords to act as the shame vector
    3. **Train and eval 4 annotation methods**:
        1. Simple cosine similarity between paragraph vector and shame vector
        2. kNearestNeighbors using shame vector(s) as target(s)
        3. kMeans - unsupervised, but then tag the cluster with the shame vector(s)
        4. Community detection - TBD
    
    The choice of target keywords can strongly change the (semi-)supervised methods. \
    This is why we picked three options and experimented with them:

    | Vector Name        | Description                                                                                             | Keywords Used                                        | Shortcode in Code |
    | :----------------- | :------------------------------------------------------------------------------------------------------ | :--------------------------------------------------- | :---------------- |
    | **Stem words vector** | Words in the corpus that had the stem "sram" in them and appeared more than 50 times.                 | 'sram', 'sramota', 'sramovati', 'sramoten', 'sramezljivo', 'sramezljivost', 'sramezljiv', 'sramotno', 'sramotiti' | `stem_sram`       |
    | **Babel vector** | Words that are most similar to "sram" according to [BabelNet](https://babelnet.org/)                   | 'sram', 'skesan', 'osramočen', 'ponižan', 'kazniv' | `babel_words`     |
    | **kontekst.io vector** | Words that are most similar to "sram" according to [kontekst.io](https://www.kontekst.io/)              | 'strah', 'groza', 'motilo', 'nerodno', 'bolelo', 'zaskrbelo' | `kontekstio_words` |
        
    For full implementation details, check [3_word_embeddings_infer.ipynb](https://github.com/teoradetic/shame_sl_lit/blob/main/notebooks/3_word_embeddings_infer.ipynb)\
     on GitHub.
    """)


# ─── Prep annotations df ─────────────────────────────────────────────────

# --- LOAD NESTED ANNOTATIONS ---
with open("annotations/main_shame_annotations.json", "r", encoding="utf-8") as f:
    annotations = json.load(f)

# --- FLATTEN FOR DATAFRAME ---
flat_records = []
for doc_id, doc_info in annotations.items():
    author = doc_info.get("author", "")
    title = doc_info.get("title", "")
    
    for para_id, para_info in doc_info["paragraphs"].items():
        # Start with the common fields for each paragraph
        record = {
            "shame_id": doc_id,
            "author": author,
            "title": title,
            "paragraph_id": para_id, 
            "paragraph_text": para_info.get("text", "") 
        }
        
        # Loop through all other items in para_info and add them to the record
        for key, value in para_info.items():
            if key != 'text': # We've already handled 'text'
                record[key] = value 
        
        flat_records.append(record) 

df = pd.DataFrame(flat_records)

# ─── Text Annotations ─────────────────────────────────────────────────

st.header("Method #1: N most similar paragraphs")
st.markdown("""
**Pipeline overview**:
1. Use FastText model to vectorize each paragraph in the corpus.
2. Compute a "shame vector" - use shame-related words and turn them into a vector.
3. Measure the (cosine) distance between the paragraph vector and the shame vector.
4. Pick the topN most similar paragraphs as "shame related".
            
Check the results below :point_down: for different shame vectors (target keywords)
""")

st.subheader("Similar paragraph picker")
# --- GET SHAME VECTORS ---
shame_vectors = [x for x in df.columns if x.startswith("cos_sim_")]
shame_vectors_pretty = [x.replace('cos_sim_', '') for x in shame_vectors]

# --- PICKERS IN SAME ROW ---
col1, col2 = st.columns(2)
with col1:
    vec_choice = st.selectbox("Shame vector", shame_vectors_pretty)
with col2:
    topN = st.number_input("Top N", min_value=1, value=100)

# --- FILTERS FOR AUTHOR/TITLE ---
authors = ["All"] + sorted(df["author"].dropna().unique())
titles = ["All"] + sorted(df["title"].dropna().unique())

col3, col4 = st.columns(2)
with col3:
    author_choice = st.selectbox("Filter by author", authors)
with col4:
    title_choice = st.selectbox("Filter by title", titles)

# --- FILTER DATAFRAME ---
keep_cols = ['shame_id', 'author', 'title', 
             'paragraph_text', f'cos_sim_{vec_choice}']
display_df = df[keep_cols]

if author_choice != "All":
    display_df = display_df[display_df["author"] == author_choice]
if title_choice != "All":
    display_df = display_df[display_df["title"] == title_choice]

# --- SORT AND SHOW ---
display_df = display_df.sort_values(f"cos_sim_{vec_choice}", ascending=False)
display_df = display_df[["shame_id", "author", "title", #"paragraph_id", 
                         "paragraph_text", f"cos_sim_{vec_choice}"]]
st.dataframe(display_df.head(topN), 
             hide_index=True,
             )


st.subheader("Top N most similar (cosine distance) texts evaluation")

# quickly check if the sram_stem vector works because of stem detection
_df = df.copy()
_df['par_contains_sram_stem'] = _df['paragraph_text'].str.contains(
    'sram', case=False, na=False).astype(int)
par_num = len(_df)
par_over_0612 = len(_df[_df.cos_sim_stem_sram >= 0.612])
pct_par_over_0612 = round(100 * par_over_0612 / par_num, 2)
par_over_0612_with_sram_stem = _df[_df.cos_sim_stem_sram >= 0.612].par_contains_sram_stem.sum()
pct_pars_sram = round(100 * par_over_0612_with_sram_stem / par_over_0612, 2)

st.markdown(f"""
* Model effectively captured "shame" context.
* Good results using `stem_sram` with a 0.612 similarity threshold (or top 1000 entries).
* Unclear if model can generalize outside of the target seed keywords used in shame vectors.
  * Quick math to check: 
       * total number of paragraphs in corpus: {par_num}
       * number of paragraphs in corpus with cos_sim_stem_sram >= 0.612: {par_over_0612} \
        (aka, {pct_par_over_0612}% of all paragraphs)
       * number of paragraphs with cos_sim_stem_sram >= 0.612 _and_ 'sram' in text: \
        {par_over_0612_with_sram_stem} (aka, {pct_pars_sram}% of paragraphs with \
        cos_sim_stem_sram >= 0.612)
  * Conclusion: model captured semantic similarity, not just lemmantic ones.
* **Next step:** Define a clear cutoff. For example, min/max similarity or top N results. \
But cos_sim_stem_sram >= 0.612 seems a good candidate.
""")

# quickly check if the sram_stem vector works because of stem detection
_df = df.copy()
_df['par_contains_sram_stem'] = _df['paragraph_text'].str.contains(
    'sram', case=False, na=False).astype(int)
total_pars = len(_df)
sim_over_0612 = _df[_df.cos_sim_stem_sram>=0.612]
total_pars_sram = len(sim_over_0612)
pct_pars_sram = round(100  * total_pars_sram / total_pars, 2)

st.header("Method #2: kMeans clustering")
st.markdown("""
Goal: Cluster all paragraph vectors into _k_ groups, without \
using the shame vector.

One can then:
1. See if one cluster aligns with high shame similarity
2. Use clusters as soft annotation (e.g., “cluster 2 = likely shame-related”)
3. Visualize clusters in 2D (with PCA or t-SNE)
""")


# df["kmeans_label"] = pd.to_numeric(df["kmeans_label"], errors='coerce')
# df["paragraph_id"] = pd.to_numeric(df["paragraph_id"], errors='coerce')
# df = df.dropna(subset=["kmeans_label"])

# # --- UI: Cluster picker, shame cluster only, author, title ---
# clusters = sorted(df["kmeans_label"].dropna().unique().astype(int))
# cluster_choice = st.selectbox("Choose cluster", clusters)

# shame_only = st.checkbox("Show only shame cluster", value=False)
# authors1 = ["All"] + sorted(df["author"].unique())
# titles1 = ["All"] + sorted(df["title"].unique())

# col1, col2 = st.columns(2)
# with col1:
#     author_choice = st.selectbox("Filter author", authors1)
# with col2:
#     title_choice = st.selectbox("Filter title", titles1)

# display_df = df[df["kmeans_label"] == cluster_choice]
# if shame_only:
#     display_df = display_df[display_df["is_shame_cluster"]]

# if author_choice != "All":
#     display_df = display_df[display_df["author"] == author_choice]
# if title_choice != "All":
#     display_df = display_df[display_df["title"] == title_choice]

# display_df = display_df.sort_values("similarity", ascending=False)

# topN = st.slider("Show top N paragraphs", min_value=1, max_value=min(100, len(display_df)), value=min(20, len(display_df)))

# for i, row in display_df.head(topN).iterrows():
#     shade = "#ffeaea" if row["is_shame_cluster"] else "#f5f5f5"
#     st.markdown(
#         f"<div style='background-color:{shade};padding:12px;border-radius:8px;margin-bottom:8px;'>"
#         f"<b>Doc:</b> {row['doc_id']}<br>"
#         f"<b>Author:</b> {row['author']}<br>"
#         f"<b>Title:</b> {row['title']}<br>"
#         f"<b>Paragraph {int(row['paragraph_id'])}</b> "
#         f"(Similarity: <b>{row['similarity']:.2f}</b>; " if row['similarity'] is not None else "(Similarity: <b>N/A</b>; "
#         f"Shame cluster: <b>{'YES' if row['is_shame_cluster'] else 'NO'}</b>)<br>"
#         f"<span style='font-size:1.1em'>{row['text']}</span>"
#         f"</div>",
#         unsafe_allow_html=True
#     )
# 
# st.success(f"Showing {min(topN, len(display_df))} paragraphs from cluster {cluster_choice}.")

st.header("Method #3: kNN neighbors")
st.header("Method #4: Community detection")