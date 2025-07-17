import json
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.title("Word embeddings annotations")
st.subheader("GOAL: Find shame-related paragraphs in corpus")
st.text("Use the word embeddings model (FastText) to annotate the corpus and evaluate annotation quality")

# ─── Annotated text ─────────────────────────────────────────────────
st.header("Method #1: N most similar (cosine) paragraphs")
st.markdown("""
Pipeline:
1. Use FastText model (300 ep, original text, stop words removed) to vectorize \
each paragraph in the corpus.
2. Compute a "shame vector" - use seed words and turn them into a "shame" vector.
3. Measure the (cosine) distance between the paragraph vector and the shame vector.
            
What's the "shame vector": 3 different sets of shame-related words were used \
to compute three different vectors:
1. Stem words vector - words in the corpus that had the stem "sram" in them and \
appeared more than 50 times.
2. Babel vector - words that are most similar to "sram" according to [BabelNet](https://babelnet.org/)
3. kontekst.io vector - ords that are most similar to "sram" according to [kontekst.io](https://www.kontekst.io/)
""")

# --- LOAD NESTED ANNOTATIONS ---
with open("annotations/main_shame_annotations.json", "r", encoding="utf-8") as f:
    annotations = json.load(f)

# --- GET SHAME VECTORS ---
sample_doc = next(iter(annotations.values()))
sample_para = next(iter(sample_doc["paragraphs"].values()))
shame_vectors = [k.replace("cos_sim_", "") for k in sample_para.keys() if k.startswith("cos_sim_")]

# --- FLATTEN FOR DATAFRAME ---
flat_records = []
for doc_id, doc_info in annotations.items():
    author = doc_info.get("author", "")
    title = doc_info.get("title", "")
    for para_id, para_info in doc_info["paragraphs"].items():
        for vec in shame_vectors:
            sim = para_info.get(f"cos_sim_{vec}", None)
            if sim is not None:
                flat_records.append({
                    "shame_id": doc_id,
                    "author": author,
                    "title": title,
                    #"paragraph_id": para_id, #uncomment if needed
                    "paragraph_text": para_info["text"],
                    f"similarity_{vec}": sim,
                    "shame_vector": vec  # For easier filtering
                })

df = pd.DataFrame(flat_records)

# --- PICKERS IN SAME ROW ---
col1, col2 = st.columns(2)
with col1:
    vec_choice = st.selectbox("Shame vector", shame_vectors)
with col2:
    topN = st.number_input("Top N", min_value=1, value=50)

# --- FILTERS FOR AUTHOR/TITLE ---
authors = ["All"] + sorted(df["author"].dropna().unique())
titles = ["All"] + sorted(df["title"].dropna().unique())

col3, col4 = st.columns(2)
with col3:
    author_choice = st.selectbox("Filter by author", authors)
with col4:
    title_choice = st.selectbox("Filter by title", titles)

# --- FILTER DATAFRAME ---
display_df = df[df["shame_vector"] == vec_choice]

if author_choice != "All":
    display_df = display_df[display_df["author"] == author_choice]
if title_choice != "All":
    display_df = display_df[display_df["title"] == title_choice]

# --- SORT AND SHOW ---
display_df = display_df.sort_values(f"similarity_{vec_choice}", ascending=False)
display_df = display_df[["shame_id", "author", "title", #"paragraph_id", 
                         "paragraph_text", f"similarity_{vec_choice}"]]
st.dataframe(display_df.head(topN), 
             hide_index=True,
             )

st.markdown("""
**Quick eval**: A quick manual inspection shows that the model captured the shame \
context surprisingly well. Setting up stem_sram and a minimum similarity \
threshold of 0.612 (or equivalently, picking the top 1000 entries) gives \
relatively good paragraphs.
""")

st.header("Method #2: kMeans clustering")
st.markdown("""
Goal: Cluster all paragraph vectors into _k_ groups, without \
using the shame vector.

One can then:
1. See if one cluster aligns with high shame similarity
2. Use clusters as soft annotation (e.g., “cluster 2 = likely shame-related”)
3. Visualize clusters in 2D (with PCA or t-SNE)
""")

data_means = "output/nested_paragraph_shame_annotations_kmeans.json"
with open(data_means, "r", encoding="utf-8") as f:
    annots = json.load(f)

# --- Flatten to DataFrame for easy filtering ---
records = []
for doc_id, doc_info in annots.items():
    author = doc_info.get("author", "")
    title = doc_info.get("title", "")
    for para_id, para_info in doc_info["paragraphs"].items():
        records.append({
            "doc_id": doc_id,
            "author": author,
            "title": title,
            "paragraph_id": para_id,
            "text": para_info.get("text", ""),
            "kmeans_label": para_info.get("kmeans_cluster", None),
            "is_shame_cluster": para_info.get("is_shame_cluster", False),
            "similarity": para_info.get("similarity", None)
        })
df = pd.DataFrame(records)
df["kmeans_label"] = pd.to_numeric(df["kmeans_label"], errors='coerce')
df["paragraph_id"] = pd.to_numeric(df["paragraph_id"], errors='coerce')
df = df.dropna(subset=["kmeans_label"])

# --- UI: Cluster picker, shame cluster only, author, title ---
clusters = sorted(df["kmeans_label"].dropna().unique().astype(int))
cluster_choice = st.selectbox("Choose cluster", clusters)

shame_only = st.checkbox("Show only shame cluster", value=False)
authors1 = ["All"] + sorted(df["author"].unique())
titles1 = ["All"] + sorted(df["title"].unique())

col1, col2 = st.columns(2)
with col1:
    author_choice = st.selectbox("Filter author", authors1)
with col2:
    title_choice = st.selectbox("Filter title", titles1)

display_df = df[df["kmeans_label"] == cluster_choice]
if shame_only:
    display_df = display_df[display_df["is_shame_cluster"]]

if author_choice != "All":
    display_df = display_df[display_df["author"] == author_choice]
if title_choice != "All":
    display_df = display_df[display_df["title"] == title_choice]

display_df = display_df.sort_values("similarity", ascending=False)

topN = st.slider("Show top N paragraphs", min_value=1, max_value=min(100, len(display_df)), value=min(20, len(display_df)))

for i, row in display_df.head(topN).iterrows():
    shade = "#ffeaea" if row["is_shame_cluster"] else "#f5f5f5"
    st.markdown(
        f"<div style='background-color:{shade};padding:12px;border-radius:8px;margin-bottom:8px;'>"
        f"<b>Doc:</b> {row['doc_id']}<br>"
        f"<b>Author:</b> {row['author']}<br>"
        f"<b>Title:</b> {row['title']}<br>"
        f"<b>Paragraph {int(row['paragraph_id'])}</b> "
        f"(Similarity: <b>{row['similarity']:.2f}</b>; " if row['similarity'] is not None else "(Similarity: <b>N/A</b>; "
        f"Shame cluster: <b>{'YES' if row['is_shame_cluster'] else 'NO'}</b>)<br>"
        f"<span style='font-size:1.1em'>{row['text']}</span>"
        f"</div>",
        unsafe_allow_html=True
    )

st.success(f"Showing {min(topN, len(display_df))} paragraphs from cluster {cluster_choice}.")

st.header("Method #3: kNN neighbors")
st.header("Method #4: Community detection")