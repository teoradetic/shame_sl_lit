import json
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ─── Intro ─────────────────────────────────────────────────
st.title("Word embeddings evaluation")
st.markdown("""
**GOAL**: Use word embeddings models (FastText) to automatically identify shame.
This notebook shows how the model was trained and how its best version was picked.
""")

st.header("1. Methodological overview")
with st.expander("Show detailed explanation of the pipeline (click to expand)"):
    st.markdown("""
    1. **Data Ingestion**: All .txt files in the corpus directory are loaded, \
                split by sentence or paragraph, then tokenized.
    2. **Preprocessing Options**:
        - Text can be _lemmatized_ (root forms) or kept as original.
        - Stop words (common words like "in", "ali") can be removed or kept.
        - Granularity can be sentence or paragraph.
    3. **Seed Word Extraction**: 
        - Look for all words with the “sram” stem (Slovene for shame) and keep \
                those occurring at least 50 times in the corpus.
        - Seed word list: ['sram', 'sramota', 'sramovati', 'sramoten', \
                'sramezljivo', 'osramocen', 'sramezljivost', 'sramezljiv', \
                'sramotno', 'osramotiti', 'nesramnez', 'sramotiti', 'zasramovati']
    4. **Word Embedding Training**:
        - For every configuration (3 dimensions: lemma/original × \
                sentence/paragraph × stopwords on/off), FastText is trained.
        - For each epoch batch (every 5 up to 300), the app saves the top-N (100) \
                most similar words to each seed.
    5. **Model evaluation**: 
        1. Embedding stability tracking
            - At each epoch batch (aka, every 5 epochs) the model predicts the \
                top-N (100) most similar words to the seed keywords at that \
                training step.
            - The top-N (100) most similar words are compared for the same seed \
                keyword in the current vs previous epoch batch.
                - Comparison is done with Jaccard Similarity: which similar words \
                appear in both the previous and current batch vs just one of them.
            - Overall Jaccard similarity is computed as the average of current \
                vs previous batch.
            - When Jaccard similarity plateaus, the embedding is stable.
        2. Manual inspection of most similar words to seed words.
    6. **Final result**: Choose most stable model for future steps.
    
    For full implementation details, check [2_word_embeddings_training.ipynb](https://github.com/teoradetic/shame_sl_lit/blob/main/notebooks/2_word_embeddings_training.ipynb)\
     on GitHub
    """)

# ─── Embedding stability ─────────────────────────────────────────────────

st.header("2. Evaluate word embedding quality")
st.text("GOAL: Find best configuration for FastText model to be used in corpus \
annotation.")

st.subheader("2.1 Embedding stability across epochs")
st.markdown("""
- _What 'embedding stability' means_: For robust word embeddings, the \
            nearest-neighbor lists for important words (e.g. "sram") shouldn’t \
            change much as you keep training.
- _How measured_: Jaccard similarity between the top-N (100) neighbors after each \
            5-epoch jump. Closer to 1 = stable, closer to 0 = neighbors keep changing.
- _Config_: 3 dimensions: lemma/original × sentence/paragraph × stopwords on/off
            
Try it out yourself :point_down:
""")

st.markdown("""
##### Embedding stability across epochs chart
**What does this chart show?**
- Each line is a different embedding model configuration (lemmatized vs. original, \
            sentence vs. paragraph, stopwords kept vs. removed).
- Higher **mean Jaccard similarity** means the words that a model outputs as most \
            similar to the seed keywords are mostly shared with the previous model \
            output.
""")

# Load combined data
all_jaccard = pd.read_csv("streamlit_app/data/all_jaccard_ft_epochs.csv")
with open("streamlit_app/data/neighbors_by_epoch_ft_300ep_original_txt_paragraph_remove_stopwords.json", "r", encoding="utf-8") as f:
    neighbors_by_epoch = json.load(f)

##### Stability Visualization ######
# Example options
content_types = ["both", "lemma", "original"]
granularities = ["both", "sentence", "paragraph"]
stopword_options = ["both", "remove", "keep"]

# Make three columns
col1, col2, col3 = st.columns(3)

with col1:
    c1 = st.selectbox("Text type", content_types, index=0)
with col2:
    g1 = st.selectbox("Granularity", granularities, index=0)
with col3:
    s1 = st.selectbox("Stopwords treatment", stopword_options, index=0)

# Filter DataFrame based on user choice
def filter_df(df, col, val):
    if val == "both":
        return df
    return df[df[col] == val]

filtered = filter_df(all_jaccard, "txt_type", c1)
filtered = filter_df(filtered, "granularity", g1)
filtered = filter_df(filtered, "stopwords", s1)

# Create one line per configuration
fig, ax = plt.subplots()
for name, group in filtered.groupby(["txt_type", "granularity", "stopwords"]):
    label = f"{name[0]}/{name[1]}/{name[2]}"
    ax.plot(group["curr_epoch"], group["mean_jaccard"], marker="o", label=label)

ax.set_xlabel("Epoch")
ax.set_ylabel("Mean Jaccard similarity")
ax.set_title("Embedding Stability Across Epochs")
ax.set_ylim(0, 1)
ax.legend()
st.pyplot(fig)

st.text("Comment: Irrespective of configuration choices, the models reach stability after cca 100 epochs.")

st.subheader("2.2 Model's predicted nearest words to target seed word")
st.markdown("""
Below you can pick the target seed word and what the model (orig/paragraph/stop \
            words removed) predicted as the 100 most similar words at each epoch.
""")
##### Word similarity picker ######
# Get all epochs and seed words available:
# Prepare options
epochs = sorted(map(int, neighbors_by_epoch.keys()))
epochs_str = [str(e) for e in epochs]
seed_words = sorted(neighbors_by_epoch[epochs_str[0]].keys())

# Three selection widgets in a row
col1, col2, col3 = st.columns(3)
with col1:
    selected_seed = st.selectbox("Seed word", seed_words, 
                                 index=seed_words.index("sram") if "sram" in seed_words else 0)
with col2:
    epoch1 = st.selectbox("Epoch 1", epochs_str, index=len(epochs_str)-2)
with col3:
    epoch2 = st.selectbox("Epoch 2", epochs_str, index=len(epochs_str)-1)

# Extract neighbor lists
n1 = neighbors_by_epoch[epoch1].get(selected_seed, [])
n2 = neighbors_by_epoch[epoch2].get(selected_seed, [])

# Compute and display Jaccard similarity
set1, set2 = set(n1), set(n2)
if set1 or set2:
    jaccard = len(set1 & set2) / len(set1 | set2)
    st.markdown(f"**Jaccard similarity:** {jaccard:.2f}")
    intersection = list(set1 & set2)
else:
    st.markdown("_No neighbors to compare._")
    intersection = []

# Output similar words side by side
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"**Epoch {epoch1}**")
    st.write(n1)
with c2:
    st.markdown(f"**Epoch {epoch2}**")
    st.write(n2)
with c3:
    st.markdown(f"**Words present in both epochs:**")
    st.write(intersection)

def jaccard_matrix(neighbors_by_epoch, seed_word):
    epochs = sorted(map(int, neighbors_by_epoch.keys()))
    epochs_str = [str(e) for e in epochs]
    n = len(epochs)
    matrix = pd.DataFrame(index=epochs, columns=epochs, dtype=float)

    for i, ei in enumerate(epochs_str):
        set_i = set(neighbors_by_epoch[ei].get(seed_word, []))
        for j, ej in enumerate(epochs_str):
            set_j = set(neighbors_by_epoch[ej].get(seed_word, []))
            if set_i or set_j:
                jac = len(set_i & set_j) / len(set_i | set_j)
            else:
                jac = 1.0
            matrix.iloc[i, j] = jac
    return matrix

st.text("Comment: The models mostly predict words with the stem 'sram' in them.")

# Usage in Streamlit:
matrix = jaccard_matrix(neighbors_by_epoch, selected_seed)
st.markdown(f"""#### Jaccard similarity matrix for seed word '{selected_seed}' \
            between all epochs
Use: Check how "correlated" (similar acc. to Jaccard similarity) each epoch's \
            predictions are with each other epoch.
""")
st.dataframe(matrix)

st.subheader("2.3 FastText model evaluation")
st.markdown(
"""
What can we infer from the data above about the optimal FastText model configuration?
1. **The model stabilizes early**: Jaccard similarity plateaus after ~100 epochs \
(cf. 2.1), indicating the model’s word neighborhoods stabilize quickly. There're \
_diminishing returns in training after ~100 epochs_.
    - Confirmed with Jaccard matrix - after 100, the values are higher, showing more \
robust embeddings.
2. **Consistent embeddings**: The top neighbors for each seed are \
themselves close (other “shame” words). It's unclear if the model has learnt \
meaningful “shame” cluster(s), and doesn’t drift away during training, or if \
the model has learned subword tokens (a FastText tokenization option). Either way \
the list of most similar words are moslty about the same words as the seeds.
3. **Preprocessing choices matter less after stabilization**: Once the model \
stabilizes, preprocessing choices (lemmatization, stopword removal, granularity) \
have less impact on epoch-to-epoch stability—but. Note, they may affect which \
words appear as neighbors, but this is unclear. 

Conclusion: 
- No model configuration is better than the other. This is why we pick the following configurations:
    - original texts: more available texts - Wikivir is not lemmatized, 
    - remove stop words: cleaner (and faster), 
    - paragraph: more semantic context, even for downstream tasks (human annotation).
- Unclear whether the model can be a good predictor of semantically similar words \
(therefore check ch. 3 below).

#### Next steps
Use the chosen model to annotate the corpus and evaluate the quality of annotations.
To see the annotations (and evaluations), check: """)
st.page_link("pages/3_word_embeddings_annotations.py", label=":blue[_Word Embeddings Annotations_]")
