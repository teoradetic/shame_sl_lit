from collections import Counter
from gensim.models import FastText
from gensim.utils import simple_preprocess
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def read_corpus(directory, stopwords=None):
    """
    Reads and tokenizes all `.txt` files in a directory, yielding 
    one tokenized sentence per line.

    Each line is tokenized using Gensim's `simple_preprocess`, 
    and optional stopword removal is applied.

    Args:
        directory (str): Path to the directory containing `.txt` files.
        stopwords (set, optional): A set of stopwords to remove from each 
                                   tokenized sentence. Defaults to None 
                                   (no stopword filtering).

    Yields:
        list[str]: A list of lowercase tokens for each sentence, with 
                   stopwords removed if provided.
    """
    if stopwords:
        stopwords = set(simple_preprocess(' '.join(stopwords), deacc=True))


    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                for line in file:
                    tokens = simple_preprocess(line, deacc=True)
                    if stopwords:
                        tokens = [w for w in tokens if w not in stopwords]
                    if tokens:
                        yield tokens


def generate_seed_words_from_stem(stem="sram", 
                                  corpus_path="data/lemma_txt_corpus/sentences",
                                  n=100,
                                  out_path=None):
    """
    Generate seed words from a given stem and keep those with >= n occurrences 
    in the corpus.

    Args:
        stem (str): The word stem to search for.
        corpus_path (str): Path to the corpus file.
        n (int): Minimum number of occurrences required.
        out_path (str): If specified, it saves the list of seed words to out_path.

    Returns:
        List[Tuple[str, int]]: Sorted list of (word, count) for words matching 
        the stem.
    """
    # manually inspected the list below and generated the excluded words
    non_shame_words = ['nesramnost', 'narnesramen', 'nasramen', 'nesramen',
                    'nesramno']

    # Read and flatten corpus
    sentences_lemma = read_corpus(corpus_path)
    flat_words = [word for sent in sentences_lemma for word in sent if word not in non_shame_words] #TODO: no magic vals
    word_counts = Counter(flat_words)
    

    # Filter for words starting with the stem and with at least n occurrences
    seed_words = [(word, count) for word, count in word_counts.items()
                  if word.startswith(stem) and count >= n]
    seed_words.sort(key=lambda x: x[1], reverse=True)

    # If specified, saved the list of seed words to the location
    if out_path:
        clean_seed_words = [w for w, _ in seed_words]
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            for w in clean_seed_words:
                f.write(f"{w},\n")  
    return seed_words


def get_similar_words_fasttext(model, seed_words, topn=10):
    """
    Given a list of seed words and a trained FastText model,
    return a list of similar words and their similarity scores.
    """
    results = []

    for word in seed_words:
        if word in model.wv:
            similar = model.wv.most_similar(word, topn=topn)
            for sim_word, score in similar:
                results.append({
                    "seed": word,
                    "similar": sim_word,
                    "similarity": score
                })
        else:
            print(f"'{word}' not found in vocabulary.")

    return results


def get_topn_neighbors(model, seed_words, topn=10):
    """
    Retrieves the top-N most similar words for each seed word using a 
    trained FastText model.

    Args:
        model (gensim.models.FastText): A trained FastText model with word vectors.
        seed_words (list[str]): List of seed words to find neighbors for.
        topn (int): Number of nearest neighbors to retrieve for each seed word.

    Returns:
        dict[str, set[str]]: Dictionary mapping each seed word to a set of its top-N 
                             similar words. If a seed word is not in the vocabulary, 
                             an empty set is returned.
    """
    neighbors = {}
    for word in seed_words:
        if word in model.wv:
            top_similars = model.wv.most_similar(word, topn=topn)
            neighbors[word] = {w for w, _ in top_similars}
        else:
            neighbors[word] = set()
    return neighbors


def jaccard(set1, set2):
    """
    Computes the Jaccard similarity between two sets. Jaccard similarity is 
    defined as the size of the intersection divided by the size of the union.

    Used primarily to compare two consecutive trainings of a FastText word embedding
    model, as a measure of embedding stability (logic: if the embedding space is stable
    the Jaccard similarity between the top n closes words won't change much).

    Args:
        set1 (set): First set.
        set2 (set): Second set.

    Returns:
        float: Jaccard similarity score in the range [0, 1]. 
               Returns 1.0 if both sets are empty.
    """
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)


def compute_epochwise_jaccard_similarity(
    neighbor_json_path,
    seed_words,
    output_csv_path=None
    ):
    """
    Computes Jaccard similarity between top-N neighbor sets of seed words
    across consecutive FastText training epochs, using a JSON file.

    Args:
        neighbor_json_path (str): Path to a JSON file with structure:
                                  {epoch: {seed_word: [similar_words]}}.
        seed_words (list[str]): List of seed words to compare.
        output_csv_path (str, optional): Path to save the output CSV of Jaccard scores.
                                         If None, the file is not saved.

    Returns:
        pd.DataFrame: DataFrame with mean/median Jaccard scores for each 
        epoch transition.
    """
    with open(neighbor_json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Convert lists to sets
    neighbor_dict = {
        int(epoch): {k: set(v) for k, v in seed_dict.items()}
        for epoch, seed_dict in raw_data.items()
    }

    rows = []
    epochs = sorted(neighbor_dict.keys())

    for prev, curr in zip(epochs, epochs[1:]):
        jaccards = [
            jaccard(
                neighbor_dict[prev].get(w, set()), 
                neighbor_dict[curr].get(w, set())
                )
            for w in seed_words
        ]
        rows.append({
            "prev_epoch": prev,
            "curr_epoch": curr,
            "mean_jaccard": np.mean(jaccards),
            "median_jaccard": np.median(jaccards)
        })

    stability_df = pd.DataFrame(rows)

    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        stability_df.to_csv(output_csv_path, index=False)
        print(f"✅ Jaccard similarity CSV saved to: {output_csv_path}")

    return stability_df



def fasttext_incr_train_and_predict(
    sentences,
    seed_words,
    output_json_fname="neighbors_by_epoch.json",
    output_json_dir=".",
    output_model_fname="model",
    output_model_dir=".",
    vector_size=300,
    window=5,
    min_count=1,
    sg=1,
    workers=4,
    seed=42,
    step_epochs=5,
    max_epochs=50,
    topn=10
):
    """
    Incrementally trains a FastText model and tracks top-N neighbors of seed words at each epoch checkpoint.

    Args:
        sentences (list[list[str]]): Preprocessed tokenized sentences.
        seed_words (list[str]): Words to track similar words for during training.
        output_json_fname (str): Filename for output JSON file storing neighbors across epochs.
        output_json_dir (str): Directory to save output file.
        output_model_fname (str): Filename for saving model after all epochs.
        output_model_dir (str): Directory to save model.
        vector_size (int): Dimensionality of embeddings.
        window (int): Context window size.
        min_count (int): Minimum word count threshold.
        sg (int): 1 for skip-gram, 0 for CBOW.
        workers (int): Number of worker threads.
        seed (int): Random seed for reproducibility.
        step_epochs (int): Number of epochs per training step.
        max_epochs (int): Total number of training epochs.
        topn (int): Number of top similar words to store for each seed.

    Returns:
        dict: Dictionary of neighbors by epoch for further use.
    """
    output_json_path = os.path.join(output_json_dir, output_json_fname)
    os.makedirs(output_json_dir, exist_ok=True)
    os.makedirs(output_model_dir, exist_ok=True)
    output_model_path = os.path.join(output_model_dir, output_model_fname)

    print("🔧 Initializing FastText model…")
    model = FastText(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        seed=seed
    )
    model.build_vocab(sentences)

    neighbors_by_epoch = {}

    print("🚀 Starting incremental training:")
    for curr_epoch in tqdm(range(step_epochs, max_epochs + 1, step_epochs), desc="Epochs"):
        model.train(sentences, total_examples=len(sentences), epochs=step_epochs)

        epoch_neighbors = {}
        for word in seed_words:
            if word in model.wv:
                top = model.wv.most_similar(word, topn=topn)
                epoch_neighbors[word] = [w for w, _ in top]
            else:
                epoch_neighbors[word] = []

        neighbors_by_epoch[curr_epoch] = epoch_neighbors

    # Save single JSON file
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(neighbors_by_epoch, f, ensure_ascii=False, indent=2)
    print(f"✅ Neighbors saved to {output_json_path}")

    # Save model after training for reuse
    model.save(f"{output_model_path}.model")
    print(f"✅ Model saved to {output_model_path}")

    return neighbors_by_epoch

