import streamlit as st
import pandas as pd
import os

st.title("Shame Corpus Analysis")
df = pd.read_csv('corpus_metadata.csv')

# --- Clean ---
rmv_cols = ['author_and_title', 'publ_year']
df = df[[x for x in df.columns if x not in rmv_cols]]

# --- Filter UI ---
st.subheader("1. Filter")
col1, col2 = st.columns([1, 2])
with col1:
    column = st.selectbox("Column", df.columns)
with col2:
    value = st.text_input(f"Filter {column}")
if value:
    st.dataframe(df[df[column].astype(str).str.contains(value, case=False)])

# --- Corpus Preview ---
st.subheader("2. Corpus Preview")
st.dataframe(df, hide_index=True)

# --- 1. Table: Group by source_corpus ---
st.subheader("3. Overview by Source Corpus")
if 'source_corpus' in df.columns and 'word_count' in df.columns:
    group1 = df.groupby('source_corpus').agg(
        num_docs=('source_corpus', 'count'),
        num_words=('word_count', 'sum')
    ).reset_index()

    # Compute percentages
    total_docs = group1['num_docs'].sum()
    total_words = group1['num_words'].sum()
    group1['pct_docs'] = (group1['num_docs'] / total_docs * 100).round(2)
    group1['pct_words'] = (group1['num_words'] / total_words * 100).round(2)

    # Add TOTAL row to the end of the same dataframe
    total_row = pd.DataFrame([{
        'source_corpus': 'TOTAL',
        'num_docs': total_docs,
        'num_words': total_words,
        'pct_docs': 100.0,
        'pct_words': 100.0
    }])

    group1 = pd.concat([group1, total_row], ignore_index=True)

    st.dataframe(group1, 
                    hide_index=True,
                    column_config={
                            "pct_docs": st.column_config.NumberColumn(format="%.1f%%"),
                            "pct_words": st.column_config.NumberColumn(format="%.1f%%"),
                            }
                    )


# --- 2. Table: Group by author_gender ---
st.subheader("4. Overview by Author Gender")
if 'author_gender' in df.columns and 'word_count' in df.columns:
    # Number of unique authors per gender
    if 'author' in df.columns:
        num_authors = df.groupby('author_gender')['author'].nunique()
        total_authors = num_authors.sum()
        num_docs = df.groupby('author_gender').size()
        total_docs = num_docs.sum()
        num_words = df.groupby('author_gender')['word_count'].sum()
        total_words = num_words.sum()
        percent_authors = num_authors / total_authors 
        percent_docs = num_docs / total_docs 
        percent_words = num_words / total_words 
        group2 = (
            pd.DataFrame({
                'num_authors': num_authors,
                'pct_authors': percent_authors * 100,
                'num_docs': num_docs,
                'pct_docs': percent_docs * 100,
                'num_words': num_words,
                'pct_words': percent_words * 100
            })
            .reset_index()
        )

        # Add TOTAL row to the end
        total_row2 = pd.DataFrame([{
            'author_gender': 'TOTAL',
            'num_authors': total_authors,
            'pct_authors': 100.0,
            'num_docs': total_docs,
            'pct_docs': 100.0,
            'num_words': total_words,
            'pct_words': 100.0
        }])

        group2 = pd.concat([group2, total_row2], ignore_index=True)

        st.dataframe(group2, 
                        hide_index=True,
                        column_config={
                            "pct_authors": st.column_config.NumberColumn(format="%.1f%%"),
                            "pct_docs": st.column_config.NumberColumn(format="%.1f%%"),
                            "pct_words": st.column_config.NumberColumn(format="%.1f%%"),
                            })

# --- 3. Charts: Bar charts by publ_year_clean ---
st.subheader("5. Publishing Time Analysis")
if 'publ_year_clean' in df.columns:
    # a) num documents per year
    docs_by_year = df.groupby('publ_year_clean').size()
    st.markdown("##### Number of Documents by Year")
    st.bar_chart(docs_by_year, x_label="Publishing year", y_label="Num documents")
    # b) num words per year
    if 'word_count' in df.columns:
        words_by_year = df.groupby('publ_year_clean')['word_count'].sum()
        st.markdown("##### Number of Words by Year")
        st.bar_chart(words_by_year, x_label="Publishing year", y_label="Num words")