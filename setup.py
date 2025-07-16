import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

from helpers.data_fetchers import fetch_corpus_metadata, fetch_sl_stopwords, fetch_raw_corpora
from helpers.data_extractors import extract_wikivir_html_to_txt, extract_clarin_txt_paragraphs, batch_extract_corpus_xml_to_txt

def main():
    steps = [
        "Fetch corpus metadata",
        "Fetch (raw) source corpora",
        "Fetch Slovenian stopwords",
        "Extract Wikivir txt",
        "Extract KDSP txt",
        "Extract PriLit txt",
        "Extract ElTeC txt"
    ]

    pbar = tqdm(steps)
    tqdm.write("===# START SETUP SCRIPT #===")

    load_dotenv()
    CORPUS_INDEX_ID = os.getenv("CORPUS_INDEX_ID")
    CORPUS_INDEX_SHEET = os.getenv("CORPUS_INDEX_SHEET")
    RAW_CORPUS_DIR = os.getenv("RAW_CORPUS_DIR")
    ORIG_CORPUS_DIR = os.getenv("ORIG_CORPUS_DIR")

    tqdm.write("\n===# DATA FETCH #===")

    # Fetch corpus metadata, aka index
    # Uncomment the line below to import corpus from Google Sheets
    #df = fetch_corpus_metadata(CORPUS_INDEX_ID, CORPUS_INDEX_SHEET)
    df = pd.read_csv('corpus_metadata.csv')
    tqdm.write("✅ Fetch corpus metadata.")
    pbar.update()

    # Fetch Slovenian stopwords for NLP preprocessing
    fetch_sl_stopwords()
    tqdm.write("✅ Fetch Slovenian stopwords")
    pbar.update()

    # Download and all source corpora
    # Note: This will download all source corpora, not just the ones in the index.
    # If you want to download only specific corpora, you can filter the df before passing
    # it to the fetch_raw_corpora function.
    # Uncomment the next line to fetch only the raw corpora listed in the index.
    fetch_raw_corpora(df, RAW_CORPUS_DIR)
    tqdm.write("✅ Fetch (raw) source corpora.")
    pbar.update()

    # extract Wikivir HTMLs to TXT files
    extract_wikivir_html_to_txt(df, 
                                RAW_CORPUS_DIR + '/wikivir', 
                                ORIG_CORPUS_DIR + '/paragraph',
                                ORIG_CORPUS_DIR + '/sentence')
    tqdm.write("\n===# DATA EXTRACT #===")
    tqdm.write("✅ Extract Wikivir files to shame corpus.")
    pbar.update()

    # extract texts from KDSP corpus
    extract_clarin_txt_paragraphs(
        df=df,
        raw_corpus_dir=os.path.join(RAW_CORPUS_DIR, 'kdsp', 'KDSP.txt'),
        output_dir=ORIG_CORPUS_DIR + '/paragraph',
        df_source_corpus_val='KDSP'
    )
    #TODO: move tqdm into function, to make progress more intuitive
    batch_extract_corpus_xml_to_txt(df, 'KDSP', RAW_CORPUS_DIR, './data', 
                                    granularity='sentence', content_type='word') 
    batch_extract_corpus_xml_to_txt(df, 'KDSP', RAW_CORPUS_DIR, './data', 
                                    granularity='paragraph', content_type='lemma')
    batch_extract_corpus_xml_to_txt(df, 'KDSP', RAW_CORPUS_DIR, './data', 
                                    granularity='sentence', content_type='lemma')
    tqdm.write("✅ Extract KDSP files to shame corpus.")
    pbar.update()

    # extract texts from PriLit corpus
    extract_clarin_txt_paragraphs(
        df=df,
        raw_corpus_dir=os.path.join(RAW_CORPUS_DIR, 'prilit', 'PriLit.txt'),
        output_dir=ORIG_CORPUS_DIR + '/paragraph',
        df_source_corpus_val='PriLit'
    )
    batch_extract_corpus_xml_to_txt(df, 'PriLit', RAW_CORPUS_DIR, './data', 
                                    granularity='sentence', content_type='word')
    batch_extract_corpus_xml_to_txt(df, 'PriLit', RAW_CORPUS_DIR, './data', 
                                    granularity='paragraph', content_type='lemma')
    batch_extract_corpus_xml_to_txt(df, 'PriLit', RAW_CORPUS_DIR, './data', 
                                    granularity='sentence', content_type='lemma')
    tqdm.write("✅ Extract PriLit files to shame corpus.")
    pbar.update()

    # extract texts from PELTeC corpus
    batch_extract_corpus_xml_to_txt(df, 'ELTeC', RAW_CORPUS_DIR, './data', 
                                    granularity='paragraph', content_type='word')
    batch_extract_corpus_xml_to_txt(df, 'ELTeC', RAW_CORPUS_DIR, './data', 
                                    granularity='sentence', content_type='word')
    batch_extract_corpus_xml_to_txt(df, 'ELTeC', RAW_CORPUS_DIR, './data', 
                                    granularity='paragraph', content_type='lemma')
    batch_extract_corpus_xml_to_txt(df, 'ELTeC', RAW_CORPUS_DIR, './data', 
                                    granularity='sentence', content_type='lemma')
    tqdm.write("✅ Extract ElTeC files to shame corpus.")
    pbar.update()

    #TODO: Add tests after import to verify data quality.

    tqdm.write("===# SETUP COMPLETE #===")
    pbar.close()


if __name__ == "__main__":
    main()