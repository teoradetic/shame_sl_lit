from bs4 import BeautifulSoup
import os
import pandas as pd
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

from helpers.string_utils import sanitize_filename, split_paragraphs_to_sentences

#########################################################
# Helper Functions for Wikivir Processing HTML --> .txt #
#########################################################


def _slice_wikivir_fragment(html_content: str, source_name: str | None = None) -> str:
    """
    Slice HTML to relevant Wikivir fragment.

    - If `source_name` starts with "Miha_Remec", return the second <table> in <body>.
    - Otherwise find text between the first <a href*="leposl.html"> and the license marker.
    - Falls back to full <body> HTML if markers not found.
    """
    soup_full = BeautifulSoup(html_content, "html.parser")
    body = soup_full.body or soup_full

    # Miha Remec special case: filename starts with Miha_Remec
    if source_name and source_name.startswith("Miha_Remec"):
        tables = body.find_all("table")
        if len(tables) < 2:
            raise ValueError("Expected at least two <table> elements in Remec page.")
        return str(tables[1])

    # Default Wikivir logic: anchor + license markers
    start_tag = soup_full.find("a", href=lambda h: h and "leposl.html" in h)
    end_tag = soup_full.find(string=lambda t: t and "To delo je licencirano" in t)
    if start_tag and end_tag:
        html_str = str(soup_full)
        start_idx = html_str.find(str(start_tag)) + len(str(start_tag))
        end_idx = html_str.find(str(end_tag))
        return html_str[start_idx:end_idx]

    # Fallback: entire body HTML
    return str(body)


def extract_paragraphs(html_content: str, source_name: str | None = None) -> list[str]:
    """
    Parse HTML and return one line per tag with non-empty text.
    All inner newlines are replaced by spaces, and whitespace is collapsed.
    """
    # 1) Slice to relevant fragment
    fragment = _slice_wikivir_fragment(html_content, source_name)

    # 2) Parse
    soup = BeautifulSoup(fragment, "html.parser")
    content = soup.body or soup

    # Remove non-text elements
    for tag in content.find_all(["b", "i", "big", "small", "cite", "span", "a"]):
        tag.unwrap()

    # Replace existing newlines within tags with space 
    text = str(content).replace("\n", " ")

    # replace all tags wuth newlines (to mimic paragraphs)
    text = re.sub(r"<[^>]+>", "\n", text)

    # remove double+ white space (artefact)
    text = re.sub(r' {2,}', ' ', text)

    # clean trtailing whitespace and remove empty newlines
    clean = [x.strip() for x in text.split("\n")]
    clean = [x for x in clean if len(x) > 1]

    # custom removal of annotators
    annotators = ["Miran Hladnik", "Andreja Musar"]
    clean = [x for x in clean if not any(substr in x for substr in annotators)]
    
    return clean


def extract_wikivir_html_to_txt(
    df: pd.DataFrame,
    in_dir: str,
    out_dir_paragraph: str,
    out_dir_sentence: str,
    verbose: bool = False
) -> None:
    """
    For each row in df with source_corpus=="Wikivir - leposlovje":
      - Locate the raw HTML file in `in_dir` named <sanitize(author_and_title)>.html
      - Extract cleaned text (paragraphs)
      - Save to `out_dir_paragraph/<shame_id>.txt`
      - Then extract the sentences from the paragraphs and save them to 
      `out_dir_sentence/<shame_id>.txt`
    """
    Path(out_dir_paragraph).mkdir(parents=True, exist_ok=True)
    Path(out_dir_sentence).mkdir(parents=True, exist_ok=True)
    subset = df[df["source_corpus"] == "Wikivir - leposlovje"]

    for _, row in subset.iterrows():
        in_fname = sanitize_filename(row["author_and_title"])
        in_path  = os.path.join(in_dir, f"{in_fname}.html")
        out_fname = row.shame_id
        out_path = os.path.join(out_dir_paragraph,  f"{out_fname}.txt")

        try:
            with open(in_path, encoding="utf-8") as f:
                raw_html = f.read()

            blocks = extract_paragraphs(
                raw_html,
                source_name=in_fname
            )

            with open(out_path, "w", encoding="utf-8") as out_f:
                out_f.write("\n".join(blocks))

            if verbose:
                print(f"[EXTRACTED PARAGRAPHS] {in_path} to {out_path}")

        except Exception as e:
            print(f"[FAILED PARAGRAPHS] {in_fname}: {e}")
        
        # Now extract sentences from the paragraphs
        try:
            out_path = os.path.join(out_dir_sentence,  f"{out_fname}.txt")

            sentences = split_paragraphs_to_sentences(blocks)
                
            with open(out_path, "w", encoding="utf-8") as out_f:
                out_f.write("\n".join(sentences))

            if verbose:
                print(f"[EXTRACTED SENTENCES] {out_path}")

        except Exception as e:
                print(f"[FAILED SENTENCES] {out_path}: {e}")




##############################################################################
# Helper Functions for Processing XML --> .txt (for PriLit, KDSP, and ElTeC) #
##############################################################################
def extract_xml_to_text(
    in_path,
    in_fname,
    out_path,
    out_fname,
    granularity='sentence',
    content_type='word'
):
    """
    Extracts text or lemmas from XML TEI files by sentence or paragraph.

    Parameters:
        in_path (str): Directory of the input file.
        in_fname (str): Input XML file name.
        out_path (str): Directory to save output file.
        out_fname (str): Output txt file name.
        granularity (str): 'sentence' or 'paragraph'
        content_type (str): 'word' or 'lemma'
    """
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

    tree = ET.parse(os.path.join(in_path, in_fname))
    root = tree.getroot()

    # helper function to remove space before punctation
    def _rmv_space_bfr_punct(txt:str) -> str:
        # Remove spaces after all openers
        txt = re.sub(r'(\(|\[|\{|"|\'|`|„|»|<<|<) +', r'\1', txt)
        # Remove spaces before all closers and punctuation marks
        txt = re.sub(r' +(\)|\]|\}|\"|\'|`|”|«|>>|>|\.|,|;|:|!|\?)', r'\1', txt)
        return txt


    # Find all paragraph-level blocks: <p> and <ab>
    para_blocks = []
    para_blocks.extend(root.findall('.//tei:p', ns))
    para_blocks.extend(root.findall('.//tei:ab', ns))

    # Remove duplicates, keep order (in case both are present in mixed order)
    all_blocks = sorted(para_blocks, key=lambda el: el.sourceline if hasattr(el, 'sourceline') else 0)

    # If there are no <p> or <ab>, maybe treat <div> as paragraph?
    if not all_blocks:
        all_blocks = root.findall('.//tei:div', ns)

    result_lines = []

    if granularity == 'paragraph':
        # Extract per paragraph/ab block
        for block in all_blocks:
            words = []
            # Sentences in this paragraph/ab, or just words/pcs directly inside
            for el in block.iter():
                if el.tag.endswith('w'):
                    if content_type == 'word':
                        words.append(el.text)
                    else:  # lemma requested
                        lemma = el.attrib.get('lemma')
                        words.append(lemma if lemma else el.text)
                elif el.tag.endswith('pc'):  # punctuation
                    words.append(el.text)
            para_str = ' '.join(words)
            # Clean double-spaces before punctuation
            para_str = _rmv_space_bfr_punct(para_str)
            result_lines.append(para_str.strip())

    elif granularity == 'sentence':
        # Extract per sentence inside <p>, <ab>, or <div>
        # We look inside all paragraph-level blocks
        for block in all_blocks:
            for s in block.findall('.//tei:s', ns):
                words = []
                for el in s:
                    if el.tag.endswith('w'):
                        if content_type == 'word':
                            words.append(el.text)
                        else:  # lemma requested
                            lemma = el.attrib.get('lemma')
                            words.append(lemma if lemma else el.text)
                    elif el.tag.endswith('pc'):
                        words.append(el.text)
                sent_str = ' '.join(words)
                sent_str = _rmv_space_bfr_punct(sent_str)
                result_lines.append(sent_str.strip())
    else:
        raise ValueError("granularity must be 'sentence' or 'paragraph'")

    # Save to output file, one per line, UTF-8
    os.makedirs(out_path, exist_ok=True)
    out_file = os.path.join(out_path, out_fname)
    with open(out_file, 'w', encoding='utf-8') as f:
        for line in result_lines:
            f.write(line + '\n')


def batch_extract_corpus_xml_to_txt(
    df,
    source_corpus,
    in_dir,
    out_dir,
    granularity='sentence',
    content_type='word'
):
    """
    Loops through df, extracts texts from XML based on row parameters
    using extract_xml_to_text().

    Args:
        df (pd.DataFrame): DataFrame of corpus metadata with columns needed for 
        source_corpus (str): Name of source corpus to be extracted
        in_dir (str): Base input directory for corpora
        out_dir (str): Base output directory
        xml_extract_func (function): Your XML extraction function
        granularity (str): 'sentence' or 'paragraph'
        content_type (str): 'word' or 'lemma'
    """
    # set path/fname of source corpora - they use different nomenclature
    if source_corpus == 'KDSP':
        in_dir = os.path.join(in_dir, 'kdsp', 'KDSP.TEI.ana')
        fname_ext = '.ana.xml'
    elif source_corpus == 'PriLit':
        in_dir = os.path.join(in_dir, 'prilit', 'PriLit.ana')
        fname_ext = '.ana.xml'
    elif source_corpus == 'ELTeC':
        in_dir = os.path.join(in_dir, 'eltec', 'l2')
        fname_ext = '-L2.xml'
    else:
        raise ValueError(f"{source_corpus} not found in dataframe.")
    
    # organize into different files based on content extracted
    if content_type == 'lemma':
        out_dir = os.path.join(out_dir, 'lemma_txt_corpus')
    elif content_type == 'word':
        out_dir = os.path.join(out_dir, 'original_txt_corpus')
    else:
        raise ValueError(f"{content_type} not one of 'lemma' or 'word'.")
    
    # organize into different files based on granularity extracted
    if granularity not in ['sentence', 'paragraph']:
        raise ValueError(f"{granularity} not one of 'sentence' or 'paragraph'.")
    out_dir = os.path.join(out_dir, granularity)

    # select just the rows relevant to the source corpus
    subset = df[df["source_corpus"] == source_corpus]
    
    for _, row in subset.iterrows():
        in_fname = row['original_id'] + fname_ext
        out_fname = row['shame_id'] + '.txt'

        extract_xml_to_text(
            in_dir,
            in_fname,
            out_dir,
            out_fname,
            granularity=granularity,
            content_type=content_type
        )


##########################################################################
# Helper Functions for Clarin corpora (KDSP, PriLit) TXT extraction --> .txt #
##########################################################################


def copy_file(src_path: str, dest_path: str) -> None:
    """
    Copy a file from src_path to dest_path, creating directories if needed.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy(src_path, dest_path)


def copy_clarin_corpus_file(original_id: str, 
                            shame_id: str, 
                            input_dir: str, 
                            output_dir: str) -> None:
    """
    Copy a clarin corpus .txt file named '{original_id}.txt' to '{shame_id}.txt'.
    """
    if not original_id.startswith("KDSP"): # for PriLit
        original_id = original_id + ".orig"
    src = os.path.join(input_dir, f"{original_id}.txt")
    dst = os.path.join(output_dir, f"{shame_id}.txt")
    copy_file(src, dst)


def extract_clarin_txt_paragraphs(df: pd.DataFrame,
                                  raw_corpus_dir: str, 
                                  output_dir: str,
                                  df_source_corpus_val: str = 'KDSP') -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # KDSP/PriLit text location
    input_dir = Path(raw_corpus_dir)
    input_rows = df[df['source_corpus'] == df_source_corpus_val]

    # extract paragraphs and sentences
    for _, row in input_rows.iterrows():
        if pd.notna(row['original_id']):
            copy_clarin_corpus_file(
                original_id=row['original_id'],
                shame_id=row['shame_id'],
                input_dir=input_dir,
                output_dir=output_dir
            )
