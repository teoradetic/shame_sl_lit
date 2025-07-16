import os
import pandas as pd
import requests
import zipfile
from pathlib import Path
from helpers.string_utils import sanitize_filename


def fetch_corpus_metadata(
    sheet_id: str,
    sheet_name: str,
    force_reimport: bool = False,
    csv_path: str = "corpus_metadata.csv"
) -> pd.DataFrame:
    """
    Fetch and cache corpus metadata from Google Sheets.

    If `csv_path` exists and `force_reimport` is False, load from that CSV.
    Otherwise fetch from Google Sheets, save to CSV, and return.

    Parameters:
    - sheet_id (str): ID of the Google Sheet.
    - sheet_name (str): Name of the sheet/tab.
    - force_reimport (bool): If True, always re-fetch and overwrite CSV.
    - csv_path (str): Path to local CSV cache.

    Returns:
    - pd.DataFrame: The corpus metadata.
    """
    # Load from cache if available and not forced
    if os.path.exists(csv_path) and not force_reimport:
        try:
            return pd.read_csv(csv_path, encoding="utf-8")
        except Exception as e:
            raise RuntimeError(f"Failed to read cached CSV '{csv_path}': {e}")

    # Otherwise fetch from Google Sheets
    export_url = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?"
        f"sheet={sheet_name}&tq=select*&tqx=out:csv"
    )
    try:
        df = pd.read_csv(export_url, encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to load sheet from URL '{export_url}': {e}")

    # Cache to local CSV
    try:
        df.to_csv(csv_path, index=False, encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to write CSV cache '{csv_path}': {e}")

    return df


def fetch_sl_stopwords(path='data/stopwords_sl.txt', verbose=False):
    # Make sure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not os.path.exists(path):
        if verbose: print(f"⚠️  Stopwords file not found at '{path}'. Downloading...")
        url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-sl/master/stopwords-sl.txt"
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            if verbose: print(f"✅ Downloaded and saved stopwords to '{path}'")
        except requests.RequestException as e:
            if verbose: print("❌ Failed to download stopwords list:", e)
            return set()

    with open(path, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())


def fetch_html(url: str, timeout: int = 10) -> str:
    """
    Download the page at `url`, detect its declared encoding (from HTTP headers or meta),
    decode the raw bytes accordingly (falling back to requests’ apparent_encoding or UTF-8),
    and return the HTML text.
    """
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    # 1) Check HTTP Content-Type header for charset
    content_type = resp.headers.get("Content-Type", "")
    if "charset=" in content_type:
        charset = content_type.split("charset=")[-1].split(";")[0].strip()
        html = resp.content.decode(charset, errors="replace")
    else:
        # 2) Fallback: use requests’ own guess at encoding (or default to utf-8)
        enc = resp.apparent_encoding or "utf-8"
        html = resp.content.decode(enc, errors="replace")

    return html


def download_file(
    url: str,
    dest_folder: str,
    skip_if_exists: bool = False,
    verbose: bool = False
    ) -> None:
    """Download a file from `url` into `dest_folder`."""
    os.makedirs(dest_folder, exist_ok=True)
    local_path = os.path.join(dest_folder, os.path.basename(url))
    if os.path.exists(local_path) and skip_if_exists:
        if verbose:
            print(f"[SKIP] {local_path} already exists")
        return
    if verbose:
        print(f"[DL] {url} → {local_path}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)


def fetch_clarin_archives(
    handle_id: str,
    filenames: list[str],
    dest_dir: Path,
    verbose: bool = False
    ) -> None:
    """
    Download and unzip CLARIN ZIPs from a handle into `dest_dir`.
    """
    base_url = f"https://www.clarin.si/repository/xmlui/bitstream/handle/{handle_id}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Downloading CLARIN archives to {dest_dir}")
    for fname in filenames:
        url = f"{base_url}/{fname}"
        download_file(url, str(dest_dir), skip_if_exists=False, verbose=verbose)
        zip_path = dest_dir / fname
        if zip_path.suffix.lower() == ".zip":
            if verbose:
                print(f"[UNZIP] {zip_path.name}")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(dest_dir)
            if verbose:
                print(f"[DEL]   {zip_path.name}")
            zip_path.unlink()


def fetch_eltec_xmls(
    repo_owner: str,
    repo_name: str,
    tag: str,
    path: str,
    dest_dir: str,
    verbose: bool = False
    ) -> None:
    """
    List and download all .xml files under `path` at `tag` from GitHub repo.
    """
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{path}"
    params = {"ref": tag}
    resp = requests.get(api_url, params=params)
    resp.raise_for_status()
    items = resp.json()
    os.makedirs(dest_dir, exist_ok=True)
    if verbose:
        print(f"Downloading ELTeC XMLs to {dest_dir}")
    for item in items:
        if item.get("type") == "file" and item["name"].endswith(".xml"):
            download_file(item["download_url"], dest_dir, verbose=verbose)


def fetch_raw_corpora(
    df: pd.DataFrame,
    raw_dir: str = "data/raw_corpora",
    verbose: bool = False
    ) -> dict[str, list[str]]:
    """
    Fetch raw corpus files/fragments into `raw_dir`.

    The @df is the corpus metadata DataFrame.

    Returns a dict mapping corpus keys to lists of file paths.
    """
    os.makedirs(raw_dir, exist_ok=True)
    results: dict[str, list[str]] = {}

    # === Wikivir (HTML) ===
    wikivir_dir = os.path.join(raw_dir, "wikivir")
    os.makedirs(wikivir_dir, exist_ok=True)
    wikivir_paths: list[str] = []
    subset = df[df.get("source_corpus") == "Wikivir - leposlovje"]
    for _, row in subset.iterrows():
        url = row.get("doc_link", "").strip()
        fragment = fetch_html(url)
        fname = sanitize_filename(row.get("author_and_title", "")) + ".html"
        path = os.path.join(wikivir_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(fragment)
        wikivir_paths.append(path)
        if verbose:
            print(f"[RAW HTML] {path}")
    results["wikivir"] = wikivir_paths

    # === CLARIN archives ===
    clarin_corpora = {
        "PriLit": (
            "11356/1319",
            ["PriLit.TEI.zip", "PriLit.ana.zip", "PriLit.txt.zip", "PriLit.vert.zip"],
        ),
        "KDSP": (
            "11356/1823",
            ["KDSP.TEI.zip", "KDSP.TEI.ana.zip", "KDSP.txt.zip", "KDSP.vert.zip"],
        ),
    }
    clarin_results: list[str] = []
    for corpus_name, (handle, files) in clarin_corpora.items():
        dest = Path(raw_dir) / corpus_name.lower()
        fetch_clarin_archives(handle, files, dest, verbose=verbose)
        # collect extracted files
        for p in dest.rglob("*.*"):
            clarin_results.append(str(p))
    results["clarin"] = clarin_results

    # === ELTeC-slv XMLs ===
    # L1 is basic xml, one paragraph per line in xml
    eltec_dir_1 = os.path.join(raw_dir, "eltec", "l1")
    fetch_eltec_xmls(
        repo_owner="COST-ELTeC",
        repo_name="ELTeC-slv",
        tag="v2.0.0",
        path="level1",
        dest_dir=eltec_dir_1,
        verbose=verbose
    )
    eltec_paths = [
        os.path.join(eltec_dir_1, f)
        for f in os.listdir(eltec_dir_1)
        if f.endswith(".xml")
    ]
    results["eltec_l1"] = eltec_paths

    # L2 is processed xml, with lemmas and modernization
    eltec_dir_2 = os.path.join(raw_dir, "eltec", "l2")
    fetch_eltec_xmls(
        repo_owner="COST-ELTeC",
        repo_name="ELTeC-slv",
        tag="v2.0.0",
        path="level2",
        dest_dir=eltec_dir_2,
        verbose=verbose
    )
    eltec_paths = [
        os.path.join(eltec_dir_2, f)
        for f in os.listdir(eltec_dir_2)
        if f.endswith(".xml")
    ]
    results["eltec_l2"] = eltec_paths

    return results