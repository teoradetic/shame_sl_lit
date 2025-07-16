import os
import pandas as pd

def count_words_in_directory(directory_path: str) -> pd.DataFrame:
    """
    Counts the number of words in each .txt file within the specified directory and
    returns a pandas DataFrame with columns:
      - filename: the base name of the file without the .txt extension
      - num_words: the total number of words in the file

    Parameters:
    directory_path (str): Path to the directory containing .txt files.

    Returns:
    pd.DataFrame: DataFrame with one row per text file and columns ['filename', 'num_words'].

    Example:
    >>> df = count_words_in_directory('/path/to/texts')
    >>> print(df)
        filename  num_words
    0   document1        345
    1   notes            120
    2   summary          210
    """
    records = []

    # Iterate over files in the directory
    for entry in os.listdir(directory_path):
        if entry.lower().endswith('.txt'):
            filepath = os.path.join(directory_path, entry)

            # Read file contents
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()

            # Split on whitespace to count words
            words = text.split()
            num_words = len(words)

            # Strip the .txt extension
            base_name = os.path.splitext(entry)[0]

            # Store the result
            records.append({'filename': base_name, 'num_words': num_words})

    # Create a DataFrame from the records
    df = pd.DataFrame(records, columns=['filename', 'num_words'])
    return df