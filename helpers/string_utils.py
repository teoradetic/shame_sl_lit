import re

def sanitize_filename(author_title: str) -> str:
    """Turn “Author Title” into a filesystem-safe filename."""
    name = author_title.replace(" ", "_").replace("/", "-")
    return re.sub(r"[^\w\-_\.]", "", name) 


def split_paragraphs_to_sentences(paragraphs: list[str]
                                 ) -> list[str]:
    """
    Split paragraphs into sentences, preserving punctuation and
    avoiding splits inside any of multiple quote types.
    """
    sentences: list[str] = []
    # Define opening to closing quote pairs
    quote_pairs = {
        '>>': '<<',
        '"': '"',
        "'": "'",
        '`': '`',
        '„': '“',
        '«': '»', 
        '»': '«', 
    }
    # Sort markers by length to match multi-char first
    openers = sorted(quote_pairs.keys(), key=len, reverse=True)

    for para in paragraphs:
        buf = []
        stack: list[str] = []
        i = 0
        length = len(para)
        while i < length:
            # Check for opening/closing markers
            matched = False
            for om in openers:
                if para.startswith(om, i):
                    cm = quote_pairs[om]
                    if stack and stack[-1] == cm:
                        stack.pop()
                    else:
                        stack.append(cm)
                    buf.append(om)
                    i += len(om)
                    matched = True
                    break
            if matched:
                continue

            ch = para[i]
            buf.append(ch)
            # Potential sentence boundary
            if not stack and ch in '.!?':
                # Look ahead over whitespace
                j = i + 1
                while j < length and para[j].isspace():
                    j += 1
                # If next char is uppercase, digit, or quote opener, split
                if j < length and (
                    para[j].isupper() or
                    para[j].isdigit() or
                    any(para.startswith(o, j) for o in openers)
                ):
                    sent = ''.join(buf).strip()
                    if sent:
                        sentences.append(sent)
                    buf = []
                    i = j - 1
            i += 1
        # Remaining buffer
        tail = ''.join(buf).strip()
        if tail:
            sentences.append(tail)
    return sentences