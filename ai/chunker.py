from typing import List, Dict


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """Split text into overlapping chunks with simple sentence-boundary awareness.

    Returns list of dicts: {"id": int, "text": str}
    """
    if not text:
        return []
    # naive split by whitespace into tokens
    tokens = text.split()
    chunks = []
    i = 0
    cid = 0
    while i < len(tokens):
        end = min(i + chunk_size, len(tokens))
        chunk_tokens = tokens[i:end]
        chunk_text = " ".join(chunk_tokens)
        chunks.append({"id": cid, "text": chunk_text})
        cid += 1
        # move pointer with overlap
        i = end - overlap if end - overlap > i else end
    return chunks


def chunk_text_by_sentences(text: str, chunk_size_chars: int = 1000, overlap_chars: int = 200):
    """Alternative chunking using character counts and sentence boundaries."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    cur = ""
    cid = 0
    for s in sentences:
        if len(cur) + len(s) <= chunk_size_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append({"id": cid, "text": cur})
                cid += 1
            # start new
            cur = s
    if cur:
        chunks.append({"id": cid, "text": cur})
    # apply small overlap by merging last chars
    if overlap_chars > 0 and len(chunks) > 1:
        out = []
        for i, c in enumerate(chunks):
            if i == 0:
                out.append(c)
            else:
                prev = out[-1]
                # add overlap from current to prev
                overlap_text = c['text'][:overlap_chars]
                prev['text'] = (prev['text'] + ' ' + overlap_text).strip()
                out[-1] = prev
                out.append(c)
        chunks = out
    return chunks
