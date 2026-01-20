from typing import List
from utils.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[dict]:
    """Return list of chunks with metadata (start, end, text)"""
    tokens = text.split()  # simple whitespace split — replace with tokenizer for better results
    chunks = []
    i = 0
    doc_len = len(tokens)
    while i < doc_len:
        start = i
        end = min(i + chunk_size, doc_len)
        chunk_tokens = tokens[start:end]
        chunk_text = " ".join(chunk_tokens)
        chunks.append({"start_token": start, "end_token": end, "text": chunk_text})
        if end == doc_len:
            break
        i += chunk_size - overlap
    return chunks
