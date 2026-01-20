# backend/context_compressor.py

def simple_compress(chunks, max_chars=15000):
    """
    Compress long contexts into a manageable size before sending to the LLM.
    Handles both dict-based and plain-text chunks.
    """
    if not chunks:
        return ""

    # Extract text from chunks whether they are strings or dicts
    texts = []
    for c in chunks:
        if isinstance(c, dict):
            text = c.get("text") or c.get("page_content") or ""
        else:
            text = str(c)
        texts.append(text.strip())

    # Join all text with spacing
    joined = "\n\n".join(texts)

    # Trim to max length
    if len(joined) > max_chars:
        joined = joined[:max_chars] + "..."
    return joined
