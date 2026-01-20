import os
import pdfplumber
import docx
from typing import List, Dict


def extract_text_from_pdf(path: str) -> str:
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text() or "")
    return "\n".join(texts)


def extract_text_from_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs])


def load_documents(folder_or_files) -> List[Dict]:
    """Return list of dicts: {"id":.., "text":.., "source":.., "meta":{}}"""
    docs = []
    if isinstance(folder_or_files, str) and os.path.isdir(folder_or_files):
        files = [os.path.join(folder_or_files, f) for f in os.listdir(folder_or_files)]
    elif isinstance(folder_or_files, list):
        files = folder_or_files
    else:
        files = [folder_or_files]

    for f in files:
        if not os.path.isfile(f):
            continue
        ext = os.path.splitext(f)[1].lower()
        text = ""
        if ext == ".pdf":
            text = extract_text_from_pdf(f)
        elif ext in [".docx", ".doc"]:
            text = extract_text_from_docx(f)
        elif ext in [".txt"]:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        else:
            # skip unknown types
            continue
        docs.append({
            "id": os.path.basename(f),
            "text": text,
            "source": f,
            "meta": {"filename": os.path.basename(f)}
        })
    return docs
