from sentence_transformers import SentenceTransformer
from typing import List
from utils.config import EMBEDDING_MODEL

_model = None


def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: List[str]):
    model = get_embedding_model()
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
