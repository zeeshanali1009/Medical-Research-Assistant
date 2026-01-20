from backend.loader import load_documents
from backend.chunker import chunk_text
from backend.embedder import embed_texts
from backend.vector_store import FaissStore
from backend.retriever import HybridRetriever
from backend.context_compressor import simple_compress
from backend.memory import load_memory, append_memory
from backend.llm_engine_groq import generate_answer_with_groq
from utils.config import CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_STORE_PATH, TOP_K
import numpy as np


class RAGPipeline:
    def __init__(self):
        self.chunks = []
        self.store = None

    def ingest(self, files):
        docs = load_documents(files)
        all_chunks = []
        metadatas = []
        for doc in docs:
            cks = chunk_text(doc['text'])
            for i, c in enumerate(cks):
                chunk_id = f"{doc['id']}_chunk_{i}"
                cmeta = {
                    "chunk_id": chunk_id,
                    "text": c['text'],
                    "source": doc['source'],
                    "meta": {**doc.get('meta', {}), "chunk_id": chunk_id}
                }
                all_chunks.append(cmeta)
                metadatas.append(cmeta)
        self.chunks = all_chunks

        if not all_chunks:
            return

        # embeddings
        texts = [c['text'] for c in all_chunks]
        embs = embed_texts(texts)
        if embs.ndim == 1:
            embs = np.expand_dims(embs, 0)

        dim = embs.shape[1]
        self.store = FaissStore(dim)
        self.store.add(embs, metadatas)

    def query(self, question: str, session_id: str = "default", top_k: int = TOP_K):
        if not self.store or not self.chunks:
            return {"answer": "[No documents ingested]", "sources": []}

        retriever = HybridRetriever(self.chunks)
        results = retriever.retrieve(question, top_k)

        # deterministic context compression
        compressed_context = simple_compress(results, max_chars=15000)

        # load chat history for this session
        history = load_memory(session_id)

        # call Groq with compressed context + history
        answer = generate_answer_with_groq(compressed_context, question, history=history)

        # append to memory
        append_memory(session_id, question, answer)

        # collect sources safely
        sources = []
        for r in results:
            if isinstance(r, dict):
                src = r.get("source", "Unknown source")
                content = r.get("content", r.get("text", ""))
            else:
                src = "Unknown source"
                content = str(r)
            sources.append({"source": src, "content": content})

        return {"answer": answer, "sources": sources, "history": history}
