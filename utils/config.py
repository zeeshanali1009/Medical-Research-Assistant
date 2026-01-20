# Configuration for Advanced Medical RAG
# Replace placeholders with your values where applicable

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # lightweight, fast
VECTOR_STORE_PATH = "./data/faiss_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 8
USE_BM25 = True  # enable hybrid retrieval
BM25_K = 10

# LLM provider settings (we use Groq by default)
LLM_PROVIDER = "groq"  # options: groq, openai, local
# If you ever use OpenAI, keep placeholders here
OPENAI_API_KEY = ""
OPENAI_MODEL = "gpt-4o-mini"

# Local model options (if using local inference)
LOCAL_MODEL_PATH = "./models/gpt4all-model"  # example

# Streamlit app settings
STREAMLIT_PORT = 8501
