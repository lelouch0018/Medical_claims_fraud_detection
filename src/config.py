from pathlib import Path
import os

BASE_DIR = Path(os.environ.get("FRAUD_BASE_DIR", os.path.dirname(__file__) + "/.."))

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DOCS_RAW_DIR = RAW_DIR / "docs"
DOCS_EXTRACTED_DIR = BASE_DIR / "data" / "docs"
MODELS_DIR = BASE_DIR / "models"

# Embeddings and vectorstore files
EMBED_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = MODELS_DIR / "faiss_index.idx"
EMBEDDINGS_NPY = MODELS_DIR / "embeddings.npy"
DOCS_CHUNKS_JSON = MODELS_DIR / "docs_metadata.json"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # Optional
