# config.py
# Single source of truth for all settings.
# Change model names, chunk sizes, paths here — updates everywhere.

from dotenv import load_dotenv
import os

load_dotenv()

# ── LLM ───────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "openai/gpt-oss-120b"


# ── Embedding model (runs locally — no API key, no cost) ──────────────────────
EMBED_MODEL  = "all-MiniLM-L6-v2"

# ── ChromaDB ──────────────────────────────────────────────────────────────────
DB_BASE_PATH    = "./db"
COLLECTION_NAME = "memories"

# ── Chunking ──────────────────────────────────────────────────────────────────
# chunk_size    : max characters per chunk
# chunk_overlap : characters shared between adjacent chunks (prevents boundary loss)
CHUNK_SIZE    = 300
CHUNK_OVERLAP = 50
