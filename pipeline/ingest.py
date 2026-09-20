# pipeline/ingest.py
# Data Ingestion Pipeline — based on brain.py's store_text().
#
# Supports:
#   - Plain text  → ingest_text()
#   - PDF files   → ingest_pdf()
#
# Pipeline flow (both inputs):
#   Parse → Chunk → Embed → Store
#
# What is identical to brain.py:
#   - SentenceTransformer("all-MiniLM-L6-v2")
#   - embedder.encode(chunks).tolist()
#   - collection.add(documents, embeddings, metadatas, ids)
#
# What is new vs brain.py:
#   - PDF support via PyMuPDF (fitz)
#   - Proper chunking with LangChain RecursiveCharacterTextSplitter
#   - category metadata field
#   - source_name stored (filename for PDFs)
#   - per-user isolation via user_id

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import fitz          # PyMuPDF — pip install pymupdf
import uuid, re

from database.chroma_client import get_collection
from config import EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

# ── Load embedding model once at startup ──────────────────────────────────────
# Loading is slow (~2s). Doing it at module level means it loads once,
# not on every store action.
print("[getsetgo] Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)
print("[getsetgo] Embedding model ready.")

# ── Text splitter ─────────────────────────────────────────────────────────────
# Splits at natural boundaries (paragraph → sentence → word → character)
# chunk_overlap prevents meaning loss at boundaries
_splitter = RecursiveCharacterTextSplitter(
    chunk_size    = CHUNK_SIZE,
    chunk_overlap = CHUNK_OVERLAP,
    separators    = ["\n\n", "\n", ". ", " ", ""]
)


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED CORE — called by both ingest_text() and ingest_pdf()
# ══════════════════════════════════════════════════════════════════════════════

def _run_pipeline(chunks: list,
                  source_name: str,
                  category: str,
                  user_id: str,
                  doc_type: str = "text") -> dict:
    """
    Core pipeline: Embed → Store.
    Parse and Chunk happen before this is called.

    Args:
        chunks      : list of text strings (already chunked)
        source_name : "text_input" for typed text, filename for PDFs
        category    : user-chosen label (movies, books, etc.)
        user_id     : routes to the user's private ChromaDB folder
        doc_type    : "text" or "pdf" — used by the UI's scope filter
    """
    if not chunks:
        return {"success": False, "error": "No content to store after processing."}

    # ── Embed ──────────────────────────────────────────────────────────────────
    # Same as brain.py: embedder.encode(chunks).tolist()
    # .tolist() converts numpy array → plain Python list (ChromaDB needs this)
    vectors = embedder.encode(chunks, show_progress_bar=False).tolist()

    # ── Build metadata per chunk ───────────────────────────────────────────────
    now = datetime.now()
    metadatas = [
        {
            "timestamp"    : now.isoformat(),
            "date"         : now.strftime("%Y-%m-%d"),
            "time"         : now.strftime("%H:%M"),
            "source"       : "user_input",
            "source_name"  : source_name,
            "category"     : category,
            "type"         : doc_type,
            "chunk_index"  : i,
            "total_chunks" : len(chunks),
            "original_text": chunks[i][:200]
        }
        for i in range(len(chunks))
    ]

    # ── Unique IDs — uuid4 guarantees no collisions ───────────────────────────
    doc_ids = [str(uuid.uuid4()) for _ in chunks]

    # ── Store in this user's ChromaDB folder ──────────────────────────────────
    collection = get_collection(user_id)
    collection.add(
        documents  = chunks,
        embeddings = vectors,
        metadatas  = metadatas,
        ids        = doc_ids
    )

    return {
        "success"     : True,
        "chunks"      : len(chunks),
        "source_name" : source_name,
        "category"    : category,
        "timestamp"   : now.isoformat()
    }


# ══════════════════════════════════════════════════════════════════════════════
#  INGEST TEXT — your original store_text() from brain.py
# ══════════════════════════════════════════════════════════════════════════════

def ingest_text(text: str,
                category: str = "general",
                user_id: str = "default") -> str:
    """
    Store plain text in the vector database.
    Direct evolution of brain.py's store_text().

    Returns a status string (same style as brain.py).
    """
    try:
        # ── Parse: clean the raw input ─────────────────────────────────
        cleaned = text.strip()
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # max 2 blank lines
        cleaned = re.sub(r' {2,}', ' ', cleaned)       # collapse spaces

        if not cleaned:
            return "ERROR: Empty input — nothing to store."

        # ── Chunk: split into overlapping pieces ───────────────────────
        chunks = [c.strip() for c in _splitter.split_text(cleaned) if c.strip()]

        result = _run_pipeline(chunks, "text_input", category, user_id, doc_type="text")

        if result["success"]:
            return f"OK: Stored! {result['chunks']} chunk(s) saved at {result['timestamp']}"
        return f"ERROR: {result['error']}"

    except Exception as e:
        return f"ERROR: {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
#  INGEST PDF — new capability
# ══════════════════════════════════════════════════════════════════════════════

def ingest_pdf(pdf_bytes: bytes,
               filename: str,
               category: str = "document",
               user_id: str = "default") -> str:
    """
    Extract text from a PDF and store it in the vector database.
    Uses PyMuPDF (fitz) for text extraction.

    After extraction, runs the same chunking → embedding → storage
    pipeline as ingest_text() — so retrieval works identically.

    Args:
        pdf_bytes : raw bytes from Streamlit's file_uploader (.read())
        filename  : original filename (stored in metadata for citations)
        category  : user-chosen label
        user_id   : routes to the user's private ChromaDB folder
    """
    try:
        # ── Parse PDF: extract text page by page ───────────────────────
        # fitz.open(stream=...) reads from bytes — no temp file needed
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(pdf_doc)

        all_text_parts = []
        for page_num in range(total_pages):
            page      = pdf_doc[page_num]
            page_text = page.get_text("text").strip()

            if page_text:
                # Tag each page so citations can reference "page 3 of report.pdf"
                all_text_parts.append(f"[Page {page_num + 1} of {filename}]\n{page_text}")

        pdf_doc.close()

        if not all_text_parts:
            return ("ERROR: No extractable text found in this PDF. "
                    "It may be a scanned image — OCR support coming soon.")

        # ── Combine all pages → chunk → embed → store ──────────────────
        full_text = "\n\n".join(all_text_parts)
        chunks    = [c.strip() for c in _splitter.split_text(full_text) if c.strip()]

        result = _run_pipeline(chunks, filename, category, user_id, doc_type="pdf")

        if result["success"]:
            return (f"OK: PDF stored! '{filename}' — "
                    f"{total_pages} page(s) → {result['chunks']} chunk(s) "
                    f"saved at {result['timestamp']}")
        return f"ERROR: {result['error']}"

    except Exception as e:
        return f"ERROR: {str(e)}"


def count_memories(user_id: str = "default") -> int:
    """Same as brain.py's count_memories() — per user."""
    from database.chroma_client import get_memory_count
    return get_memory_count(user_id)
