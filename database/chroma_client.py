# database/chroma_client.py
# Shared ChromaDB connection — per-user isolation built in.
#
# Every user gets their own folder: ./db/{user_id}/
# user_id="default" is used when auth is not yet set up.
# When Supabase auth is added, pass the real user UUID.

import chromadb
from chromadb.config import Settings
from config import DB_BASE_PATH, COLLECTION_NAME
import os


def get_collection(user_id: str = "default"):
    """
    Returns the ChromaDB collection for a specific user.
    Creates the folder and collection if they don't exist.

    Args:
        user_id : "default" for single-user mode,
                  or a UUID string from Supabase auth for multi-user mode.
    """
    db_path = os.path.join(DB_BASE_PATH, user_id)
    os.makedirs(db_path, exist_ok=True)

    client = chromadb.PersistentClient(
        path     = db_path,
        settings = Settings(anonymized_telemetry=False)
    )

    # hnsw:space=cosine: measures angle between vectors (better for text than euclidean)
    return client.get_or_create_collection(
        name     = COLLECTION_NAME,
        metadata = {"hnsw:space": "cosine"}
    )


def get_memory_count(user_id: str = "default") -> int:
    """Returns how many chunks are stored for this user."""
    try:
        return get_collection(user_id).count()
    except Exception:
        return 0
