import time
from difflib import SequenceMatcher
from typing import Dict, Any, List, Optional

import chromadb
from chromadb.config import Settings

from models import embed_texts

DB_DIR = "db"
COLLECTION_NAME = "manual_chunks"

chroma_client = chromadb.Client(
    Settings(persist_directory=DB_DIR, is_persistent=True)
)
collection = chroma_client.get_collection(name=COLLECTION_NAME)


def fuzzy_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def detect_chapter_from_query(query: str) -> Optional[str]:
    res = collection.get(include=["metadatas"])
    chapters = {m["chapter"] for m in res["metadatas"] if "chapter" in m}

    q = query.lower()

    # 🔥 Exact / substring match FIRST
    for chapter in chapters:
        if chapter.lower() in q or q in chapter.lower():
            return chapter

    # 🔥 Then fallback to fuzzy match
    best_match = None
    best_score = 0
    for chapter in chapters:
        score = fuzzy_sim(chapter, q)
        if score > 0.55 and score > best_score:
            best_match = chapter
            best_score = score

    return best_match


def retrieve(query: str, k: int = 5) -> Dict[str, Any]:
    """
    Returns:
    {
      "results": [{"text": ..., "metadata": {...}}, ...],
      "latency": float,
      "used_filter": Optional[str]
    }
    """
    query_emb = embed_texts([query])[0]
    chapter = detect_chapter_from_query(query)

    where_filter = {"chapter": chapter} if chapter else None

    start = time.time()
    res = collection.query(
        query_embeddings=[query_emb],
        n_results=k,
        where=where_filter,
    )
    latency = time.time() - start

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    results: List[Dict[str, Any]] = []
    for d, m in zip(docs, metas):
        results.append({"text": d, "metadata": m})

    return {
    "results": results,
    "latency": latency,
    "used_filter": {"chapter": chapter} if chapter else None
}

