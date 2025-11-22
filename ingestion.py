import os
import re
import time
from typing import List, Dict

import pdfplumber
import chromadb
from chromadb.config import Settings

from models import embed_texts

DB_DIR = "db"
PDF_PATH = os.path.join("data", "manual.pdf")
COLLECTION_NAME = "manual_chunks"


# ---------- Chapter Detection Helpers ----------

def is_table_of_contents_page(text: str, page_num: int) -> bool:
    """
    Detect TOC pages (many dotted lines and page number references).
    Only applies to first 10 pages to ensure no real content is skipped later.
    """
    if page_num > 10:  # TOC always early in manuals
        return False

    dotted_lines = sum(
        1 for line in text.splitlines()
        if ("..." in line) or re.search(r"\s\d{1,3}$", line)
    )
    return dotted_lines >= 5


def is_probable_chapter_title(line: str) -> bool:
    """
    Detect REAL chapter/section titles, not TOC lines or labels.
    Works for Nikon camera manuals.
    """
    line = line.strip()
    if not line:
        return False

    # Example: "Guide Mode 44", "Special Effects 52", "Technical Notes 89"
    if re.match(r"^[A-Za-z][A-Za-z\s]+ \d{1,3}$", line):
        return True

    # MENU sections (with or without page numbers)
    menu_keywords = ["menu", "mode", "playback", "shooting", "movie", "setup", "guide"]
    if any(kw.lower() in line.lower() for kw in menu_keywords) and len(line) < 45:
        return True

    return False


# ---------- PDF helpers ----------

def load_pdf_pages(path: str) -> List[Dict]:
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({"page_number": i + 1, "text": text})
    return pages


def split_into_chunks_with_chapters(
    pages: List[Dict],
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> List[Dict]:

    chunks = []
    current_chapter = "Unknown"
    buffer = ""
    emb_id = 0

    for p in pages:
        page_num = p["page_number"]
        text = p["text"]

        # ---------- Skip Table of Contents ----------
        if is_table_of_contents_page(text, page_num):
            continue

        # ---------- Chunk assembly ----------
        lines = text.splitlines()
        for line in lines:

            if is_probable_chapter_title(line):
                # Clean chapter title: remove trailing page number if present
                chapter = re.sub(r"\s\d{1,3}$", "", line.strip())
                current_chapter = chapter
                continue

            clean_line = line.strip()
            if not clean_line:
                continue

            buffer = f"{buffer} {clean_line}".strip()

            if len(buffer.split()) >= chunk_size:
                chunks.append(
                    {
                        "id": f"chunk_{emb_id}",
                        "text": buffer,
                        "chapter": current_chapter,
                        "page": page_num,
                    }
                )
                emb_id += 1
                # overlapping words
                buffer = " ".join(buffer.split()[-chunk_overlap:])

        # flush per page
        if buffer:
            chunks.append(
                {
                    "id": f"chunk_{emb_id}",
                    "text": buffer,
                    "chapter": current_chapter,
                    "page": page_num,
                }
            )
            emb_id += 1
            buffer = ""

    return chunks


# ---------- Main Ingestion Runner ----------

def main():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(
            f"PDF not found at {PDF_PATH}. Put your manual as 'data/manual.pdf'."
        )

    os.makedirs(DB_DIR, exist_ok=True)

    print("[INGEST] Loading PDF...")
    pages = load_pdf_pages(PDF_PATH)
    print(f"[INGEST] Loaded {len(pages)} pages")

    print("[INGEST] Splitting into chapter-aware, TOC-filtered chunks...")
    chunks = split_into_chunks_with_chapters(pages)
    print(f"[INGEST] Created {len(chunks)} chunks")

    chroma_client = chromadb.Client(
        Settings(persist_directory=DB_DIR, is_persistent=True)
    )

    # Recreate collection
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print("[INGEST] Embedding and storing chunks...")
    start = time.time()

    batch_size = 64
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metadatas = [
            {"source": "manual.pdf", "chapter": c["chapter"], "page": c["page"]}
            for c in batch
        ]
        embeddings = embed_texts(texts)
        collection.add(
            documents=texts,
            ids=ids,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    print(f"[INGEST] Finish. Stored {len(chunks)} chunks in {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
