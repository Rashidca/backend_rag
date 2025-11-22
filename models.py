# models.py
import os
import time
from typing import List, Tuple

import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ========= Embedding Model (unchanged) =========
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_embed_model = None

def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(
            EMBED_MODEL_NAME,
            device="cpu"  # change to "cuda" if you have GPU
        )
    return _embed_model


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_embed_model()
    emb = model.encode(texts, normalize_embeddings=True)
    return emb.tolist()


# ========= Local LLM (Ollama) =========
LLM_MODEL_NAME = "qwen2.5:3b-instruct"
  # or "mistral" if that's what you downloaded

import json

def call_llm(system_prompt: str, user_prompt: str) -> Tuple[str, float]:
    """
    Calls a local Ollama model via HTTP (streamed JSON chunks from Ollama).
    Returns (answer, generation_latency_seconds).
    """
    print("🔍 USING MODEL:", LLM_MODEL_NAME)

    final_prompt = system_prompt + "\n\n" + user_prompt

    start = time.time()
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": LLM_MODEL_NAME, "prompt": final_prompt},
        stream=True
    )
    latency = time.time() - start

    answer = ""
    for line in response.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line.decode("utf-8"))
            token = data.get("response", "")
            answer += token
            if data.get("done", False):
                break
        except Exception:
            # Skip malformed chunk
            continue

    return answer.strip(), latency
