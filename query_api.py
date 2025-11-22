# query_api.py
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from retriever import retrieve
from models import call_llm

app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    max_chunks: int = 5


class ContextSnippet(BaseModel):
    text: str
    metadata: dict


class QueryResponse(BaseModel):
    answer: str
    retrieval_latency: float
    generation_latency: float
    used_filter: Optional[dict]
    context_snippets: List[ContextSnippet]


SYSTEM_PROMPT = """
You are a highly accurate assistant that answers questions ONLY using the provided manual context.
If the answer is not present in the context, you MUST reply:
"I don't know based on this manual."

Do not invent features or steps that are not clearly supported by the context.
Always stay grounded in the text chunks.
"""


def build_user_prompt(query: str, contexts: List[dict]) -> str:
    context_str = ""
    for i, c in enumerate(contexts):
        meta = c["metadata"]
        context_str += (
            f"[CHUNK {i+1} | Chapter: {meta.get('chapter')} | Page: {meta.get('page')}]\n"
            f"{c['text']}\n\n"
        )

    return (
        "CONTEXT:\n"
        + context_str
        + "\n\nUSER QUESTION:\n"
        + query
        + "\n\nAnswer strictly based on the context above."
    )


@app.post("/query", response_model=QueryResponse)
def query_manual(req: QueryRequest):
    # 1) Retrieval
    retrieval_result = retrieve(req.query, k=req.max_chunks)
    contexts = retrieval_result["results"]
    retrieval_latency = retrieval_result["latency"]
    used_filter = retrieval_result["used_filter"]

    print(
        f"[RAG] Retrieval latency: {retrieval_latency:.4f}s | filter={used_filter}"
    )

    if not contexts:
        answer = "I don't know based on this manual."
        generation_latency = 0.0
    else:
        # 2) LLM generation
        user_prompt = build_user_prompt(req.query, contexts)
        answer, generation_latency = call_llm(SYSTEM_PROMPT, user_prompt)

    print(f"[RAG] Generation latency: {generation_latency:.4f}s")

    snippets = [ContextSnippet(**c) for c in contexts]

    return QueryResponse(
        answer=answer,
        retrieval_latency=retrieval_latency,
        generation_latency=generation_latency,
        used_filter=used_filter,
        context_snippets=snippets,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("query_api:app", host="0.0.0.0", port=8000, reload=True)
