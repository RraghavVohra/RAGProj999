"""
Reranking module for our modular RAG pipeline (using Cohere Rerank).

Why rerank?
  The retriever compares a question vector with chunk vectors that were
  made SEPARATELY. That's fast, but rough.
  A reranker reads the question and each chunk TOGETHER and scores how well
  the chunk answers the question. Much more accurate, but slower - so we
  only run it on a small candidate list.

    retriever: top 10 (fast, rough)  -->  reranker: best 3 (slow, accurate)  -->  LLM
"""

import os

import cohere
from dotenv import load_dotenv

from utils.chunking import Chunk
from utils.embedding import OpenAIEmbedder
from utils.retrieval import Retriever
from utils.vector_store import VectorStore

load_dotenv()   # reads COHERE_API_KEY from .env


class CohereReranker:
    def __init__(self, model_name: str = "rerank-v3.5", top_n: int = 3):
        self.model_name = model_name
        self.top_n = top_n
        self.client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

    def rerank(self, question: str, chunks: list[Chunk]) -> list[Chunk]:
        """Return the `top_n` most relevant chunks, best first, with a `rerank_score` in metadata."""
        response = self.client.rerank(
            model=self.model_name,
            query=question,
            documents=[chunk.text for chunk in chunks],
            top_n=self.top_n,
        )

        # Cohere returns the position (index) of each chosen chunk + its relevance score (0 to 1)
        reranked = []
        for result in response.results:
            original = chunks[result.index]
            reranked.append(Chunk(
                text=original.text,
                metadata={**original.metadata, "rerank_score": round(result.relevance_score, 4)},
            ))
        return reranked


# ---------------------------------------------------------------------------
# Quick manual check (run from project root:  python -m utils.reranking)
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    question = "What are AI leaders doing differently from other companies?"

    # 1. Retrieve a WIDE candidate list (top 10) -----------------------------
    embedder = OpenAIEmbedder()
    retriever = Retriever(embedder, VectorStore(f"bain_{embedder.model_name}"), top_k=10)
    candidates = retriever.retrieve(question)

    print("=" * 60)
    print(f"Q: {question}")
    print("=" * 60)
    print("\nBEFORE reranking (vector search order, top 10):")
    for rank, chunk in enumerate(candidates, start=1):
        print(f"  #{rank:<2} page {chunk.metadata['page']:<3} vector score {chunk.metadata['score']}")

    # 2. Rerank down to the best 3 -------------------------------------------
    reranker = CohereReranker(top_n=10)
    best = reranker.rerank(question, candidates)

    print(f"\nAFTER reranking ({reranker.model_name}, top {reranker.top_n}):")
    for rank, chunk in enumerate(best, start=1):
        old_rank = next(i for i, c in enumerate(candidates, start=1) if c.metadata["page"] == chunk.metadata["page"])
        preview = chunk.text[:80].replace("\n", " ")
        print(f"  #{rank}  page {chunk.metadata['page']:<3} rerank score {chunk.metadata['rerank_score']:<7} (was #{old_rank})")
        print(f"      {preview}...")
