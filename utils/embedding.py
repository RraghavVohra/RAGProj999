"""
Embedding module for our modular RAG pipeline.

An embedding turns text into a list of numbers (a vector). Texts with
similar meaning get vectors that are close to each other, which is what
lets us search by meaning later.

Every embedder has the same method:   embed(texts) -> list of vectors
So the pipeline can swap a local model for OpenAI (or anything else)
without changing any other code.
"""

import os

from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI

from utils.chunking import Chunk, character_chunking, page_chunking

load_dotenv()   # reads OPENAI_API_KEY from the .env file in the project root


# ---------------------------------------------------------------------------
# Embedder 1: Local model (sentence-transformers) - free, runs on your laptop
# ---------------------------------------------------------------------------

class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer   # imported here so OpenAI-only users don't need it
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()


# ---------------------------------------------------------------------------
# Embedder 2: OpenAI API - needs OPENAI_API_KEY, costs a tiny amount per call
# ---------------------------------------------------------------------------

class OpenAIEmbedder:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# Helper: embed a list of Chunk objects (output of any chunking strategy)
# ---------------------------------------------------------------------------

def embed_chunks(chunks: list[Chunk], embedder) -> list[list[float]]:
    """Returns one vector per chunk, in the same order as `chunks`."""
    texts = [chunk.text for chunk in chunks]
    return embedder.embed(texts)


# ---------------------------------------------------------------------------
# Quick manual check (run from project root:  python -m utils.embedding)
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    PROVIDER = "openai"   # change to "openai" to use the OpenAI embedder

    # 1. Get chunks using BOTH chunking strategies -------------------------
    pdf_path = Path(__file__).parent.parent / "assets" / "bain_report_technology_report_2025.pdf"

    page_chunks = page_chunking(str(pdf_path))[:3]                         # first 3 pages only, to keep output small
    # char_chunks = character_chunking(page_chunks[1].text, chunk_size=500, overlap=50)

    print(f"Page-wise chunks     : {len(page_chunks)}")
    # print(f"Character-wise chunks: {len(char_chunks)}  (from page 2 text)\n")

    # 2. Pick the embedder --------------------------------------------------
    embedder = SentenceTransformerEmbedder() if PROVIDER == "local" else OpenAIEmbedder()
    print(f"Embedder: {type(embedder).__name__}  (model = {embedder.model_name})\n")

    # 3. Embed and inspect --------------------------------------------------
    for name, chunks in [("PAGE-WISE", page_chunks)]:
        vectors = embed_chunks(chunks, embedder)

        print("=" * 60)
        print(f"{name}: {len(vectors)} vectors, each with {len(vectors[0])} numbers")
        print("=" * 60)

        for chunk, vector in zip(chunks, vectors):
            preview = chunk.text[:60].replace("\n", " ")
            numbers = ", ".join(f"{x:.4f}" for x in vector[:5])
            print(f"chunk {chunk.metadata['chunk_id']}: '{preview}...'")
            print(f"   -> [{numbers}, ...]\n")
