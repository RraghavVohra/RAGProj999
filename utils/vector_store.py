"""
Vector store module for our modular RAG pipeline (backed by ChromaDB).

A vector DB stores, for every chunk:
    id        -> unique name of the chunk
    embedding -> the vector (list of numbers) from our embedder
    document  -> the chunk text itself
    metadata  -> page number, source file, chunking strategy, ...

We compute embeddings ourselves (utils/embedding.py) and only use Chroma
for storing + searching. That keeps the embedder swappable.
"""

from pathlib import Path

import chromadb

from utils.chunking import Chunk
from utils.chunking import page_chunking
from utils.embedding import OpenAIEmbedder, SentenceTransformerEmbedder, embed_chunks

# Chroma saves everything inside this folder, so data survives restarts
DEFAULT_DB_PATH = Path(__file__).parent.parent / "vector_store"


class VectorStore:
    def __init__(self, collection_name: str, db_path: Path = DEFAULT_DB_PATH):
        """
        One collection = one embedding model.
        (MiniLM vectors have 384 numbers, OpenAI's have 1536 - they can't be mixed.)
        """
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            configuration={"hnsw": {"space": "cosine"}},   # compare vectors by cosine distance
            embedding_function=None,                       # we bring our own vectors
        )

    def add_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Save chunks + their vectors. Re-running with the same chunks won't create duplicates."""
        ids = [f"{c.metadata['source']}-{c.metadata['strategy']}-{c.metadata['chunk_id']}" for c in chunks]

        self.collection.upsert(          # upsert = insert new ids, overwrite existing ones
            ids=ids,
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )

    def count(self) -> int:
        return self.collection.count()

    def search(self, query_vector: list[float], top_k: int = 3) -> list[Chunk]:
        """
        Find the `top_k` chunks whose vectors are closest to `query_vector`.
        Returns plain Chunk objects (not Chroma's raw format), with a
        similarity `score` added to the metadata: 1.0 = identical meaning, 0 = unrelated.
        """
        results = self.collection.query(query_embeddings=[query_vector], n_results=top_k)

        # Chroma returns lists-of-lists (one inner list per query); we sent one query -> take [0]
        chunks = []
        for text, metadata, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            chunks.append(Chunk(
                text=text,
                metadata={**metadata, "score": round(1 - distance, 4)},   # cosine distance -> similarity
            ))
        return chunks


# ---------------------------------------------------------------------------
# Quick manual check (run from project root:  python -m utils.vector_store)
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    PROVIDER = "openai"   # "local" or "openai"

    # 1. Chunk ---------------------------------------------------------------
    pdf_path = Path(__file__).parent.parent / "assets" / "bain_report_technology_report_2025.pdf"
    chunks = page_chunking(str(pdf_path))
    print(f"Chunks created : {len(chunks)}")

    # 2. Embed ---------------------------------------------------------------
    embedder = SentenceTransformerEmbedder() if PROVIDER == "local" else OpenAIEmbedder()
    vectors = embed_chunks(chunks, embedder)
    print(f"Vectors created: {len(vectors)}  (each {len(vectors[0])} numbers, model = {embedder.model_name})")

    # 3. Store ---------------------------------------------------------------
    store = VectorStore(collection_name=f"bain_{embedder.model_name}")
    store.add_chunks(chunks, vectors)
    print(f"Stored in collection '{store.collection.name}': {store.count()} items")
    print(f"DB folder: {DEFAULT_DB_PATH}\n")

    # 4. Peek at what Chroma actually saved for one chunk ---------------------
    record = store.collection.get(ids=[f"{pdf_path.name}-page-wise-1"], include=["documents", "metadatas", "embeddings"])
    print("=" * 60)
    print("ONE STORED RECORD")
    print("=" * 60)
    print(f"id        : {record['ids'][0]}")
    print(f"metadata  : {record['metadatas'][0]}")
    print(f"document  : {record['documents'][0][:150]!r}...")
    print(f"embedding : {[round(x, 4) for x in record['embeddings'][0][:5]]} ... ({len(record['embeddings'][0])} numbers)")
