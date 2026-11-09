"""
Retrieval module for our modular RAG pipeline.

Retrieval = "given a question, find the most relevant chunks".

    question --(embedder)--> query vector --(vector store)--> top-k chunks

IMPORTANT: the question must be embedded with the SAME model that was used
for the stored chunks, otherwise the vectors live in different "spaces"
and can't be compared.
"""

from utils.chunking import Chunk
from utils.vector_store import VectorStore


class Retriever:
    def __init__(self, embedder, store: VectorStore, top_k: int = 3):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def retrieve(self, question: str) -> list[Chunk]:
        query_vector = self.embedder.embed([question])[0]   # embed() takes a list -> take the single vector
        return self.store.search(query_vector, top_k=self.top_k)


# ---------------------------------------------------------------------------
# Quick manual check (run from project root:  python -m utils.retrieval)
# Needs the collection built first:            python -m utils.vector_store
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from utils.embedding import OpenAIEmbedder, SentenceTransformerEmbedder

    PROVIDER = "openai"   # must match the provider used in vector_store.py

    embedder = SentenceTransformerEmbedder() if PROVIDER == "local" else OpenAIEmbedder()
    store = VectorStore(collection_name=f"bain_{embedder.model_name}")
    retriever = Retriever(embedder, store, top_k=3)

    print(f"Searching collection '{store.collection.name}' ({store.count()} chunks)\n")

    questions = [
        "who is Senior Operations Analyst"
        "What are AI leaders doing differently from other companies?",
        "How much money is being invested in data centers?",
        "What is happening with quantum computing?",
    ]

    for question in questions:
        print("=" * 60)
        print(f"Q: {question}")
        print("=" * 60)

        for rank, chunk in enumerate(retriever.retrieve(question), start=1):
            preview = chunk.text[:120].replace("\n", " ")
            print(f"#{rank}  page {chunk.metadata['page']}  (score {chunk.metadata['score']})")
            print(f"    {preview}...\n")
