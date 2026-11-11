"""
Pipeline module: glues all our components together using the settings
the user picked on the frontend's configuration page.

Batch 2: the config (what settings exist) + turning PDFs into chunks.
Batch 3: the RAGPipeline class (build = embed + store, ask = retrieve + rerank + generate).
"""

import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from utils.chunking import Chunk, character_chunking, page_chunking
from utils.embedding import OpenAIEmbedder, SentenceTransformerEmbedder, embed_chunks
from utils.generation import Generator
from utils.reranking import CohereReranker
from utils.retrieval import Retriever
from utils.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Config: one small class per pipeline stage.
#
# The frontend sends JSON like:
#   {"chunking": {"strategy": "page-wise", ...}, "embedding": {"model": "..."}, ...}
# Pydantic turns that JSON into these Python objects AND checks it:
#   wrong strategy name, negative top_k, missing fields -> clear error, automatically.
# ---------------------------------------------------------------------------

class ChunkingConfig(BaseModel):
    strategy: Literal["page-wise", "character-wise"] = "page-wise"   # only these two values are allowed
    chunk_size: int = Field(200, ge=50, le=5000)                     # ge = "greater or equal", le = "less or equal"
    overlap: int = Field(50, ge=0)


class EmbeddingConfig(BaseModel):
    model: Literal["text-embedding-3-small", "all-MiniLM-L6-v2"] = "text-embedding-3-small"


class RetrievalConfig(BaseModel):
    top_k: int = Field(10, ge=1, le=50)


class RerankingConfig(BaseModel):
    enabled: bool = True
    model: str = "rerank-v3.5"
    top_n: int = Field(3, ge=1, le=20)


class GenerationConfig(BaseModel):
    model: str = "gpt-5.6-luna"


class PipelineConfig(BaseModel):
    """The full set of settings = one object per stage. Every field has a default."""
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)


# ---------------------------------------------------------------------------
# Chunking PDFs according to the config
# ---------------------------------------------------------------------------

def chunk_pdf(pdf_path: str | Path, chunking: ChunkingConfig) -> list[Chunk]:
    """Split ONE PDF into chunks using the chosen strategy."""
    pages = page_chunking(str(pdf_path))   # always start with pages: they know their page number + file name

    if chunking.strategy == "page-wise":
        return pages

    # Character-wise: split EACH PAGE separately, so every small piece still
    # knows which page it came from (needed for "(page 5)" citations).
    chunks = []
    for page in pages:
        for piece in character_chunking(page.text, chunking.chunk_size, chunking.overlap):
            piece.metadata.update(
                page=page.metadata["page"],
                source=page.metadata["source"],
                chunk_id=len(chunks),          # number chunks across the whole file, not per page
            )
            chunks.append(piece)
    return chunks


def chunk_documents(pdf_paths: list[str | Path], chunking: ChunkingConfig) -> list[Chunk]:
    """Chunk several PDFs and return all chunks in one list."""
    all_chunks = []
    for path in pdf_paths:
        all_chunks.extend(chunk_pdf(path, chunking))
    return all_chunks


# ---------------------------------------------------------------------------
# RAGPipeline: one object = one configured pipeline, ready to answer questions
#
#   build(pdfs):    PDFs -> chunks -> vectors -> ChromaDB          (done once)
#   ask(question):  question -> retrieve -> rerank -> LLM answer   (done per question)
# ---------------------------------------------------------------------------

EMBED_BATCH_SIZE = 100   # embed 100 chunks per API call (big documents would exceed OpenAI's request limit)


def make_embedder(model_name: str):
    """Pick the embedder class that matches the model name from the config."""
    if model_name == "all-MiniLM-L6-v2":
        return SentenceTransformerEmbedder(model_name)
    return OpenAIEmbedder(model_name)


class RAGPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.id = uuid.uuid4().hex[:8]   # short random id, e.g. "3f9a1c2e"
        self.documents: list[str] = []
        self.num_chunks = 0

        # Create every component from the config - these are the classes we wrote earlier
        self.embedder = make_embedder(config.embedding.model)
        self.store = VectorStore(f"pipeline_{self.id}")   # own collection: documents of different pipelines never mix
        self.retriever = Retriever(self.embedder, self.store, top_k=config.retrieval.top_k)
        self.reranker = (
            CohereReranker(config.reranking.model, top_n=min(config.reranking.top_n, config.retrieval.top_k))
            if config.reranking.enabled else None    # None = "skip reranking"
        )
        self.generator = Generator(config.generation.model)

    def build(self, pdf_paths: list[str | Path]) -> int:
        """Chunk, embed and store the PDFs. Returns how many chunks were stored."""
        chunks = chunk_documents(pdf_paths, self.config.chunking)
        if not chunks:
            raise ValueError("No text found in the PDFs (are they scanned images?)")

        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[start:start + EMBED_BATCH_SIZE]
            vectors = embed_chunks(batch, self.embedder)
            self.store.add_chunks(batch, vectors)
            print(f"[pipeline {self.id}] stored {start + len(batch)}/{len(chunks)} chunks")

        self.documents = [Path(p).name for p in pdf_paths]
        self.num_chunks = len(chunks)
        return self.num_chunks

    def ask(self, question: str) -> dict:
        """Answer a question. Returns the answer AND the chunks it was based on."""
        chunks = self.retriever.retrieve(question)              # 1. fast vector search
        if self.reranker:
            chunks = self.reranker.rerank(question, chunks)     # 2. (optional) re-score, keep the best
        answer = self.generator.generate(question, chunks)      # 3. LLM writes the answer

        return {"answer": answer, "sources": [to_source(chunk) for chunk in chunks]}


def to_source(chunk: Chunk) -> dict:
    """The few fields the frontend shows under each answer."""
    return {
        "source": chunk.metadata.get("source"),
        "page": chunk.metadata.get("page"),
        "text": chunk.text,
        "score": chunk.metadata.get("score"),                # vector similarity (retriever)
        "rerank_score": chunk.metadata.get("rerank_score"),  # None when reranking is off
    }


# ---------------------------------------------------------------------------
# Quick manual check (run from project root:  python -m utils.pipeline)
# Uses your OpenAI + Cohere keys from .env
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # 1. Config, exactly like the frontend would send it ----------------------
    config = PipelineConfig.model_validate_json(
        '{"chunking": {"strategy": "page-wise"}, "retrieval": {"top_k": 10}, "reranking": {"enabled": true, "top_n": 3}}'
    )

    # 2. Build -------------------------------------------------------------------
    pipeline = RAGPipeline(config)
    pdf_path = Path(__file__).parent.parent / "assets" / "bain_report_technology_report_2025.pdf"

    print("=" * 60)
    print(f"BUILD  pipeline {pipeline.id}")
    print("=" * 60)
    print(f"chunking  : {config.chunking.strategy}")
    print(f"embedding : {config.embedding.model}")
    print(f"reranking : {'top ' + str(config.reranking.top_n) if pipeline.reranker else 'off'}")
    num_chunks = pipeline.build([pdf_path])
    print(f"done      : {num_chunks} chunks in collection '{pipeline.store.collection.name}'")

    # 3. Ask ---------------------------------------------------------------------
    question = "What are AI leaders doing differently from other companies?"
    result = pipeline.ask(question)

    print("\n" + "=" * 60)
    print(f"Q: {question}")
    print("=" * 60)
    print(result["answer"])

    print("\nSources (this list is what the frontend will show):")
    for s in result["sources"]:
        print(f"  page {s['page']:<3} vector {s['score']}  rerank {s['rerank_score']}  {s['text'][:60]!r}...")
