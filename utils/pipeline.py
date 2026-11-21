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

from utils.chunking import Chunk, character_chunking, page_chunking, recursive_chunking, semantic_chunking
from utils.embedding import LOCAL_MODELS, OpenAIEmbedder, SentenceTransformerEmbedder, embed_chunks
from utils.generation import Generator
from utils.parsing import read_pages
from utils.reranking import make_reranker
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
    strategy: Literal["page-wise", "character-wise", "recursive", "semantic"] = "page-wise"   # only these values are allowed
    chunk_size: int = Field(200, ge=50, le=5000)                     # ge = "greater or equal", le = "less or equal"
    overlap: int = Field(50, ge=0)
    breakpoint_percentile: int = Field(90, ge=50, le=99)             # semantic only: cut at the top 10% topic jumps


class EmbeddingConfig(BaseModel):
    model: Literal[
        "text-embedding-3-small",                     # OpenAI (paid API)
        "all-MiniLM-L6-v2",                           # the rest: local Hugging Face models (see utils/embedding.py)
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "intfloat/e5-base-v2",
        "sentence-transformers/all-mpnet-base-v2",
    ] = "text-embedding-3-small"


class RetrievalConfig(BaseModel):
    top_k: int = Field(10, ge=1, le=50)


class RerankingConfig(BaseModel):
    enabled: bool = True
    model: Literal[
        "rerank-v3.5",                                # Cohere (paid API)
        "cross-encoder/ms-marco-MiniLM-L-6-v2",       # the rest: open-source, run locally (see utils/reranking.py)
        "BAAI/bge-reranker-base",
        "BAAI/bge-reranker-v2-m3",
    ] = "rerank-v3.5"
    top_n: int = Field(3, ge=1, le=20)


class ParsingConfig(BaseModel):
    mode: Literal["fast", "accurate"] = "fast"   # fast = PyMuPDF + cleaning (seconds), accurate = Docling (minutes, cached)


class ImagesConfig(BaseModel):
    enabled: bool = False          # off by default: it costs one vision call per visual page (cached after the first build)
    model: str = "gpt-5.6-luna"    # must be a vision-capable OpenAI model


class GenerationConfig(BaseModel):
    model: str = "gpt-5.6-luna"


class PipelineConfig(BaseModel):
    """The full set of settings = one object per stage. Every field has a default."""
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    images: ImagesConfig = Field(default_factory=ImagesConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)


# ---------------------------------------------------------------------------
# Chunking PDFs according to the config
# ---------------------------------------------------------------------------

def split_pages(pages: list[Chunk], chunking: ChunkingConfig, embedder) -> list[Chunk]:
    """Turn page chunks into the chunks of the chosen strategy."""
    if chunking.strategy == "page-wise":
        return pages

    if chunking.strategy == "semantic":   # works across all pages at once (see utils/chunking.py)
        return semantic_chunking(pages, embedder, chunking.breakpoint_percentile)

    # Character-wise / recursive: split EACH PAGE separately, so every small
    # piece still knows which page it came from (needed for "(page 5)" citations).
    splitter = character_chunking if chunking.strategy == "character-wise" else recursive_chunking
    chunks = []
    for page in pages:
        for piece in splitter(page.text, chunking.chunk_size, chunking.overlap):
            piece.metadata.update(page=page.metadata["page"], source=page.metadata["source"])
            chunks.append(piece)
    return chunks


def chunk_pdf(pdf_path: str | Path, config: PipelineConfig, embedder) -> list[Chunk]:
    """Split ONE PDF into chunks: text chunks (chosen strategy) + figure chunks (if image parsing is on)."""
    # STEP 1 (document parsing): clean text per page, the USER picks how (config.parsing.mode):
    #   before    : pages = page_chunking(str(pdf_path))  -> raw pypdf text, page numbers + headers glued into every page
    #   "fast"    : PyMuPDF + 5 general cleaning rules (repeated headers/footers, page numbers, broken words, ...)
    #               ~1 second for a whole book. Refuses scanned PDFs with a clear message.
    #   "accurate": Docling (layout-aware AI parser): keeps only the real content, tables as tables,
    #               reads scanned pages. ~5 s per page the first time, then loaded from cache/docling/.
    #   (details: utils/parsing.py)
    # We still start with pages: they know their page number + file name.
    pages = read_pages(pdf_path, config.parsing.mode)

    figures = []
    if config.images.enabled:
        from utils.image_parsing import parse_visual_pages   # imported here: only needed when images are on
        scanned_pages, figures = parse_visual_pages(pdf_path, config.images.model)

        # Avoid DUPLICATE text for scanned pages (a scanned page = a photo of paper, no real text inside).
        # Two parts of the app can now read such a page:
        #   1. Docling, with its built-in OCR (free)  -> already inside `pages`
        #   2. image parsing, with the OpenAI vision model (when the Images switch is on) -> `scanned_pages`
        # If both were kept, the page's text would be stored twice and show up twice in search results.
        # Rule: if Docling already got text from a page, ignore image parsing's transcription of that page.
        # (Chart descriptions in `figures` are NOT affected - they are still added below.)
        pages_docling_read = {p.metadata["page"] for p in pages}
        scanned_pages = [p for p in scanned_pages if p.metadata["page"] not in pages_docling_read]

        pages = sorted(pages + scanned_pages, key=lambda p: p.metadata["page"])   # scanned text joins normal text

    chunks = split_pages(pages, config.chunking, embedder)
    chunks += figures   # figure descriptions are kept whole: splitting a table in half helps nobody

    for i, chunk in enumerate(chunks):   # number chunks across the whole file (ids in Chroma must be unique)
        chunk.metadata["chunk_id"] = i
        chunk.metadata["strategy"] = config.chunking.strategy
        chunk.metadata.setdefault("type", "text")
    return chunks


def chunk_documents(pdf_paths: list[str | Path], config: PipelineConfig, embedder) -> list[Chunk]:
    """Chunk several PDFs and return all chunks in one list."""
    all_chunks = []
    for path in pdf_paths:
        all_chunks.extend(chunk_pdf(path, config, embedder))
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
    if model_name in LOCAL_MODELS:
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
            make_reranker(config.reranking.model, top_n=min(config.reranking.top_n, config.retrieval.top_k))
            if config.reranking.enabled else None    # None = "skip reranking"
        )
        self.generator = Generator(config.generation.model)

    def build(self, pdf_paths: list[str | Path]) -> int:
        """Chunk, embed and store the PDFs. Returns how many chunks were stored."""
        chunks = chunk_documents(pdf_paths, self.config, self.embedder)
        if not chunks:
            raise ValueError("No text found in the PDFs (are they scanned images? Turn on image parsing.)")

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
        "type": chunk.metadata.get("type", "text"),          # "text" or "figure" (description of a chart/table)
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
