"""
Embedding module for our modular RAG pipeline.

An embedding turns text into a list of numbers (a vector). Texts with
similar meaning get vectors that are close to each other, which is what
lets us search by meaning later.

Every embedder has the same two methods:
    embed(texts)       -> list of vectors   (for the document chunks)
    embed_query(text)  -> one vector        (for the user's question)
So the pipeline can swap a local model for OpenAI (or anything else)
without changing any other code.

Why two methods? Some models (BGE, E5) were TRAINED with a short
instruction in front of questions and/or passages. Leaving it out still
"works" but gives noticeably worse retrieval. Keeping the two methods
separate means each model gets exactly the prefix it expects.
"""

import os

from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI

from utils.chunking import Chunk, character_chunking, page_chunking

load_dotenv()   # reads OPENAI_API_KEY from the .env file in the project root


# ---------------------------------------------------------------------------
# Local models from Hugging Face (downloaded once, then run on your laptop)
#
#   dims       = numbers per vector (bigger = more detail, more storage)
#   max_tokens = longer text is SILENTLY CUT OFF at this length
#                (~4 characters per token, so 256 tokens = ~1000 characters)
#   prefixes   = the instruction each model was trained with
# ---------------------------------------------------------------------------

LOCAL_MODELS = {
    "all-MiniLM-L6-v2": {                       # tiny + fast, the classic baseline
        "dims": 384, "max_tokens": 256,
        "query_prefix": "", "doc_prefix": "",
    },
    "BAAI/bge-small-en-v1.5": {                 # same size as MiniLM, clearly better retrieval
        "dims": 384, "max_tokens": 512,
        "query_prefix": "Represent this sentence for searching relevant passages: ", "doc_prefix": "",
    },
    "BAAI/bge-base-en-v1.5": {                  # 2x bigger + slower, best quality of this list
        "dims": 768, "max_tokens": 512,
        "query_prefix": "Represent this sentence for searching relevant passages: ", "doc_prefix": "",
    },
    "intfloat/e5-base-v2": {                    # strong alternative, needs prefixes on BOTH sides
        "dims": 768, "max_tokens": 512,
        "query_prefix": "query: ", "doc_prefix": "passage: ",
    },
    "sentence-transformers/all-mpnet-base-v2": {  # older but solid general-purpose model
        "dims": 768, "max_tokens": 384,
        "query_prefix": "", "doc_prefix": "",
    },
}

# Loading a model takes seconds (and the first time, a download).
# Keep loaded models here so building a second pipeline is instant.
_loaded_models = {}


def load_sentence_transformer(model_name: str):
    if model_name not in _loaded_models:
        from sentence_transformers import SentenceTransformer   # imported here so OpenAI-only users don't need it
        print(f"[embedding] loading {model_name} (first time downloads it from Hugging Face)")
        _loaded_models[model_name] = SentenceTransformer(model_name)
    return _loaded_models[model_name]


# ---------------------------------------------------------------------------
# Embedder 1: Local model (sentence-transformers) - free, runs on your laptop
# ---------------------------------------------------------------------------

class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        settings = LOCAL_MODELS.get(model_name, {})   # unknown model -> no prefixes
        self.model_name = model_name
        self.query_prefix = settings.get("query_prefix", "")
        self.doc_prefix = settings.get("doc_prefix", "")
        self.model = load_sentence_transformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        # normalize = every vector gets length 1, so cosine similarity is just a dot product
        texts = [self.doc_prefix + text for text in texts]
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(self.query_prefix + text, normalize_embeddings=True).tolist()


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

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]   # OpenAI models need no prefix


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
