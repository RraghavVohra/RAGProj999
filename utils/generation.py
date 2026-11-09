"""
Generation module for our modular RAG pipeline.

Generation = "give the LLM the question + the retrieved chunks, and let it
write an answer using ONLY those chunks".

    question + retrieved chunks --(build prompt)--> LLM --> answer with page citations

Batch 1: building the prompt (what the LLM will actually read).
Batch 2: the Generator class that sends the prompt to the LLM.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

from utils.chunking import Chunk
from utils.embedding import OpenAIEmbedder
from utils.retrieval import Retriever
from utils.vector_store import VectorStore

# Rules for the LLM. Telling it to use only the context and cite pages
# is what stops it from making things up ("hallucinating").
SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context.

Rules:
- Use only the information in the context. Do not use outside knowledge.
- Cite the page number for every fact, like this: (page 5).
- If the context does not contain the answer, say: "I couldn't find this in the document."
- Keep the answer short and clear."""


def build_context(chunks: list[Chunk]) -> str:
    """Join the retrieved chunks into one text block, each labelled with its source page."""
    blocks = []
    for chunk in chunks:
        header = f"[Source: {chunk.metadata['source']}, page {chunk.metadata['page']}]"
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


def build_user_prompt(question: str, chunks: list[Chunk]) -> str:
    """The message we send to the LLM: context first, then the question."""
    return f"Context:\n{build_context(chunks)}\n\nQuestion: {question}"


# ---------------------------------------------------------------------------
# Generator: sends the prompt to the LLM and returns its answer
# ---------------------------------------------------------------------------

load_dotenv()   # reads OPENAI_API_KEY from .env



class Generator:
    def __init__(self, model_name: str = "gpt-5.6-luna"):
        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, question: str, chunks: list[Chunk]) -> str:
        response = self.client.responses.create(
            model=self.model_name,
            instructions=SYSTEM_PROMPT,                     # the rules (system prompt)
            input=build_user_prompt(question, chunks),      # context + question
        )
        return response.output_text
    
# HERE I HAVE EXPLAINED OBJECT AND CLASS
# raghav = Generator()
# ans = raghav.generate(question, chunks)
# print(ans)


# ---------------------------------------------------------------------------
# Quick manual check (run from project root:  python -m utils.generation)
# ---------------------------------------------------------------------------

if __name__ == "__main__":


    question = "What are AI leaders doing differently from other companies?"

    # 1. Retrieve (same as retrieval.py) ------------------------------------
    embedder = OpenAIEmbedder()
    retriever = Retriever(embedder, VectorStore(f"bain_{embedder.model_name}"), top_k=3)
    chunks = retriever.retrieve(question)
    print(f"Retrieved pages: {[c.metadata['page'] for c in chunks]}\n")

    # 2. What the LLM will receive (short summary) ---------------------------
    user_prompt = build_user_prompt(question, chunks)
    print(f"Prompt size: {len(user_prompt)} chars of context + question\n")

    # 3. Generate the answer -------------------------------------------------
    generator = Generator()
    answer = generator.generate(question, chunks)

    print("=" * 60)
    print(f"Q: {question}")
    print(f"Model: {generator.model_name}")
    print("=" * 60)
    print(answer)

    # 4. Out-of-scope question: the LLM should refuse, not make things up ----
    off_topic = "What is the capital of France?"
    print("\n" + "=" * 60)
    print(f"Q: {off_topic}   (not in the document)")
    print("=" * 60)
    print(generator.generate(off_topic, retriever.retrieve(off_topic)))



