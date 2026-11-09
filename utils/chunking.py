"""
Chunking module for our modular RAG pipeline.

Every chunking strategy returns the same thing: a list of `Chunk` objects.
That way the rest of the pipeline (embedding, retrieval, ...) doesn't care
which strategy produced the chunks.
"""

from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader


# ---------------------------------------------------------------------------
# Common data structure
# ---------------------------------------------------------------------------


#  here we have not used @dataclass decorator hence we have to write constructor
#  whenevr you create an object consytructor is needed, eitther you write one constructor or python will intialize its own constructor
# class Chunk:
    # def __init__(self,text,metadata):
    #     self.text = text
    #     self.metadata = metadata  
        

    # def __init__(self):
    #     return


@dataclass
class Chunk:
    text: str                                      # the actual chunk content
    metadata: dict = field(default_factory=dict)   # e.g. strategy, position, page


# ---------------------------------------------------------------------------
# Strategy 1: Character-wise chunking
# ---------------------------------------------------------------------------

def character_chunking(text: str, chunk_size: int = 200, overlap: int = 50) -> list[Chunk]:
    """
    Split text into fixed-size pieces of `chunk_size` characters.
    Consecutive chunks share `overlap` characters so that a sentence cut
    at a boundary still appears whole in at least one chunk.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - overlap   # how far we move forward each time

    for start in range(0, len(text), step):
        end = start + chunk_size
        piece = text[start:end]

        chunks.append(Chunk(
            text=piece,
            metadata={"strategy": "character-wise", "chunk_id": len(chunks), "start": start, "end": min(end, len(text))},
        ))

        if end >= len(text):   # last chunk reached the end of the text
            break

    return chunks


# ---------------------------------------------------------------------------
# Strategy 2: Page-wise chunking
# ---------------------------------------------------------------------------

def page_chunking(pdf_path: str) -> list[Chunk]:
    """
    One chunk per PDF page.
    Simple and keeps the page number, which is great for citations
    ("answer found on page 3"). Downside: pages can be very long or very
    short, and a paragraph spanning two pages gets split.
    """
    reader = PdfReader(pdf_path)
    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()

        if not text:   # skip blank / image-only pages
            continue

        chunks.append(Chunk(
            text=text,
            metadata={"strategy": "page-wise", "chunk_id": len(chunks), "page": page_number, "source": Path(pdf_path).name},
        ))

    return chunks


# ---------------------------------------------------------------------------
# Quick manual check (run: python utils/chunking.py)
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
    # sample_text = (
    #     "Retrieval-Augmented Generation (RAG) combines a retriever with a language model. "
    #     "First, documents are split into chunks. Then each chunk is converted into an embedding. "
    #     "At query time, the most relevant chunks are retrieved and passed to the LLM as context. "
    #     "Good chunking is important because it decides what information the LLM gets to see."
    # )

    # print("=" * 60)
    # print("CHARACTER CHUNKING  (chunk_size=100, overlap=20)")
    # print("=" * 60)
    # print(f"Input length: {len(sample_text)} characters\n")

    # chunks = character_chunking(sample_text, chunk_size=100, overlap=20)
    # print(f"Number of chunks: {len(chunks)}\n")

    # for chunk in chunks:
    #     print(f"--- Chunk {chunk.metadata['chunk_id']} "
    #           f"[{chunk.metadata['start']}:{chunk.metadata['end']}] "
    #           f"({len(chunk.text)} chars) ---")
    #     print(repr(chunk.text))
    #     print()

    # -------------------------------------------------------------------
    # pdf_path = Path(__file__).parent.parent / "assets" / "bain_report_technology_report_2025.pdf"

    # print("=" * 60)
    # print(f"PAGE-WISE CHUNKING  ({pdf_path.name})")
    # print("=" * 60)

    # chunks = page_chunking(str(pdf_path))
    # print(f"Number of chunks: {len(chunks)}\n")

    # for chunk in chunks:
    #     if chunk.metadata['page']>4:
    #         break

    #     print(f"--- Chunk {chunk.metadata['chunk_id']} "
    #           f"(page {chunk.metadata['page']} of {chunk.metadata['source']}, "
    #           f"{len(chunk.text)} chars) ---")
    #     print(chunk.text)
    #     print()
