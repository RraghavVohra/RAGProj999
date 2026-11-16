"""
Chunking module for our modular RAG pipeline.

Every chunking strategy returns the same thing: a list of `Chunk` objects.
That way the rest of the pipeline (embedding, retrieval, ...) doesn't care
which strategy produced the chunks.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
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
# Strategy 3: Recursive chunking
# ---------------------------------------------------------------------------
#
# Character chunking cuts blindly every N characters, often mid-word.
# Recursive chunking tries the "nicest" place to cut first, and only falls
# back to uglier cuts when a piece is still too big:
#
#     paragraphs ("\n\n")  ->  lines ("\n")  ->  sentences (". ")  ->  words (" ")  ->  characters ("")
#
# Then the small pieces are glued back together into chunks of up to
# `chunk_size` characters, with `overlap` characters repeated between chunks.
# (This is the same idea as LangChain's RecursiveCharacterTextSplitter.)

RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_into_pieces(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Cut `text` into pieces that are each <= chunk_size, using the nicest separator possible."""
    if len(text) <= chunk_size:
        return [text]

    # Find the first separator that actually appears in the text
    for i, separator in enumerate(separators):
        if separator == "":   # last resort: hard cut every chunk_size characters
            return [text[start:start + chunk_size] for start in range(0, len(text), chunk_size)]
        if separator in text:
            remaining_separators = separators[i + 1:]
            break

    # Split, keeping the separator at the end of each part (so no text is lost)
    parts = text.split(separator)
    parts = [part + separator for part in parts[:-1]] + [parts[-1]]

    pieces = []
    for part in parts:
        if len(part) <= chunk_size:
            pieces.append(part)
        else:   # this part is still too big -> split it with the NEXT separator
            pieces.extend(_split_into_pieces(part, chunk_size, remaining_separators))
    return pieces


def _merge_pieces(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Glue small pieces together into chunks of up to chunk_size, repeating ~overlap characters."""
    chunks = []
    current = []   # pieces in the chunk we are building
    length = 0     # total characters in `current`

    for piece in pieces:
        if current and length + len(piece) > chunk_size:   # adding this piece would overflow
            chunks.append("".join(current))
            # Keep the last few pieces (up to `overlap` characters) as the start of the next chunk
            while current and (length > overlap or length + len(piece) > chunk_size):
                length -= len(current.pop(0))
        current.append(piece)
        length += len(piece)

    if current:
        chunks.append("".join(current))
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def recursive_chunking(text: str, chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    pieces = _split_into_pieces(text, chunk_size, RECURSIVE_SEPARATORS)
    return [
        Chunk(text=piece, metadata={"strategy": "recursive", "chunk_id": i})
        for i, piece in enumerate(_merge_pieces(pieces, chunk_size, overlap))
    ]


# ---------------------------------------------------------------------------
# Strategy 4: Semantic chunking
# ---------------------------------------------------------------------------
#
# All strategies above cut by SIZE. Semantic chunking cuts by MEANING:
#
#   1. split the text into sentences
#   2. embed every sentence (together with its neighbours, to reduce noise)
#   3. compare each sentence with the next one: distance = 1 - cosine similarity
#   4. a BIG jump in distance = the topic changed -> start a new chunk there
#
# "Big" is relative to the document: we cut wherever the distance is in the
# top (100 - breakpoint_percentile)% of all distances. So percentile 90 =
# cut at the 10% biggest topic changes. Lower percentile = more, smaller chunks.
#
# Guard rails: never make a chunk shorter than min_chunk_size (one-line chunks
# have too little context), never longer than max_chunk_size (the embedder
# would cut it off), and always cut at a page boundary (so page citations stay correct).

def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()   # PDFs put line breaks mid-sentence; flatten them
    # Table-of-contents "dot leaders" ("Quantum Computing . . . . . 41") are NOT sentence ends.
    # Turn 3+ dots into one "…" so the split below ignores them.
    text = re.sub(r"(?:\.\s?){3,}", "… ", text)
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


SEMANTIC_MAX_CHARS = 1500   # upper limit for every model: bigger chunks mix too many topics
CHARS_PER_TOKEN = 4         # rough average for English text


def semantic_max_chunk_size(embedder, min_chunk_size: int = 200) -> int:
    """
    The biggest chunk the embedder can read completely.
      MiniLM: 256 tokens x 4 = 1024 chars -> minus 200 margin = 824
      BGE   : 512 tokens x 4 = 2048 chars -> minus 200 = 1848 -> capped at 1500
    The 200 margin (= min_chunk_size) is room for fix 2: a tiny leftover glued
    onto a full chunk can make it up to 199 characters longer.
    """
    max_tokens = getattr(embedder, "max_tokens", None)   # unknown embedder -> just use the cap
    if max_tokens is None:
        return SEMANTIC_MAX_CHARS
    readable_chars = max_tokens * CHARS_PER_TOKEN
    return min(SEMANTIC_MAX_CHARS, readable_chars - min_chunk_size)


def semantic_chunking(
    pages: list[Chunk],
    embedder,
    breakpoint_percentile: int = 90,
    min_chunk_size: int = 200,
    max_chunk_size: int | None = None,   # None = decide from the embedding model (see below)
    batch_size: int = 100,
) -> list[Chunk]:
    """
    Takes PAGE chunks (from page_chunking) instead of raw text, because it
    works across the whole document: "big jump" is judged against all of
    its sentences, not just one page.
    """
    # 0. How big may a chunk be? Never bigger than the embedding model can READ.
    if max_chunk_size is None:
        max_chunk_size = semantic_max_chunk_size(embedder, min_chunk_size)

    # 1. Sentences, remembering which page each one came from
    sentences, page_of = [], []
    for page in pages:
        for sentence in split_sentences(page.text):
            sentences.append(sentence)
            page_of.append(page.metadata["page"])
    if not sentences:
        return []
    source = pages[0].metadata.get("source")

    # 2. Embed each sentence together with its neighbours on the same page.
    #    A single short sentence ("See Figure 2.") has a noisy vector; a window of 3 is far more stable.
    windows = []
    for i in range(len(sentences)):
        neighbours = [sentences[j] for j in (i - 1, i, i + 1)
                      if 0 <= j < len(sentences) and page_of[j] == page_of[i]]
        windows.append(" ".join(neighbours))

    vectors = []
    for start in range(0, len(windows), batch_size):   # batches keep API requests small
        vectors.extend(embedder.embed(windows[start:start + batch_size]))
    vectors = np.array(vectors)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    # 3. distances[i] = how different sentence i is from sentence i+1
    distances = 1 - np.sum(vectors[:-1] * vectors[1:], axis=1)
    threshold = np.percentile(distances, breakpoint_percentile) if len(distances) else 1.0

    # 4. Walk through the sentences and cut where the topic jumps
    chunks = []

    def close_chunk(sentence_list, page):
        text = " ".join(sentence_list)
        # A tiny leftover (e.g. the last short sentence before a page break) has too little
        # context to be useful on its own -> glue it onto the previous chunk of the same page.
        if len(text) < min_chunk_size and chunks and chunks[-1].metadata["page"] == page:
            chunks[-1].text += " " + text
            return
        chunks.append(Chunk(
            text=text,
            metadata={"strategy": "semantic", "chunk_id": len(chunks), "page": page, "source": source},
        ))

    current = [sentences[0]]
    current_length = len(sentences[0])
    for i in range(1, len(sentences)):
        new_page = page_of[i] != page_of[i - 1]
        topic_changed = distances[i - 1] > threshold and current_length >= min_chunk_size
        too_long = current_length + 1 + len(sentences[i]) > max_chunk_size

        if new_page or topic_changed or too_long:
            close_chunk(current, page_of[i - 1])
            current, current_length = [], 0

        current.append(sentences[i])
        current_length += len(sentences[i]) + (1 if current_length else 0)

    close_chunk(current, page_of[-1])
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
