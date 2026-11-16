"""
Chunking strategies - a small demo to SEE how each one cuts the text.

Part 1: character vs recursive on the same page -> where does each chunk START and END?
Part 2: semantic chunking -> how the breakpoint percentile changes the number/size of chunks
Part 3: semantic chunks of one page -> where did it decide the topic changed?
Part 4: all strategies on the whole PDF -> how many chunks, how big?

Run from the project root:   python -m evals.chunking_demo
(Uses the local MiniLM model for semantic chunking - free, no API.)
"""

from utils.chunking import character_chunking, page_chunking, recursive_chunking, semantic_chunking
from utils.embedding import SentenceTransformerEmbedder

PDF = "assets/bain_report_technology_report_2025.pdf"
PAGE = 7          # the page we look at closely
SIZE, OVERLAP = 500, 50


def edges(text, n=45):
    """Show how a chunk starts and ends, on one line."""
    flat = text.replace("\n", " ")
    return f"START: {flat[:n]!r:<50} END: {flat[-n:]!r}"


def stats(chunks):
    sizes = [len(c.text) for c in chunks]
    return f"{len(chunks):>4} chunks   avg {sum(sizes) // len(sizes):>5} chars   min {min(sizes):>5}   max {max(sizes):>5}"


pages = page_chunking(PDF)
page = next(p for p in pages if p.metadata["page"] == PAGE)

# --- Part 1 -------------------------------------------------------------------
print("=" * 100)
print(f"PART 1   Page {PAGE} ({len(page.text)} chars), size {SIZE}, overlap {OVERLAP}")
print("=" * 100)
for name, splitter in [("CHARACTER", character_chunking), ("RECURSIVE", recursive_chunking)]:
    print(f"\n{name}")
    for i, chunk in enumerate(splitter(page.text, SIZE, OVERLAP)):
        print(f"  chunk {i}  ({len(chunk.text)} chars)  {edges(chunk.text)}")
print("\n  -> does each chunk start/end at a clean word or sentence? Compare the two.")

# --- Part 2 -------------------------------------------------------------------
print("\n" + "=" * 100)
print("PART 2   Semantic chunking on the whole PDF, different breakpoint percentiles")
print("=" * 100)
embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
for percentile in (80, 90, 95):
    print(f"  percentile {percentile}:  {stats(semantic_chunking(pages, embedder, percentile))}")
print("  -> lower percentile = cut more often = more, smaller chunks")

# --- Part 3 -------------------------------------------------------------------
print("\n" + "=" * 100)
print(f"PART 3   Semantic chunks of page {PAGE} (percentile 90)")
print("=" * 100)
semantic = semantic_chunking(pages, embedder, 90)
for chunk in [c for c in semantic if c.metadata["page"] == PAGE]:
    print(f"\n  --- {len(chunk.text)} chars ---")
    print("  " + chunk.text[:300] + (" ..." if len(chunk.text) > 300 else ""))

# --- Part 4 -------------------------------------------------------------------
print("\n" + "=" * 100)
print("PART 4   Whole PDF, every strategy")
print("=" * 100)
character = [c for p in pages for c in character_chunking(p.text, SIZE, OVERLAP)]
recursive = [c for p in pages for c in recursive_chunking(p.text, SIZE, OVERLAP)]
print(f"  page-wise  {stats(pages)}")
print(f"  character  {stats(character)}")
print(f"  recursive  {stats(recursive)}")
print(f"  semantic   {stats(semantic)}")

# --- Part 5 -------------------------------------------------------------------
print("\n" + "=" * 100)
print("PART 5   Does every semantic chunk FIT in what MiniLM can read? (real token count)")
print("=" * 100)
tokenizer = embedder.model.tokenizer
token_counts = [len(tokenizer(c.text)["input_ids"]) for c in semantic]
too_long = sum(1 for n in token_counts if n > embedder.max_tokens)
print(f"  model reads        : {embedder.max_tokens} tokens")
print(f"  biggest chunk      : {max(token_counts)} tokens")
print(f"  chunks cut off     : {too_long} of {len(semantic)}")
print("  -> 0 cut off = the model sees every chunk completely")
