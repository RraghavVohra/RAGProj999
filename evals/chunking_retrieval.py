"""
Which chunking strategy RETRIEVES best?

Same PDF, same embedding model (BGE small), same questions - only the chunking changes.
No reranker and no LLM, so we see the effect of chunking alone.

For every question we know the page AND a few words of the answer, so we can check
two things for the top 3 chunks:
    right page?      - did retrieval find the right page?
    contains answer? - is the answer actually INSIDE the chunk the LLM would get?

Run from the project root:   python -m evals.chunking_retrieval
(Local model only - free, no API.)
"""

from utils.embedding import SentenceTransformerEmbedder
from utils.pipeline import ChunkingConfig, PipelineConfig, chunk_pdf
from utils.retrieval import Retriever
from utils.vector_store import VectorStore

PDF = "assets/bain_report_technology_report_2025.pdf"
EMBEDDER = "BAAI/bge-small-en-v1.5"
TOP_K = 3

# (question, page with the answer, words that must be in the chunk, what kind of test it is)
QUESTIONS = [
    ("How much annual revenue is needed to fund new data centers by 2030?",
     35, "$2 trillion", "easy: answer at the TOP of the page"),
    ("How should software companies change their pricing for AI agent solutions?",
     26, "outcome-based", "hard: answer in the LAST lines of a 3067-char page"),
    ("What share of North American private equity deals went to technology in 2025?",
     18, "22%", "chart: answer is in a figure's text"),
]

STRATEGIES = {   # name -> chunking settings (same as you would pick in the app)
    "page-wise": ChunkingConfig(strategy="page-wise"),
    "character": ChunkingConfig(strategy="character-wise", chunk_size=500, overlap=50),
    "recursive": ChunkingConfig(strategy="recursive", chunk_size=500, overlap=50),
    "semantic":  ChunkingConfig(strategy="semantic", breakpoint_percentile=90),
}

embedder = SentenceTransformerEmbedder(EMBEDDER)

# 1. Build one small index per strategy ------------------------------------------
retrievers = {}
for name, chunking in STRATEGIES.items():
    chunks = chunk_pdf(PDF, PipelineConfig(chunking=chunking), embedder)
    collection = f"demo_chunking_{name.replace('-', '_')}"
    store = VectorStore(collection)
    store.client.delete_collection(collection)          # start clean every run
    store = VectorStore(collection)
    store.add_chunks(chunks, embedder.embed([c.text for c in chunks]))
    retrievers[name] = Retriever(embedder, store, top_k=TOP_K)
    print(f"{name:<10} {len(chunks):>4} chunks indexed")

# 2. Ask every question with every strategy -----------------------------------------
summary = {name: {"page": 0, "answer": 0} for name in STRATEGIES}

for question, page, answer, kind in QUESTIONS:
    print("\n" + "=" * 100)
    print(f"Q: {question}")
    print(f"   expected: page {page}, chunk must contain {answer!r}   ({kind})")
    print("=" * 100)

    for name, retriever in retrievers.items():
        results = retriever.retrieve(question)
        print(f"\n  {name.upper()}")
        found_page = found_answer = False
        for rank, chunk in enumerate(results, start=1):
            right_page = chunk.metadata["page"] == page
            has_answer = answer in chunk.text
            found_page |= right_page
            found_answer |= has_answer
            marks = ("page OK " if right_page else "        ") + ("ANSWER" if has_answer else "")
            preview = chunk.text.replace("\n", " ")[:70]
            print(f"    #{rank}  p.{chunk.metadata['page']:<3} {chunk.metadata['score']:.2f}  "
                  f"{len(chunk.text):>5} chars  {marks:<14} {preview!r}")
        summary[name]["page"] += found_page
        summary[name]["answer"] += found_answer

# 3. Scoreboard ----------------------------------------------------------------------
print("\n" + "=" * 100)
print(f"SCOREBOARD   (out of {len(QUESTIONS)} questions, top {TOP_K})")
print("=" * 100)
print(f"  {'strategy':<10}  {'right page found':>17}  {'answer inside a chunk':>22}")
for name, s in summary.items():
    print(f"  {name:<10}  {s['page']:>17}  {s['answer']:>22}")

for retriever in retrievers.values():   # clean up the demo collections
    retriever.store.client.delete_collection(retriever.store.collection.name)
