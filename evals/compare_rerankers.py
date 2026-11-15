"""
Compare rerankers: Cohere (paid API) vs open-source cross-encoders (local).

How it works:
  1. chunk + embed the Bain report ONCE (same candidates for every reranker = fair comparison)
  2. for each question, retrieve the top-k candidates from ChromaDB
  3. let each reranker pick its best top-n, and check them against the page(s)
     we KNOW contain the answer (ground truth below, checked by hand in the PDF)

Metrics:
  Hit@n  - % of questions where at least one correct page is in the top n   (higher = better)
  MRR    - mean of 1/rank of the first correct chunk: 1.0 = always ranked #1 (higher = better)
  ms/q   - average reranking time per question                              (lower = better)

"no rerank" is the baseline: plain vector-search order. A reranker that
doesn't beat it isn't worth its cost.

Run from the project root:
    python -m evals.compare_rerankers
    python -m evals.compare_rerankers --rerankers cohere ms-marco bge-base bge-m3 --top-k 20
Cohere is skipped automatically if COHERE_API_KEY isn't set.
"""

import argparse
import os
import time
from pathlib import Path

from utils.embedding import SentenceTransformerEmbedder, OpenAIEmbedder, LOCAL_MODELS
from utils.pipeline import ChunkingConfig, PipelineConfig, chunk_pdf
from utils.reranking import CohereReranker, CrossEncoderReranker
from utils.retrieval import Retriever
from utils.vector_store import VectorStore

PDF_PATH = Path(__file__).parent.parent / "assets" / "bain_report_technology_report_2025.pdf"

# (question, pages that contain the answer). Page = PDF page number, as shown in citations.
QUESTIONS = [
    ("How much annual revenue is needed to fund new data centers by 2030?", [35]),
    ("Who is the Senior Operations Analyst credited in the report?", [2]),
    ("What productivity boost do teams using AI coding assistants see?", [64, 65]),
    ("Which tech companies are valued above $2 trillion?", [8]),
    ("What would a car achieve if Moore's law applied to it?", [34]),
    ("How are tariffs and export controls affecting the technology industry?", [11, 12]),
    ("How could quantum computing be used in financial services?", [44, 45, 46]),
    ("What is holding back humanoid robots from wide adoption?", [37, 38, 39, 40, 41, 42]),
]

RERANKERS = {   # short name -> how to create it
    "cohere":   lambda n: CohereReranker("rerank-v3.5", top_n=n),
    "ms-marco": lambda n: CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=n),
    "bge-base": lambda n: CrossEncoderReranker("BAAI/bge-reranker-base", top_n=n),
    "bge-m3":   lambda n: CrossEncoderReranker("BAAI/bge-reranker-v2-m3", top_n=n),   # ~2.2 GB download
}


def score(ranked_chunks, correct_pages):
    """(hit, reciprocal rank) for one question."""
    for rank, chunk in enumerate(ranked_chunks, start=1):
        if chunk.metadata["page"] in correct_pages:
            return 1, 1 / rank
    return 0, 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rerankers", nargs="+", default=["cohere", "ms-marco", "bge-base"], choices=list(RERANKERS))
    parser.add_argument("--embedder", default="BAAI/bge-small-en-v1.5", help="local model name or text-embedding-3-small")
    parser.add_argument("--chunking", default="recursive", choices=["page-wise", "character-wise", "recursive", "semantic"])
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--top-k", type=int, default=20, help="candidates retrieved before reranking")
    parser.add_argument("--top-n", type=int, default=3, help="chunks kept after reranking")
    args = parser.parse_args()

    if "cohere" in args.rerankers and not os.getenv("COHERE_API_KEY"):
        print("COHERE_API_KEY not set -> skipping Cohere\n")
        args.rerankers.remove("cohere")

    # 1. Build the candidate index once ---------------------------------------------
    embedder = SentenceTransformerEmbedder(args.embedder) if args.embedder in LOCAL_MODELS else OpenAIEmbedder(args.embedder)
    config = PipelineConfig(chunking=ChunkingConfig(strategy=args.chunking, chunk_size=args.chunk_size, overlap=100))
    chunks = chunk_pdf(PDF_PATH, config, embedder)

    store = VectorStore("eval_rerankers")
    store.client.delete_collection("eval_rerankers")    # start clean every run
    store = VectorStore("eval_rerankers")
    for start in range(0, len(chunks), 100):
        batch = chunks[start:start + 100]
        store.add_chunks(batch, embedder.embed([c.text for c in batch]))
    retriever = Retriever(embedder, store, top_k=args.top_k)
    print(f"Index: {len(chunks)} {args.chunking} chunks, embedder {args.embedder}, "
          f"retrieve top {args.top_k} -> keep top {args.top_n}\n")

    candidates = {question: retriever.retrieve(question) for question, _ in QUESTIONS}

    # 2. Run every reranker on the same candidates ---------------------------------
    results = {}       # name -> (hit rate, mrr, ms per question)
    top_pages = {}     # name -> {question: [pages]}  (to compare rerankers with each other)

    systems = [("no rerank", None)] + [(name, RERANKERS[name](args.top_n)) for name in args.rerankers]
    for name, reranker in systems:
        if reranker is not None:   # warm-up: the first call of a local model is always slow
            reranker.rerank("warm up", candidates[QUESTIONS[0][0]][:2])

        hits, reciprocal_ranks, seconds = [], [], 0.0
        top_pages[name] = {}
        for question, correct_pages in QUESTIONS:
            start = time.perf_counter()
            ranked = reranker.rerank(question, candidates[question]) if reranker else candidates[question][:args.top_n]
            seconds += time.perf_counter() - start

            hit, rr = score(ranked, correct_pages)
            hits.append(hit)
            reciprocal_ranks.append(rr)
            top_pages[name][question] = [c.metadata["page"] for c in ranked]

        results[name] = (sum(hits) / len(hits), sum(reciprocal_ranks) / len(reciprocal_ranks), 1000 * seconds / len(QUESTIONS))

    # 3. Report ------------------------------------------------------------------
    print(f"{'reranker':<12} {'Hit@' + str(args.top_n):>7} {'MRR':>6} {'ms/q':>8}")
    print("-" * 36)
    for name, (hit_rate, mrr, ms) in results.items():
        print(f"{name:<12} {hit_rate:>7.0%} {mrr:>6.2f} {ms:>8.0f}")

    print("\nTop pages per question (correct pages in brackets):")
    for question, correct_pages in QUESTIONS:
        print(f"\n  {question}  {correct_pages}")
        for name in results:
            pages = top_pages[name][question]
            marks = " ".join(f"[{p}]" if p in correct_pages else f" {p} " for p in pages)
            print(f"    {name:<12} {marks}")

    store.client.delete_collection("eval_rerankers")


if __name__ == "__main__":
    main()
