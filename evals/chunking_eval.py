"""
Full chunking test: which strategy + settings retrieves best?

  8 chunking setups  x  2 embedding models  x  22 questions
  (no reranker, no LLM -> we measure the effect of chunking alone)

THE ANSWER KEY
Every question has a short answer phrase (a regex). A chunk "contains the answer" if the phrase
is inside it. Pages are NOT fixed by hand: the script finds every page that contains the phrase
itself, so facts that the report repeats (e.g. in "At a Glance" summaries) count automatically.

METRICS (per setup)
  Hit@1 / Hit@3 / Hit@5  % of questions where a chunk WITH the answer is in the top 1 / 3 / 5
  MRR                    average of 1/rank of the first chunk with the answer (1.0 = always #1)
  ctx                    average characters in the top 3 chunks = how much text the LLM must read
  end@3                  Hit@3 only on the 7 questions whose answer is at the END of a long page

Run from the project root:   python -m evals.chunking_eval
Local models only (free). Takes ~10 minutes: every setup x every embedder.

Only some setups / one embedder (much faster):
    python -m evals.chunking_eval --setups "recursive 500" "recursive 500/150" "character 500" --embedders BAAI/bge-small-en-v1.5
"""

import argparse
import re
import time

from utils.chunking import page_chunking
from utils.embedding import SentenceTransformerEmbedder
from utils.pipeline import ChunkingConfig, PipelineConfig, chunk_pdf
from utils.retrieval import Retriever
from utils.vector_store import VectorStore

PDF = "assets/bain_report_technology_report_2025.pdf"
EMBEDDERS = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]
TOP_K = 5

SETUPS = {
    "page-wise":     ChunkingConfig(strategy="page-wise"),
    "character 500": ChunkingConfig(strategy="character-wise", chunk_size=500, overlap=50),
    "recursive 300": ChunkingConfig(strategy="recursive", chunk_size=300, overlap=50),
    "recursive 500": ChunkingConfig(strategy="recursive", chunk_size=500, overlap=50),
    "recursive 500/150": ChunkingConfig(strategy="recursive", chunk_size=500, overlap=150),   # overlap test
    "recursive 800": ChunkingConfig(strategy="recursive", chunk_size=800, overlap=100),
    "semantic 80":   ChunkingConfig(strategy="semantic", breakpoint_percentile=80),
    "semantic 90":   ChunkingConfig(strategy="semantic", breakpoint_percentile=90),
    "semantic 95":   ChunkingConfig(strategy="semantic", breakpoint_percentile=95),
}

# (question, answer phrase as regex, where the answer sits)
#   top / middle / end = position on the page,  chart = text from a figure,  paraphrase = different words than the report
QUESTIONS = [
    ("How much annual revenue is needed to fund new data centers by 2030?", r"\$2 trillion in annual revenue", "top"),
    ("How much is OpenAI valued at?", r"\$300 billion", "middle"),
    ("How much has China invested in semiconductor manufacturing since 2019?", r"250 billion in semiconductor", "top"),
    ("How much unspent capital do tech-focused private equity funds have?", r"\$476 billion", "paraphrase"),
    ("What share of value creation since 2010 came from revenue growth?", r"53% of the total value", "middle"),
    ("How quickly did the price of OpenAI's o3 model fall?", r"dropped 80%", "paraphrase"),
    ("How much global compute capacity could AI need by 2030?", r"200 gigawatts", "top"),
    ("How much venture capital did humanoid robots attract in 2024?", r"\$2\.5 billion", "middle"),
    ("How big is the market for quantum computing hardware and services today?", r"less than \$1 billion", "middle"),
    ("What percentage of tech leaders have a plan for quantum cybersecurity risk?", r"only 9%", "chart"),
    ("How much of their time do salespeople actually spend selling?", r"25% of their time", "end"),
    ("How much old or inaccurate content may need to be removed before using AI?", r"as much as 80%", "middle"),
    ("What share of the time from idea to product launch is spent writing and testing code?", r"25% to 35%", "top"),
    ("What productivity gains come from combining generative AI with end-to-end process redesign?", r"25% to 30%", "middle"),
    ("How should software companies change their pricing for AI agents?", r"outcome-based|based on outcomes", "end"),
    ("What share of North American private equity deals went to technology in 2025?", r"22%", "chart"),
    ("How fast is AI compute demand growing each year?", r"4\.5x", "chart"),
    ("Who is the Senior Operations Analyst credited in the report?", r"Raghav Sharma", "end"),
    ("How many employees does Mistral have?", r"fewer than 500 employees", "end"),
    ("Is the development of new quantum algorithms speeding up or slowing down?", r"QA development has slowed", "end"),
    ("What is still holding back humanoid robots in handling objects?", r"fine-motor control", "end"),
    ("How should leaders reassure engineers about AI coding tools?", r"not a replacement", "end"),
]


parser = argparse.ArgumentParser()
parser.add_argument("--setups", nargs="+", choices=list(SETUPS), help="run only these setups")
parser.add_argument("--embedders", nargs="+", help="run only these embedding models")
args = parser.parse_args()
if args.setups:
    SETUPS = {name: SETUPS[name] for name in args.setups}
if args.embedders:
    EMBEDDERS = args.embedders


def flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)   # line breaks inside a phrase must not hide it


def has_answer(chunk_text: str, pattern: str) -> bool:
    return re.search(pattern, flat(chunk_text)) is not None


# --- 0. Check the answer key against the PDF -------------------------------------------
pages = page_chunking(PDF)
print("ANSWER KEY (pages that contain each answer, found automatically)")
for question, pattern, kind in QUESTIONS:
    found = [p.metadata["page"] for p in pages if has_answer(p.text, pattern)]
    assert found, f"answer {pattern!r} is not in the PDF - fix the answer key"
    print(f"  {kind:<10} pages {str(found):<10} {question}")

# --- 1. Run every setup with every embedder ---------------------------------------------
results = {}   # (embedder, setup) -> metrics
ranks = {}     # (embedder, setup) -> rank of the first answer chunk per question (None = not in top 5)

for embedder_name in EMBEDDERS:
    embedder = SentenceTransformerEmbedder(embedder_name)
    for setup_name, chunking in SETUPS.items():
        start = time.perf_counter()
        chunks = chunk_pdf(PDF, PipelineConfig(chunking=chunking), embedder)

        collection = "eval_chunking"
        store = VectorStore(collection)
        store.client.delete_collection(collection)   # fresh index for every setup
        store = VectorStore(collection)
        for i in range(0, len(chunks), 100):
            batch = chunks[i:i + 100]
            store.add_chunks(batch, embedder.embed([c.text for c in batch]))
        retriever = Retriever(embedder, store, top_k=TOP_K)

        question_ranks, ctx_chars = [], []
        for question, pattern, kind in QUESTIONS:
            top = retriever.retrieve(question)
            rank = next((r for r, c in enumerate(top, start=1) if has_answer(c.text, pattern)), None)
            question_ranks.append(rank)
            ctx_chars.append(sum(len(c.text) for c in top[:3]))
        store.client.delete_collection(collection)

        n = len(QUESTIONS)
        end_questions = [r for r, (_, _, kind) in zip(question_ranks, QUESTIONS) if kind == "end"]
        results[(embedder_name, setup_name)] = {
            "chunks": len(chunks),
            "hit1": sum(r == 1 for r in question_ranks if r) / n,
            "hit3": sum(r <= 3 for r in question_ranks if r) / n,
            "hit5": sum(1 for r in question_ranks if r) / n,
            "mrr": sum(1 / r for r in question_ranks if r) / n,
            "ctx": sum(ctx_chars) // n,
            "end": sum(1 for r in end_questions if r and r <= 3) / len(end_questions),
            "secs": time.perf_counter() - start,
        }
        ranks[(embedder_name, setup_name)] = question_ranks
        m = results[(embedder_name, setup_name)]
        print(f"  done: {embedder_name:<24} {setup_name:<14} Hit@3 {m['hit3']:.0%}  ({m['secs']:.0f}s)")

# --- 2. Scoreboard ----------------------------------------------------------------------
for embedder_name in EMBEDDERS:
    print("\n" + "=" * 92)
    print(f"SCOREBOARD   embedder: {embedder_name}   ({len(QUESTIONS)} questions)")
    print("=" * 92)
    print(f"  {'setup':<14} {'chunks':>6} {'Hit@1':>6} {'Hit@3':>6} {'Hit@5':>6} {'MRR':>5} {'end@3':>6} {'ctx':>6}")
    for setup_name in SETUPS:
        m = results[(embedder_name, setup_name)]
        print(f"  {setup_name:<14} {m['chunks']:>6} {m['hit1']:>6.0%} {m['hit3']:>6.0%} {m['hit5']:>6.0%} "
              f"{m['mrr']:>5.2f} {m['end']:>6.0%} {m['ctx']:>6}")

# --- 3. Which questions did each setup miss? --------------------------------------------
for embedder_name in EMBEDDERS:
    print("\n" + "=" * 92)
    print(f"RANK OF THE ANSWER PER QUESTION   embedder: {embedder_name}   (- = not in top {TOP_K})")
    print("=" * 92)
    short = [name.replace("recursive", "rec").replace("semantic", "sem").replace("character", "char") for name in SETUPS]
    print(f"  {'':<10} {'question':<44}" + "".join(f"{s:>12}" for s in short))
    for i, (question, _, kind) in enumerate(QUESTIONS):
        cells = "".join(f"{ranks[(embedder_name, s)][i] or '-':>12}" for s in SETUPS)
        print(f"  {kind:<10} {question[:42]:<44}{cells}")
