"""
Local embedding models - a small demo to SEE how they behave.

Part 1: one question, three sentences. Which sentence does each model think is closest?
Part 2: prefixes. Same BGE model, with and without its query prefix.
Part 3: length limit. How much of a page-wise chunk does each model actually read?

Run from the project root:   python -m evals.embedding_demo
(First run downloads the models from Hugging Face, after that it's offline.)
"""

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

QUESTION = "How much money is needed to build new data centers?"
SENTENCES = [
    "Building new data centers will require $500 billion of annual capital spending by 2030.",   # the answer
    "Data centers use huge amounts of electricity and water for cooling.",                     # same topic, wrong answer
    "Humanoid robots still struggle with tasks that need a precise grip.",                     # unrelated
]
MODELS = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]
BGE_PREFIX = "Represent this sentence for searching relevant passages: "


def similarities(model, question, sentences):
    """Cosine similarity between the question and each sentence (1.0 = same meaning)."""
    q = model.encode(question, normalize_embeddings=True)
    s = model.encode(sentences, normalize_embeddings=True)
    return s @ q   # dot product of normalized vectors = cosine similarity


models = {name: SentenceTransformer(name) for name in MODELS}

# --- Part 1 -------------------------------------------------------------------
print("=" * 70)
print(f"PART 1   Q: {QUESTION}")
print("=" * 70)
for name, model in models.items():
    print(f"\n{name}  (vector size {model.get_embedding_dimension()})")
    for sentence, score in zip(SENTENCES, similarities(model, QUESTION, SENTENCES)):
        print(f"   {score:.3f}  {sentence[:65]}")

# --- Part 2 -------------------------------------------------------------------
print("\n" + "=" * 70)
print("PART 2   BGE small: without vs with the query prefix")
print("=" * 70)
bge = models["BAAI/bge-small-en-v1.5"]
without = similarities(bge, QUESTION, SENTENCES)
with_prefix = similarities(bge, BGE_PREFIX + QUESTION, SENTENCES)
print(f"   {'without':>8} {'with':>8}")
for sentence, a, b in zip(SENTENCES, without, with_prefix):
    print(f"   {a:>8.3f} {b:>8.3f}  {sentence[:55]}")
print("   -> look at the GAP between the answer (row 1) and the wrong ones, not the raw numbers")

# --- Part 3 -------------------------------------------------------------------
print("\n" + "=" * 70)
print("PART 3   Length limit, using page 35 of the Bain report as one page-wise chunk")
print("=" * 70)
page_text = PdfReader("assets/bain_report_technology_report_2025.pdf").pages[34].extract_text()
for name, model in models.items():
    tokens = len(model.tokenizer(page_text)["input_ids"])
    limit = model.max_seq_length
    seen = min(1.0, limit / tokens)
    print(f"   {name:<25} page = {tokens} tokens, model reads {limit}  -> sees {seen:.0%} of the page")
