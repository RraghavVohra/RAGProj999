"""
Document parsing - BEFORE (pypdf) vs FAST (PyMuPDF + cleaning) vs ACCURATE (Docling), on both PDFs.

Part 1: time per mode
Part 2: the same pages side by side - is the noise gone?
Part 3: whole-document counts (header lines, page-number lines, tabs, broken words)
Part 4: (accurate only) what Docling LEFT OUT = its "furniture" layer (headers, footers, page numbers)

Run from the project root:
    python -m evals.parsing_demo               -> before + fast         (a few seconds)
    python -m evals.parsing_demo --accurate    -> also Docling          (minutes the first time, then cached)
"""

import re
import sys
import time
from collections import Counter
from pathlib import Path

from utils.chunking import page_chunking
from utils.parsing import fast_page_texts, page_texts, parse_pdf, removed_blocks

PDFS = {
    "assets/bain_report_technology_report_2025.pdf": [35, 34],   # page 35 started with "33 Technology Report 2025"
    "assets/Atomic Habits - James Clear.pdf": [4, 21],           # copyright page ("P ENGUIN" in pypdf), a text page
}
WITH_ACCURATE = "--accurate" in sys.argv


def show(text: str, n: int = 300) -> str:
    return repr(text[:n]) + (" ..." if len(text) > n else "")


def counts(pages: dict[int, str]) -> dict[str, int]:
    texts = list(pages.values())
    return {
        "pages with text": len(texts),
        "characters": sum(map(len, texts)),
        '"Technology Report 2025"': sum(t.count("Technology Report 2025") for t in texts),
        "lines that are only a number": sum(len(re.findall(r"(?m)^\s*\d{1,4}\s*$", t)) for t in texts),
        "tabs": sum(t.count("\t") for t in texts),
        "broken words (x-\\ny)": sum(len(re.findall(r"[a-z]-\n[a-z]", t)) for t in texts),
    }


for pdf, sample_pages in PDFS.items():
    name = Path(pdf).name
    print("\n" + "#" * 100)
    print(f"#  {name}")
    print("#" * 100)

    # --- Part 1: time per mode ----------------------------------------------------
    modes = {}
    timings = {}

    start = time.perf_counter()
    modes["before"] = {p.metadata["page"]: p.text for p in page_chunking(pdf)}
    timings["before"] = time.perf_counter() - start

    start = time.perf_counter()
    modes["fast"] = {p.metadata["page"]: p.text for p in fast_page_texts(pdf)}
    timings["fast"] = time.perf_counter() - start

    if WITH_ACCURATE:
        start = time.perf_counter()
        document = parse_pdf(pdf)
        modes["accurate"] = {p.metadata["page"]: p.text for p in page_texts(document, name)}
        timings["accurate"] = time.perf_counter() - start

    print("\nPART 1   time")
    for mode, seconds in timings.items():
        print(f"   {mode:<9} {seconds:>7.1f}s")

    # --- Part 2: same pages side by side --------------------------------------------
    print("\nPART 2   Same page in every mode")
    for page in sample_pages:
        print(f"\n   --- page {page} ---")
        for mode, pages in modes.items():
            print(f"   {mode.upper():<9}: {show(pages.get(page, '(no text)'))}")

    # --- Part 3: whole-document counts ----------------------------------------------
    print("\nPART 3   Whole document")
    table = {mode: counts(pages) for mode, pages in modes.items()}
    print(f"   {'':<32}" + "".join(f"{mode:>11}" for mode in modes))
    for key in table["before"]:
        print(f"   {key:<32}" + "".join(f"{table[mode][key]:>11}" for mode in modes))

    # --- Part 4: what Docling left out ------------------------------------------------
    if WITH_ACCURATE:
        removed = removed_blocks(document)
        print(f"\nPART 4   Docling left out {len(removed)} blocks. Most common:")
        for text, count in Counter(t for _, t in removed).most_common(6):
            print(f"   {count:>4} x  {text[:70]!r}")
