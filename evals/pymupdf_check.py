"""
What does PyMuPDF's raw text look like? (a quick look BEFORE writing any cleaning code)

For each PDF it prints:
  1. speed        - seconds to read all pages
  2. tabs         - tabs between words (pypdf gave 69,238 in the book)
  3. broken words - words split across lines with a hyphen ("self-" + new line + "improvement")
  4. repeated lines at the TOP and BOTTOM of pages (numbers replaced by #)
                    -> this is how headers / footers / page numbers can be found
  5. a few sample pages

Run from the project root:   python -m evals.pymupdf_check
Reads only - changes nothing, saves nothing.
"""

import re
import time
from collections import Counter

import pymupdf

PDFS = {
    "assets/bain_report_technology_report_2025.pdf": [35],
    "assets/Atomic Habits - James Clear.pdf": [4, 21],
}

for path, sample_pages in PDFS.items():
    start = time.perf_counter()
    document = pymupdf.open(path)
    texts = [page.get_text() for page in document]
    seconds = time.perf_counter() - start

    tabs = sum(text.count("\t") for text in texts)
    broken_words = sum(len(re.findall(r"[a-z]-\n[a-z]", text)) for text in texts)

    top_lines, bottom_lines = Counter(), Counter()
    for text in texts:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in lines[:2]:
            top_lines[re.sub(r"\d+", "#", line)] += 1      # "33" and "34" both become "#"
        for line in lines[-2:]:
            bottom_lines[re.sub(r"\d+", "#", line)] += 1

    print("\n" + "=" * 90)
    print(f"{path}")
    print("=" * 90)
    print(f"  1. speed        : {len(document)} pages in {seconds:.1f}s")
    print(f"  2. tabs         : {tabs}")
    print(f"  3. broken words : {broken_words}")
    print(f"  4. most common TOP lines    : {top_lines.most_common(4)}")
    print(f"     most common BOTTOM lines : {bottom_lines.most_common(4)}")
    for page in sample_pages:
        print(f"\n  5. page {page}: {texts[page - 1][:300]!r}")
