"""
Document parsing module for our modular RAG pipeline: turns a PDF into clean text per page.

Two modes - the USER picks one in the app:

  FAST (default)   PyMuPDF + 5 general cleaning rules.        ~1 second for a whole book.
  ACCURATE         Docling, a layout-aware AI parser.          ~5 s per page the first time, then cached.

Why not just pypdf (what we used before)?
  It only pulls out raw text. Page numbers and running headers end up glued into every
  chunk ("33 Technology Report 2025 ..."), and some PDFs come out with tabs between
  every word or letters split apart ("P ENGUIN").

FAST mode - the 5 cleaning rules (general: they work for any PDF, not just ours)
  1. read each page as PARAGRAPHS (PyMuPDF "blocks"), blank line between paragraphs
  2. tabs -> space, special characters normalized (e.g. the ligature "ﬁ" -> "fi")
  3. words broken across lines are rejoined: "self-" + new line + "improvement" -> "self-improvement"
  4. REPEATED header/footer: a line in the top 2 / bottom 2 lines of a page that appears on
     30%+ of all pages (numbers ignored, so "33" and "34" count as the same line) is removed
  5. a line that is ONLY a number at the very top/bottom of a page (a page number) is removed
  What FAST can't do: tables come out as plain lines, multi-column pages may be read in the
  wrong order, scanned pages (photos of paper) have no text -> that's what ACCURATE is for.

ACCURATE mode - Docling
  AI models label every block on the page (paragraph, heading, list, table, picture, caption,
  page header, page footer) and put them in reading order. Headers/footers/page numbers go
  into a separate "furniture" layer, so we simply read only the "body" layer.
  OCR (reading text from images) is only switched on when the PDF has scanned pages:
  measured on 10 Bain pages, OCR made Docling 3x slower (16.3 -> 5.2 s/page without it)
  while adding only 6 characters of text.
"""

import hashlib
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf

from utils.chunking import Chunk

CACHE_DIR = Path(__file__).parent.parent / "cache" / "docling"

HEADER_FOOTER_SHARE = 0.30   # rule 4: a line on 30%+ of pages is a header/footer
EDGE_LINES = 2               # rule 4/5: only look at the top 2 and bottom 2 lines of a page
SCANNED_MAX_CHARS = 20       # a page with less text than this but with an image = probably a scan
SCANNED_SHARE = 0.5          # FAST mode refuses a PDF if half or more of its pages look scanned

PAGE_NUMBER = re.compile(r"^\d{1,4}$")


class ScannedPDFError(ValueError):
    """FAST mode found (almost) no text: the PDF is probably scanned -> use ACCURATE mode."""


def looks_scanned(page) -> bool:
    """No real text, but an image = a photo of paper. (Blank pages have no image, so they don't count.)"""
    return len(page.get_text().strip()) < SCANNED_MAX_CHARS and len(page.get_images()) > 0


# ===========================================================================
# FAST mode: PyMuPDF + cleaning rules
# ===========================================================================

def _clean_line(line: str) -> str:
    line = unicodedata.normalize("NFKC", line)   # rule 2: "ﬁ" -> "fi", odd spaces -> normal ones
    line = line.replace("\t", " ")
    return re.sub(r" {2,}", " ", line).strip()


def _join_lines(lines: list[str]) -> str:
    """Rule 3: one paragraph = one line of text; a word broken with a hyphen is glued back together."""
    text = ""
    for line in lines:
        if text.endswith("-") and line[:1].islower():
            text += line                  # "self-" + "improvement" -> "self-improvement"
        else:
            text += (" " if text else "") + line
    return text


def fast_page_texts(pdf_path: str | Path) -> list[Chunk]:
    pdf_path = Path(pdf_path)
    document = pymupdf.open(pdf_path)

    # Rule 1: every page as a list of paragraphs, every paragraph as a list of lines
    pages = []
    for page in document:
        paragraphs = []
        for block in page.get_text("blocks", sort=True):   # sort=True: top-to-bottom, left-to-right
            if block[6] != 0:                               # block type 0 = text (1 = image)
                continue
            lines = [_clean_line(line) for line in block[4].split("\n")]
            lines = [line for line in lines if line]
            if lines:
                paragraphs.append(lines)
        pages.append(paragraphs)

    if pages and sum(looks_scanned(page) for page in document) / len(pages) >= SCANNED_SHARE:
        raise ScannedPDFError(f"{pdf_path.name} looks scanned (pages are images, no real text). "
                              "Choose 'Accurate' parsing - it can read scanned pages.")

    # Rule 4: which top/bottom lines repeat on many pages?
    def pattern(line: str) -> str:
        return re.sub(r"\d+", "#", line)   # "33" and "34" -> "#", "Report 2025" -> "Report #"

    def edge_lines(paragraphs):
        flat = [line for paragraph in paragraphs for line in paragraph]
        return flat[:EDGE_LINES] + flat[-EDGE_LINES:]

    counts = Counter()
    for paragraphs in pages:
        counts.update({pattern(line) for line in edge_lines(paragraphs)})   # count each line once per page
    min_pages = max(3, HEADER_FOOTER_SHARE * len(pages))
    repeated = {p for p, n in counts.items() if n >= min_pages}

    # Rules 4 + 5: remove repeated headers/footers and page numbers - only at the top/bottom of a page
    chunks = []
    for page_number, paragraphs in enumerate(pages, start=1):
        flat = [(i, j) for i, paragraph in enumerate(paragraphs) for j in range(len(paragraph))]
        edges = set(flat[:EDGE_LINES] + flat[-EDGE_LINES:])
        kept = []
        for i, paragraph in enumerate(paragraphs):
            lines = [line for j, line in enumerate(paragraph)
                     if (i, j) not in edges or not (pattern(line) in repeated or PAGE_NUMBER.match(line))]
            if lines:
                kept.append(_join_lines(lines))
        text = "\n\n".join(kept)
        if text:
            chunks.append(Chunk(
                text=text,
                metadata={"strategy": "page-wise", "chunk_id": len(chunks), "page": page_number, "source": pdf_path.name},
            ))
    return chunks


# ===========================================================================
# ACCURATE mode: Docling
# ===========================================================================

_converters = {}   # loading Docling's models takes time -> one converter per OCR setting, reused


def _get_converter(ocr: bool):
    if ocr not in _converters:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        options = PdfPipelineOptions()
        options.do_ocr = ocr                     # table model stays on its default (accurate) mode
        print(f"[parsing] loading Docling models (OCR {'on' if ocr else 'off'}; first time downloads them)")
        _converters[ocr] = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    return _converters[ocr]


def parse_pdf(pdf_path: str | Path):
    """Read a PDF with Docling -> DoclingDocument (from cache/docling/ if parsed before with the same settings)."""
    from docling_core.types.doc import DoclingDocument

    pdf_path = Path(pdf_path)
    ocr = any(looks_scanned(page) for page in pymupdf.open(pdf_path))   # OCR only if there ARE scanned pages

    # Cache file name = PDF content + settings, so a result made with other settings is never reused
    pdf_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]
    cache_file = CACHE_DIR / f"{pdf_hash}_{'ocr' if ocr else 'noocr'}.json"
    if cache_file.exists():
        return DoclingDocument.load_from_json(cache_file)

    print(f"[parsing] {pdf_path.name}: parsing with Docling, OCR {'on' if ocr else 'off'} "
          f"(~5 s per page the first time, cached afterwards) ...")
    start = time.perf_counter()
    document = _get_converter(ocr).convert(pdf_path).document
    print(f"[parsing] {pdf_path.name}: done in {time.perf_counter() - start:.0f}s")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    document.save_as_json(cache_file)
    return document


def page_texts(document, source: str) -> list[Chunk]:
    """
    Clean text per page from a DoclingDocument, same shape as fast_page_texts().
      - only the BODY layer: page headers, footers and page numbers are left out
      - tables are written as Markdown tables (rows and columns stay readable)
      - blocks are separated by a blank line, so recursive chunking can cut between them
      - pictures are skipped here: their content comes from image parsing (utils/image_parsing.py)
    """
    from docling_core.types.doc import TableItem, TextItem

    blocks_per_page: dict[int, list[str]] = {}
    for item, _level in document.iterate_items():   # default = body layer only
        if isinstance(item, TableItem):
            text = item.export_to_markdown(doc=document)
        elif isinstance(item, TextItem):             # paragraphs, headings, list items, captions, ...
            text = item.text
        else:
            continue
        if not text.strip() or not item.prov:        # prov = where on which page the block was found
            continue
        blocks_per_page.setdefault(item.prov[0].page_no, []).append(text.strip())

    return [
        Chunk(
            text="\n\n".join(blocks),
            metadata={"strategy": "page-wise", "chunk_id": i, "page": page, "source": source},
        )
        for i, (page, blocks) in enumerate(sorted(blocks_per_page.items()))
    ]


def removed_blocks(document) -> list[tuple[int, str]]:
    """What Docling put in the 'furniture' layer (= left OUT of our text): (page, text) pairs."""
    from docling_core.types.doc import TextItem
    from docling_core.types.doc.document import ContentLayer

    return [
        (item.prov[0].page_no, item.text.strip())
        for item, _level in document.iterate_items(included_content_layers={ContentLayer.FURNITURE})
        if isinstance(item, TextItem) and item.prov
    ]


# ===========================================================================
# The one function the pipeline calls
# ===========================================================================

def read_pages(pdf_path: str | Path, mode: str = "fast") -> list[Chunk]:
    """Clean text per page. mode = "fast" (PyMuPDF + cleaning rules) or "accurate" (Docling)."""
    if mode == "accurate":
        return page_texts(parse_pdf(pdf_path), Path(pdf_path).name)
    return fast_page_texts(pdf_path)
