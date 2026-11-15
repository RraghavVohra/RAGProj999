"""
Image parsing module for our modular RAG pipeline.

Problem: pypdf only reads TEXT. Charts, tables drawn as pictures, diagrams
and scanned pages are invisible to it, so questions about them can't be answered.

Solution: "look" at the page the way a human would.

    1. find the pages that contain visual content
    2. render each of those pages to a PNG image (PyMuPDF)
    3. ask a vision LLM to describe the charts/tables/diagrams, with all their numbers
    4. store that description as an extra chunk, tagged  type="figure" + page number

Why render the WHOLE PAGE instead of pulling out the embedded images?
Many reports (like the Bain one) draw their charts as vector graphics,
not as image files. Page 34 of the Bain report has 400+ drawing commands
and zero images. Extracting images would miss it completely. A rendered
page also includes the figure title and axis labels, which are plain text
sitting next to the image.

Scanned pages (no text layer at all) get a different prompt: transcribe
everything. Their text then goes through normal chunking like any other page.

Descriptions are cached on disk, so rebuilding a pipeline with the same
PDF doesn't pay for the same vision calls again.
"""

import base64
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from openai import OpenAI

from utils.chunking import Chunk

load_dotenv()

CACHE_DIR = Path(__file__).parent.parent / "cache" / "vision"

# --- When does a page count as "visual"? -------------------------------------
MIN_IMAGE_SHARE = 0.05      # an image covering >= 5% of the page (skips small logos and icons)
MIN_VECTOR_DRAWINGS = 30    # >= 30 lines/shapes usually means a vector chart or diagram
SCANNED_MAX_CHARS = 20      # a page with (almost) no text but some content = scanned page
RENDER_DPI = 150            # sharp enough to read chart labels, small enough to send quickly
NOTHING = "NONE"            # what the model answers when a page has nothing useful to describe

FIGURE_PROMPT = """This is page {page} of a PDF document. Describe every chart, graph, table, diagram or infographic on this page, so the description can be found by search and used to answer questions.
- Start with the figure title (if any) and what the figure shows.
- List the key numbers, labels, categories, axes, legends and trends exactly as shown.
- For tables, write out the rows and columns as text.
- Ignore logos, decorative photos, page numbers and running headers/footers.
- Do not repeat the page's ordinary body text paragraphs.
If the page has no informative visual content, reply with exactly: NONE"""

SCAN_PROMPT = """This is page {page} of a scanned PDF. Transcribe ALL the text on it exactly, in reading order.
If there are charts or tables, describe them too, including their numbers.
If the page is blank, reply with exactly: NONE"""


def classify_page(page) -> str | None:
    """Return "scanned", "visual" or None (plain text page, nothing to do)."""
    has_text = len(page.get_text().strip()) > SCANNED_MAX_CHARS
    page_area = page.rect.width * page.rect.height

    big_image = any(
        (info["bbox"][2] - info["bbox"][0]) * (info["bbox"][3] - info["bbox"][1]) / page_area >= MIN_IMAGE_SHARE
        for info in page.get_image_info()
    )
    vector_graphics = len(page.get_drawings()) >= MIN_VECTOR_DRAWINGS

    if not has_text:
        return "scanned" if (big_image or vector_graphics or page.get_images()) else None
    if big_image or vector_graphics:
        return "visual"
    return None


class VisionDescriber:
    """Sends one rendered page image to an OpenAI vision model and returns its description."""

    def __init__(self, model_name: str = "gpt-5.6-luna"):
        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def describe(self, png_bytes: bytes, prompt: str) -> str:
        image_b64 = base64.b64encode(png_bytes).decode()
        response = self.client.responses.create(
            model=self.model_name,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{image_b64}"},
                ],
            }],
        )
        return response.output_text.strip()


def _cache_file(pdf_path: Path, model_name: str) -> Path:
    """One cache file per (PDF content, model). Same file uploaded again = cache hit."""
    pdf_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]
    safe_model = model_name.replace("/", "_")
    return CACHE_DIR / f"{pdf_hash}_{safe_model}.json"


def parse_visual_pages(pdf_path: str | Path, model_name: str = "gpt-5.6-luna", max_workers: int = 6):
    """
    Returns two lists of Chunks:
      scanned_pages -> text transcribed from scanned pages (to be chunked like normal page text)
      figures       -> one description chunk per visual page (kept whole, never split)
    """
    pdf_path = Path(pdf_path)
    document = pymupdf.open(pdf_path)

    # 1. Which pages need the vision model? ---------------------------------------
    todo = []   # (page_number, kind)
    for page in document:
        kind = classify_page(page)
        if kind:
            todo.append((page.number + 1, kind))
    print(f"[images] {pdf_path.name}: {len(todo)} of {len(document)} pages have visual content")

    # 2. Describe them (cached results first, then the rest in parallel) ------------
    cache_file = _cache_file(pdf_path, model_name)
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    missing = [(number, kind) for number, kind in todo if str(number) not in cache]

    if missing:
        describer = VisionDescriber(model_name)

        # Render all pages first, here in the main thread (PyMuPDF isn't thread-safe) ...
        jobs = []
        for number, kind in missing:
            png = document[number - 1].get_pixmap(dpi=RENDER_DPI).tobytes("png")
            prompt = (SCAN_PROMPT if kind == "scanned" else FIGURE_PROMPT).format(page=number)
            jobs.append((number, png, prompt))

        # ... then only the slow network calls run in parallel (6 at a time)
        def describe_job(job):
            number, png, prompt = job
            try:
                return str(number), describer.describe(png, prompt), None
            except Exception as error:   # one failed page shouldn't throw away the others
                return str(number), None, error

        errors = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for page_key, description, error in pool.map(describe_job, jobs):
                if error:
                    errors.append(error)
                    print(f"[images] page {page_key} failed: {error!r}")
                else:
                    cache[page_key] = description   # only successes are cached, failures are retried next build

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")

        if errors and len(errors) == len(jobs):   # nothing worked: bad API key, model without vision, no internet...
            raise RuntimeError(f"Image parsing failed for every page: {errors[0]}")
        print(f"[images] described {len(jobs) - len(errors)} page(s) with {model_name} "
              f"({len(todo) - len(missing)} from cache, {len(errors)} failed)")

    # 3. Turn the descriptions into Chunks --------------------------------------
    scanned_pages, figures = [], []
    for number, kind in todo:
        description = cache.get(str(number), "").strip()
        if not description or description.upper().startswith(NOTHING):
            continue   # decorative photo / logo only

        metadata = {"page": number, "source": pdf_path.name}
        if kind == "scanned":
            scanned_pages.append(Chunk(text=description, metadata={**metadata, "strategy": "page-wise", "type": "text"}))
        else:
            figures.append(Chunk(
                text=f"[Visual content on page {number}]\n{description}",
                metadata={**metadata, "type": "figure"},
            ))
    return scanned_pages, figures


# ---------------------------------------------------------------------------
# Quick manual check (run from project root:  python -m utils.image_parsing)
# Step 1 is free; step 2 calls the OpenAI vision model (needs OPENAI_API_KEY).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    pdf_path = Path(__file__).parent.parent / "assets" / "bain_report_technology_report_2025.pdf"

    print("Pages that would be sent to the vision model:")
    for page in pymupdf.open(pdf_path):
        kind = classify_page(page)
        if kind:
            print(f"  page {page.number + 1:<3} {kind:<8} images={len(page.get_images())}  drawings={len(page.get_drawings())}")

    if "--page" in sys.argv:   # python -m utils.image_parsing --page 34   -> test ONE page (one API call)
        number = int(sys.argv[sys.argv.index("--page") + 1])
        page = pymupdf.open(pdf_path)[number - 1]

        png = page.get_pixmap(dpi=RENDER_DPI).tobytes("png")          # what we send = exactly this picture
        preview = CACHE_DIR.parent / f"page_{number}.png"
        preview.parent.mkdir(parents=True, exist_ok=True)
        preview.write_bytes(png)
        print(f"\nRendered page {number}: {len(png) // 1024} KB  -> saved to {preview} (open it to see what the AI sees)")

        print(f"Asking the vision model ...\n")
        description = VisionDescriber().describe(png, FIGURE_PROMPT.format(page=number))
        print("=" * 60)
        print(description)

    if "--describe" in sys.argv:   # python -m utils.image_parsing --describe   -> ALL visual pages
        _, figures = parse_visual_pages(pdf_path)
        for figure in figures[:5]:
            print("\n" + "=" * 60)
            print(figure.text[:800])
