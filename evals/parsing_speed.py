"""
How fast is Docling with different settings? MEASURE instead of guessing.

Same 10 Bain pages (30-39: text, charts, photos), 3 settings:
  1. current      : OCR on every picture + accurate table model   (what we used, ~12 s/page)
  2. no OCR       : OCR off (these pages already contain real text)
  3. no OCR + fast: OCR off + fast table model

For each one we also check the result is still the same quality:
  characters of text extracted, number of tables found.

Run from the project root:   python -m evals.parsing_speed
(Takes a few minutes. Nothing is cached, nothing in the app changes.)
"""

import time

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import TableItem, TextItem

PDF = "assets/bain_report_technology_report_2025.pdf"
PAGES = (30, 39)


def make_options(ocr: bool, fast_tables: bool) -> PdfPipelineOptions:
    options = PdfPipelineOptions()
    options.do_ocr = ocr
    options.table_structure_options.mode = TableFormerMode.FAST if fast_tables else TableFormerMode.ACCURATE
    return options


SETTINGS = {
    "1. current":       make_options(ocr=True, fast_tables=False),
    "2. no OCR":        make_options(ocr=False, fast_tables=False),
    "3. no OCR + fast": make_options(ocr=False, fast_tables=True),
}

n_pages = PAGES[1] - PAGES[0] + 1
results = []
for name, options in SETTINGS.items():
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    converter.initialize_pipeline(InputFormat.PDF)   # load the models BEFORE the stopwatch starts

    print(f"running {name} ...")
    start = time.perf_counter()
    document = converter.convert(PDF, page_range=PAGES).document
    seconds = time.perf_counter() - start

    items = [item for item, _ in document.iterate_items()]
    characters = sum(len(i.text) for i in items if isinstance(i, TextItem))
    tables = sum(isinstance(i, TableItem) for i in items)
    results.append((name, seconds, characters, tables))

print("\n" + "=" * 70)
print(f"Bain pages {PAGES[0]}-{PAGES[1]} ({n_pages} pages)")
print("=" * 70)
print(f"  {'setting':<18} {'total':>7} {'per page':>9} {'text chars':>11} {'tables':>7}")
for name, seconds, characters, tables in results:
    print(f"  {name:<18} {seconds:>6.0f}s {seconds / n_pages:>8.1f}s {characters:>11} {tables:>7}")
print("\n  -> same text chars + tables = no quality lost, only time saved")
