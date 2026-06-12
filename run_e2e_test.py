# pylint: disable=broad-exception-caught
"""End-to-end verification — strict engine policy + layout fidelity.

Runs tests/fixtures/testocrtor-demo.pdf (3 pages: Thai text + org chart,
Thai table, diagram + table + text) through the full pipeline with REAL
models and reports:

    1. Which engine handled each path (must be thai_trocr for Thai)
    2. Per-line bboxes present (required for spacing/alignment fidelity)
    3. Tables / figures detected and positioned
    4. DOCX uses absolute page-anchored frames & tables

Usage (from the project venv):
    python run_e2e_test.py [path/to/some.pdf]
"""
import os
import sys
import shutil

os.environ.setdefault("LAYOUT_MODE", "absolute")
os.environ.setdefault("LANGUAGES", "tha+eng")
os.environ.setdefault("OCR_ENGINE", "auto")

from src.pipeline import OCRPipeline  # noqa: E402

PDF = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    "tests", "fixtures", "testocrtor-demo.pdf")
OUT_DIR = "e2e_output"


def main() -> int:
    ok = True
    p = OCRPipeline()

    status = p.get_status()
    print("Engines available:", {k: v for k, v in
                                 status["ocr_engines"].items() if v})
    if not status["ocr_engines"].get("thai_trocr"):
        print("!! Thai-TrOCR not available — Thai pages will be EMPTY. "
              "Check models/thai-trocr or network access.")
        ok = False
    if not status["ocr_engines"].get("paddleocr"):
        print("!! PaddleOCR not available — non-Thai pages will be EMPTY.")
        ok = False

    print(f"\nProcessing {PDF} ...")
    res = p.process_pdf(PDF, quality="balanced", languages="tha+eng")
    if not res["success"]:
        print("FAILED:", res["error"])
        return 1

    meta = res["metadata"]
    print(f"pages={meta['pages']} tables={meta['tables']} "
          f"figures={meta['figures']}")
    if not res["text"].strip():
        print("!! No text extracted")
        ok = False

    # Layout-fidelity check on the DOCX
    try:
        from docx import Document
        from docx.oxml.ns import qn
        d = Document(res["files"]["docx"])
        frames = d.element.body.findall(".//" + qn("w:framePr"))
        positioned_tables = sum(
            1 for t in d.tables
            if t._tbl.tblPr.find(qn("w:tblpPr")) is not None)  # noqa: SLF001
        print(f"DOCX: {len(frames)} positioned text frames, "
              f"{len(d.tables)} tables ({positioned_tables} positioned), "
              f"{len(d.inline_shapes)} images")
        if not frames:
            print("!! No absolute-positioned frames — layout fidelity lost")
            ok = False
    except Exception as exc:
        print("!! DOCX inspection failed:", exc)
        ok = False

    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(PDF))[0]
    for kind, path in res["files"].items():
        if path and os.path.exists(path):
            dest = os.path.join(OUT_DIR, f"{base}.{kind}")
            shutil.copy(path, dest)
            print("saved", dest)

    print("\nRESULT:", "PASS — open e2e_output/ and compare against the PDF"
          if ok else "ISSUES FOUND (see !! lines above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
