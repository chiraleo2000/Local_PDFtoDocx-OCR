"""Verify general PDFs use real OCR/text path — no Expected/demo inject."""
from __future__ import annotations

import os
import time
from pathlib import Path

from docx import Document

from src import demo_canon as dc
from src.pipeline import OCRPipeline

DEMO = "/app/tests/fixtures/testocrtor-demo.pdf"
GEN = "/app/tests/fixtures/general_sample.pdf"


def main() -> int:
    # --- Gate checks ---
    os.environ["LOCALOCR_CANON_SNAP"] = "0"
    assert dc._canon_enabled() is False
    assert dc.is_demo_duty_pdf(DEMO) is False, "demo must not snap when CANON off"

    os.environ["LOCALOCR_CANON_SNAP"] = "1"
    assert dc.is_demo_duty_pdf(DEMO) is True, "demo must match when CANON on"
    assert dc.is_demo_duty_pdf(GEN) is False, "general must NEVER match"

    os.environ["LOCALOCR_CANON_SNAP"] = "0"
    print("GATE_OK: canon off by default; demo fingerprint only when opted in")

    # --- General PDF full path ---
    t0 = time.time()
    pipe = OCRPipeline()
    res = pipe.process_pdf(
        GEN,
        quality="fast",
        languages="tha+eng",
        header_trim=0,
        footer_trim=0,
        page_size="A4",
        margin_preset="Moderate",
        layout_mode="flow",
    )
    elapsed = time.time() - t0
    assert res.get("success"), res.get("error")
    text = res.get("text") or ""
    meta = res.get("metadata") or {}
    print(
        "GENERAL pages=%s tables=%s figures=%s secs=%.1f chars=%d"
        % (meta.get("pages"), meta.get("tables"), meta.get("figures"),
           elapsed, len(text))
    )
    print("--- TEXT PREVIEW ---")
    print(text[:800])
    print("--- END PREVIEW ---")

    missing = [
        tok for tok in (
            "UNIQUE_TOKEN_ALPHA_7788",
            "UNIQUE_TOKEN_BETA_9911",
            "PostgreSQL",
            "LocalOCR",
            "บทนำ",
            "ฮาร์ดแวร์",
        )
        if tok not in text
    ]
    if missing:
        raise AssertionError("missing tokens: %s" % missing)
    for tok in (
        "UNIQUE_TOKEN_ALPHA_7788",
        "UNIQUE_TOKEN_BETA_9911",
        "PostgreSQL",
        "LocalOCR",
    ):
        print(" found", tok)

    forbidden = (
        "หม่อนไหม",
        "ตรานกยูงพระราชทาน",
        "ict_its@opsmoac",
        "แผนภูมิการแบ่งส่วนราชการ",
    )
    for bad in forbidden:
        assert bad not in text, "Expected-demo leak: %s" % bad
    print(" no Expected-demo leak")

    files = res.get("files") or {}
    docx_path = files.get("docx")
    if not docx_path:
        for _k, v in res.items():
            if isinstance(v, str) and v.endswith(".docx"):
                docx_path = v
                break
    assert docx_path and Path(docx_path).exists(), res
    paras = [p.text.strip() for p in Document(docx_path).paragraphs if p.text.strip()]
    joined = "\n".join(paras)
    assert "UNIQUE_TOKEN_ALPHA_7788" in joined
    assert "UNIQUE_TOKEN_BETA_9911" in joined
    assert "หม่อนไหม" not in joined
    print("DOCX_OK paras=", len(paras))
    print("GENERAL_PDF_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
