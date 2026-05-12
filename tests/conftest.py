"""Shared pytest setup for LocalOCR tests."""
import os
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
TEST_PDF = TESTS_DIR / "testocrtor.pdf"


def _set_test_environment() -> None:
    os.environ.setdefault("DISABLE_TROCR_PRELOAD", "1")
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("USE_GPU", "false")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("LOCALOCR_SKIP_HEAVY_IMPORTS", "1")
    os.environ.setdefault("LOCALOCR_ALLOW_LAYOUT_FALLBACK", "1")


def _ensure_test_pdf() -> None:
    if TEST_PDF.exists() and TEST_PDF.stat().st_size > 0:
        return

    import fitz

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(
        (72, 96),
        "LocalOCR generated test PDF\nHello OCR pipeline\nThis document has selectable text for tests.",
        fontsize=18,
        fontname="helv",
    )
    page.draw_rect(fitz.Rect(72, 190, 300, 260), color=(0, 0, 0), width=1)
    page.insert_text((84, 225), "Sample table area", fontsize=12, fontname="helv")
    doc.save(TEST_PDF)
    doc.close()


def pytest_sessionstart(session):
    session.config.cache.set("localocr/test_pdf", str(TEST_PDF))
    _set_test_environment()
    _ensure_test_pdf()