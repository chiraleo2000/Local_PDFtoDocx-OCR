"""Regression tests for Docling adapter + table-drop fixes."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.docling_adapter import (
    docling_to_blocks,
    detections_from_docling,
    _suppress_text_in_structure,
    _looks_garbled_for_thai,
    _has_usable_thai,
    _ocr_text_block,
    _thai_reocr_enabled,
)
from src.pipeline import ContentBlock
from src.exporter import DocumentExporter

docx = pytest.importorskip("docx")
from docx import Document  # noqa: E402


class _BBox:
    def __init__(self, l, t, r, b):
        self.l, self.t, self.r, self.b = l, t, r, b

    def to_top_left_origin(self, page_h):
        return self


class _Prov:
    def __init__(self, page_no, bbox):
        self.page_no = page_no
        self.bbox = bbox


class _Label:
    def __init__(self, value):
        self.value = value


def _fake_docling_doc():
    """Minimal DoclingDocument-like object with text/table/picture."""
    text = SimpleNamespace(
        label=_Label("text"),
        text="วิเคราะห์มาตรฐานการผลิตหม่อนไหม",
        prov=[_Prov(1, _BBox(50, 40, 400, 80))],
    )
    table = SimpleNamespace(
        label=_Label("table"),
        text="",
        prov=[_Prov(1, _BBox(50, 120, 500, 300))],
        data=SimpleNamespace(
            num_cols=2,
            grid=[
                [SimpleNamespace(text="ชื่อ"), SimpleNamespace(text="จำนวน")],
                [SimpleNamespace(text="ข้าว"), SimpleNamespace(text="10")],
            ],
        ),
        export_to_html=lambda: (
            "<table><tr><th>ชื่อ</th><th>จำนวน</th></tr>"
            "<tr><td>ข้าว</td><td>10</td></tr></table>"
        ),
    )
    picture = SimpleNamespace(
        label=_Label("picture"),
        text="",
        prov=[_Prov(1, _BBox(60, 340, 260, 480))],
        image=None,
    )
    page = SimpleNamespace(size=SimpleNamespace(width=600, height=800))
    doc = SimpleNamespace(
        pages={1: page},
        texts=[text],
        tables=[table],
        pictures=[picture],
    )
    doc.iterate_items = lambda: [
        (text, 0), (table, 0), (picture, 0)]
    return doc


def test_docling_adapter_maps_table_figure_text():
    doc = _fake_docling_doc()
    blocks = docling_to_blocks(doc, ocr=None, page_images={}, languages="tha+eng")
    types = {b.block_type for b in blocks}
    assert "table" in types
    assert "figure" in types
    assert "text" in types
    table = next(b for b in blocks if b.block_type == "table")
    assert "<table>" in table.table_html
    assert "ข้าว" in table.table_html
    assert table.bbox == [50, 120, 500, 300]
    n_tables = sum(1 for b in blocks if b.block_type == "table")
    n_figures = sum(1 for b in blocks if b.block_type == "figure")
    assert n_tables == 1
    assert n_figures == 1


def test_detections_from_docling_match_adapter_counts():
    doc = _fake_docling_doc()
    dets = detections_from_docling(doc, page_idx=0)
    assert len(dets["tables"]) == 1
    assert len(dets["figures"]) == 1
    assert len(dets["text_regions"]) >= 1


def test_suppress_text_inside_table():
    blocks = [
        ContentBlock("text", 0, 130, 60, text="cell junk",
                     bbox=[60, 130, 200, 160], page_width=600, page_height=800),
        ContentBlock("table", 0, 120, 50, text="A\tB",
                     table_html="<table><tr><td>A</td><td>B</td></tr></table>",
                     bbox=[50, 120, 500, 300], page_width=600, page_height=800),
        ContentBlock("text", 0, 40, 50, text="Title outside",
                     bbox=[50, 40, 400, 80], page_width=600, page_height=800),
    ]
    out = _suppress_text_in_structure(blocks)
    texts = [b.text for b in out if b.block_type == "text"]
    assert "Title outside" in texts
    assert "cell junk" not in texts


def test_exporter_rebuilds_table_from_plain_text():
    """Absolute export must not drop tables that lack table_html."""
    blocks = [
        ContentBlock(
            "table", 0, 100, 50,
            text="Name\tQty\nRice\t10",
            table_html="",
            bbox=[50, 100, 500, 300],
            page_width=600, page_height=800,
        ),
    ]
    exp = DocumentExporter()
    files = exp.create_all_from_blocks(blocks, "Pages: 1", "A4", "Normal",
                                       layout_mode="absolute")
    doc = Document(files["docx"])
    assert len(doc.tables) == 1
    flat = "\n".join(c.text for row in doc.tables[0].rows for c in row.cells)
    assert "Rice" in flat or "Name" in flat


def test_layout_backend_env_default(monkeypatch):
    monkeypatch.delenv("LAYOUT_BACKEND", raising=False)
    from src.docling_backend import layout_backend
    assert layout_backend() == "docling"
    monkeypatch.setenv("LAYOUT_BACKEND", "yolo")
    assert layout_backend() == "yolo"


def test_garbled_latin_detected():
    # RapidOCR-on-Thai soup must be rejected
    assert _looks_garbled_for_thai(
        "COMMSSUBLMACLUNGMUNEUSLUOBLUMLABEMUI ENUCSH")
    # Real English / product names must be KEPT on tha+eng jobs
    assert not _looks_garbled_for_thai("Hello English only")
    assert not _looks_garbled_for_thai("Microsoft Windows Server")
    assert not _looks_garbled_for_thai(
        "วิเคราะห์และตรวจสอบมาตรฐานการผลิต")
    assert _has_usable_thai("ชื่อ\tจำนวน\nข้าว\t10", min_chars=3)
    assert not _has_usable_thai("A B C ข้าว", min_chars=6, min_density=0.25)


def test_thai_reocr_force_off(monkeypatch):
    monkeypatch.setenv("DOCLING_REOCR", "force-off")
    assert _thai_reocr_enabled("tha+eng") is False
    monkeypatch.setenv("DOCLING_REOCR", "0")
    assert _thai_reocr_enabled("tha+eng") is True


def test_ocr_text_block_refuses_latin_fallback():
    """Failed Thai-TrOCR must not keep RapidOCR Latin garbage."""
    ocr = MagicMock()
    ocr.ocr_image.return_value = {"text": "", "lines": []}
    ocr.ocr_full_page.return_value = {"text": "", "lines": []}
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    block = _ocr_text_block(
        ocr, img, [10, 10, 200, 40], 0, 400, 200, "tha+eng",
        fallback_text="COMMSSUBLMACLUNGMUNEUSLUOBLUMLABEMUI")
    assert block.text.strip() == ""
    assert not block.lines
