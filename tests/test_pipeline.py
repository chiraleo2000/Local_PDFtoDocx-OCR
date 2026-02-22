"""
Backend unit / integration tests for the OCR pipeline.
Tests the pipeline, services, preprocessor, OCR engine, and exporter.
"""
import os
import shutil
import tempfile
from pathlib import Path

import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

TESTS_DIR = Path(__file__).parent
TEST_PDF = TESTS_DIR / "testocrtor.pdf"


@pytest.fixture(scope="module")
def pipeline():
    from src.pipeline import OCRPipeline
    return OCRPipeline()


@pytest.fixture(scope="module")
def pipeline_result(pipeline):
    assert TEST_PDF.exists(), f"Test PDF not found at {TEST_PDF}"
    result = pipeline.process_pdf(str(TEST_PDF), quality="fast")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline tests
# ══════════════════════════════════════════════════════════════════════════════

class TestOCRPipeline:
    def test_process_pdf_success(self, pipeline_result):
        """Pipeline should succeed on the test PDF."""
        assert pipeline_result["success"] is True
        assert pipeline_result["error"] is None

    def test_process_pdf_has_text(self, pipeline_result):
        """Extracted text should be non-empty."""
        assert isinstance(pipeline_result["text"], str)
        assert len(pipeline_result["text"]) > 0

    def test_process_pdf_metadata(self, pipeline_result):
        """Metadata should contain expected keys."""
        meta = pipeline_result["metadata"]
        assert "pages" in meta
        assert meta["pages"] >= 1
        assert "tables" in meta
        assert "figures" in meta

    def test_process_pdf_files_dict(self, pipeline_result):
        """Files dict should contain paths for txt and docx."""
        files = pipeline_result["files"]
        assert "txt" in files
        assert "docx" in files

    def test_process_pdf_txt_exists(self, pipeline_result):
        """Generated TXT file should exist on disk."""
        txt_path = pipeline_result["files"].get("txt")
        assert txt_path is not None
        assert os.path.exists(txt_path), f"TXT file not found: {txt_path}"
        assert os.path.getsize(txt_path) > 0

    def test_process_pdf_docx_exists(self, pipeline_result):
        """Generated DOCX file should exist on disk."""
        docx_path = pipeline_result["files"].get("docx")
        assert docx_path is not None
        assert os.path.exists(docx_path), f"DOCX file not found: {docx_path}"
        assert os.path.getsize(docx_path) > 0

    def test_process_pdf_docx_readable(self, pipeline_result):
        """Generated DOCX should be a valid Word document."""
        from docx import Document
        docx_path = pipeline_result["files"].get("docx")
        doc = Document(docx_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        assert len(paragraphs) > 0

    def test_process_pdf_invalid_path(self, pipeline):
        """Pipeline should return success=False for a missing file."""
        result = pipeline.process_pdf("/nonexistent/file.pdf")
        assert result["success"] is False
        assert result["error"] is not None

    def test_get_status(self, pipeline):
        """get_status should return a dict with expected keys."""
        status = pipeline.get_status()
        assert "ocr_engines" in status
        assert "tesseract" in status["ocr_engines"]


# ══════════════════════════════════════════════════════════════════════════════
# OCR Engine tests
# ══════════════════════════════════════════════════════════════════════════════

class TestOCREngine:
    def test_available_engines(self):
        from src.ocr_engine import OCREngine
        engine = OCREngine()
        engines = engine.get_available_engines()
        assert isinstance(engines, dict)
        assert "tesseract" in engines

    def test_tesseract_available(self):
        from src.ocr_engine import TESSERACT_AVAILABLE
        assert TESSERACT_AVAILABLE, "Tesseract should be available in the test environment"

    def test_ocr_image(self):
        """OCR should return text from a simple image."""
        import numpy as np
        import cv2
        from src.ocr_engine import OCREngine

        engine = OCREngine()
        # Create a white image with black text pattern
        img = np.ones((100, 400), dtype=np.uint8) * 255
        cv2.putText(img, "Hello OCR", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 0, 0), 2)
        result = engine.ocr_image(img)
        assert isinstance(result, dict)
        assert "text" in result
        assert "confidence" in result
        assert "engine_used" in result

    def test_is_available(self):
        from src.ocr_engine import OCREngine
        engine = OCREngine()
        assert engine.is_available() is True


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessor tests
# ══════════════════════════════════════════════════════════════════════════════

class TestOpenCVPreprocessor:
    def test_preprocess_fast(self):
        import numpy as np
        from src.preprocessor import OpenCVPreprocessor

        pp = OpenCVPreprocessor(quality="fast")
        img = np.ones((200, 300, 3), dtype=np.uint8) * 200
        out = pp.preprocess(img)
        assert out is not None
        assert out.size > 0

    def test_preprocess_balanced(self):
        import numpy as np
        from src.preprocessor import OpenCVPreprocessor

        pp = OpenCVPreprocessor(quality="balanced")
        img = np.ones((200, 300, 3), dtype=np.uint8) * 200
        out = pp.preprocess(img)
        assert out is not None
        assert out.ndim in (2, 3)

    def test_preprocess_accurate(self):
        import numpy as np
        from src.preprocessor import OpenCVPreprocessor

        pp = OpenCVPreprocessor(quality="accurate")
        img = np.ones((200, 300, 3), dtype=np.uint8) * 200
        out = pp.preprocess(img)
        assert out is not None


# ══════════════════════════════════════════════════════════════════════════════
# Exporter tests
# ══════════════════════════════════════════════════════════════════════════════

class TestDocumentExporter:
    SAMPLE_TEXT = "Hello, World!\n\nThis is a test document.\n## Section\nSome content."

    def test_create_txt(self):
        from src.exporter import DocumentExporter
        exp = DocumentExporter()
        path = exp.create_txt(self.SAMPLE_TEXT, metadata="Test metadata")
        assert os.path.exists(path)
        content = Path(path).read_text(encoding="utf-8")
        assert "Hello, World!" in content
        assert "Test metadata" in content
        os.unlink(path)

    def test_create_docx(self):
        from src.exporter import DocumentExporter
        from docx import Document
        exp = DocumentExporter()
        path = exp.create_docx(self.SAMPLE_TEXT)
        assert path is not None
        assert os.path.exists(path)
        doc = Document(path)
        full_text = "\n".join(p.text for p in doc.paragraphs)
        assert "Hello, World!" in full_text
        os.unlink(path)

    def test_create_html(self):
        from src.exporter import DocumentExporter
        exp = DocumentExporter()
        path = exp.create_html(self.SAMPLE_TEXT)
        assert os.path.exists(path)
        content = Path(path).read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "Hello, World!" in content
        os.unlink(path)

    def test_create_all(self):
        from src.exporter import DocumentExporter
        exp = DocumentExporter()
        files = exp.create_all(self.SAMPLE_TEXT)
        assert "txt" in files
        assert "docx" in files
        assert "html" in files
        for fmt, path in files.items():
            if path:
                assert os.path.exists(path), f"{fmt} file not found"
                os.unlink(path)


# ══════════════════════════════════════════════════════════════════════════════
# Auth & History tests
# ══════════════════════════════════════════════════════════════════════════════

class TestAuthManager:
    def test_default_guest_login(self):
        from src.services import AuthManager
        auth = AuthManager()
        result = auth.login("guest", "guest123")
        assert result["success"] is True
        assert "token" in result
        assert result["username"] == "guest"

    def test_invalid_login(self):
        from src.services import AuthManager
        auth = AuthManager()
        result = auth.login("guest", "wrongpassword")
        assert result["success"] is False

    def test_register_and_login(self):
        import uuid
        from src.services import AuthManager
        auth = AuthManager()
        uname = f"testuser_{uuid.uuid4().hex[:8]}"
        reg = auth.register(uname, "pass1234")
        assert reg["success"] is True
        login = auth.login(uname, "pass1234")
        assert login["success"] is True

    def test_session_validate(self):
        from src.services import AuthManager
        auth = AuthManager()
        login = auth.login("guest", "guest123")
        token = login["token"]
        username = auth.validate_session(token)
        assert username == "guest"

    def test_logout_invalidates_session(self):
        from src.services import AuthManager
        auth = AuthManager()
        login = auth.login("guest", "guest123")
        token = login["token"]
        auth.logout(token)
        assert auth.validate_session(token) is None


class TestHistoryManager:
    def test_save_and_list(self, tmp_path, monkeypatch):
        import tempfile
        from src.services import HistoryManager

        monkeypatch.setenv("HISTORY_RETENTION_DAYS", "30")
        # Patch tempdir so history goes into tmp_path
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

        hist = HistoryManager()
        # Create dummy output files
        txt_file = tmp_path / "out.txt"
        docx_file = tmp_path / "out.docx"
        txt_file.write_text("hello")
        docx_file.write_bytes(b"PK\x03\x04")  # minimal docx/zip header

        entry_id = hist.save_result(
            "testuser", "sample.pdf",
            {"txt": str(txt_file), "docx": str(docx_file)},
            {"pages": 1}
        )
        assert entry_id is not None

        entries = hist.list_entries("testuser")
        assert len(entries) >= 1
        assert entries[0]["original_filename"] == "sample.pdf"

    def test_get_file_path(self, tmp_path, monkeypatch):
        import tempfile
        from src.services import HistoryManager

        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

        hist = HistoryManager()
        txt_file = tmp_path / "out2.txt"
        docx_file = tmp_path / "out2.docx"
        txt_file.write_text("text content")
        docx_file.write_bytes(b"PK\x03\x04")

        entry_id = hist.save_result(
            "user2", "doc.pdf",
            {"txt": str(txt_file), "docx": str(docx_file)},
        )
        docx_path = hist.get_file_path("user2", entry_id, "docx")
        txt_path = hist.get_file_path("user2", entry_id, "txt")
        assert docx_path is not None and os.path.exists(docx_path)
        assert txt_path is not None and os.path.exists(txt_path)
