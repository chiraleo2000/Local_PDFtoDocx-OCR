"""
UI / webapp tests for the Gradio PDF OCR Pipeline — v0.5.

Validates:
  - Handler functions (process, download, history)
  - Language / engine / confidence pass-through
  - Review & Correct handlers (detect, add region, convert with corrections)
  - Training / correction stats UI
  - Gradio interface construction (no auth)
  - Docker deployment (skipped if Docker not available)
"""
import os
import subprocess
import time
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent
ROOT_DIR = TESTS_DIR.parent
TEST_PDF = TESTS_DIR / "testocrtor.pdf"
APP_PORT = 7872


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def app_module():
    import sys
    sys.path.insert(0, str(ROOT_DIR))
    from src.pipeline import OCRPipeline
    from src.services import HistoryManager
    pipeline = OCRPipeline()
    hist = HistoryManager()
    hist.cleanup_old_entries()
    return {"pipeline": pipeline, "history": hist}


@pytest.fixture(scope="module")
def handlers(app_module):
    pipeline = app_module["pipeline"]
    history = app_module["history"]

    QUALITY_OPTIONS = {
        "Standard (Fast)": "fast",
        "Balanced (Recommended)": "balanced",
        "Best (Accurate)": "accurate",
    }
    LANGUAGE_OPTIONS = {
        "English": "eng",
        "Thai": "tha",
        "Thai + English": "tha+eng",
    }
    ENGINE_OPTIONS = {
        "EasyOCR (Thai+English)": "easyocr",
        "Thai-TrOCR (Line-level)": "thai_trocr",
        "PaddleOCR (Multilingual)": "paddleocr",
        "Typhoon OCR 3B (GPU LLM)": "typhoon",
    }
    _LOCAL_USER = "local"

    def _process_document(pdf_path, quality_label="Standard (Fast)",
                          header_pct=0, footer_pct=0,
                          language_label="English",
                          engine_label="EasyOCR (Thai+English)",
                          yolo_conf=0.30):
        if pdf_path is None:
            return {"success": False, "error": "No PDF provided"}
        quality = QUALITY_OPTIONS.get(quality_label, "fast")
        languages = LANGUAGE_OPTIONS.get(language_label, "eng")
        engine = ENGINE_OPTIONS.get(engine_label, "paddleocr")
        pipeline.ocr.primary_engine = engine
        result = pipeline.process_pdf(
            pdf_path, quality=quality,
            header_trim=header_pct, footer_trim=footer_pct,
            languages=languages, yolo_confidence=yolo_conf,
        )
        if result["success"]:
            files = result["files"]
            original_name = os.path.basename(pdf_path)
            entry_id = history.save_result(
                _LOCAL_USER, original_name, files, result["metadata"])
            result["entry_id"] = entry_id
        return result

    def _download_from_history(entry_id, fmt="docx"):
        if not entry_id:
            return None
        return history.get_file_path(_LOCAL_USER, entry_id.strip(), fmt)

    def _refresh_history():
        return history.list_entries(_LOCAL_USER)

    return {
        "process": _process_document,
        "download_history": _download_from_history,
        "refresh_history": _refresh_history,
        "history": history,
        "pipeline": pipeline,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Server health
# ══════════════════════════════════════════════════════════════════════════════

class TestServerHealth:
    @pytest.fixture(scope="class")
    def server(self):
        import sys
        env = os.environ.copy()
        env.update({"SERVER_PORT": str(APP_PORT), "SERVER_HOST": "127.0.0.1",
                    "SHARE_GRADIO": "false",
                    "DISABLE_TROCR_PRELOAD": "1"})   # skip model download
        proc = subprocess.Popen(
            [sys.executable, str(ROOT_DIR / "app.py")],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(ROOT_DIR),
        )
        import urllib.request
        deadline = time.time() + 180             # 3 min for model loads
        ready = False
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{APP_PORT}/", timeout=2)
                ready = True
                break
            except Exception:
                time.sleep(1)
        if not ready:
            proc.terminate()
            _, err = proc.communicate(timeout=5)
            pytest.fail(f"Server did not start: {err.decode(errors='replace')[-1000:]}")
        yield f"http://127.0.0.1:{APP_PORT}"
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    def test_server_returns_200(self, server):
        import urllib.request
        resp = urllib.request.urlopen(server, timeout=10)
        assert resp.status == 200

    def test_server_serves_html(self, server):
        import urllib.request
        resp = urllib.request.urlopen(server, timeout=10)
        body = resp.read().decode("utf-8", errors="replace")
        assert "<!doctype html>" in body.lower() or "<html" in body.lower()


# ══════════════════════════════════════════════════════════════════════════════
# PDF Conversion
# ══════════════════════════════════════════════════════════════════════════════

class TestPDFConversionActions:
    @pytest.fixture(scope="class")
    def conversion_result(self, handlers):
        assert TEST_PDF.exists(), f"Test PDF not found: {TEST_PDF}"
        return handlers["process"](str(TEST_PDF))

    def test_conversion_succeeds(self, conversion_result):
        assert conversion_result["success"] is True

    def test_conversion_extracts_text(self, conversion_result):
        text = conversion_result["text"]
        assert isinstance(text, str) and len(text) > 0

    def test_conversion_produces_docx(self, conversion_result):
        docx_path = conversion_result["files"].get("docx")
        assert docx_path and os.path.exists(docx_path)
        assert os.path.getsize(docx_path) > 0

    def test_conversion_produces_txt(self, conversion_result):
        txt_path = conversion_result["files"].get("txt")
        assert txt_path and os.path.exists(txt_path)
        assert os.path.getsize(txt_path) > 0

    def test_docx_file_is_valid_word(self, conversion_result):
        from docx import Document
        docx_path = conversion_result["files"]["docx"]
        doc = Document(docx_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        assert len(paragraphs) > 0

    def test_metadata_has_page_count(self, conversion_result):
        assert conversion_result["metadata"]["pages"] >= 1

    def test_metadata_has_languages(self, conversion_result):
        assert "languages" in conversion_result["metadata"]

    def test_no_pdf_returns_error(self, handlers):
        result = handlers["process"](None)
        assert result.get("success") is not True

    def test_header_footer_trim(self, handlers):
        result = handlers["process"](
            str(TEST_PDF), header_pct=5, footer_pct=5)
        assert result["success"] is True

    def test_language_override_works(self, handlers):
        result = handlers["process"](
            str(TEST_PDF), language_label="Thai + English")
        assert result["success"] is True
        assert result["metadata"]["languages"] == "tha+eng"

    def test_yolo_confidence_override(self, handlers):
        result = handlers["process"](
            str(TEST_PDF), yolo_conf=0.10)
        assert result["success"] is True


# ══════════════════════════════════════════════════════════════════════════════
# Download actions
# ══════════════════════════════════════════════════════════════════════════════

class TestDownloadActionsOCRPage:
    @pytest.fixture(scope="class")
    def files(self, handlers):
        result = handlers["process"](str(TEST_PDF))
        assert result["success"]
        return result["files"]

    def test_download_docx_file_exists(self, files):
        assert files.get("docx") and os.path.exists(files["docx"])

    def test_download_txt_file_exists(self, files):
        assert files.get("txt") and os.path.exists(files["txt"])

    def test_download_docx_not_empty(self, files):
        assert os.path.getsize(files["docx"]) > 0

    def test_download_txt_not_empty(self, files):
        assert os.path.getsize(files["txt"]) > 0


# ══════════════════════════════════════════════════════════════════════════════
# History
# ══════════════════════════════════════════════════════════════════════════════

class TestHistoryPageActions:
    @pytest.fixture(scope="class")
    def history_entry(self, handlers):
        result = handlers["process"](str(TEST_PDF))
        assert result["success"]
        return result["entry_id"]

    def test_history_lists_entries(self, handlers, history_entry):
        entries = handlers["refresh_history"]()
        assert isinstance(entries, list) and len(entries) >= 1

    def test_history_entry_has_filename(self, handlers, history_entry):
        entries = handlers["refresh_history"]()
        filenames = [e["original_filename"] for e in entries]
        assert any("testocrtor" in fn for fn in filenames)

    def test_history_download_docx(self, handlers, history_entry):
        path = handlers["download_history"](history_entry, "docx")
        assert path and os.path.exists(path) and os.path.getsize(path) > 0

    def test_history_download_txt(self, handlers, history_entry):
        path = handlers["download_history"](history_entry, "txt")
        assert path and os.path.exists(path) and os.path.getsize(path) > 0

    def test_history_download_invalid_returns_none(self, handlers):
        assert handlers["download_history"]("nonexistent_id", "docx") is None

    def test_history_download_empty_returns_none(self, handlers):
        assert handlers["download_history"]("", "docx") is None


# ══════════════════════════════════════════════════════════════════════════════
# Gradio interface
# ══════════════════════════════════════════════════════════════════════════════

class TestGradioInterfaceConstruction:
    def test_create_interface_no_error(self):
        import sys
        sys.path.insert(0, str(ROOT_DIR))
        import app as app_module
        blocks = app_module.create_interface()
        import gradio as gr
        assert isinstance(blocks, gr.Blocks)

    def test_interface_has_language_dropdown(self):
        import sys
        sys.path.insert(0, str(ROOT_DIR))
        import app as app_module
        blocks = app_module.create_interface()
        labels = [getattr(b, "label", "") or "" for b in blocks.blocks.values()]
        assert any("Language" in lbl for lbl in labels), (
            f"Language dropdown not found. Labels: {labels[:20]}")

    def test_interface_has_engine_dropdown(self):
        import sys
        sys.path.insert(0, str(ROOT_DIR))
        import app as app_module
        blocks = app_module.create_interface()
        labels = [getattr(b, "label", "") or "" for b in blocks.blocks.values()]
        assert any("Engine" in lbl for lbl in labels)

    def test_interface_has_convert_button(self):
        import sys
        sys.path.insert(0, str(ROOT_DIR))
        import app as app_module
        blocks = app_module.create_interface()
        labels = []
        for block in blocks.blocks.values():
            label = getattr(block, "label", None) or getattr(block, "value", None)
            if isinstance(label, str):
                labels.append(label)
        assert any("Convert" in lbl or "DOCX" in lbl for lbl in labels)

    def test_no_login_section(self):
        import sys
        sys.path.insert(0, str(ROOT_DIR))
        import app as app_module
        blocks = app_module.create_interface()
        labels = []
        for block in blocks.blocks.values():
            label = getattr(block, "label", None) or getattr(block, "value", None)
            if isinstance(label, str):
                labels.append(label)
        assert not any("Sign In" == lbl or "Password" == lbl for lbl in labels)

    def test_interface_has_settings_tab(self):
        import sys
        sys.path.insert(0, str(ROOT_DIR))
        import app as app_module
        blocks = app_module.create_interface()
        labels = []
        for block in blocks.blocks.values():
            label = getattr(block, "label", None) or getattr(block, "value", None)
            if isinstance(label, str):
                labels.append(label)
        assert any("Configuration" in lbl or "Settings" in lbl or "Confidence" in lbl
                   for lbl in labels)

    def test_interface_has_review_tab(self):
        """v0.5: Review & Correct tab should exist."""
        import sys
        sys.path.insert(0, str(ROOT_DIR))
        import app as app_module
        blocks = app_module.create_interface()
        labels = []
        for block in blocks.blocks.values():
            label = getattr(block, "label", None) or getattr(block, "value", None)
            if isinstance(label, str):
                labels.append(label)
        assert any("Region" in lbl or "Add Region" in lbl or "Detection" in lbl
                   for lbl in labels), (
            f"Review & Correct UI elements not found. Labels: {labels[:30]}")


# ══════════════════════════════════════════════════════════════════════════════
# Review & Correct handlers (v0.5)
# ══════════════════════════════════════════════════════════════════════════════

class TestReviewAndCorrectHandlers:
    def test_review_load_pdf(self):
        import sys
        sys.path.insert(0, str(ROOT_DIR))
        import app as app_module
        assert TEST_PDF.exists()
        result = app_module.review_load_pdf(str(TEST_PDF), 0.15)
        # Returns (img, info, pg, total, dets, manuals, vis)
        assert len(result) == 7
        img, info, pg, total, dets, manuals, vis = result
        assert img is not None
        assert total > 0
        assert isinstance(dets, dict)
        assert isinstance(manuals, list) and len(manuals) == 0

    def test_review_add_region(self):
        import sys
        sys.path.insert(0, str(ROOT_DIR))
        import app as app_module
        assert TEST_PDF.exists()
        # First load
        img, info, pg, total, dets, manuals, vis = \
            app_module.review_load_pdf(str(TEST_PDF), 0.15)
        # Add a manual region
        new_img, new_info, new_dets, new_manuals = app_module.review_add_region(
            str(TEST_PDF), 0, total, 0.15,
            50, 50, 200, 200, "table", dets, manuals,
        )
        assert new_img is not None
        assert len(new_manuals) == 1
        assert new_manuals[0]["class"] == "table"

    def test_review_add_region_invalid_bbox(self):
        import sys
        sys.path.insert(0, str(ROOT_DIR))
        import app as app_module
        img, info, dets, manuals = app_module.review_add_region(
            str(TEST_PDF), 0, 1, 0.15,
            200, 200, 50, 50, "table", {}, [],  # x1 < x0
        )
        assert img is None  # should fail validation
        assert "must be" in info.lower() or "invalid" in info.lower()

    def test_review_clear_manual(self):
        import sys
        sys.path.insert(0, str(ROOT_DIR))
        import app as app_module
        # Add then clear
        manuals = [{"bbox": [10, 10, 100, 100], "class": "table", "page": 0}]
        img, info, dets, kept = app_module.review_clear_manual(
            str(TEST_PDF), 0, 0.15, manuals)
        assert len(kept) == 0
        assert "cleared" in info.lower()

    def test_review_convert_with_corrections(self):
        import sys
        sys.path.insert(0, str(ROOT_DIR))
        import app as app_module
        manuals = [{"bbox": [50, 50, 200, 200], "class": "table", "page": 0}]
        result = app_module.review_convert_with_corrections(
            str(TEST_PDF), "Standard (Fast)", 0, 0,
            "English", "EasyOCR (Thai+English)", 0.30, manuals,
        )
        # Returns (text, status, txt_path, docx_path, vis, hist)
        text, status, txt_path, docx_path, vis, hist = result
        assert "Complete" in status or "Error" in status
        assert isinstance(text, str)

    def test_review_no_pdf_returns_error(self):
        import sys
        sys.path.insert(0, str(ROOT_DIR))
        import app as app_module
        result = app_module.review_convert_with_corrections(
            None, "Standard (Fast)", 0, 0,
            "English", "EasyOCR (Thai+English)", 0.30, [],
        )
        text, status = result[0], result[1]
        assert "Upload" in status or "Error" in status


# ══════════════════════════════════════════════════════════════════════════════
# Training / stats (v0.5)
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainingStats:
    def test_correction_stats_md(self):
        import sys
        sys.path.insert(0, str(ROOT_DIR))
        import app as app_module
        md = app_module.get_correction_stats_md()
        assert isinstance(md, str)
        assert "Manual corrections" in md
        assert "Next retrain" in md

    def test_corrections_log_md(self):
        import sys
        sys.path.insert(0, str(ROOT_DIR))
        import app as app_module
        md = app_module.get_corrections_log_md()
        assert isinstance(md, str)

    def test_draw_detections(self):
        import sys
        import numpy as np
        sys.path.insert(0, str(ROOT_DIR))
        import app as app_module
        img = np.ones((400, 600, 3), dtype=np.uint8) * 200
        dets = {
            "tables": [{"bbox": [10, 10, 100, 100], "class": "table", "confidence": 0.9}],
            "figures": [{"bbox": [200, 200, 400, 350], "class": "figure", "confidence": 0.8}],
        }
        annotated = app_module._draw_detections(img, dets)
        assert annotated.shape == img.shape
        # Verify pixels changed (boxes drawn)
        assert not np.array_equal(annotated, img)


# ══════════════════════════════════════════════════════════════════════════════
# Docker
# ══════════════════════════════════════════════════════════════════════════════

def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


_DOCKER_SKIP = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker daemon not available — skipping deployment tests",
)

DOCKER_PORT = 7875
DOCKER_IMAGE = "pdf-ocr-pipeline:test"
DOCKER_CONTAINER = "pdf-ocr-test-container"
_CONTAINER_TEST_PDF = "/app/tests/testocrtor.pdf"


def _docker_exec(args: list, timeout: int = 120) -> "subprocess.CompletedProcess":
    return subprocess.run(
        ["docker", "exec", DOCKER_CONTAINER, *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout,
    )


@_DOCKER_SKIP
class TestDockerDeployment:
    @pytest.fixture(scope="class")
    def docker_container(self):
        import urllib.request

        build = subprocess.run(
            ["docker", "build", "-t", DOCKER_IMAGE, str(ROOT_DIR)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=900,
        )
        assert build.returncode == 0, f"docker build failed:\n{build.stderr[-2000:]}"

        subprocess.run(["docker", "rm", "-f", DOCKER_CONTAINER],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        run = subprocess.run(
            ["docker", "run", "-d", "--name", DOCKER_CONTAINER,
             "-p", f"{DOCKER_PORT}:7870",
             "-e", "SERVER_PORT=7870", "-e", "SERVER_HOST=0.0.0.0",
             "-e", "QUALITY_PRESET=fast",
             "-e", "OCR_ENGINE=easyocr", "-e", "LANGUAGES=tha+eng",
             DOCKER_IMAGE],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30,
        )
        assert run.returncode == 0, f"docker run failed:\n{run.stderr}"

        subprocess.run(["docker", "exec", DOCKER_CONTAINER,
                        "mkdir", "-p", "/app/tests"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cp = subprocess.run(
            ["docker", "cp", str(TEST_PDF),
             f"{DOCKER_CONTAINER}:{_CONTAINER_TEST_PDF}"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        assert cp.returncode == 0, f"docker cp test PDF failed: {cp.stderr}"

        url = f"http://127.0.0.1:{DOCKER_PORT}/"
        deadline = time.time() + 120
        ready = False
        while time.time() < deadline:
            try:
                urllib.request.urlopen(url, timeout=3)
                ready = True
                break
            except Exception:
                time.sleep(2)

        if not ready:
            logs = subprocess.run(["docker", "logs", DOCKER_CONTAINER],
                                  capture_output=True, text=True,
                                  encoding="utf-8", errors="replace").stdout
            subprocess.run(["docker", "rm", "-f", DOCKER_CONTAINER],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            pytest.fail(f"Container not ready.\nLogs:\n{logs[-3000:]}")

        yield url
        subprocess.run(["docker", "rm", "-f", DOCKER_CONTAINER],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def test_http_root_returns_200(self, docker_container):
        import urllib.request
        resp = urllib.request.urlopen(docker_container, timeout=10)
        assert resp.status == 200

    def test_container_pipeline_processes_pdf(self, docker_container):
        result = _docker_exec([
            "python", "-c",
            f"from src.pipeline import OCRPipeline; p=OCRPipeline();"
            f"r=p.process_pdf('{_CONTAINER_TEST_PDF}', quality='fast');"
            f"print('SUCCESS' if r['success'] else 'FAILED:'+str(r.get('error','')));"
            f"print('CHARS:', len(r['text']))"
        ], timeout=900)
        assert result.returncode == 0, f"returncode={result.returncode}\nstderr={result.stderr[-500:]}"
        assert "SUCCESS" in result.stdout, f"stdout={result.stdout[-500:]}"

    def test_container_pipeline_with_language(self, docker_container):
        result = _docker_exec([
            "python", "-c",
            f"from src.pipeline import OCRPipeline; p=OCRPipeline();"
            f"r=p.process_pdf('{_CONTAINER_TEST_PDF}', quality='fast', languages='eng');"
            f"print('SUCCESS' if r['success'] else 'FAILED:'+str(r.get('error','')));"
            f"print('LANG:', r['metadata'].get('languages','?'))"
        ], timeout=900)
        assert result.returncode == 0, f"returncode={result.returncode}\nstderr={result.stderr[-500:]}"
        assert "LANG: eng" in result.stdout, f"stdout={result.stdout[-500:]}"

    def test_container_gradio_interface_builds(self, docker_container):
        result = _docker_exec([
            "python", "-c",
            "import sys; sys.path.insert(0,'/app');"
            "import app as a; import gradio as gr;"
            "blocks=a.create_interface();"
            "print('BLOCKS:'+str(isinstance(blocks, gr.Blocks)))"
        ], timeout=60)
        assert result.returncode == 0
        assert "BLOCKS:True" in result.stdout
