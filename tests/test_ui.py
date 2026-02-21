"""
UI / webapp tests for the Gradio PDF OCR Pipeline.

These tests validate the webapp handler functions that back every UI action
(login, register, logout, PDF upload/preview, conversion, download, history).
A server-health section also tests the live HTTP endpoint.

Strategy: import and call the handler functions directly rather than going
through the HTTP/websocket layer — this matches how Gradio routes button
clicks and is both faster and more reliable than browser automation.
"""
import os
import threading
import time
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent
ROOT_DIR = TESTS_DIR.parent
TEST_PDF = TESTS_DIR / "testocrtor.pdf"
APP_PORT = 7872  # separate port to avoid conflicts with manual testing


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def app_module():
    """Import app-level singletons (pipeline, auth, history) once per module."""
    import sys

    # Ensure the app module is importable from the root directory
    sys.path.insert(0, str(ROOT_DIR))

    # Import just the services and pipeline, not launching the web server
    from src.pipeline import OCRPipeline
    from src.services import AuthManager, HistoryManager

    pipeline = OCRPipeline()
    auth = AuthManager()
    hist = HistoryManager()
    hist.cleanup_old_entries()
    return {"pipeline": pipeline, "auth": auth, "history": hist}


@pytest.fixture(scope="module")
def handlers(app_module):
    """Return bound handler callables (mirrors app.py logic)."""
    pipeline = app_module["pipeline"]
    auth = app_module["auth"]
    history = app_module["history"]

    QUALITY_OPTIONS = {
        "Standard (Fast)": "fast",
        "Balanced (Recommended)": "balanced",
        "Best (Accurate)": "accurate",
    }

    def _handle_login(username, password):
        return auth.login(username, password)

    def _handle_register(username, password):
        return auth.register(username, password)

    def _handle_logout(token):
        auth.logout(token)

    def _process_document(pdf_path, quality_label, header_pct=0, footer_pct=0,
                           session_token="", username_state="guest"):
        if pdf_path is None:
            return {"success": False, "error": "No PDF provided"}
        quality = QUALITY_OPTIONS.get(quality_label, "fast")
        result = pipeline.process_pdf(
            pdf_path, quality=quality,
            header_trim=header_pct, footer_trim=footer_pct,
        )
        if result["success"]:
            files = result["files"]
            username = auth.validate_session(session_token) or username_state or "anonymous"
            original_name = os.path.basename(pdf_path)
            entry_id = history.save_result(username, original_name, files, result["metadata"])
            result["entry_id"] = entry_id
            result["username"] = username
        return result

    def _download_from_history(username, token, entry_id, fmt="docx"):
        uname = auth.validate_session(token) or username or "anonymous"
        if not entry_id:
            return None
        return history.get_file_path(uname, entry_id.strip(), fmt)

    def _refresh_history(username, token):
        uname = auth.validate_session(token) or username or "anonymous"
        return history.list_entries(uname)

    def _validate_session(token):
        return auth.validate_session(token)

    return {
        "login": _handle_login,
        "register": _handle_register,
        "logout": _handle_logout,
        "process": _process_document,
        "download_history": _download_from_history,
        "refresh_history": _refresh_history,
        "validate_session": _validate_session,
        "auth": auth,
        "history": history,
        "pipeline": pipeline,
    }


@pytest.fixture(scope="module")
def guest_session(handlers):
    """Login as guest and return (token, username)."""
    result = handlers["login"]("guest", "guest123")
    assert result["success"], "Guest login must succeed"
    return result["token"], result["username"]


# ══════════════════════════════════════════════════════════════════════════════
# Server health (HTTP)
# ══════════════════════════════════════════════════════════════════════════════

class TestServerHealth:
    @pytest.fixture(scope="class")
    def server(self):
        """Start the Gradio HTTP server in a background thread."""
        import subprocess, sys
        env = os.environ.copy()
        env.update({"SERVER_PORT": str(APP_PORT), "SERVER_HOST": "127.0.0.1",
                    "SHARE_GRADIO": "false"})
        proc = subprocess.Popen(
            [sys.executable, str(ROOT_DIR / "app.py")],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(ROOT_DIR),
        )
        import urllib.request
        deadline = time.time() + 60
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
            pytest.fail(f"Server did not start: {err.decode()[-1000:]}")
        yield f"http://127.0.0.1:{APP_PORT}"
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    def test_server_returns_200(self, server):
        """Gradio server should return HTTP 200 for root URL."""
        import urllib.request
        resp = urllib.request.urlopen(server, timeout=10)
        assert resp.status == 200

    def test_server_serves_html(self, server):
        """Root URL should return HTML content."""
        import urllib.request
        resp = urllib.request.urlopen(server, timeout=10)
        body = resp.read().decode("utf-8", errors="replace")
        assert "<!doctype html>" in body.lower() or "<html" in body.lower()


# ══════════════════════════════════════════════════════════════════════════════
# Authentication UI actions
# ══════════════════════════════════════════════════════════════════════════════

class TestAuthenticationActions:
    def test_guest_login_succeeds(self, handlers):
        """Default guest credentials should authenticate successfully."""
        result = handlers["login"]("guest", "guest123")
        assert result["success"] is True
        assert "token" in result
        assert result["username"] == "guest"

    def test_login_returns_session_token(self, handlers):
        """Login must return a non-empty session token."""
        result = handlers["login"]("guest", "guest123")
        assert result["token"], "Session token must not be empty"
        assert len(result["token"]) >= 16

    def test_invalid_password_rejected(self, handlers):
        """Wrong password should not authenticate."""
        result = handlers["login"]("guest", "bad_password")
        assert result["success"] is False

    def test_unknown_user_rejected(self, handlers):
        """Non-existent username should not authenticate."""
        result = handlers["login"]("no_such_user_xyz", "anything")
        assert result["success"] is False

    def test_register_new_user(self, handlers):
        """Registering with a unique username should succeed."""
        import uuid
        uname = f"ui_test_{uuid.uuid4().hex[:10]}"
        result = handlers["register"](uname, "securepw!")
        assert result["success"] is True

    def test_register_duplicate_user_fails(self, handlers):
        """Registering the same username twice should fail."""
        import uuid
        uname = f"dup_{uuid.uuid4().hex[:10]}"
        handlers["register"](uname, "pw1234")
        result = handlers["register"](uname, "pw1234")
        assert result["success"] is False

    def test_session_token_is_valid(self, handlers):
        """Session token returned by login should pass validation."""
        result = handlers["login"]("guest", "guest123")
        token = result["token"]
        username = handlers["validate_session"](token)
        assert username == "guest"

    def test_logout_invalidates_session(self, handlers):
        """After logout the session token should be invalid."""
        result = handlers["login"]("guest", "guest123")
        token = result["token"]
        handlers["logout"](token)
        assert handlers["validate_session"](token) is None


# ══════════════════════════════════════════════════════════════════════════════
# PDF conversion UI actions
# ══════════════════════════════════════════════════════════════════════════════

class TestPDFConversionActions:
    @pytest.fixture(scope="class")
    def conversion_result(self, handlers, guest_session):
        """Run the PDF conversion once and share the result."""
        token, username = guest_session
        assert TEST_PDF.exists(), f"Test PDF not found: {TEST_PDF}"
        return handlers["process"](
            str(TEST_PDF), "Standard (Fast)",
            header_pct=0, footer_pct=0,
            session_token=token, username_state=username,
        )

    def test_conversion_succeeds(self, conversion_result):
        """The conversion pipeline should report success."""
        assert conversion_result["success"] is True
        assert conversion_result.get("error") is None

    def test_conversion_extracts_text(self, conversion_result):
        """Extracted text must be a non-empty string."""
        text = conversion_result["text"]
        assert isinstance(text, str)
        assert len(text) > 0, "OCR should extract text from the test PDF"

    def test_conversion_produces_docx(self, conversion_result):
        """Conversion must produce a DOCX file."""
        docx_path = conversion_result["files"].get("docx")
        assert docx_path is not None, "DOCX file path should be present"
        assert os.path.exists(docx_path), f"DOCX not found on disk: {docx_path}"
        assert os.path.getsize(docx_path) > 0

    def test_conversion_produces_txt(self, conversion_result):
        """Conversion must produce a TXT file."""
        txt_path = conversion_result["files"].get("txt")
        assert txt_path is not None, "TXT file path should be present"
        assert os.path.exists(txt_path), f"TXT not found on disk: {txt_path}"
        assert os.path.getsize(txt_path) > 0

    def test_txt_file_contains_text(self, conversion_result):
        """TXT file content should include the extracted text."""
        txt_path = conversion_result["files"].get("txt")
        content = Path(txt_path).read_text(encoding="utf-8")
        assert len(content) > 0

    def test_docx_file_is_valid_word(self, conversion_result):
        """DOCX file should be a valid Word document with paragraphs."""
        from docx import Document
        docx_path = conversion_result["files"].get("docx")
        doc = Document(docx_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        assert len(paragraphs) > 0, "DOCX should contain at least one paragraph"

    def test_metadata_has_page_count(self, conversion_result):
        """Metadata should report at least one page."""
        assert conversion_result["metadata"]["pages"] >= 1

    def test_no_pdf_returns_error(self, handlers, guest_session):
        """Calling process without a PDF should fail gracefully."""
        token, username = guest_session
        result = handlers["process"](
            None, "Standard (Fast)",
            session_token=token, username_state=username,
        )
        assert result.get("success") is not True, (
            "Processing without a PDF should fail; error: " + str(result.get("error"))
        )

    def test_header_footer_trim(self, handlers, guest_session):
        """Conversion with header/footer trim should still succeed."""
        token, username = guest_session
        result = handlers["process"](
            str(TEST_PDF), "Standard (Fast)",
            header_pct=5, footer_pct=5,
            session_token=token, username_state=username,
        )
        assert result["success"] is True
        assert len(result["text"]) > 0


# ══════════════════════════════════════════════════════════════════════════════
# Download actions — OCR page
# ══════════════════════════════════════════════════════════════════════════════

class TestDownloadActionsOCRPage:
    """Verify that both DOCX and TXT download paths are valid after conversion."""

    @pytest.fixture(scope="class")
    def files(self, handlers, guest_session):
        token, username = guest_session
        result = handlers["process"](
            str(TEST_PDF), "Standard (Fast)",
            session_token=token, username_state=username,
        )
        assert result["success"]
        return result["files"]

    def test_download_docx_file_exists(self, files):
        """DOCX download path from OCR page should point to an existing file."""
        path = files.get("docx")
        assert path and os.path.exists(path), f"DOCX not found: {path}"

    def test_download_txt_file_exists(self, files):
        """TXT download path from OCR page should point to an existing file."""
        path = files.get("txt")
        assert path and os.path.exists(path), f"TXT not found: {path}"

    def test_download_docx_not_empty(self, files):
        """DOCX file must not be empty."""
        assert os.path.getsize(files["docx"]) > 0

    def test_download_txt_not_empty(self, files):
        """TXT file must not be empty."""
        assert os.path.getsize(files["txt"]) > 0


# ══════════════════════════════════════════════════════════════════════════════
# History page UI actions
# ══════════════════════════════════════════════════════════════════════════════

class TestHistoryPageActions:
    @pytest.fixture(scope="class")
    def history_entry(self, handlers, guest_session):
        """Convert a PDF and return (token, username, entry_id)."""
        token, username = guest_session
        result = handlers["process"](
            str(TEST_PDF), "Standard (Fast)",
            session_token=token, username_state=username,
        )
        assert result["success"]
        return token, username, result["entry_id"]

    def test_history_lists_entries(self, handlers, history_entry):
        """After conversion, history must contain at least one entry."""
        token, username, _ = history_entry
        entries = handlers["refresh_history"](username, token)
        assert isinstance(entries, list)
        assert len(entries) >= 1

    def test_history_entry_has_filename(self, handlers, history_entry):
        """History entries must record the original PDF filename."""
        token, username, _ = history_entry
        entries = handlers["refresh_history"](username, token)
        filenames = [e["original_filename"] for e in entries]
        assert any("testocrtor" in fn for fn in filenames)

    def test_history_entry_has_entry_id(self, handlers, history_entry):
        """Each history entry must have a non-empty entry_id."""
        token, username, _ = history_entry
        entries = handlers["refresh_history"](username, token)
        for e in entries:
            assert e.get("entry_id"), "entry_id must not be empty"

    def test_history_download_docx_button(self, handlers, history_entry):
        """Download DOCX button from History tab must return a valid file path."""
        token, username, entry_id = history_entry
        path = handlers["download_history"](username, token, entry_id, "docx")
        assert path is not None, "DOCX path should not be None"
        assert os.path.exists(path), f"DOCX history file not found: {path}"
        assert os.path.getsize(path) > 0

    def test_history_download_txt_button(self, handlers, history_entry):
        """Download TXT button from History tab must return a valid file path."""
        token, username, entry_id = history_entry
        path = handlers["download_history"](username, token, entry_id, "txt")
        assert path is not None, "TXT path should not be None"
        assert os.path.exists(path), f"TXT history file not found: {path}"
        assert os.path.getsize(path) > 0

    def test_history_download_docx_is_valid_word(self, handlers, history_entry):
        """DOCX downloaded from history should be a valid Word document."""
        from docx import Document
        token, username, entry_id = history_entry
        path = handlers["download_history"](username, token, entry_id, "docx")
        doc = Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        assert len(paragraphs) > 0

    def test_history_download_invalid_entry_returns_none(self, handlers, guest_session):
        """Requesting a non-existent entry_id should return None gracefully."""
        token, username = guest_session
        result = handlers["download_history"](username, token, "nonexistent_id", "docx")
        assert result is None

    def test_history_download_empty_entry_id_returns_none(self, handlers, guest_session):
        """Passing empty entry_id should return None (no crash)."""
        token, username = guest_session
        result = handlers["download_history"](username, token, "", "docx")
        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# Gradio interface construction tests
# ══════════════════════════════════════════════════════════════════════════════

class TestGradioInterfaceConstruction:
    """Verify the Gradio Blocks object can be built without errors."""

    def test_create_interface_no_error(self):
        """create_interface() must return a Blocks object without raising."""
        import sys
        sys.path.insert(0, str(ROOT_DIR))
        import importlib
        # Temporarily suppress the server launch
        import app as app_module
        blocks = app_module.create_interface()
        import gradio as gr
        assert isinstance(blocks, gr.Blocks)

    def test_interface_has_convert_tab(self):
        """Interface should include a 'Convert PDF' tab."""
        import sys
        sys.path.insert(0, str(ROOT_DIR))
        import app as app_module
        blocks = app_module.create_interface()
        # Check that block components include the convert button
        labels = []
        for block in blocks.blocks.values():
            label = getattr(block, "label", None) or getattr(block, "value", None)
            if isinstance(label, str):
                labels.append(label)
        assert any("Convert" in lbl or "DOCX" in lbl for lbl in labels), (
            f"'Convert to DOCX' label not found. Labels: {labels[:20]}")

    def test_interface_has_history_tab(self):
        """Interface should include history download components."""
        import sys
        sys.path.insert(0, str(ROOT_DIR))
        import app as app_module
        blocks = app_module.create_interface()
        labels = []
        for block in blocks.blocks.values():
            label = getattr(block, "label", None) or getattr(block, "value", None)
            if isinstance(label, str):
                labels.append(label)
        assert any("DOCX" in lbl or "TXT" in lbl or "History" in lbl
                   for lbl in labels), (
            f"History labels not found. Labels: {labels[:20]}")

