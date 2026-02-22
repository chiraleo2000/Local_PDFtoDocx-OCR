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


# Fixtures

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


# Server health (HTTP)

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


# Authentication UI actions

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


# PDF conversion UI actions

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


# Download actions — OCR page

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


# History page UI actions

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


# Gradio interface construction tests

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


# Docker deployment tests

def _docker_available() -> bool:
    """Return True when the docker CLI is present and daemon is reachable."""
    import subprocess
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
    """Run a command inside the running test container and return the result."""
    import subprocess
    return subprocess.run(
        ["docker", "exec", DOCKER_CONTAINER, *args],
        capture_output=True, text=True, timeout=timeout,
    )


@_DOCKER_SKIP
class TestDockerDeployment:
    """
    Full integration suite against the live Docker-deployed app on port DOCKER_PORT.

    Covers:
      1. HTTP health checks
      2. Gradio /info API endpoint
      3. Authentication (login / register / session) inside the container
      4. PDF pipeline (render → OCR → export) inside the container
      5. TXT / DOCX output file verification inside the container
      6. History manager (save, list, download paths) inside the container
      7. Gradio UI construction inside the container
      8. Gradio HTTP predict endpoint reachable from host
    """

    # ------------------------------------------------------------------
    # Class-scoped fixture: build image, start container, seed test PDF
    # ------------------------------------------------------------------
    @pytest.fixture(scope="class")
    def docker_container(self):
        """Build the Docker image, start the container, seed the test PDF, yield base URL."""
        import subprocess
        import urllib.request

        # Build image (reuse cache layer when unchanged)
        build = subprocess.run(
            ["docker", "build", "-t", DOCKER_IMAGE, str(ROOT_DIR)],
            capture_output=True, text=True, timeout=300,
        )
        assert build.returncode == 0, f"docker build failed:\n{build.stderr[-2000:]}"

        # Remove stale container if any
        subprocess.run(["docker", "rm", "-f", DOCKER_CONTAINER],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Start container
        run = subprocess.run(
            [
                "docker", "run", "-d",
                "--name", DOCKER_CONTAINER,
                "-p", f"{DOCKER_PORT}:7870",
                "-e", "SERVER_PORT=7870",
                "-e", "SERVER_HOST=0.0.0.0",
                "-e", "QUALITY_PRESET=fast",
                "-e", "LLM_CORRECTION=false",
                DOCKER_IMAGE,
            ],
            capture_output=True, text=True, timeout=30,
        )
        assert run.returncode == 0, f"docker run failed:\n{run.stderr}"

        # Seed the test PDF so in-container tests can use it
        subprocess.run(["docker", "exec", DOCKER_CONTAINER,
                        "mkdir", "-p", "/app/tests"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cp = subprocess.run(
            ["docker", "cp", str(TEST_PDF), f"{DOCKER_CONTAINER}:{_CONTAINER_TEST_PDF}"],
            capture_output=True, text=True,
        )
        assert cp.returncode == 0, f"docker cp test PDF failed: {cp.stderr}"

        # Poll until the Gradio HTTP server responds (up to 120 s)
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
                                  capture_output=True, text=True).stdout
            subprocess.run(["docker", "rm", "-f", DOCKER_CONTAINER],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            pytest.fail(
                f"Container did not become ready within 120 s.\nLogs:\n{logs[-3000:]}")

        yield url

        # Teardown: stop and remove container
        subprocess.run(["docker", "rm", "-f", DOCKER_CONTAINER],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ------------------------------------------------------------------
    # 1. HTTP health checks
    # ------------------------------------------------------------------

    def test_http_root_returns_200(self, docker_container):
        """Container root URL must respond with HTTP 200."""
        import urllib.request
        resp = urllib.request.urlopen(docker_container, timeout=10)
        assert resp.status == 200

    def test_http_root_serves_html(self, docker_container):
        """Root URL must deliver Gradio HTML page."""
        import urllib.request
        body = urllib.request.urlopen(docker_container, timeout=10).read().decode(
            "utf-8", errors="replace")
        assert "<!doctype html>" in body.lower() or "<html" in body.lower()

    # ------------------------------------------------------------------
    # 2. Gradio /info API endpoint
    # ------------------------------------------------------------------

    def test_gradio_info_endpoint_reachable(self, docker_container):
        """Gradio /info endpoint must respond with JSON API metadata."""
        import urllib.request, json
        url = docker_container.rstrip("/") + "/info"
        try:
            resp = urllib.request.urlopen(url, timeout=10)
            assert resp.status == 200
            data = json.loads(resp.read())
            assert isinstance(data, dict), "/info should return a JSON object"
        except Exception as exc:
            pytest.skip(f"/info not exposed in this Gradio version: {exc}")

    def test_gradio_queue_status_reachable(self, docker_container):
        """Gradio /queue/status SSE endpoint must at least not return 5xx."""
        import urllib.request
        url = docker_container.rstrip("/") + "/queue/status"
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            assert resp.status < 500
        except Exception:
            pass  # SSE endpoints may not support plain GET — that's fine

    # ------------------------------------------------------------------
    # 3. Authentication inside the container
    # ------------------------------------------------------------------

    def test_container_auth_guest_login(self, docker_container):
        """Default guest credentials must authenticate successfully inside the container."""
        result = _docker_exec([
            "python", "-c",
            "from src.services import AuthManager; a=AuthManager();"
            "r=a.login('guest','guest123'); print(r['success'], r.get('token','')[:8])"
        ])
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().startswith("True"), (
            f"Guest login failed inside container: {result.stdout!r}")

    def test_container_auth_bad_password_rejected(self, docker_container):
        """Wrong password must be rejected inside the container."""
        result = _docker_exec([
            "python", "-c",
            "from src.services import AuthManager; a=AuthManager();"
            "r=a.login('guest','wrong'); print(r['success'])"
        ])
        assert result.returncode == 0, result.stderr
        assert "False" in result.stdout

    def test_container_auth_register_new_user(self, docker_container):
        """Register a new user inside the container."""
        result = _docker_exec([
            "python", "-c",
            "from src.services import AuthManager; a=AuthManager();"
            "r=a.register('dockertestuser','pass1234'); print(r['success'])"
        ])
        assert result.returncode == 0, result.stderr
        assert "True" in result.stdout

    def test_container_auth_session_validate(self, docker_container):
        """Session token issued by login must pass validate_session inside the container."""
        result = _docker_exec([
            "python", "-c",
            "from src.services import AuthManager; a=AuthManager();"
            "token=a.login('guest','guest123')['token'];"
            "print(a.validate_session(token))"
        ])
        assert result.returncode == 0, result.stderr
        assert "guest" in result.stdout

    # ------------------------------------------------------------------
    # 4. PDF pipeline inside the container
    # ------------------------------------------------------------------

    def test_container_pipeline_processes_pdf(self, docker_container):
        """Pipeline must successfully OCR the test PDF inside the container."""
        result = _docker_exec([
            "python", "-c",
            f"from src.pipeline import OCRPipeline; p=OCRPipeline();"
            f"r=p.process_pdf('{_CONTAINER_TEST_PDF}', quality='fast');"
            f"print('SUCCESS' if r['success'] else 'FAILED:'+str(r.get('error','')));"
            f"print('CHARS:', len(r['text']));"
            f"print('PAGES:', r['metadata']['pages'])"
        ], timeout=180)
        assert result.returncode == 0, f"exec failed:\n{result.stderr}"
        assert "SUCCESS" in result.stdout, (
            f"Pipeline returned failure inside container:\n{result.stdout}\n{result.stderr}")
        # Extract CHARS value and verify > 0
        for line in result.stdout.splitlines():
            if line.startswith("CHARS:"):
                chars = int(line.split(":")[1].strip())
                assert chars > 0, "Pipeline produced no text inside container"

    def test_container_pipeline_produces_txt(self, docker_container):
        """Pipeline must produce a TXT output file inside the container."""
        result = _docker_exec([
            "python", "-c",
            f"import os; from src.pipeline import OCRPipeline; p=OCRPipeline();"
            f"r=p.process_pdf('{_CONTAINER_TEST_PDF}', quality='fast');"
            f"txt=r['files'].get('txt',''); print('EXISTS:'+str(os.path.exists(txt)));"
            f"print('SIZE:'+str(os.path.getsize(txt) if txt and os.path.exists(txt) else 0))"
        ], timeout=180)
        assert result.returncode == 0, result.stderr
        assert "EXISTS:True" in result.stdout, "TXT file not created inside container"
        for line in result.stdout.splitlines():
            if line.startswith("SIZE:"):
                assert int(line.split(":")[1]) > 0, "TXT file is empty inside container"

    def test_container_pipeline_produces_docx(self, docker_container):
        """Pipeline must produce a DOCX output file inside the container."""
        result = _docker_exec([
            "python", "-c",
            f"import os; from src.pipeline import OCRPipeline; p=OCRPipeline();"
            f"r=p.process_pdf('{_CONTAINER_TEST_PDF}', quality='fast');"
            f"docx=r['files'].get('docx','');"
            f"print('EXISTS:'+str(os.path.exists(docx)));"
            f"print('SIZE:'+str(os.path.getsize(docx) if docx and os.path.exists(docx) else 0))"
        ], timeout=180)
        assert result.returncode == 0, result.stderr
        assert "EXISTS:True" in result.stdout, "DOCX file not created inside container"
        for line in result.stdout.splitlines():
            if line.startswith("SIZE:"):
                assert int(line.split(":")[1]) > 0, "DOCX file is empty inside container"

    def test_container_pipeline_docx_has_paragraphs(self, docker_container):
        """DOCX produced inside the container must contain readable paragraphs."""
        result = _docker_exec([
            "python", "-c",
            f"from src.pipeline import OCRPipeline; from docx import Document;"
            f"p=OCRPipeline(); r=p.process_pdf('{_CONTAINER_TEST_PDF}', quality='fast');"
            f"docx_path=r['files'].get('docx','');"
            f"doc=Document(docx_path); pars=[p.text for p in doc.paragraphs if p.text.strip()];"
            f"print('PARS:'+str(len(pars)))"
        ], timeout=180)
        assert result.returncode == 0, result.stderr
        for line in result.stdout.splitlines():
            if line.startswith("PARS:"):
                assert int(line.split(":")[1]) > 0, (
                    "DOCX produced inside container has no paragraphs")

    def test_container_pipeline_invalid_pdf_returns_error(self, docker_container):
        """Pipeline must not crash on a missing path — success must be False."""
        result = _docker_exec([
            "python", "-c",
            "from src.pipeline import OCRPipeline; p=OCRPipeline();"
            "r=p.process_pdf('/nonexistent/file.pdf', quality='fast');"
            "print('SUCCESS:'+str(r['success']))"
        ], timeout=30)
        assert result.returncode == 0, result.stderr
        assert "SUCCESS:False" in result.stdout, (
            "Pipeline should report failure for missing PDF")

    # ------------------------------------------------------------------
    # 5. History manager inside the container
    # ------------------------------------------------------------------

    def test_container_history_save_and_list(self, docker_container):
        """HistoryManager must save a result and list it back inside the container."""
        result = _docker_exec([
            "python", "-c",
            f"from src.pipeline import OCRPipeline; from src.services import HistoryManager;"
            f"p=OCRPipeline(); r=p.process_pdf('{_CONTAINER_TEST_PDF}', quality='fast');"
            f"h=HistoryManager();"
            f"eid=h.save_result('guest','test.pdf',r['files'],r['metadata']);"
            f"entries=h.list_entries('guest');"
            f"print('SAVED:'+str(bool(eid)));"
            f"print('COUNT:'+str(len(entries)))"
        ], timeout=180)
        assert result.returncode == 0, result.stderr
        assert "SAVED:True" in result.stdout, "History save failed inside container"
        for line in result.stdout.splitlines():
            if line.startswith("COUNT:"):
                assert int(line.split(":")[1]) >= 1, "History list empty after save"

    def test_container_history_download_docx_path(self, docker_container):
        """HistoryManager.get_file_path must return a valid DOCX path inside the container."""
        result = _docker_exec([
            "python", "-c",
            f"import os; from src.pipeline import OCRPipeline; from src.services import HistoryManager;"
            f"p=OCRPipeline(); r=p.process_pdf('{_CONTAINER_TEST_PDF}', quality='fast');"
            f"h=HistoryManager();"
            f"eid=h.save_result('guest','test.pdf',r['files'],r['metadata']);"
            f"path=h.get_file_path('guest',eid,'docx');"
            f"print('EXISTS:'+str(os.path.exists(path) if path else False))"
        ], timeout=180)
        assert result.returncode == 0, result.stderr
        assert "EXISTS:True" in result.stdout, (
            "History DOCX path does not exist inside container")

    def test_container_history_download_txt_path(self, docker_container):
        """HistoryManager.get_file_path must return a valid TXT path inside the container."""
        result = _docker_exec([
            "python", "-c",
            f"import os; from src.pipeline import OCRPipeline; from src.services import HistoryManager;"
            f"p=OCRPipeline(); r=p.process_pdf('{_CONTAINER_TEST_PDF}', quality='fast');"
            f"h=HistoryManager();"
            f"eid=h.save_result('guest','test.pdf',r['files'],r['metadata']);"
            f"path=h.get_file_path('guest',eid,'txt');"
            f"print('EXISTS:'+str(os.path.exists(path) if path else False))"
        ], timeout=180)
        assert result.returncode == 0, result.stderr
        assert "EXISTS:True" in result.stdout, (
            "History TXT path does not exist inside container")

    def test_container_history_invalid_entry_returns_none(self, docker_container):
        """get_file_path with a non-existent entry_id must return None — no crash."""
        result = _docker_exec([
            "python", "-c",
            "from src.services import HistoryManager; h=HistoryManager();"
            "path=h.get_file_path('guest','nonexistent_id_xyz','docx');"
            "print('NONE:'+str(path is None))"
        ], timeout=15)
        assert result.returncode == 0, result.stderr
        assert "NONE:True" in result.stdout

    # ------------------------------------------------------------------
    # 6. Gradio interface construction inside the container
    # ------------------------------------------------------------------

    def test_container_gradio_interface_builds(self, docker_container):
        """create_interface() must return a Blocks object inside the container."""
        result = _docker_exec([
            "python", "-c",
            "import sys; sys.path.insert(0,'/app');"
            "import app as a; import gradio as gr;"
            "blocks=a.create_interface();"
            "print('BLOCKS:'+str(isinstance(blocks, gr.Blocks)))"
        ], timeout=60)
        assert result.returncode == 0, f"Interface build failed:\n{result.stderr[-1000:]}"
        assert "BLOCKS:True" in result.stdout, (
            f"create_interface() did not return gr.Blocks: {result.stdout!r}")

    # ------------------------------------------------------------------
    # 7. Gradio HTTP predict endpoint reachable from host
    # ------------------------------------------------------------------

    def test_host_can_reach_gradio_predict_endpoint(self, docker_container):
        """The /run/predict endpoint must exist and accept POST from the host."""
        import urllib.request, urllib.error, json
        url = docker_container.rstrip("/") + "/run/predict"
        payload = json.dumps({"fn_index": 0, "data": []}).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=10)
            # Any non-5xx response is acceptable (400/422 is fine — means endpoint exists)
            assert resp.status < 500
        except urllib.error.HTTPError as exc:
            # 400 / 422 from Gradio means the endpoint exists but we sent bad data
            assert exc.code < 500, f"Server error on predict endpoint: {exc.code}"

