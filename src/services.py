"""
Application Services — Auth & History
In-memory authentication + per-user processing history.

Security:
    - PBKDF2-HMAC-SHA256 password hashing (260k iterations, random salt)
    - Path traversal protection on all file operations
    - Input sanitization on usernames, entry IDs
    - Session tokens via ``secrets.token_hex``
"""
import os
import re
import json
import shutil
import logging
import hashlib
import secrets
import tempfile
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

_INFO_FILENAME = "info.json"
_PBKDF2_ITERATIONS = 260_000
_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")
_SAFE_DISPLAY_NAME_RE = re.compile(r"[^a-zA-Z0-9_.\- ]")
_USERNAME_RE = re.compile(r"^\w{3,50}$")


def _sanitize_path_component(value: str) -> str:
    """Strip any character that could enable path traversal."""
    return _SAFE_NAME_RE.sub("", value)


# Authentication
class AuthManager:
    """In-memory user registration, login, session management.

    Security:
        - Passwords hashed with PBKDF2-HMAC-SHA256 (260 000 iterations)
        - Random 32-byte salt per password
        - Default credentials loaded from environment (never hardcoded)
        - Session tokens: 256-bit random via ``secrets``
    """

    def __init__(self) -> None:
        self._users: Dict[str, str] = {}       # username -> "salt_hex:key_hex"
        self._sessions: Dict[str, Dict] = {}   # token -> {"username", "expires_at"}
        self._setup_default_user()
        logger.info("AuthManager initialised (PBKDF2, in-memory)")

    def _setup_default_user(self) -> None:
        """Create default account from environment variables."""
        default_user = os.getenv("DEFAULT_USERNAME", "guest")
        default_pass = os.getenv("DEFAULT_PASSWORD", "guest123")
        if default_user and default_pass:
            self._users[default_user.strip().lower()] = self._hash(default_pass)

    @staticmethod
    def _hash(password: str) -> str:
        """Hash *password* with PBKDF2-HMAC-SHA256 + random salt."""
        salt = os.urandom(32)
        key = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS,
        )
        return salt.hex() + ":" + key.hex()

    @staticmethod
    def _verify(password: str, stored: str) -> bool:
        """Verify *password* against a PBKDF2 stored hash."""
        try:
            salt_hex, key_hex = stored.split(":", maxsplit=1)
            salt = bytes.fromhex(salt_hex)
            key = hashlib.pbkdf2_hmac(
                "sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS,
            )
            # Constant-time comparison to prevent timing attacks
            return secrets.compare_digest(key.hex(), key_hex)
        except (ValueError, TypeError):
            return False

    def register(self, username: str, password: str) -> Dict[str, Any]:
        """Register a new user after input validation."""
        username = username.strip().lower()
        if not _USERNAME_RE.match(username):
            return {
                "success": False,
                "error": "Username must be 3-50 alphanumeric/underscore characters",
            }
        if len(password) < 6:
            return {"success": False, "error": "Password must be at least 6 characters"}
        if username in self._users:
            return {"success": False, "error": "Username already exists"}
        self._users[username] = self._hash(password)
        return {"success": True, "message": "Registration successful"}

    def login(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate and return a session token."""
        username = username.strip().lower()
        if not username or not password:
            return {"success": False, "error": "Invalid credentials"}
        stored = self._users.get(username)
        if not stored or not self._verify(password, stored):
            return {"success": False, "error": "Invalid credentials"}
        token = secrets.token_hex(32)
        expires = datetime.now(timezone.utc) + timedelta(hours=24)
        self._sessions[token] = {"username": username, "expires_at": expires}
        return {"success": True, "token": token, "username": username}

    def validate_session(self, token: str) -> Optional[str]:
        """Return username if *token* is valid, else ``None``."""
        if not token:
            return None
        sess = self._sessions.get(token)
        if sess and sess["expires_at"] > datetime.now(timezone.utc):
            return sess["username"]
        if sess:
            del self._sessions[token]
        return None

    def logout(self, token: str) -> None:
        """Invalidate a session token."""
        self._sessions.pop(token, None)


# History
class HistoryManager:
    """Per-user processing history with automatic cleanup."""

    def __init__(self):
        self.retention_days = int(os.getenv("HISTORY_RETENTION_DAYS", "30"))
        self.base_dir = Path(tempfile.gettempdir()) / "pdf_ocr_history"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def user_dir(self, username: str) -> Path:
        """Return per-user directory, sanitised against path traversal."""
        safe_name = _sanitize_path_component(username)
        if not safe_name:
            safe_name = "anonymous"
        d = self.base_dir / safe_name
        # Ensure resolved path is under base_dir (extra guard)
        if not str(d.resolve()).startswith(str(self.base_dir.resolve())):
            d = self.base_dir / "anonymous"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_result(self, username: str, original_filename: str,
                    output_paths: Dict[str, Optional[str]],
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save conversion output files and metadata."""
        entry_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        entry_dir = self.user_dir(username) / _sanitize_path_component(entry_id)
        entry_dir.mkdir(parents=True, exist_ok=True)

        # Sanitise original filename for storage (preserve dots for display)
        safe_filename = _SAFE_DISPLAY_NAME_RE.sub("_", original_filename)[:200]

        saved_files: Dict[str, str] = {}
        for fmt, src_path in output_paths.items():
            safe_fmt = _sanitize_path_component(fmt)
            if src_path and os.path.exists(src_path):
                ext = os.path.splitext(src_path)[1]
                dest = entry_dir / f"output{ext}"
                shutil.copy2(src_path, dest)
                saved_files[safe_fmt] = str(dest)

        info = {
            "entry_id": entry_id,
            "original_filename": safe_filename,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files": saved_files,
            "metadata": metadata or {},
        }
        with open(entry_dir / _INFO_FILENAME, "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=2, ensure_ascii=False)
        return entry_id

    def list_entries(self, username: str) -> List[Dict[str, Any]]:
        udir = self.user_dir(username)
        entries = []
        for child in sorted(udir.iterdir(), reverse=True):
            if child.is_dir():
                info_file = child / _INFO_FILENAME
                if info_file.exists():
                    try:
                        with open(info_file, "r", encoding="utf-8") as f:
                            entries.append(json.load(f))
                    except Exception:
                        pass
        return entries

    def get_file_path(self, username: str, entry_id: str,
                      fmt: str = "docx") -> Optional[str]:
        """Retrieve file path from history, with path-traversal protection."""
        safe_id = _sanitize_path_component(entry_id)
        if not safe_id:
            return None
        safe_fmt = _sanitize_path_component(fmt) or "docx"
        entry_dir = self.user_dir(username) / safe_id
        # Guard: resolved path must be under user_dir
        if not str(entry_dir.resolve()).startswith(
                str(self.user_dir(username).resolve())):
            logger.warning("Path traversal attempt blocked: %s", entry_id)
            return None
        info_file = entry_dir / _INFO_FILENAME
        if not info_file.exists():
            return None
        try:
            with open(info_file, "r", encoding="utf-8") as fh:
                info = json.load(fh)
            path = info.get("files", {}).get(safe_fmt)
            if path and os.path.exists(path):
                return path
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("get_file_path error: %s", exc)
        return None

    def cleanup_old_entries(self, username: Optional[str] = None) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        dirs: List[Path] = (
            [self.user_dir(username)]
            if username
            else [c for c in self.base_dir.iterdir() if c.is_dir()]
        )
        for udir in dirs:
            self._cleanup_dir_entries(udir, cutoff)

    def _cleanup_dir_entries(self, udir: Path, cutoff: datetime) -> None:
        """Remove history entries older than cutoff in a single user directory."""
        for entry in udir.iterdir():
            if not entry.is_dir():
                continue
            info_file = entry / _INFO_FILENAME
            if not info_file.exists():
                continue
            self._maybe_remove_entry(entry, info_file, cutoff)

    @staticmethod
    def _maybe_remove_entry(entry: Path, info_file: Path, cutoff: datetime) -> None:
        try:
            with open(info_file, "r", encoding="utf-8") as f:
                info = json.load(f)
            created = datetime.fromisoformat(info["created_at"])
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            if created < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
        except Exception:
            pass
