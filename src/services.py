"""
Application Services — Auth & History
In-memory authentication + per-user processing history.
"""
import os
import json
import shutil
import logging
import hashlib
import secrets
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Authentication
# ══════════════════════════════════════════════════════════════════════════════
class AuthManager:
    """In-memory user registration, login, session management."""

    def __init__(self):
        self._users: Dict[str, str] = {}       # username -> password_hash
        self._sessions: Dict[str, Dict] = {}   # token -> {"username", "expires_at"}
        # Default guest user
        self._users["guest"] = self._hash("guest123")
        logger.info("AuthManager initialised (in-memory)")

    @staticmethod
    def _hash(password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

    @staticmethod
    def _verify(password: str, hashed: str) -> bool:
        return hashlib.sha256(password.encode()).hexdigest() == hashed

    def register(self, username: str, password: str) -> Dict[str, Any]:
        username = username.strip().lower()
        if len(username) < 3:
            return {"success": False, "error": "Username must be at least 3 characters"}
        if len(password) < 4:
            return {"success": False, "error": "Password must be at least 4 characters"}
        if username in self._users:
            return {"success": False, "error": "Username already exists"}
        self._users[username] = self._hash(password)
        return {"success": True, "message": "Registration successful"}

    def login(self, username: str, password: str) -> Dict[str, Any]:
        username = username.strip().lower()
        stored = self._users.get(username)
        if not stored or not self._verify(password, stored):
            return {"success": False, "error": "Invalid credentials"}
        token = secrets.token_hex(32)
        expires = datetime.utcnow() + timedelta(hours=24)
        self._sessions[token] = {"username": username, "expires_at": expires}
        return {"success": True, "token": token, "username": username}

    def validate_session(self, token: str) -> Optional[str]:
        if not token:
            return None
        sess = self._sessions.get(token)
        if sess and sess["expires_at"] > datetime.utcnow():
            return sess["username"]
        if sess:
            del self._sessions[token]
        return None

    def logout(self, token: str):
        self._sessions.pop(token, None)


# ══════════════════════════════════════════════════════════════════════════════
# History
# ══════════════════════════════════════════════════════════════════════════════
class HistoryManager:
    """Per-user processing history with automatic cleanup."""

    def __init__(self):
        self.retention_days = int(os.getenv("HISTORY_RETENTION_DAYS", "30"))
        self.base_dir = Path(tempfile.gettempdir()) / "pdf_ocr_history"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def user_dir(self, username: str) -> Path:
        d = self.base_dir / username
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_result(self, username: str, original_filename: str,
                    output_paths: Dict[str, Optional[str]],
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        entry_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        entry_dir = self.user_dir(username) / entry_id
        entry_dir.mkdir(parents=True, exist_ok=True)

        saved_files: Dict[str, str] = {}
        for fmt, src_path in output_paths.items():
            if src_path and os.path.exists(src_path):
                ext = os.path.splitext(src_path)[1]
                dest = entry_dir / f"output{ext}"
                shutil.copy2(src_path, dest)
                saved_files[fmt] = str(dest)

        info = {
            "entry_id": entry_id,
            "original_filename": original_filename,
            "created_at": datetime.utcnow().isoformat(),
            "files": saved_files,
            "metadata": metadata or {},
        }
        with open(entry_dir / "info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        return entry_id

    def list_entries(self, username: str) -> List[Dict[str, Any]]:
        udir = self.user_dir(username)
        entries = []
        for child in sorted(udir.iterdir(), reverse=True):
            if child.is_dir():
                info_file = child / "info.json"
                if info_file.exists():
                    try:
                        with open(info_file, "r", encoding="utf-8") as f:
                            entries.append(json.load(f))
                    except Exception:
                        pass
        return entries

    def get_file_path(self, username: str, entry_id: str,
                      fmt: str = "docx") -> Optional[str]:
        entry_dir = self.user_dir(username) / entry_id
        info_file = entry_dir / "info.json"
        if not info_file.exists():
            return None
        try:
            with open(info_file, "r", encoding="utf-8") as f:
                info = json.load(f)
            path = info.get("files", {}).get(fmt)
            if path and os.path.exists(path):
                return path
        except Exception:
            pass
        return None

    def cleanup_old_entries(self, username: Optional[str] = None):
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
        dirs = []
        if username:
            dirs.append(self.user_dir(username))
        else:
            for child in self.base_dir.iterdir():
                if child.is_dir():
                    dirs.append(child)
        for udir in dirs:
            for entry in udir.iterdir():
                if not entry.is_dir():
                    continue
                info_file = entry / "info.json"
                if info_file.exists():
                    try:
                        with open(info_file, "r", encoding="utf-8") as f:
                            info = json.load(f)
                        created = datetime.fromisoformat(info["created_at"])
                        if created < cutoff:
                            shutil.rmtree(entry, ignore_errors=True)
                    except Exception:
                        pass
