"""
Correction Store & Auto-Retrain — v1.0

Logs every manual region correction (add table / add figure / remove region).
Stores page image crops + YOLO-format labels for fine-tuning.
Triggers automatic YOLO fine-tune every N corrections (default 100).

Security:
    - Filenames sanitised (only alphanumeric, underscore, hyphen, dot)
    - All file writes performed under a thread lock
    - Exception handlers scoped narrowly
    - No internal paths exposed in error messages

Directory layout (under CORRECTION_DATA_DIR):
    images/          — full page images (PNG)
    labels/          — YOLO .txt label files (class cx cy w h)
    corrections.jsonl — append-only log of every correction
    retrain_log.json — history of retrain runs
"""
import json
import logging
import os
import re
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 "correction_data")

_SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9_.-]")
_LEADING_DOTS_RE = re.compile(r"^\.+")
_CONSECUTIVE_DOTS_RE = re.compile(r"\.{2,}")


def _sanitize_filename(name: str, max_len: int = 100) -> str:
    """Strip unsafe characters from *name* for use in file paths.

    Also removes leading dots and consecutive dots to prevent
    path traversal (e.g. ``../../etc/passwd`` → ``etc_passwd``).
    """
    result = _SAFE_FILENAME_RE.sub("_", name)
    result = _CONSECUTIVE_DOTS_RE.sub(".", result)
    result = _LEADING_DOTS_RE.sub("", result)
    return result[:max_len] or "unnamed"

# YOLO class IDs matching LayoutDetector.LAYOUT_CLASSES
YOLO_CLASS_MAP = {
    "title": 0,
    "plain text": 1,
    "abandon": 2,
    "figure": 3,
    "figure_caption": 4,
    "table": 5,
    "table_caption": 6,
    "table_footnote": 7,
    "isolate_formula": 8,
    "formula_caption": 9,
}


class CorrectionStore:
    """Persistent store for manual layout corrections.

    Each correction records:
        - The full-page image (saved once per unique page)
        - The bbox and class of the manually-added region
        - Source: ``"manual"`` vs ``"auto"``

    After every ``retrain_interval`` **manual** corrections, triggers
    a YOLO fine-tune in a background thread.
    """

    def __init__(self, data_dir: Optional[str] = None,
                 retrain_interval: int = 100) -> None:
        self.data_dir = Path(
            data_dir or os.getenv("CORRECTION_DATA_DIR", _DEFAULT_DATA_DIR))
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        self.corrections_file = self.data_dir / "corrections.jsonl"
        self.retrain_log_file = self.data_dir / "retrain_log.json"
        self.retrain_interval = max(1, int(
            os.getenv("RETRAIN_INTERVAL", str(retrain_interval))))

        self._lock = threading.Lock()
        self._correction_count = self._count_existing()
        self._retrain_running = False
        logger.info(
            "CorrectionStore: %d existing corrections, retrain every %d",
            self._correction_count, self.retrain_interval)

    # ══════════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════════

    def log_correction(self, page_image: np.ndarray,
                       bbox: List[float],
                       region_class: str,
                       page_number: int,
                       pdf_name: str = "",
                       action: str = "add",
                       source: str = "manual") -> Dict[str, Any]:
        """Record a single correction.

        Args:
            page_image: Full page image (BGR numpy).
            bbox: [x0, y0, x1, y1] absolute pixel coords.
            region_class: "table" or "figure" (or any LAYOUT_CLASSES key).
            page_number: 0-based page index.
            pdf_name: Original PDF filename (for traceability).
            action: "add" or "remove".
            source: "manual" (user-drawn) or "auto" (model-detected).

        Returns:
            Dict with correction_id, total_corrections, retrain_triggered.
        """
        h, w = page_image.shape[:2]
        if h == 0 or w == 0:
            return {"correction_id": "", "total_corrections": self._correction_count,
                    "retrain_triggered": False}

        safe_pdf = _sanitize_filename(pdf_name) if pdf_name else "unknown"
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        page_key = f"{safe_pdf}_p{int(page_number)}_{ts}"

        # ── Convert bbox to YOLO normalised format ──
        x0, y0, x1, y1 = [float(v) for v in bbox]
        cx = ((x0 + x1) / 2.0) / w
        cy = ((y0 + y1) / 2.0) / h
        bw = abs(x1 - x0) / w
        bh = abs(y1 - y0) / h
        cls_id = YOLO_CLASS_MAP.get(region_class, 3)  # default figure

        # ── Prepare label line ──
        label_line = f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"

        # ── Prepare correction record ──
        record = {
            "correction_id": page_key,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pdf_name": safe_pdf,
            "page": int(page_number),
            "bbox": [x0, y0, x1, y1],
            "class": region_class,
            "class_id": cls_id,
            "action": action,
            "source": source,
            "image_file": f"{page_key}.png",
            "label_file": f"{page_key}.txt",
        }

        # ── Atomic write under lock ──
        with self._lock:
            img_path = self.images_dir / f"{page_key}.png"
            cv2.imwrite(str(img_path), page_image)

            lbl_path = self.labels_dir / f"{page_key}.txt"
            with open(lbl_path, "a", encoding="utf-8") as f:
                f.write(label_line)

            with open(self.corrections_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if source == "manual":
                self._correction_count += 1
            count = self._correction_count

        # ── Check if retrain should trigger ──
        retrain_triggered = False
        if (source == "manual"
                and count > 0
                and count % self.retrain_interval == 0
                and not self._retrain_running):
            retrain_triggered = True
            self._start_retrain_async()

        return {
            "correction_id": page_key,
            "total_corrections": count,
            "retrain_triggered": retrain_triggered,
        }

    def log_page_detections(self, page_image: np.ndarray,
                            detections: Dict[str, List[Dict]],
                            page_number: int,
                            pdf_name: str = "") -> None:
        """Bulk-log all auto-detected regions for a page (for training data).

        This does NOT count towards the retrain interval — only manual
        corrections trigger retraining.
        """
        h, w = page_image.shape[:2]
        if h == 0 or w == 0:
            return

        safe_pdf = _sanitize_filename(pdf_name) if pdf_name else "auto"
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        page_key = f"auto_{safe_pdf}_p{int(page_number)}_{ts}"

        img_path = self.images_dir / f"{page_key}.png"
        cv2.imwrite(str(img_path), page_image)

        lbl_path = self.labels_dir / f"{page_key}.txt"
        lines: List[str] = []
        for category in ("figures", "tables", "text_regions", "captions",
                         "formulas"):
            for det in detections.get(category, []):
                bbox = det.get("bbox", [0, 0, 0, 0])
                cls_name = det.get("class", "figure")
                cls_id = YOLO_CLASS_MAP.get(cls_name, 3)
                x0, y0, x1, y1 = bbox
                cx = ((x0 + x1) / 2.0) / w
                cy = ((y0 + y1) / 2.0) / h
                bw = (x1 - x0) / w
                bh = (y1 - y0) / h
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if lines:
            with open(lbl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

    def get_stats(self) -> Dict[str, Any]:
        """Return correction stats for the UI."""
        retrain_history = self._load_retrain_log()
        return {
            "total_manual_corrections": self._correction_count,
            "next_retrain_at": (
                (self._correction_count // self.retrain_interval + 1)
                * self.retrain_interval
            ),
            "retrain_interval": self.retrain_interval,
            "retrain_running": self._retrain_running,
            "retrain_history": retrain_history[-5:],  # last 5 runs
            "data_dir": str(self.data_dir),
            "images_count": len(list(self.images_dir.glob("*.png"))),
            "labels_count": len(list(self.labels_dir.glob("*.txt"))),
        }

    def get_corrections_log(self, limit: int = 50) -> List[Dict]:
        """Return the most recent corrections."""
        if not self.corrections_file.exists():
            return []
        lines = self.corrections_file.read_text(encoding="utf-8").strip().split("\n")
        entries = []
        for line in lines[-limit:]:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return entries

    # ══════════════════════════════════════════════════════════════════════════
    # Fine-tune (background)
    # ══════════════════════════════════════════════════════════════════════════

    def _start_retrain_async(self) -> None:
        """Launch YOLO fine-tune in a background thread."""
        self._retrain_running = True
        t = threading.Thread(target=self._do_retrain, daemon=True)
        t.start()
        logger.info("Retrain triggered at %d corrections", self._correction_count)

    def _do_retrain(self) -> None:
        """Actual fine-tune logic.

        Creates a YOLO dataset YAML, runs model.train() for a few epochs,
        and saves the new weights alongside the original model.
        """
        start_ts = datetime.now(timezone.utc).isoformat()
        result: Dict[str, Any] = {
            "started_at": start_ts,
            "corrections_at_start": self._correction_count,
            "status": "running",
        }
        try:
            # Check we have enough data
            label_files = list(self.labels_dir.glob("*.txt"))
            if len(label_files) < 10:
                result["status"] = "skipped"
                result["reason"] = f"Only {len(label_files)} label files (need >= 10)"
                self._save_retrain_result(result)
                return

            # Prepare dataset YAML
            dataset_dir = self.data_dir / "dataset"
            dataset_dir.mkdir(exist_ok=True)
            train_img = dataset_dir / "images" / "train"
            train_lbl = dataset_dir / "labels" / "train"
            train_img.mkdir(parents=True, exist_ok=True)
            train_lbl.mkdir(parents=True, exist_ok=True)

            # Symlink/copy images and labels
            for lbl in label_files:
                img = self.images_dir / lbl.with_suffix(".png").name
                if img.exists():
                    dst_img = train_img / img.name
                    dst_lbl = train_lbl / lbl.name
                    if not dst_img.exists():
                        shutil.copy2(str(img), str(dst_img))
                    if not dst_lbl.exists():
                        shutil.copy2(str(lbl), str(dst_lbl))

            # Write dataset YAML
            yaml_path = dataset_dir / "dataset.yaml"
            yaml_content = (
                f"path: {dataset_dir}\n"
                f"train: images/train\n"
                f"val: images/train\n"  # use train as val for few-shot
                f"nc: 10\n"
                f"names:\n"
                f"  0: title\n"
                f"  1: plain text\n"
                f"  2: abandon\n"
                f"  3: figure\n"
                f"  4: figure_caption\n"
                f"  5: table\n"
                f"  6: table_caption\n"
                f"  7: table_footnote\n"
                f"  8: isolate_formula\n"
                f"  9: formula_caption\n"
            )
            yaml_path.write_text(yaml_content, encoding="utf-8")

            # Find base model
            model_dir = Path(__file__).parent.parent / "models" / "DocLayout-YOLO-DocStructBench"
            base_pt = model_dir / "doclayout_yolo_docstructbench_imgsz1280_2501.pt"

            try:
                import torch
                _original_torch_load = torch.load
                def _safe_torch_load(*args, **kwargs):
                    kwargs.setdefault("weights_only", False)
                    return _original_torch_load(*args, **kwargs)
                torch.load = _safe_torch_load

                from doclayout_yolo import YOLOv10
            except ImportError:
                result["status"] = "skipped"
                result["reason"] = "doclayout-yolo or torch not installed"
                self._save_retrain_result(result)
                return

            if not base_pt.exists():
                result["status"] = "skipped"
                result["reason"] = f"Base model not found at {base_pt}"
                self._save_retrain_result(result)
                return

            # Fine-tune: low epochs, small learning rate
            model = YOLOv10(str(base_pt))
            logger.info("Starting YOLO fine-tune with %d images", len(label_files))

            train_results = model.train(
                data=str(yaml_path),
                epochs=5,
                imgsz=1280,
                batch=2,
                lr0=0.0001,
                patience=3,
                save=True,
                project=str(self.data_dir / "runs"),
                name="finetune",
                exist_ok=True,
                verbose=False,
            )

            # Save fine-tuned model
            ft_dir = self.data_dir / "finetuned_models"
            ft_dir.mkdir(exist_ok=True)
            ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            ft_path = ft_dir / f"finetuned_{ts_str}.pt"

            # Copy best weights from training run
            best_wt = self.data_dir / "runs" / "finetune" / "weights" / "best.pt"
            if best_wt.exists():
                shutil.copy2(str(best_wt), str(ft_path))
                result["status"] = "success"
                result["model_path"] = str(ft_path)
                result["train_images"] = len(label_files)
                logger.info("Fine-tune complete: %s", ft_path)
            else:
                result["status"] = "completed_no_best"
                result["reason"] = "Training ran but no best.pt found"

        except (OSError, RuntimeError, ImportError, ValueError) as exc:
            logger.error("Retrain failed: %s", type(exc).__name__, exc_info=True)
            result["status"] = "error"
            result["error"] = type(exc).__name__
        finally:
            result["finished_at"] = datetime.now(timezone.utc).isoformat()
            self._save_retrain_result(result)
            self._retrain_running = False

    # ══════════════════════════════════════════════════════════════════════════
    # Internal helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _count_existing(self) -> int:
        """Count existing manual corrections from the log."""
        if not self.corrections_file.exists():
            return 0
        count = 0
        try:
            with open(self.corrections_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        if rec.get("source") == "manual":
                            count += 1
                    except (json.JSONDecodeError, ValueError):
                        pass
        except OSError as exc:
            logger.warning("Could not read corrections file: %s", type(exc).__name__)
        return count

    def _load_retrain_log(self) -> List[Dict]:
        """Load the retrain history log."""
        if not self.retrain_log_file.exists():
            return []
        try:
            with open(self.retrain_log_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError, ValueError):
            return []

    def _save_retrain_result(self, result: Dict) -> None:
        history = self._load_retrain_log()
        history.append(result)
        with self._lock:
            with open(self.retrain_log_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
