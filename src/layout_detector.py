"""
Layout Detection & Table Extraction Module — v0.4
DocLayout-YOLO for AI-powered layout detection with OpenCV contour fallback.
Detects text blocks, tables, figures, formulas, headings on page images.
Includes table grid extraction and per-cell OCR with language support.

v0.4 changes:
  - Fix torch.load weights_only issue for PyTorch >= 2.6
  - Improved OpenCV fallback: multi-scale line detection, table scoring
  - Smarter figure-vs-table classification
"""
import os
import logging
from typing import Dict, List, Any, Optional

import numpy as np
import cv2

logger = logging.getLogger(__name__)

_PLAIN_TEXT = "plain text"

# ── Optional heavy imports ────────────────────────────────────────────────────
YOLO_AVAILABLE = False
try:
    # Patch torch.load for PyTorch >= 2.6 (weights_only default changed to True)
    import torch
    _original_torch_load = torch.load
    def _safe_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original_torch_load(*args, **kwargs)
    torch.load = _safe_torch_load

    from doclayout_yolo import YOLOv10
    YOLO_AVAILABLE = True
    logger.info("DocLayout-YOLO available")
except Exception:
    pass

TESSERACT_AVAILABLE = False
try:
    import pytesseract
    pytesseract.get_tesseract_version()
    TESSERACT_AVAILABLE = True
except Exception:
    pass

PPSTRUCTURE_AVAILABLE = False
try:
    from paddleocr import PPStructure
    PPSTRUCTURE_AVAILABLE = True
except Exception:
    pass


# ══════════════════════════════════════════════════════════════════════════════
# Layout Detector
# ══════════════════════════════════════════════════════════════════════════════
class LayoutDetector:
    """Document layout detector: YOLO-based with OpenCV fallback."""

    LAYOUT_CLASSES = {
        0: "title", 1: _PLAIN_TEXT, 2: "abandon", 3: "figure",
        4: "figure_caption", 5: "table", 6: "table_caption",
        7: "table_footnote", 8: "isolate_formula", 9: "formula_caption",
    }
    FIGURE_CLASSES = {"figure"}
    TABLE_CLASSES = {"table"}
    TEXT_CLASSES = {"title", _PLAIN_TEXT}

    def __init__(self, model_path: Optional[str] = None,
                 confidence_threshold: float = 0.15,
                 iou_threshold: float = 0.45):
        self.confidence = float(os.getenv("YOLO_CONFIDENCE", str(confidence_threshold)))
        self.iou = float(os.getenv("YOLO_NMS", str(iou_threshold)))
        self.model = None
        self.model_loaded = False

        if YOLO_AVAILABLE:
            try:
                if model_path and os.path.exists(model_path):
                    self.model = YOLOv10(model_path)
                else:
                    local_pt = os.path.join(
                        os.path.dirname(os.path.dirname(__file__)),
                        "models", "DocLayout-YOLO-DocStructBench",
                        "doclayout_yolo_docstructbench_imgsz1280_2501.pt",
                    )
                    if os.path.exists(local_pt):
                        self.model = YOLOv10(local_pt)
                    else:
                        self.model = YOLOv10.from_pretrained(
                            "juliozhao/DocLayout-YOLO-DocStructBench-imgsz1280-2501"
                        )
                self.model_loaded = True
                logger.info("DocLayout-YOLO model loaded")
            except Exception as e:
                logger.warning(f"Failed to load YOLO: {e} — using OpenCV fallback")

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_layout(self, image: np.ndarray, page_number: int = 0,
                      confidence: Optional[float] = None) -> Dict[str, Any]:
        """Detect layout elements in a page image.

        Args:
            confidence: Override YOLO confidence threshold (lower = more detections).
        """
        if self.model_loaded:
            return self._detect_yolo(image, page_number, confidence)
        return self._detect_opencv_fallback(image, page_number)

    # ── YOLO detection ────────────────────────────────────────────────────────

    def _detect_yolo(self, image: np.ndarray, page_number: int,
                     confidence: Optional[float] = None) -> Dict[str, Any]:
        try:
            import tempfile, uuid
            conf = confidence if confidence is not None else self.confidence
            tmp = os.path.join(tempfile.gettempdir(), f"yolo_{uuid.uuid4().hex}.png")
            cv2.imwrite(tmp, image)
            results = self.model.predict(tmp, imgsz=1280,
                                         conf=conf, iou=self.iou)
            try:
                os.remove(tmp)
            except OSError:
                pass

            detections = {
                "figures": [], "tables": [], "text_regions": [],
                "formulas": [], "captions": [], "other": [],
            }
            for r in results:
                boxes = r.boxes
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    box_conf = float(boxes.conf[i].item())
                    bbox = boxes.xyxy[i].cpu().numpy().tolist()
                    cls_name = self.LAYOUT_CLASSES.get(cls_id, "unknown")
                    det = {"bbox": bbox, "confidence": box_conf,
                           "class": cls_name, "class_id": cls_id}
                    if cls_name in self.FIGURE_CLASSES:
                        detections["figures"].append(det)
                    elif cls_name in self.TABLE_CLASSES:
                        detections["tables"].append(det)
                    elif cls_name in self.TEXT_CLASSES:
                        detections["text_regions"].append(det)
                    elif "caption" in cls_name:
                        detections["captions"].append(det)
                    elif "formula" in cls_name:
                        detections["formulas"].append(det)
                    else:
                        detections["other"].append(det)

            # ── Reclassify figures that look like tables ──────────────
            self._reclassify_figures_as_tables(image, detections)

            return {
                "page": page_number,
                "detections": detections,
                "total": sum(len(v) for v in detections.values()),
            }
        except Exception as exc:
            logger.error(f"YOLO detection failed: {exc}")
            return self._detect_opencv_fallback(image, page_number)

    def _reclassify_figures_as_tables(self, image: np.ndarray,
                                      detections: Dict[str, List]) -> None:
        """Move figure detections that contain grid/ruled lines to the tables list.

        Uses three heuristics (any one triggers reclassification):
          1. Both H and V grid lines present (classic bordered table).
          2. Multiple horizontal rules only (borderless form or row-striped table).
          3. High text-pixel density + wide aspect ratio (text-heavy table region).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        remaining_figures = []
        for det in detections["figures"]:
            x0, y0, x1, y1 = [int(v) for v in det["bbox"]]
            ih, iw = gray.shape[:2]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(iw, x1), min(ih, y1)
            roi = gray[y0:y1, x0:x1]
            if roi.size == 0:
                remaining_figures.append(det)
                continue

            rh, rw = roi.shape[:2]
            area = rh * rw
            if area < 2000:
                remaining_figures.append(det)
                continue

            _, thresh = cv2.threshold(roi, 0, 255,
                                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # ── Detect horizontal / vertical lines ──
            hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, rw // 5), 1))
            vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, rh // 5)))
            hl = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, hk)
            vl = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vk)
            h_density = cv2.countNonZero(hl) / max(area, 1)
            v_density = cv2.countNonZero(vl) / max(area, 1)

            is_table = False
            reason = ""

            # Heuristic 1: bordered table (both directions)
            if h_density > 0.005 and v_density > 0.005:
                is_table = True
                reason = f"grid h={h_density:.4f} v={v_density:.4f}"

            # Heuristic 2: horizontal rules only (forms, row tables)
            elif h_density > 0.012:
                is_table = True
                reason = f"h-rules h={h_density:.4f}"

            # Heuristic 3: wide text-heavy region (borderless table)
            else:
                aspect = rw / max(rh, 1)
                text_density = cv2.countNonZero(thresh) / max(area, 1)
                if aspect > 1.8 and text_density > 0.08 and rh > 60:
                    is_table = True
                    reason = f"text-dense aspect={aspect:.1f} td={text_density:.3f}"

            if is_table:
                det["class"] = "table"
                det["class_id"] = 5
                det["reclassified"] = True
                detections["tables"].append(det)
                logger.info(f"Reclassified figure -> table ({reason})")
            else:
                remaining_figures.append(det)

        detections["figures"] = remaining_figures

    # ── OpenCV fallback ───────────────────────────────────────────────────────

    def _detect_opencv_fallback(self, image: np.ndarray,
                                page_number: int) -> Dict[str, Any]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape[:2]

        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (max(30, w // 20), 8))
        dilated = cv2.dilate(thresh, kernel, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        text_regions: List[Dict] = []
        tables: List[Dict] = []
        figures: List[Dict] = []

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            if cw < 40 or ch < 10 or area < 500:
                continue
            if cw > w * 0.98 and ch > h * 0.98:
                continue
            bbox = [x, y, x + cw, y + ch]
            aspect = cw / max(ch, 1)

            if aspect > 1.5 and ch < h * 0.05 or not (0.7 < aspect < 1.5 and area > w * h * 0.05):
                text_regions.append({"bbox": bbox, "confidence": 0.6,
                                     "class": _PLAIN_TEXT, "class_id": 1})
            else:
                det = self._classify_mixed_region(thresh, bbox, x, y, cw, ch, area)
                if det["class"] == "table":
                    tables.append(det)
                else:
                    figures.append(det)

        text_regions.sort(key=lambda r: r["bbox"][1])
        return {
            "page": page_number,
            "detections": {
                "figures": figures, "tables": tables,
                "text_regions": text_regions,
                "formulas": [], "captions": [], "other": [],
            },
            "total": len(text_regions) + len(tables) + len(figures),
        }

    @staticmethod
    def _classify_mixed_region(
        thresh: np.ndarray, bbox: List, x: int, y: int,
        cw: int, ch: int, area: int,
    ) -> Dict[str, Any]:
        """Classify a contour as table or figure based on grid-line density."""
        roi = thresh[y:y + ch, x:x + cw]
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, cw // 4), 1))
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, ch // 4)))
        hl = cv2.morphologyEx(roi, cv2.MORPH_OPEN, hk)
        vl = cv2.morphologyEx(roi, cv2.MORPH_OPEN, vk)
        if cv2.countNonZero(hl) > area * 0.01 and cv2.countNonZero(vl) > area * 0.01:
            return {"bbox": bbox, "confidence": 0.5, "class": "table", "class_id": 5}
        return {"bbox": bbox, "confidence": 0.4, "class": "figure", "class_id": 3}


# ══════════════════════════════════════════════════════════════════════════════
# Table Extractor
# ══════════════════════════════════════════════════════════════════════════════
class TableExtractor:
    """Detect table grid structure via OpenCV and OCR each cell.

    v0.3: accepts ``languages`` parameter so cell OCR uses the correct
    language pack (critical for non-English documents).
    """

    def __init__(self):
        self.use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
        self.engine = os.getenv("TABLE_ENGINE", "opencv").lower()
        self._pp_structure = None
        self.enabled = os.getenv("TABLE_DETECTION", "true").lower() == "true"
        self._ocr_languages = "eng"  # updated per-call via extract_tables()

    def extract_tables(self, image: np.ndarray,
                       table_boxes: List[Dict],
                       languages: str = "eng") -> List[Dict[str, Any]]:
        """Extract table content from detected table regions.

        Args:
            languages: Tesseract language string for cell OCR (e.g. ``"tha+eng"``).
        """
        if not self.enabled or not table_boxes:
            return []
        self._ocr_languages = languages
        tables = []
        for tbox in table_boxes:
            x0, y0, x1, y1 = [int(v) for v in tbox["bbox"]]
            h, w = image.shape[:2]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            crop = image[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            result = self._extract_single(crop)
            result["bbox"] = tbox["bbox"]
            tables.append(result)
        return tables

    def _extract_single(self, table_img: np.ndarray) -> Dict[str, Any]:
        if self.engine == "paddleocr" and PPSTRUCTURE_AVAILABLE:
            r = self._extract_ppstructure(table_img)
            if r:
                return r
        return self._extract_opencv(table_img)

    # ── PPStructure (optional) ────────────────────────────────────────────────

    def _extract_ppstructure(self, img: np.ndarray) -> Optional[Dict[str, Any]]:
        if self._pp_structure is None:
            try:
                self._pp_structure = PPStructure(
                    table=True, ocr=True,
                    use_gpu=self.use_gpu, show_log=False,
                )
            except Exception as exc:
                logger.warning(f"PPStructure init failed: {exc}")
                return None
        try:
            result = self._pp_structure(img)
            for item in result:
                if item.get("type") == "table":
                    html = item.get("res", {}).get("html", "")
                    if html:
                        return {"html": html, "text": ""}
        except Exception as exc:
            logger.warning(f"PPStructure error: {exc}")
        return None

    # ── OpenCV grid extraction ────────────────────────────────────────────────

    def _extract_opencv(self, img: np.ndarray) -> Dict[str, Any]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        h, w = gray.shape[:2]
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 5), 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, h // 5)))
        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)

        grid = cv2.add(h_lines, v_lines)
        contours, _ = cv2.findContours(grid, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cells = [
            {"x": x, "y": y, "w": cw, "h": ch}
            for cnt in contours
            for x, y, cw, ch in [cv2.boundingRect(cnt)]
            if 15 < cw < w * 0.95 and 10 < ch < h * 0.95
        ]

        if not cells:
            # No grid lines found — try full-region OCR and return as plain text
            text = self._ocr_region(gray)
            if text.strip():
                return {"html": "", "text": text}
            return {"html": "", "text": ""}

        rows = self._group_cells_into_rows(cells)
        return self._build_html_table(rows, gray)

    @staticmethod
    def _group_cells_into_rows(cells: List[Dict]) -> List[List[Dict]]:
        cells.sort(key=lambda c: (c["y"], c["x"]))
        rows: List[List[Dict]] = []
        current_row: List[Dict] = [cells[0]]
        for cell in cells[1:]:
            if abs(cell["y"] - current_row[0]["y"]) < 15:
                current_row.append(cell)
            else:
                rows.append(sorted(current_row, key=lambda c: c["x"]))
                current_row = [cell]
        rows.append(sorted(current_row, key=lambda c: c["x"]))
        return rows

    def _build_html_table(self, rows: List[List[Dict]],
                          gray: np.ndarray) -> Dict[str, Any]:
        html_parts = ["<table border='1' style='border-collapse:collapse;'>"]
        all_text: List[str] = []
        for ri, row in enumerate(rows):
            html_parts.append("<tr>")
            row_texts: List[str] = []
            for cell in row:
                cx, cy, cw, ch = cell["x"], cell["y"], cell["w"], cell["h"]
                cell_img = gray[cy:cy + ch, cx:cx + cw]
                cell_text = self._ocr_region(cell_img) if cell_img.size > 0 else ""
                row_texts.append(cell_text)
                tag = "th" if ri == 0 else "td"
                html_parts.append(f"<{tag}>{cell_text}</{tag}>")
            html_parts.append("</tr>")
            all_text.append("\t".join(row_texts))
        html_parts.append("</table>")
        return {"html": "\n".join(html_parts), "text": "\n".join(all_text)}

    def _ocr_region(self, gray_img: np.ndarray) -> str:
        """OCR a single region using Tesseract with the active language."""
        if not TESSERACT_AVAILABLE:
            return ""
        try:
            lang = getattr(self, '_ocr_languages', 'eng')
            return pytesseract.image_to_string(
                gray_img, lang=lang, config="--oem 1 --psm 6"
            ).strip()
        except Exception:
            return ""
