"""
Layout Detection & Table Extraction Module
DocLayout-YOLO for AI-powered layout detection with OpenCV contour fallback.
Detects text blocks, tables, figures, formulas, headings on page images.
Includes table grid extraction and per-cell OCR.
"""
import os
import logging
from typing import Dict, List, Any, Optional

import numpy as np
import cv2

logger = logging.getLogger(__name__)

# ── Optional heavy imports ────────────────────────────────────────────────────
YOLO_AVAILABLE = False
try:
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
        0: "title", 1: "plain text", 2: "abandon", 3: "figure",
        4: "figure_caption", 5: "table", 6: "table_caption",
        7: "table_footnote", 8: "isolate_formula", 9: "formula_caption",
    }
    FIGURE_CLASSES = {"figure"}
    TABLE_CLASSES = {"table"}
    TEXT_CLASSES = {"title", "plain text"}

    def __init__(self, model_path: Optional[str] = None,
                 confidence_threshold: float = 0.25,
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

    def detect_layout(self, image: np.ndarray, page_number: int = 0) -> Dict[str, Any]:
        """Detect layout elements in a page image."""
        if self.model_loaded:
            return self._detect_yolo(image, page_number)
        return self._detect_opencv_fallback(image, page_number)

    # ── YOLO detection ────────────────────────────────────────────────────────

    def _detect_yolo(self, image: np.ndarray, page_number: int) -> Dict[str, Any]:
        try:
            import tempfile, uuid
            tmp = os.path.join(tempfile.gettempdir(), f"yolo_{uuid.uuid4().hex}.png")
            cv2.imwrite(tmp, image)
            results = self.model.predict(tmp, imgsz=1280,
                                         conf=self.confidence, iou=self.iou)
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
                    conf = float(boxes.conf[i].item())
                    bbox = boxes.xyxy[i].cpu().numpy().tolist()
                    cls_name = self.LAYOUT_CLASSES.get(cls_id, "unknown")
                    det = {"bbox": bbox, "confidence": conf,
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

            return {
                "page": page_number,
                "detections": detections,
                "total": sum(len(v) for v in detections.values()),
            }
        except Exception as exc:
            logger.error(f"YOLO detection failed: {exc}")
            return self._detect_opencv_fallback(image, page_number)

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

            if aspect > 1.5 and ch < h * 0.05:
                text_regions.append({"bbox": bbox, "confidence": 0.6,
                                     "class": "plain text", "class_id": 1})
            elif 0.7 < aspect < 1.5 and area > w * h * 0.05:
                roi = thresh[y:y + ch, x:x + cw]
                hk = cv2.getStructuringElement(cv2.MORPH_RECT,
                                               (max(40, cw // 4), 1))
                vk = cv2.getStructuringElement(cv2.MORPH_RECT,
                                               (1, max(40, ch // 4)))
                hl = cv2.morphologyEx(roi, cv2.MORPH_OPEN, hk)
                vl = cv2.morphologyEx(roi, cv2.MORPH_OPEN, vk)
                if (cv2.countNonZero(hl) > area * 0.01
                        and cv2.countNonZero(vl) > area * 0.01):
                    tables.append({"bbox": bbox, "confidence": 0.5,
                                   "class": "table", "class_id": 5})
                else:
                    figures.append({"bbox": bbox, "confidence": 0.4,
                                    "class": "figure", "class_id": 3})
            else:
                text_regions.append({"bbox": bbox, "confidence": 0.6,
                                     "class": "plain text", "class_id": 1})

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


# ══════════════════════════════════════════════════════════════════════════════
# Table Extractor
# ══════════════════════════════════════════════════════════════════════════════
class TableExtractor:
    """Detect table grid structure via OpenCV and OCR each cell."""

    def __init__(self):
        self.use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
        self.engine = os.getenv("TABLE_ENGINE", "opencv").lower()
        self._pp_structure = None
        self.enabled = os.getenv("TABLE_DETECTION", "true").lower() == "true"

    def extract_tables(self, image: np.ndarray,
                       table_boxes: List[Dict]) -> List[Dict[str, Any]]:
        if not self.enabled or not table_boxes:
            return []
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
        cells = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if 15 < cw < w * 0.95 and 10 < ch < h * 0.95:
                cells.append({"x": x, "y": y, "w": cw, "h": ch})

        if not cells:
            text = self._ocr_region(gray)
            return {"html": "", "text": text}

        cells.sort(key=lambda c: (c["y"], c["x"]))
        rows: List[List[Dict]] = []
        current_row = [cells[0]]
        for cell in cells[1:]:
            if abs(cell["y"] - current_row[0]["y"]) < 15:
                current_row.append(cell)
            else:
                rows.append(sorted(current_row, key=lambda c: c["x"]))
                current_row = [cell]
        rows.append(sorted(current_row, key=lambda c: c["x"]))

        html_parts = ["<table border='1' style='border-collapse:collapse;'>"]
        all_text: List[str] = []
        for ri, row in enumerate(rows):
            html_parts.append("<tr>")
            row_texts = []
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

    @staticmethod
    def _ocr_region(gray_img: np.ndarray) -> str:
        if not TESSERACT_AVAILABLE:
            return ""
        try:
            return pytesseract.image_to_string(
                gray_img, config="--oem 1 --psm 6"
            ).strip()
        except Exception:
            return ""
