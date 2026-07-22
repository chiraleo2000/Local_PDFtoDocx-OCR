"""Docling layout + TableFormer backend for LocalOCR.

Default layout backend when ``LAYOUT_BACKEND=docling``. Falls back to YOLO
when Docling is unavailable. Uses LocalOCR Thai-TrOCR/PaddleOCR via the
plugin when entry-points are registered; otherwise RapidOCR bridge + adapter
re-OCR with ``OCREngine``.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGES = "tha+eng"
DOCLING_AVAILABLE = False
_IMPORT_ERROR: Optional[str] = None

try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableStructureOptions,
        TableFormerMode,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption
    DOCLING_AVAILABLE = True
except ImportError as exc:  # pragma: no cover
    _IMPORT_ERROR = str(exc)


def layout_backend() -> str:
    """Return configured backend: ``docling`` (default) or ``yolo``."""
    raw = (os.getenv("LAYOUT_BACKEND", "docling") or "docling").strip().lower()
    if raw in ("yolo", "doclayout-yolo", "opencv"):
        return "yolo"
    return "docling"


def docling_ready() -> bool:
    return DOCLING_AVAILABLE and layout_backend() == "docling"


class DoclingBackend:
    """Thin wrapper around Docling DocumentConverter."""

    def __init__(self, ocr_engine=None, images_scale: float = 2.0) -> None:
        self.ocr_engine = ocr_engine
        self.images_scale = float(
            os.getenv("DOCLING_IMAGES_SCALE", str(images_scale)))
        self._converter: Any = None
        self._init_error: Optional[str] = None
        if DOCLING_AVAILABLE:
            try:
                self._converter = self._build_converter()
            except Exception as exc:  # noqa: BLE001
                self._init_error = str(exc)
                logger.exception("Docling converter init failed")

    def _build_converter(self):  # NOSONAR
        pipeline_options = PdfPipelineOptions()
        # Docling OCR is still needed for *region detection* on scans
        # (RapidOCR text is discarded; Thai-TrOCR re-reads in the adapter).
        # DOCLING_DO_OCR=0 collapses list/body boxes to tiny markers.
        speed = os.getenv("SPEED_MODE", "0").strip().lower() in (
            "1", "true", "yes", "on")
        do_ocr_env = (os.getenv("DOCLING_DO_OCR", "") or "").strip().lower()
        if do_ocr_env in ("0", "false", "no", "off"):
            do_ocr = False
        elif do_ocr_env in ("1", "true", "yes", "on"):
            do_ocr = True
        else:
            do_ocr = True  # default on even in SPEED_MODE (region detect)
        pipeline_options.do_ocr = do_ocr
        pipeline_options.do_table_structure = True
        if speed:
            logger.info("SPEED_MODE: Docling OCR kept for region detection")
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = self.images_scale
        logger.info("Docling do_ocr=%s (speed_mode=%s)", do_ocr, speed)

        # GPU accelerator when USE_GPU=true. Under SPEED_MODE + ≤2GB VRAM,
        # keep Docling on CPU so Thai-TrOCR can own the GPU (they fight
        # otherwise and band OCR returns empty/garbage).
        try:
            from docling.datamodel.accelerator_options import (
                AcceleratorDevice, AcceleratorOptions)
            use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
            try:
                vram = float(os.getenv("MAX_VRAM_MB", "0") or "0")
            except ValueError:
                vram = 0.0
            want = os.getenv("DOCLING_DEVICE", "").strip().lower()
            if want in ("cpu",):
                docling_cpu = True
            elif want in ("cuda", "gpu"):
                docling_cpu = False
            elif speed and 0 < vram <= 2048:
                # Don't co-reside Docling+TrOCR on a 1.5GB GPU — pick one.
                # Prefer Docling GPU when TrOCR is on CPU (compose default).
                trocr_dev = (os.getenv("TROCR_DEVICE") or "cpu").strip().lower()
                docling_cpu = trocr_dev in ("cuda", "gpu")
            else:
                docling_cpu = not use_gpu
            device = (
                AcceleratorDevice.CPU if (not use_gpu or docling_cpu)
                else AcceleratorDevice.CUDA)
            pipeline_options.accelerator_options = AcceleratorOptions(
                device=device,
                num_threads=max(1, int(os.getenv("DOCLING_NUM_THREADS", "4"))),
            )
            logger.info("Docling accelerator: %s", device.value)
        except Exception:  # noqa: BLE001
            logger.info("Docling accelerator options unavailable")

        # FAST tables by default for speed / low VRAM; ACCURATE via env
        table_mode = (os.getenv("DOCLING_TABLE_MODE", "fast") or "fast").lower()
        try:
            mode = (TableFormerMode.ACCURATE if table_mode in ("accurate", "exact")
                    else TableFormerMode.FAST)
            pipeline_options.table_structure_options = TableStructureOptions(
                mode=mode,
                do_cell_matching=True,
            )
            logger.info("Docling TableFormer mode: %s", mode.value)
        except Exception:  # noqa: BLE001
            pass

        # Prefer RapidOCR bridge for region detect (LocalOCR plugin TextCell
        # API drift breaks pages → empty boxes). Opt-in via env=1.
        use_plugin = os.getenv("DOCLING_USE_LOCALOCR_PLUGIN", "0").strip() in (
            "1", "true", "yes", "on")
        if do_ocr and use_plugin:
            try:
                from .docling_ocr_plugin import LocalOCROptions, _DOCLING_OCR, _IMPORT_ERR
                if not _DOCLING_OCR:
                    raise RuntimeError(
                        f"LocalOCR Docling plugin imports failed: {_IMPORT_ERR}")
                pipeline_options.allow_external_plugins = True
                langs = (os.getenv("LANGUAGES", DEFAULT_LANGUAGES)
                         .replace("+", ",").replace(" ", "").split(","))
                pipeline_options.ocr_options = LocalOCROptions(
                    lang=[x for x in langs if x] or ["tha", "eng"],
                    force_full_page_ocr=True,
                )
                logger.info("Docling OCR: LocalOCR plugin (Thai-TrOCR/Paddle)")
            except Exception as exc:  # noqa: BLE001
                use_plugin = False
                logger.info(
                    "LocalOCR Docling plugin not active (%s) — "
                    "using RapidOCR bridge + OCREngine re-OCR",
                    exc)

        if do_ocr and (not use_plugin or getattr(
                pipeline_options, "ocr_options", None) is None):
            try:
                from docling.datamodel.pipeline_options import RapidOcrOptions
                pipeline_options.ocr_options = RapidOcrOptions(
                    force_full_page_ocr=True,
                )
                logger.info("Docling OCR: RapidOCR bridge")
            except Exception:  # noqa: BLE001
                try:
                    from docling.datamodel.pipeline_options import EasyOcrOptions
                    pipeline_options.ocr_options = EasyOcrOptions(
                        lang=["th", "en"],
                        force_full_page_ocr=True,
                    )
                    logger.info("Docling OCR: EasyOCR bridge")
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "No Docling OCR options available; structure-only")
        elif not do_ocr:
            logger.info("Docling OCR disabled — adapter Thai-TrOCR only")

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options),
            }
        )

    @property
    def available(self) -> bool:
        return self._converter is not None

    def convert(self, pdf_path: str):
        """Run Docling on a PDF; return DoclingDocument or None."""
        if not self._converter:
            logger.error("Docling unavailable: %s",
                         self._init_error or _IMPORT_ERROR)
            return None
        result = self._converter.convert(pdf_path)
        return getattr(result, "document", result)

    def convert_to_blocks(
            self, pdf_path: str, languages: str = DEFAULT_LANGUAGES,
            page_images: Optional[Dict[int, np.ndarray]] = None,
    ) -> Tuple[List, int, int]:
        """Convert PDF → (ContentBlocks, n_tables, n_figures)."""
        from .docling_adapter import docling_to_blocks

        doc = self.convert(pdf_path)
        if doc is None:
            return [], 0, 0
        # Collect page images from Docling if not provided
        if not page_images:
            page_images = self._extract_page_images(doc)
        blocks = docling_to_blocks(
            doc, ocr=self.ocr_engine, page_images=page_images,
            languages=languages)
        n_tables = sum(1 for b in blocks if b.block_type == "table")
        n_figures = sum(1 for b in blocks if b.block_type == "figure")
        return blocks, n_tables, n_figures

    def convert_page_to_blocks(
            self, pdf_path: str, page_num: int,
            languages: str = DEFAULT_LANGUAGES,
            page_img: Optional[np.ndarray] = None,
    ) -> Tuple[List, int, int]:
        """Convert a single PDF page → blocks (progress-friendly)."""
        from .docling_adapter import docling_to_blocks
        import fitz

        tmp_path = None
        try:
            src = fitz.open(pdf_path)
            if page_num < 0 or page_num >= len(src):
                src.close()
                return [], 0, 0
            one = fitz.open()
            one.insert_pdf(src, from_page=page_num, to_page=page_num)
            src.close()
            fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
            os.close(fd)
            one.save(tmp_path)
            one.close()

            doc = self.convert(tmp_path)
            if doc is None:
                return [], 0, 0
            images = {0: page_img} if page_img is not None else (
                self._extract_page_images(doc))
            blocks = docling_to_blocks(
                doc, ocr=self.ocr_engine, page_images=images,
                languages=languages)
            for b in blocks:
                b.page = page_num
            n_tables = sum(1 for b in blocks if b.block_type == "table")
            n_figures = sum(1 for b in blocks if b.block_type == "figure")
            return blocks, n_tables, n_figures
        finally:
            if tmp_path and os.path.isfile(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            # Free GPU between pages under tight VRAM caps
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    def warm_up(self) -> None:
        """Force-load Docling layout + TableFormer + OCR weights once."""
        if not self._converter:
            return
        import fitz
        tmp_path = None
        try:
            doc = fitz.open()
            page = doc.new_page(width=400, height=600)
            page.insert_text((72, 72), "warmup")
            fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
            os.close(fd)
            doc.save(tmp_path)
            doc.close()
            logger.info("Warming Docling models (first load)…")
            self.convert(tmp_path)
            logger.info("Docling models warm — ready for convert")
        except Exception:  # noqa: BLE001
            logger.exception("Docling warm-up failed (first convert may be slow)")
        finally:
            if tmp_path and os.path.isfile(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def detect_page(
            self, pdf_path: str, page_num: int,
            page_img: Optional[np.ndarray] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Layout detections for one page (Review UI)."""
        from .docling_adapter import detections_from_docling

        # Build a one-page temp PDF for speed on large docs
        tmp_path = None
        try:
            import fitz
            src = fitz.open(pdf_path)
            if page_num < 0 or page_num >= len(src):
                src.close()
                return {"figures": [], "tables": [], "text_regions": [],
                        "formulas": [], "captions": [], "other": []}
            one = fitz.open()
            one.insert_pdf(src, from_page=page_num, to_page=page_num)
            src.close()
            fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
            os.close(fd)
            one.save(tmp_path)
            one.close()
            doc = self.convert(tmp_path)
            if doc is None:
                return {"figures": [], "tables": [], "text_regions": [],
                        "formulas": [], "captions": [], "other": []}
            return detections_from_docling(doc, page_idx=0, page_img=page_img)
        finally:
            if tmp_path and os.path.isfile(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def _extract_page_images(docling_doc) -> Dict[int, np.ndarray]:
        """Pull rendered page images from DoclingDocument when present."""
        out: Dict[int, np.ndarray] = {}
        try:
            pages = getattr(docling_doc, "pages", None) or {}
            for key, page in pages.items():
                try:
                    page_no = int(key) if not isinstance(key, int) else key
                except (TypeError, ValueError):
                    continue
                img = getattr(page, "image", None)
                if img is None:
                    continue
                pil = getattr(img, "pil_image", img)
                try:
                    arr = np.array(pil.convert("RGB"))[:, :, ::-1].copy()
                    out[page_no - 1 if page_no >= 1 else page_no] = arr
                except Exception:  # noqa: BLE001
                    continue
        except Exception:  # noqa: BLE001
            logger.exception("Failed to extract Docling page images")
        return out


def warm_docling_models() -> None:
    """Best-effort download/cache of Docling layout + TableFormer weights."""
    if not DOCLING_AVAILABLE:
        print("Docling not installed — skip model warm-up")
        return
    try:
        backend = DoclingBackend()
        if not backend.available:
            print("Docling converter unavailable:", backend._init_error)
            return
        # Touch converter so models resolve from HF cache
        print("Docling converter ready")
    except Exception as exc:  # noqa: BLE001
        print("Docling warm-up skipped:", exc)
