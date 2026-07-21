"""Docling OCR plugin wrapping LocalOCR's Thai-TrOCR / PaddleOCR engines.

Register via pyproject.toml entry-point ``docling = src.docling_ocr_plugin``
and ``PdfPipelineOptions(allow_external_plugins=True, ocr_options=...)``.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, ClassVar, Iterable, List, Optional, Type

logger = logging.getLogger(__name__)

_DOCLING_OCR = False
_IMPORT_ERR: Optional[str] = None

try:
    from docling.datamodel.pipeline_options import OcrOptions
    from docling.models.base_ocr_model import BaseOcrModel
    from docling.datamodel.base_models import BoundingBox, TextCell
    try:
        from docling_core.types.doc.base import CoordOrigin
    except ImportError:  # pragma: no cover
        from docling_core.types.doc import CoordOrigin  # type: ignore
    _DOCLING_OCR = True
except ImportError as exc:  # pragma: no cover
    _IMPORT_ERR = str(exc)
    OcrOptions = object  # type: ignore
    BaseOcrModel = object  # type: ignore
    TextCell = object  # type: ignore
    BoundingBox = object  # type: ignore
    CoordOrigin = None  # type: ignore


if _DOCLING_OCR:
    class LocalOCROptions(OcrOptions):
        """Options for LocalOCR (Thai-TrOCR + PaddleOCR) inside Docling."""

        kind: ClassVar[str] = "localocr"
        lang: List[str] = ["tha", "eng"]
        force_full_page_ocr: bool = True

    class LocalOCRModel(BaseOcrModel):
        """Docling OCR engine that delegates to ``src.ocr_engine.OCREngine``."""

        def __init__(self, *, enabled: bool, artifacts_path: Optional[Path],
                     options: "LocalOCROptions",
                     accelerator_options: Any = None, **kwargs: Any) -> None:
            # Older/newer Docling variants differ slightly on kwargs
            try:
                super().__init__(
                    enabled=enabled,
                    artifacts_path=artifacts_path,
                    options=options,
                    accelerator_options=accelerator_options,
                )
            except TypeError:
                super().__init__(enabled=enabled, options=options, **kwargs)
            self.options = options
            self._engine = None
            if enabled:
                from .ocr_engine import OCREngine
                self._engine = OCREngine()

        @classmethod
        def get_options_type(cls) -> Type[OcrOptions]:
            return LocalOCROptions

        def __call__(self, conv_res, page_batch: Iterable) -> Iterable:
            if not self.enabled or self._engine is None:
                yield from page_batch
                return
            langs = "+".join(self.options.lang) if self.options.lang else (
                os.getenv("LANGUAGES", "tha+eng"))
            for page in page_batch:
                try:
                    page = self._ocr_page(page, langs)
                except Exception:  # noqa: BLE001
                    logger.exception("LocalOCR Docling plugin failed on a page")
                yield page

        def _ocr_page(self, page, languages: str):
            """Run OCREngine on the page image and attach TextCells."""
            image = getattr(page, "image", None)
            if image is None:
                return page
            import numpy as np
            if hasattr(image, "mode"):  # PIL
                arr = np.array(image.convert("RGB"))[:, :, ::-1]
            else:
                arr = np.asarray(image)
                if arr.ndim == 3 and arr.shape[2] == 3:
                    arr = arr[:, :, ::-1].copy()

            result = self._engine.ocr_full_page(arr, languages=languages)
            cells: List[Any] = []
            for i, seg in enumerate(result.get("lines") or []):
                text = (seg.get("text") or "").strip()
                bbox = seg.get("bbox")
                if not text or not bbox:
                    continue
                try:
                    if isinstance(bbox[0], (list, tuple)):
                        xs = [float(p[0]) for p in bbox]
                        ys = [float(p[1]) for p in bbox]
                        l, t, r, b = min(xs), min(ys), max(xs), max(ys)
                    else:
                        l, t, r, b = (float(bbox[0]), float(bbox[1]),
                                      float(bbox[2]), float(bbox[3]))
                except (TypeError, ValueError, IndexError):
                    continue
                conf = seg.get("conf", seg.get("confidence", 1.0))
                try:
                    conf_f = float(conf) if conf is not None else 1.0
                except (TypeError, ValueError):
                    conf_f = 1.0
                origin = CoordOrigin.TOPLEFT if CoordOrigin is not None else None
                bb_kwargs = {"l": l, "t": t, "r": r, "b": b}
                if origin is not None:
                    bb_kwargs["coord_origin"] = origin
                cells.append(TextCell(
                    index=i,
                    text=text,
                    orig=text,
                    from_ocr=True,
                    confidence=conf_f,
                    rect=BoundingBox(**bb_kwargs),
                ))
            if hasattr(page, "cells"):
                page.cells = list(getattr(page, "cells", []) or []) + cells
            if hasattr(self, "post_process_cells") and cells:
                try:
                    self.post_process_cells(conv_res=None, page=page,
                                            ocr_cells=cells)
                except TypeError:
                    try:
                        self.post_process_cells(page, cells)
                    except Exception:  # noqa: BLE001
                        pass
                except Exception:  # noqa: BLE001
                    pass
            return page

else:  # pragma: no cover
    class LocalOCROptions:  # type: ignore
        kind = "localocr"
        lang = ["tha", "eng"]

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class LocalOCRModel:  # type: ignore
        @classmethod
        def get_options_type(cls):
            return LocalOCROptions


def ocr_engines():
    """Docling plugin factory entry."""
    if not _DOCLING_OCR:
        logger.warning("Docling OCR plugin unavailable: %s", _IMPORT_ERR)
    return {"ocr_engines": [LocalOCRModel]}
