"""
OpenCV Pre-processing Pipeline — v1.1
Deskew, denoise, CLAHE contrast, upscale, sharpen.

Quality presets:
  fast     — denoise + light CLAHE
  balanced — denoise, deskew, CLAHE, upscale, sharpen
  accurate — above + stronger enhance (OCR-friendly; optional binarise)

Neural OCR (Thai-TrOCR / PaddleOCR) prefers grayscale contrast enhancement
over hard binarisation. Set ENHANCE_BINARIZE=1 to restore adaptive threshold.
"""
import logging
import os
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class OpenCVPreprocessor:
    """Full OpenCV pre-processing pipeline for page images before OCR."""

    def __init__(self, quality: str = "accurate") -> None:
        self.quality = quality
        logger.info("OpenCV Preprocessor — quality=%s", quality)

    def preprocess(self, image: np.ndarray, *, quality: Optional[str] = None) -> np.ndarray:
        """Run the pre-processing pipeline; return cleaned image."""
        if image is None or image.size == 0:
            return image
        q = quality or self.quality
        steps = self._steps_for_quality(q)
        result = image.copy()

        for step_name in steps:
            fn = getattr(self, f"_step_{step_name}", None)
            if fn is None:
                continue
            try:
                out = fn(result)
                if out is not None and out.size > 0:
                    result = out
            except (cv2.error, ValueError, RuntimeError) as exc:
                logger.warning("OpenCV step '%s' failed: %s", step_name, type(exc).__name__)
        return result

    @staticmethod
    def enhance_figure(image: np.ndarray) -> np.ndarray:
        """Enhance a cropped figure/logo for clearer DOCX/HTML embedding.

        Soft denoise + CLAHE + mild unsharp — keeps colour when present.
        """
        if image is None or image.size == 0:
            return image
        try:
            colour = len(image.shape) == 3 and image.shape[2] >= 3
            if colour:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l_ch, a_ch, b_ch = cv2.split(lab)
                l_ch = cv2.fastNlMeansDenoising(l_ch, None, 6, 5, 15)
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                l_ch = clahe.apply(l_ch)
                blur = cv2.GaussianBlur(l_ch, (0, 0), 0.8)
                l_ch = cv2.addWeighted(l_ch, 1.35, blur, -0.35, 0)
                merged = cv2.merge([l_ch, a_ch, b_ch])
                return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            gray = image if len(image.shape) == 2 else cv2.cvtColor(
                image, cv2.COLOR_BGR2GRAY)
            gray = cv2.fastNlMeansDenoising(gray, None, 6, 5, 15)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            blur = cv2.GaussianBlur(gray, (0, 0), 0.8)
            return cv2.addWeighted(gray, 1.35, blur, -0.35, 0)
        except (cv2.error, ValueError, RuntimeError) as exc:
            logger.warning("Figure enhance failed: %s", type(exc).__name__)
            return image

    @staticmethod
    def _steps_for_quality(q: str) -> List[str]:
        binarize = os.getenv("ENHANCE_BINARIZE", "0").strip().lower() in (
            "1", "true", "yes", "on")
        if q == "fast":
            return ["denoise", "clahe"]
        if q == "accurate":
            steps = ["denoise", "deskew", "clahe", "upscale", "sharpen",
                     "enhance"]
            if binarize:
                steps.extend(["binarise", "morphology"])
            return steps
        # balanced
        return ["denoise", "deskew", "clahe", "upscale", "sharpen"]

    # ── Individual steps ──

    @staticmethod
    def _step_denoise(img: np.ndarray) -> np.ndarray:
        # Convert to grayscale first — color denoising is 10x slower
        # and we only need grayscale for OCR anyway
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Use smaller search/template windows for speed on large images
        h, w = gray.shape[:2]
        if h * w > 4_000_000:  # > ~2000x2000
            return cv2.fastNlMeansDenoising(gray, None, 8, 5, 15)
        return cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    @staticmethod
    def _step_deskew(img: np.ndarray) -> np.ndarray:
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                minLineLength=100, maxLineGap=10)
        if lines is None:
            return img
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 15:
                angles.append(angle)
        if not angles:
            return img
        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.3:
            return img
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def _step_clahe(img: np.ndarray) -> np.ndarray:
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return clahe.apply(gray)

    # Small renders make Thai tone/vowel marks only 1-2 px tall, which
    # both Thai-TrOCR and PaddleOCR misread. Upscale low-resolution pages
    # so the shortest side reaches a workable size before recognition.
    _UPSCALE_MIN_SHORT_SIDE = 2200
    _UPSCALE_MAX_FACTOR = 2.5

    @classmethod
    def _step_upscale(cls, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        short = min(h, w)
        if short >= cls._UPSCALE_MIN_SHORT_SIDE:
            return img
        scale = min(cls._UPSCALE_MAX_FACTOR,
                    cls._UPSCALE_MIN_SHORT_SIDE / max(short, 1))
        if scale <= 1.05:
            return img
        return cv2.resize(img, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _step_sharpen(img: np.ndarray) -> np.ndarray:
        """Unsharp mask — crisper glyph edges after denoise/upscale."""
        blur = cv2.GaussianBlur(img, (0, 0), 0.9)
        return cv2.addWeighted(img, 1.45, blur, -0.45, 0)

    @staticmethod
    def _step_enhance(img: np.ndarray) -> np.ndarray:
        """Accurate-mode boost: gentle gamma + local contrast for faint ink."""
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Mild gamma stretch toward mid-tones (helps washed scans)
        table = np.array(
            [((i / 255.0) ** 0.90) * 255 for i in range(256)],
            dtype=np.uint8)
        stretched = cv2.LUT(gray, table)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        return clahe.apply(stretched)

    @staticmethod
    def _step_binarise(img: np.ndarray) -> np.ndarray:
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 15)

    @staticmethod
    def _step_morphology(img: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated = cv2.dilate(img, kernel, iterations=1)
        return cv2.erode(dilated, kernel, iterations=1)
