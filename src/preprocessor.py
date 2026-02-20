"""
OpenCV Pre-processing Pipeline
Deskew, denoise, binarise, CLAHE contrast, morphological ops.
"""
import logging
import cv2
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class OpenCVPreprocessor:
    """Full OpenCV pre-processing pipeline for page images before OCR."""

    def __init__(self, quality: str = "balanced"):
        self.quality = quality
        logger.info(f"OpenCV Preprocessor — quality={quality}")

    def preprocess(self, image: np.ndarray, *, quality: Optional[str] = None) -> np.ndarray:
        """Run the pre-processing pipeline; return cleaned image."""
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
            except Exception as exc:
                logger.warning(f"OpenCV step '{step_name}' failed: {exc}")
        return result

    @staticmethod
    def _steps_for_quality(q: str):
        if q == "fast":
            return ["denoise"]
        elif q == "accurate":
            return ["denoise", "deskew", "clahe", "binarise", "morphology"]
        return ["denoise", "deskew", "clahe"]

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
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

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
