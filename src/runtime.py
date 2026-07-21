"""CPU / concurrency runtime settings for multi-user parallel OCR.

Design goals:
  - Use as many host CPUs as possible
  - Allow several users to convert PDFs at once (Gradio queue)
  - Process pages of a single PDF in parallel
  - Cap concurrent model inferences so RAM/CPU do not thrash

Env knobs (all optional):
  MAX_CONCURRENT_JOBS  — parallel PDF conversions (multi-user)
  PAGE_WORKERS         — parallel pages inside one PDF
  MODEL_SLOTS          — concurrent YOLO/OCR inferences
  OMP_NUM_THREADS / TORCH_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS
  QUEUE_MAX_SIZE       — Gradio queue depth
"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    """Parse positive int env; blank / 0 / negative → *default* (0 = auto)."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if value <= 0:
        return default
    return value


CPU_COUNT = max(1, os.cpu_count() or 1)

# Multi-user: how many PDF conversions may run at once (use half the cores)
_DEFAULT_JOBS = max(2, min(CPU_COUNT, CPU_COUNT // 2))
MAX_CONCURRENT_JOBS = max(1, _env_int("MAX_CONCURRENT_JOBS", _DEFAULT_JOBS))

# Intra-PDF: parallel pages — use half the cores so a single big PDF
# saturates the machine; multi-user still shares MODEL_SLOTS.
_DEFAULT_PAGES = max(2, min(CPU_COUNT, CPU_COUNT // 2))
PAGE_WORKERS = max(1, _env_int("PAGE_WORKERS", _DEFAULT_PAGES))

# Shared inference budget across all jobs/pages (= all logical CPUs)
_DEFAULT_SLOTS = max(1, CPU_COUNT)
MODEL_SLOTS = max(1, _env_int("MODEL_SLOTS", _DEFAULT_SLOTS))

QUEUE_MAX_SIZE = max(1, _env_int("QUEUE_MAX_SIZE", 64))

# Prefer many parallel tasks over fat BLAS/OpenMP threads per task
TORCH_NUM_THREADS = max(1, _env_int("TORCH_NUM_THREADS", 1))
OMP_NUM_THREADS = max(1, _env_int("OMP_NUM_THREADS", 1))
OPENBLAS_NUM_THREADS = max(1, _env_int("OPENBLAS_NUM_THREADS", 1))
MKL_NUM_THREADS = max(1, _env_int("MKL_NUM_THREADS", 1))
NUMEXPR_NUM_THREADS = max(1, _env_int("NUMEXPR_NUM_THREADS", max(1, CPU_COUNT)))

_model_sema = threading.BoundedSemaphore(MODEL_SLOTS)
_configured = False
_configure_lock = threading.Lock()


def configure_native_threads() -> None:
    """Apply thread env + library limits once (call before heavy imports if possible)."""
    global _configured
    with _configure_lock:
        if _configured:
            return
        os.environ.setdefault("OMP_NUM_THREADS", str(OMP_NUM_THREADS))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(OPENBLAS_NUM_THREADS))
        os.environ.setdefault("MKL_NUM_THREADS", str(MKL_NUM_THREADS))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(NUMEXPR_NUM_THREADS))
        os.environ.setdefault("TORCH_NUM_THREADS", str(TORCH_NUM_THREADS))
        # Avoid OpenMP oversubscription when many page workers run
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        os.environ.setdefault("KMP_BLOCKTIME", "0")

        try:
            import torch
            torch.set_num_threads(TORCH_NUM_THREADS)
            if hasattr(torch, "set_num_interop_threads"):
                torch.set_num_interop_threads(max(1, min(4, CPU_COUNT)))
        except Exception:
            pass

        try:
            import cv2
            # 0 = let OpenCV use all cores for its own ops; we keep BLAS thin
            cv2.setNumThreads(max(0, _env_int("CV2_NUM_THREADS", 0)))
        except Exception:
            pass

        _configured = True
        logger.info(
            "Runtime CPU: cores=%d jobs=%d page_workers=%d model_slots=%d "
            "torch_threads=%d omp=%d",
            CPU_COUNT, MAX_CONCURRENT_JOBS, PAGE_WORKERS, MODEL_SLOTS,
            TORCH_NUM_THREADS, OMP_NUM_THREADS,
        )


@contextmanager
def model_slot() -> Iterator[None]:
    """Acquire a shared inference slot (YOLO / TrOCR / Paddle)."""
    _model_sema.acquire()
    try:
        yield
    finally:
        _model_sema.release()


def summary() -> dict:
    return {
        "cpu_count": CPU_COUNT,
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
        "page_workers": PAGE_WORKERS,
        "model_slots": MODEL_SLOTS,
        "queue_max_size": QUEUE_MAX_SIZE,
        "torch_num_threads": TORCH_NUM_THREADS,
        "omp_num_threads": OMP_NUM_THREADS,
    }
