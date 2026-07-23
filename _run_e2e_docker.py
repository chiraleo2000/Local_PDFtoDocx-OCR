"""Run e2e inside the GPU container with fixture paths."""
from pathlib import Path
import run_e2e_test as t

t.PDF = Path("/tmp/testocrtor-demo.pdf")
t.GOLD = Path("/tmp/Expected-output-testocr-demon.docx")
t.OUT_DIR = Path("/tmp/e2e_out")
raise SystemExit(t.main())
