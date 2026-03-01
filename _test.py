"""Quick pipeline test on the test PDF."""
from src.pipeline import OCRPipeline
r = OCRPipeline().process_pdf("tests/testocrtor-demo.pdf")
m = r["metadata"]
print(f"Pages={m['pages']} Tables={m['tables']} Figures={m['figures']} Success={r['success']}")
assert r["success"], f"Pipeline failed: {r.get('error')}"
assert m["tables"] == 2, f"Expected 2 tables, got {m['tables']}"
assert m["figures"] == 2, f"Expected 2 figures, got {m['figures']}"
print("ALL ASSERTIONS PASSED")
