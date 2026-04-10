#!/usr/bin/env python3
"""
Score model outputs by combining ALL python code blocks into a single file
and running pytest. Matches the approach used by qwen3_8b_comparison.py and
other per-model bench scripts — avoids the missing-import problem when models
emit impl + tests in one bundled block.

Usage:
    python tools/score_combined.py <benchmark_dir>
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

BENCHMARK_INFO = {
    "expression_evaluator": {"expected": 5, "class": "ExpressionEvaluator"},
    "astar":                {"expected": 6, "class": "AStarGrid"},
    "lru_cache":            {"expected": 6, "class": "TTLCache"},
}


def extract_blocks(markdown: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", markdown, re.DOTALL)


def is_test_block(code: str) -> bool:
    return ("def test_" in code or
            "import pytest" in code or
            "from pytest" in code or
            "@pytest." in code or
            "unittest.TestCase" in code)


def combine_and_test(content: str, class_name: str) -> dict:
    blocks = extract_blocks(content)
    if not blocks:
        return {"passed": 0, "failed": 0, "errors": 0, "status": "no_code", "stdout": ""}

    # Models often emit multiple impl blocks (stubs, examples, the final rewrite).
    # Pick the LAST block that actually *defines* the canonical class as the impl,
    # and the LAST test block. This avoids duplicate-class collisions and the
    # "usage example" block being mistaken for the impl.
    class_def_re = re.compile(rf"^class {re.escape(class_name)}\b", re.MULTILINE)
    impl_candidates = [b for b in blocks if class_def_re.search(b) and not is_test_block(b)]
    test_blocks = [b for b in blocks if is_test_block(b)]

    parts = []
    if impl_candidates:
        parts.append(impl_candidates[-1].strip())
    elif blocks:
        # Fall back: last non-test block, or last block of any kind
        non_test = [b for b in blocks if not is_test_block(b)]
        parts.append((non_test[-1] if non_test else blocks[-1]).strip())

    if test_blocks:
        # If a test block also contains the class definition (bundled), use it standalone
        bundled = [t for t in test_blocks if class_def_re.search(t)]
        if bundled:
            parts = [bundled[-1].strip()]
        else:
            parts.append(test_blocks[-1].strip())

    combined = "\n\n".join(parts)

    # Strip self-imports the model may have added (e.g. `from expression_evaluator import ...`)
    for p in [
        r'from \w+ import ExpressionEvaluator.*\n',
        r'from \w+ import AStarGrid.*\n',
        r'from \w+ import TTLCache.*\n',
    ]:
        combined = re.sub(p, '', combined)

    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "test_combined.py"
        f.write_text(combined)
        try:
            r = subprocess.run(
                [sys.executable, "-m", "pytest", "-v", str(f)],
                cwd=td, capture_output=True, text=True, timeout=60,
            )
            out = r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            return {"passed": 0, "failed": 0, "errors": 0, "status": "timeout", "stdout": ""}

        passed = failed = errors = 0
        m = re.search(r"(\d+) passed", out)
        if m: passed = int(m.group(1))
        m = re.search(r"(\d+) failed", out)
        if m: failed = int(m.group(1))
        m = re.search(r"(\d+) error", out)
        if m: errors = int(m.group(1))

        return {
            "passed": passed, "failed": failed, "errors": errors,
            "status": "ok" if errors == 0 else "collection_error",
            "stdout": out[-2000:],  # tail of output for debugging
        }


def find_files(directory: Path) -> dict:
    found = {}
    for md in directory.glob("*.md"):
        for bk in BENCHMARK_INFO:
            if bk in md.stem.lower():
                found.setdefault(bk, []).append(md)
                break
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory")
    args = p.parse_args()

    bd = Path(args.directory)
    files = find_files(bd)
    if not files:
        print(f"No .md benchmark files in {bd}")
        sys.exit(1)

    all_results = []
    print(f"\n{'=' * 90}")
    print(f"{'File':55s} | {'Bench':18s} | Pass/Total | Status")
    print("-" * 90)

    for bk, mds in files.items():
        info = BENCHMARK_INFO[bk]
        for md in sorted(mds):
            r = combine_and_test(md.read_text(), info["class"])
            display = md.name[:53]
            print(f"{display:55s} | {bk:18s} |   {r['passed']}/{info['expected']}      | {r['status']}")
            all_results.append({
                "file": md.name,
                "benchmark": bk,
                "expected": info["expected"],
                **r,
            })

    print("=" * 90)

    by_model = {}
    for r in all_results:
        key = r["file"].rsplit("_" + r["benchmark"], 1)[0]
        by_model.setdefault(key, []).append(r)

    print("\nSUMMARY:")
    for k, rs in by_model.items():
        passed = sum(r["passed"] for r in rs)
        exp = sum(r["expected"] for r in rs)
        pct = 100 * passed / exp if exp else 0
        per_bench = ", ".join(f"{r['benchmark']}={r['passed']}/{r['expected']}" for r in rs)
        print(f"  {k}: {passed}/{exp} ({pct:.0f}%)  [{per_bench}]")

    out = bd / "test_results_combined.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
