#!/usr/bin/env python3
"""
Extract Python code blocks from benchmark .md output files and run pytest.

Usage:
    python tools/extract_and_test.py <benchmark_dir>

Walks the directory looking for .md files matching the benchmark patterns,
extracts the Python code blocks, splits them into impl + test files, runs
pytest, and reports pass counts. Does NOT fix any code bugs - runs exactly
as the model generated.

Output: prints a summary table and writes results to <dir>/test_results.json
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Map benchmark name in filename → expected test count
BENCHMARK_INFO = {
    "expression_evaluator": {"expected": 5, "module": "expression_evaluator"},
    "astar":                {"expected": 6, "module": "astar"},
    "lru_cache":            {"expected": 6, "module": "lru_cache"},
}


def extract_code_blocks(markdown: str) -> list[str]:
    """Extract all ```python ... ``` code blocks from markdown."""
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    return pattern.findall(markdown)


def is_test_block(code: str) -> bool:
    """Heuristic: a test block contains pytest imports or test_ functions."""
    return ("def test_" in code or
            "import pytest" in code or
            "from pytest" in code or
            "@pytest." in code)


def split_impl_and_test(blocks: list[str]) -> tuple[str, str]:
    """Split code blocks into implementation and test code.

    The model usually emits separate blocks for impl and tests, but sometimes
    bundles them. We handle both:
    - Multiple blocks: classify each block as impl or test
    - Single block: split on the first 'def test_' or 'import pytest' line
    """
    if not blocks:
        return "", ""

    impl_parts = []
    test_parts = []
    for block in blocks:
        if is_test_block(block):
            test_parts.append(block)
        else:
            impl_parts.append(block)

    # If everything ended up as test (or impl), try splitting the bundled block
    if not impl_parts and test_parts:
        bundled = "\n\n".join(test_parts)
        # Find first test marker
        m = re.search(r"^(import pytest|from pytest|def test_|@pytest\.)",
                      bundled, re.MULTILINE)
        if m:
            return bundled[:m.start()], bundled[m.start():]
        return "", bundled
    if not test_parts and impl_parts:
        bundled = "\n\n".join(impl_parts)
        return bundled, ""

    return "\n\n".join(impl_parts), "\n\n".join(test_parts)


def fix_test_imports(test_code: str, module_name: str, impl_code: str) -> str:
    """If tests import from a non-matching module name, rewrite the import.

    Per methodology: we don't fix code bugs but we DO fix import module names
    so the tests can find the implementation file.
    """
    # Find class names defined in the impl
    class_names = re.findall(r"^class (\w+)", impl_code, re.MULTILINE)
    func_names = re.findall(r"^def (\w+)", impl_code, re.MULTILINE)
    symbols = class_names + func_names

    # Find any 'from X import Y' lines in tests where X is not the right module
    def replace_import(match):
        from_module = match.group(1)
        imports = match.group(2)
        if from_module == module_name:
            return match.group(0)
        # Check if any of the imported names match our impl symbols
        imported_names = [n.strip() for n in imports.split(",")]
        if any(n.split(" as ")[0].strip() in symbols for n in imported_names):
            return f"from {module_name} import {imports}"
        return match.group(0)

    test_code = re.sub(
        r"from (\w+) import ([\w, ]+)",
        replace_import,
        test_code,
    )
    return test_code


def run_pytest(impl_code: str, test_code: str, module_name: str) -> dict:
    """Write the code to a temp dir and run pytest, return results."""
    if not impl_code or not test_code:
        return {
            "passed": 0, "failed": 0, "errors": 0, "total": 0,
            "status": "no_code",
            "stdout": "", "stderr": "",
        }

    with tempfile.TemporaryDirectory() as td:
        impl_path = Path(td) / f"{module_name}.py"
        test_path = Path(td) / f"test_{module_name}.py"
        impl_path.write_text(impl_code)
        test_path.write_text(test_code)

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-v", str(test_path)],
                cwd=td,
                capture_output=True,
                text=True,
                timeout=60,
            )
            stdout = result.stdout
            stderr = result.stderr
        except subprocess.TimeoutExpired:
            return {
                "passed": 0, "failed": 0, "errors": 0, "total": 0,
                "status": "timeout", "stdout": "", "stderr": "timeout 60s",
            }
        except Exception as e:
            return {
                "passed": 0, "failed": 0, "errors": 0, "total": 0,
                "status": "error", "stdout": "", "stderr": str(e),
            }

        # Parse pytest summary line: "5 passed", "3 passed, 2 failed", "1 error"
        passed = failed = errors = 0
        m = re.search(r"(\d+) passed", stdout)
        if m: passed = int(m.group(1))
        m = re.search(r"(\d+) failed", stdout)
        if m: failed = int(m.group(1))
        m = re.search(r"(\d+) error", stdout)
        if m: errors = int(m.group(1))

        return {
            "passed": passed, "failed": failed, "errors": errors,
            "total": passed + failed + errors,
            "status": "ok" if errors == 0 else "collection_error",
            "stdout": stdout, "stderr": stderr,
        }


def find_benchmark_files(directory: Path) -> dict:
    """Find .md files for each benchmark type in the directory."""
    found = {}
    for md_file in directory.glob("*.md"):
        name = md_file.stem.lower()
        for bench_key in BENCHMARK_INFO:
            if bench_key in name:
                found.setdefault(bench_key, []).append(md_file)
                break
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Benchmark output directory containing .md files")
    parser.add_argument("--save-extracted", action="store_true",
                        help="Save extracted .py files alongside the .md files")
    args = parser.parse_args()

    bench_dir = Path(args.directory)
    if not bench_dir.is_dir():
        print(f"Error: {bench_dir} is not a directory")
        sys.exit(1)

    files_by_bench = find_benchmark_files(bench_dir)
    if not files_by_bench:
        print(f"No benchmark .md files found in {bench_dir}")
        sys.exit(1)

    all_results = []
    print(f"\n{'=' * 90}")
    print(f"{'File':50s} | {'Bench':18s} | {'Pass/Total':>10s} | Status")
    print("-" * 90)

    for bench_key, md_files in files_by_bench.items():
        info = BENCHMARK_INFO[bench_key]
        for md_file in sorted(md_files):
            markdown = md_file.read_text()
            blocks = extract_code_blocks(markdown)
            impl, test = split_impl_and_test(blocks)
            test = fix_test_imports(test, info["module"], impl)

            if args.save_extracted:
                base = md_file.with_suffix("")
                if impl:
                    Path(f"{base}_impl.py").write_text(impl)
                if test:
                    Path(f"{base}_test.py").write_text(test)

            result = run_pytest(impl, test, info["module"])
            display_name = md_file.name[:48]
            print(
                f"{display_name:50s} | {bench_key:18s} | "
                f"{result['passed']:>2}/{info['expected']:<2}      | "
                f"{result['status']}"
            )
            all_results.append({
                "file": str(md_file.relative_to(bench_dir)),
                "benchmark": bench_key,
                "expected": info["expected"],
                **result,
            })

    print("=" * 90)

    # Summary by file
    print(f"\nSUMMARY:")
    by_file = {}
    for r in all_results:
        # Group by everything before the benchmark name
        key = r["file"].rsplit(r["benchmark"], 1)[0].rstrip("_")
        by_file.setdefault(key, []).append(r)

    for fname, results in by_file.items():
        total_passed = sum(r["passed"] for r in results)
        total_expected = sum(r["expected"] for r in results)
        pct = 100 * total_passed / total_expected if total_expected else 0
        print(f"  {fname}: {total_passed}/{total_expected} ({pct:.0f}%)")

    # Save raw results
    out_path = bench_dir / "test_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to {out_path}")


if __name__ == "__main__":
    main()
