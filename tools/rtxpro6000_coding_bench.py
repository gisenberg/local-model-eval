#!/usr/bin/env python3
"""
Coding quality benchmark for Pro 6000 runs.

Runs the 4-benchmark coding suite (string_processor, expression_evaluator,
astar_pathfinding, lru_cache_ttl) against llama-server, extracts Python
code blocks from the response, and runs pytest to score each benchmark.

Uses stdlib only for HTTP (urllib). pytest is expected on PATH (we use
the one from micromamba cuda env).

Usage:
  rtxpro6000_coding_bench.py <model_key>
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path("/home/gisenberg/git/gisenberg/local-model-eval")
LLAMA_DIR = "/home/gisenberg/llama/llama-b8826"  # Vulkan build, will override for CUDA
MODELS_ROOT = "/home/gisenberg/models"
PORT = 8080
OUTPUT_ROOT = REPO / "experiments/rtxpro6000_coding"
PYTEST = "/home/gisenberg/.micromamba/envs/cuda/bin/pytest"

# Which benchmarks to run, and the expected test count
BENCHMARKS = [
    ("string_processor", 5, "string_processor"),
    ("expression_evaluator", 5, "expression_evaluator"),
    ("astar_pathfinding", 6, "astar"),
    ("lru_cache_ttl", 6, "lru_cache"),
]

# Reuse same model configs as the throughput bench
sys.path.insert(0, str(REPO / "tools"))
from rtxpro6000_bench import MODELS, start_server, wait_for_server, stop_server, vram_used_mb


# -------- Benchmark prompt extraction from .md ---------

def load_prompt(bench_md_name: str) -> str:
    """Extract the ``` ... ``` block under the '## Prompt' header."""
    md_path = REPO / "benchmarks" / f"{bench_md_name}.md"
    text = md_path.read_text()
    # Look for ```...``` block under "## Prompt"
    m = re.search(r"## Prompt\s*\n+```\s*\n(.*?)\n```", text, re.DOTALL)
    if not m:
        raise RuntimeError(f"No prompt block in {md_path}")
    return m.group(1).strip()


# -------- Chat completion client (stdlib) ---------

def chat(prompt: str, max_tokens: int = 16384, temperature: float = 0.0) -> str:
    body = json.dumps({
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://localhost:{PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        obj = json.loads(resp.read())
    return obj["choices"][0]["message"]["content"]


# -------- Code extraction + test running (adapted from extract_and_test.py) ---------

def extract_code_blocks(markdown: str) -> list:
    return re.findall(r"```python\n(.*?)```", markdown, re.DOTALL)


def is_test_block(code: str) -> bool:
    return ("def test_" in code or
            "import pytest" in code or
            "from pytest" in code or
            "@pytest." in code)


def split_impl_test(blocks: list) -> tuple:
    impl_parts = []
    test_parts = []
    for b in blocks:
        if is_test_block(b):
            test_parts.append(b)
        else:
            impl_parts.append(b)
    if not impl_parts and test_parts:
        bundled = "\n\n".join(test_parts)
        m = re.search(r"^(import pytest|from pytest|def test_|@pytest\.)", bundled, re.MULTILINE)
        if m:
            return bundled[:m.start()], bundled[m.start():]
        return "", bundled
    if not test_parts and impl_parts:
        return "\n\n".join(impl_parts), ""
    return "\n\n".join(impl_parts), "\n\n".join(test_parts)


def fix_test_imports(test_code: str, module_name: str, impl_code: str) -> str:
    class_names = re.findall(r"^class (\w+)", impl_code, re.MULTILINE)
    func_names = re.findall(r"^def (\w+)", impl_code, re.MULTILINE)
    symbols = set(class_names + func_names)

    def repl(m):
        from_mod, imports = m.group(1), m.group(2)
        if from_mod == module_name:
            return m.group(0)
        imp_names = {n.split(" as ")[0].strip() for n in imports.split(",")}
        if imp_names & symbols:
            return f"from {module_name} import {imports}"
        return m.group(0)

    return re.sub(r"from (\w+) import ([\w, ]+)", repl, test_code)


def loosen_pytest_raises(test_code: str) -> str:
    if "pytest.raises" not in test_code:
        return test_code
    pattern = re.compile(
        r'(pytest\.raises\s*\(\s*\w+(?:\.\w+)*)\s*,\s*match\s*=\s*(?:r?"[^"]*"|r?\'[^\']*\')'
    )
    return pattern.sub(r'\1', test_code)


def _tests_reference_impl(tests: str, impl: str) -> bool:
    """Tests import (or otherwise reference via `from X import Y`) impl symbols."""
    class_names = set(re.findall(r"^class (\w+)", impl, re.MULTILINE))
    func_names = set(re.findall(r"^def (\w+)", impl, re.MULTILINE))
    symbols = class_names | func_names
    for m in re.finditer(r"from (\w+) import ([\w, ]+)", tests):
        imp_names = {n.split(" as ")[0].strip() for n in m.group(2).split(",")}
        if imp_names & symbols:
            return True
    return False


def run_pytest(impl: str, tests: str, module_name: str) -> dict:
    if not impl and not tests:
        return {"passed": 0, "failed": 0, "errors": 0, "total": 0, "status": "no_code"}
    # Single-file mode: tests don't import impl symbols (they live in same file per the model)
    single_file = bool(impl) and bool(tests) and not _tests_reference_impl(tests, impl)
    if not impl or not tests:
        single_file = True
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        if single_file:
            combined = (impl + "\n\n" + tests) if impl else tests
            (tdp / f"test_{module_name}.py").write_text(combined)
        else:
            (tdp / f"{module_name}.py").write_text(impl)
            (tdp / f"test_{module_name}.py").write_text(tests)
        try:
            res = subprocess.run(
                [PYTEST, "-v", f"test_{module_name}.py"],
                cwd=td, capture_output=True, text=True, timeout=90,
            )
        except subprocess.TimeoutExpired:
            return {"passed": 0, "failed": 0, "errors": 0, "total": 0, "status": "timeout"}
        out = res.stdout
        passed = int((re.search(r"(\d+) passed", out) or re.search(r"(0)", "")).group(1) if re.search(r"(\d+) passed", out) else 0)
        failed = int(re.search(r"(\d+) failed", out).group(1)) if re.search(r"(\d+) failed", out) else 0
        errors = int(re.search(r"(\d+) error", out).group(1)) if re.search(r"(\d+) error", out) else 0
        return {
            "passed": passed, "failed": failed, "errors": errors,
            "total": passed + failed + errors,
            "status": "ok" if errors == 0 else "collection_error",
            "single_file": single_file,
            "stdout_tail": out[-2000:],
        }


def score_response(bench_key: str, module_name: str, resp: str) -> dict:
    blocks = extract_code_blocks(resp)
    impl, tests = split_impl_test(blocks)
    if tests:
        tests = fix_test_imports(tests, module_name, impl)
        tests = loosen_pytest_raises(tests)
    result = run_pytest(impl, tests, module_name)
    result["n_code_blocks"] = len(blocks)
    result["response_len"] = len(resp)
    return result


# -------- Main driver ---------

def benchmark_model(model_key: str, ctx: int = 32768) -> dict:
    cfg = dict(MODELS[model_key])
    cfg["key"] = model_key
    if not os.path.isfile(cfg["path"]):
        raise RuntimeError(f"Missing model: {cfg['path']}")

    print(f"{'='*70}\n{cfg['name']} — coding benchmarks\n{'='*70}")
    # Use smaller ctx for coding (saves VRAM for bigger output; 32K is plenty)
    proc, log = start_server(cfg, ctx_override=ctx)
    print(f"Waiting for server (log: {log})...", flush=True)
    if not wait_for_server(timeout=900):
        stop_server(proc)
        raise RuntimeError("server start timeout")
    print("Ready.")

    vram_mb = vram_used_mb()

    bench_results = {}
    artifacts_dir = OUTPUT_ROOT / model_key
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for bench_name, expected, module_name in BENCHMARKS:
        print(f"\n--- {bench_name} ---")
        prompt = load_prompt(bench_name)
        t0 = time.perf_counter()
        try:
            resp = chat(prompt)
        except Exception as e:
            print(f"  ERROR: {e}")
            bench_results[bench_name] = {"error": str(e)}
            continue
        elapsed = time.perf_counter() - t0
        (artifacts_dir / f"{bench_name}.md").write_text(resp)
        score = score_response(bench_name, module_name, resp)
        score["expected"] = expected
        score["elapsed_s"] = round(elapsed, 2)
        print(f"  {score.get('passed', 0)}/{expected} passed "
              f"(failed={score.get('failed', 0)}, errors={score.get('errors', 0)}, "
              f"status={score.get('status')}, {elapsed:.1f}s)")
        bench_results[bench_name] = score

    stop_server(proc)
    total_passed = sum(r.get("passed", 0) for r in bench_results.values())
    total_expected = sum(r.get("expected", 0) for r in bench_results.values())
    summary = {
        "model": cfg["name"],
        "key": model_key,
        "ctx": ctx,
        "vram_mb": vram_mb,
        "backend": "vulkan",
        "total_score": f"{total_passed}/{total_expected}",
        "benchmarks": bench_results,
    }
    print(f"\nTOTAL: {total_passed}/{total_expected}")
    return summary


def main():
    if len(sys.argv) < 2:
        print("Usage: rtxpro6000_coding_bench.py <model_key>")
        print("Keys:", " ".join(MODELS))
        sys.exit(1)
    key = sys.argv[1]
    if key not in MODELS:
        print(f"Unknown key {key!r}")
        sys.exit(1)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = benchmark_model(key)
    out_path = OUTPUT_ROOT / f"{key}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
