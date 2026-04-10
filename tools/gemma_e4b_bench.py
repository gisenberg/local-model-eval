#!/usr/bin/env python3
"""Gemma 4 E4B benchmark: tiny model (2.3B active, 5.1B total) with f16 and turbo4 KV."""

import json, os, re, subprocess, sys, time, requests

LLAMA_SERVER = "T:/git/TheTom/llama-cpp-turboquant/build/bin/Release/llama-server.exe"
MODEL = os.path.expanduser("~/.lmstudio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf")
PORT = 8080
OUTPUT_DIR = "experiments/gemma_e4b_bench"
TEMP = 0.3
RUNS = 3
MAX_TOKENS = 16384

CONFIGS = [
    {"key": "f16", "ctk": "f16", "ctv": "f16", "label": "f16 KV (baseline)"},
    {"key": "turbo4", "ctk": "turbo4", "ctv": "turbo4", "label": "turbo4 KV"},
]

BENCHMARKS = {
    "expression_evaluator": {
        "name": "Expression Evaluator", "expected": 5,
        "prompt": "Build a mathematical expression evaluator in Python. Requirements:\n1. Support +, -, *, / with correct operator precedence\n2. Support parentheses for grouping\n3. Support unary minus (e.g., '-3', '-(2+1)')\n4. Support floating point numbers (e.g., '3.14')\n5. Raise ValueError for: mismatched parentheses, division by zero, invalid tokens, empty expressions\n6. Implement as class ExpressionEvaluator with evaluate(expr: str) -> float\n7. Use recursive descent parser — no eval() or ast.literal_eval()\n8. Include type hints and docstrings\n9. Write 5 pytest tests",
    },
    "astar": {
        "name": "A* Pathfinding", "expected": 6,
        "prompt": "Implement A* pathfinding on a weighted 2D grid in Python.\n1. Class AStarGrid with find_path(start, end) -> Optional[List[Tuple[int,int]]]\n2. 4-directional, Manhattan heuristic, heapq, walls (0), weighted cells\n3. Handle: start==end, walls, out-of-bounds (ValueError)\n4. Path must be optimal. Include type hints and docstrings\n5. Write 6 pytest tests",
    },
    "lru_cache": {
        "name": "LRU Cache with TTL", "expected": 6,
        "prompt": "Implement LRU cache with TTL in Python.\n1. Class TTLCache(capacity, default_ttl)\n2. get(key), put(key, value, ttl=None), delete(key), size()\n3. O(1) avg time. Doubly-linked list + hash map, no OrderedDict\n4. time.monotonic(), lazy cleanup. Type hints and docstrings\n5. Write 6 pytest tests using unittest.mock.patch on time.monotonic",
    },
    "string_processor": {
        "name": "String Processor", "expected": 5,
        "prompt": "Write class StringProcessor with:\n1. reverse_words(s) -> str (reverse word order, collapse spaces)\n2. count_vowels(s) -> int (case-insensitive)\n3. is_palindrome(s) -> bool (ignore case, spaces, punctuation)\n4. caesar_cipher(s, shift) -> str (a-z/A-Z only, support negative)\n5. most_common_word(s) -> Optional[str] (case-insensitive, first if tied)\nInclude type hints, docstrings, and 5 pytest tests.",
    },
}

def start_server(ctk, ctv):
    return subprocess.Popen(
        [LLAMA_SERVER, "-m", MODEL, "--port", str(PORT), "-c", "32768", "-ngl", "99",
         "-fa", "on", "-ctk", ctk, "-ctv", ctv, "-np", "1", "-rea", "off"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)

def wait_for_server(timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"http://localhost:{PORT}/health", timeout=2).json().get("status") == "ok":
                return True
        except: pass
        time.sleep(2)
    return False

def stop_server(proc):
    try: proc.terminate(); proc.wait(timeout=10)
    except: proc.kill(); proc.wait(timeout=5)

def get_vram_mb():
    try:
        return int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5).strip().split("\n")[0])
    except: return None

def run_inference(prompt):
    t0 = time.perf_counter()
    resp = requests.post(f"http://localhost:{PORT}/v1/chat/completions", json={
        "model": "local", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS, "temperature": TEMP}, timeout=600)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    d = resp.json(); msg = d["choices"][0]["message"]; u = d.get("usage", {})
    comp = u.get("completion_tokens", 0)
    return {"content": msg.get("content", ""), "tokens": comp,
            "elapsed_s": round(elapsed, 2), "tok_per_sec": round(comp/elapsed if elapsed > 0 else 0, 1),
            "finish_reason": d["choices"][0].get("finish_reason", "?")}

def extract_and_test(content, test_file):
    blocks = re.findall(r'```python\n(.*?)```', content, re.DOTALL)
    if not blocks: return {"passed": 0, "failed": 0, "errors": 0}
    combined = "\n\n".join(b.strip() for b in blocks)
    for p in [r'from \w+ import \w+']: combined = re.sub(p, '', combined)
    with open(test_file, "w", encoding="utf-8") as f: f.write(combined)
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
                          capture_output=True, text=True, timeout=30)
        out = r.stdout + r.stderr
        return {"passed": len(re.findall(r' PASSED', out)), "failed": len(re.findall(r' FAILED', out)),
                "errors": len(re.findall(r' ERROR', out))}
    except: return {"passed": 0, "failed": 0, "errors": 0}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}

    for cfg in CONFIGS:
        print(f"\n{'='*70}\nCONFIG: {cfg['label']}\n{'='*70}")
        proc = start_server(cfg["ctk"], cfg["ctv"])
        print("  Starting...", end="", flush=True)
        if not wait_for_server():
            print(" TIMEOUT"); stop_server(proc); continue
        vram = get_vram_mb()
        print(f" ready (VRAM: {vram} MB)")

        config_results = {}
        for bk, bench in BENCHMARKS.items():
            runs = []
            for run in range(1, RUNS + 1):
                print(f"  [{bench['name']}] Run {run}/{RUNS}...", end="", flush=True)
                result = run_inference(bench["prompt"])
                tf = f"{OUTPUT_DIR}/{cfg['key']}_{bk}_run{run}_test.py"
                tr = extract_and_test(result["content"], tf)
                print(f" {result['tok_per_sec']:.0f} tok/s | {tr['passed']}/{bench['expected']} | {result['tokens']} tok")
                runs.append({**result, **tr})
            best = max(r["passed"] for r in runs)
            avg = sum(r["passed"] for r in runs) / len(runs)
            config_results[bk] = {"best": best, "avg": round(avg, 1), "expected": bench["expected"], "runs": runs}

        total_best = sum(min(v["best"], v["expected"]) for v in config_results.values())
        total_avg = sum(min(v["avg"], v["expected"]) for v in config_results.values())
        total_exp = sum(v["expected"] for v in config_results.values())
        print(f"\n  TOTAL: Best-of-{RUNS} = {total_best}/{total_exp} ({total_best/total_exp*100:.0f}%), Avg = {total_avg:.1f}/{total_exp}")
        all_results[cfg["key"]] = {"vram_mb": vram, "results": config_results}
        stop_server(proc); time.sleep(3)

    with open(f"{OUTPUT_DIR}/results.json", "w") as f: json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}\nGEMMA 4 E4B COMPARISON (Q8_0, temp=0.3, best-of-3)")
    print(f"{'='*80}")
    print(f"{'Config':25s} | {'VRAM':>6s} | {'ExprEval':>10s} | {'A*':>10s} | {'LRU':>10s} | {'StrProc':>10s} | {'Total':>12s}")
    print("-"*100)
    for key, label in [("f16", "f16 KV (baseline)"), ("turbo4", "turbo4 KV")]:
        d = all_results.get(key, {})
        r = d.get("results", {})
        vram = d.get("vram_mb", "?")
        parts = []
        tb = 0; te = 0
        for bk in ["expression_evaluator", "astar", "lru_cache", "string_processor"]:
            b = r.get(bk, {})
            bp = min(b.get("best", 0), b.get("expected", 5))
            exp = b.get("expected", 5)
            tb += bp; te += exp
            parts.append(f"{bp}/{exp}")
        print(f"{label:25s} | {vram:>5}MB | {parts[0]:>10s} | {parts[1]:>10s} | {parts[2]:>10s} | {parts[3]:>10s} | {tb}/{te}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
