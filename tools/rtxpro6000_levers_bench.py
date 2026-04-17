#!/usr/bin/env python3
"""
Coding-bench variant with three prompting/inference levers switchable via flags:

  --thinking       enable reasoning budget on llama-server (-rea on --reasoning-budget 16384)
  --system-prompt  prepend a careful-engineer persona system message
  --agentic        on test failure, send failure output back and let model fix (up to 3 rounds)

Invocation:
  rtxpro6000_levers_bench.py <model_key> [--thinking] [--system-prompt] [--agentic]

Writes to experiments/rtxpro6000_levers/<model_key>__<mode>/.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path("/home/gisenberg/git/gisenberg/local-model-eval")
sys.path.insert(0, str(REPO / "tools"))

from rtxpro6000_bench import MODELS, start_server as _start_server, wait_for_server, stop_server, vram_used_mb
from rtxpro6000_coding_bench import (
    BENCHMARKS, load_prompt, score_response,
    extract_code_blocks, split_impl_test, fix_test_imports, loosen_pytest_raises, run_pytest,
)

OUTPUT_ROOT = REPO / "experiments/rtxpro6000_levers"
PORT = 8080

SYSTEM_PROMPT = """\
You are a precise, rigorous software engineer writing production-quality Python code.

Before you write a single line, walk through the requirements and identify:
- The exact invariants the implementation must preserve.
- All the edge cases (empty inputs, boundary values, interactions between multiple state updates).
- The semantics of each operation (e.g., "lazy cleanup" means size() should NOT count expired entries, even if they remain in the underlying map).
- Any subtle requirements that could be easily missed on a first read.

Then write the implementation carefully. Re-read each requirement against your code.

When writing tests, verify the tricky cases explicitly — not just the happy path. Each test should encode a specific semantic contract from the prompt.
"""


def start_server(model_cfg, extra_args=None, ctx_override=None):
    """Patched start_server that accepts extra CLI args."""
    cfg = dict(model_cfg)
    cfg["extra"] = (cfg.get("extra", []) or []) + (extra_args or [])
    return _start_server(cfg, ctx_override=ctx_override)


def chat_multi(messages, max_tokens=16384, temperature=0.0):
    body = json.dumps({
        "model": "local",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://localhost:{PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1200) as resp:
        obj = json.loads(resp.read())
    return obj["choices"][0]["message"]["content"]


def score_and_feedback(resp: str, module_name: str) -> tuple:
    """Score response; if failing, produce a fix-me prompt showing pytest output."""
    blocks = extract_code_blocks(resp)
    impl, tests = split_impl_test(blocks)
    if tests:
        tests = fix_test_imports(tests, module_name, impl)
        tests = loosen_pytest_raises(tests)
    result = run_pytest(impl, tests, module_name)
    return result


def make_fix_prompt(score: dict) -> str:
    tail = score.get("stdout_tail", "")[-1500:]
    return (
        "Your tests did not all pass. Here is the pytest output:\n\n"
        "```\n" + tail + "\n```\n\n"
        "Please produce a corrected version of the implementation and tests. "
        "Output exactly one ```python code block containing both the implementation "
        "and the tests, ready to run as a single file. "
        "Think carefully about why each failing test failed before rewriting."
    )


def benchmark_model(model_key: str, mode: str, thinking: bool, system_prompt: bool, agentic: bool, ctx: int = 32768) -> dict:
    cfg = dict(MODELS[model_key])
    cfg["key"] = model_key
    if not os.path.isfile(cfg["path"]):
        raise RuntimeError(f"Missing: {cfg['path']}")

    extra_args = []
    if thinking:
        extra_args += ["-rea", "on", "--reasoning-budget", "16384"]

    print(f"{'='*70}\n{cfg['name']} [mode={mode}] — coding levers\n{'='*70}")
    print(f"extra server args: {extra_args}")
    proc, log = start_server(cfg, extra_args=extra_args, ctx_override=ctx)
    print(f"Waiting for server (log: {log})...", flush=True)
    if not wait_for_server(timeout=900):
        stop_server(proc)
        raise RuntimeError("server start timeout")
    print("Ready.")
    vram_mb = vram_used_mb()

    artifacts_dir = OUTPUT_ROOT / f"{model_key}__{mode}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    bench_results = {}

    for bench_name, expected, module_name in BENCHMARKS:
        print(f"\n--- {bench_name} ---")
        user_prompt = load_prompt(bench_name)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": user_prompt})

        rounds = []
        round_idx = 0
        t_bench_start = time.perf_counter()
        max_rounds = 3 if agentic else 1

        while round_idx < max_rounds:
            try:
                resp = chat_multi(messages)
            except Exception as e:
                print(f"  round {round_idx}: ERROR {e}")
                bench_results[bench_name] = {"error": str(e), "expected": expected}
                resp = None
                break
            (artifacts_dir / f"{bench_name}__round{round_idx}.md").write_text(resp)
            score = score_and_feedback(resp, module_name)
            score["round"] = round_idx
            rounds.append(score)
            total = score.get("total", 0) or 0
            passed = score.get("passed", 0) or 0
            all_pass = passed == expected and total == expected
            print(f"  round {round_idx}: {passed}/{expected} "
                  f"(failed={score.get('failed', 0)}, errors={score.get('errors', 0)}, status={score.get('status')})")
            if all_pass or not agentic:
                break
            # Build fix-prompt conversation
            messages.append({"role": "assistant", "content": resp})
            messages.append({"role": "user", "content": make_fix_prompt(score)})
            round_idx += 1

        elapsed = time.perf_counter() - t_bench_start
        last = rounds[-1] if rounds else {"error": "no rounds"}
        entry = {
            "passed": last.get("passed", 0),
            "failed": last.get("failed", 0),
            "errors": last.get("errors", 0),
            "total": last.get("total", 0),
            "status": last.get("status"),
            "expected": expected,
            "n_rounds": len(rounds),
            "elapsed_s": round(elapsed, 2),
        }
        bench_results[bench_name] = entry
        print(f"  FINAL: {entry['passed']}/{expected} in {entry['n_rounds']} round(s), {elapsed:.1f}s")

    stop_server(proc)
    total_p = sum(r.get("passed", 0) for r in bench_results.values() if isinstance(r, dict))
    total_e = sum(r.get("expected", 0) for r in bench_results.values() if isinstance(r, dict))
    summary = {
        "model": cfg["name"],
        "key": model_key,
        "mode": mode,
        "flags": {"thinking": thinking, "system_prompt": system_prompt, "agentic": agentic},
        "vram_mb": vram_mb,
        "total_score": f"{total_p}/{total_e}",
        "benchmarks": bench_results,
    }
    print(f"\nTOTAL [{mode}]: {total_p}/{total_e}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_key")
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--system-prompt", action="store_true")
    p.add_argument("--agentic", action="store_true")
    p.add_argument("--mode", default=None, help="Label for output dir; auto-generated if omitted")
    args = p.parse_args()

    if args.model_key not in MODELS:
        print(f"Unknown model: {args.model_key}")
        sys.exit(1)

    mode = args.mode or "_".join(
        part for part, on in [("thinking", args.thinking), ("sysprompt", args.system_prompt), ("agentic", args.agentic)] if on
    ) or "baseline"

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = benchmark_model(
        args.model_key, mode=mode,
        thinking=args.thinking, system_prompt=args.system_prompt, agentic=args.agentic,
    )
    out = OUTPUT_ROOT / f"{args.model_key}__{mode}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
