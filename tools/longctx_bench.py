#!/usr/bin/env python3
"""
Long-context coding benchmark runner for Spark.

Research questions:
1. Does throughput degrade as context grows (32K vs 64K vs 128K)?
2. Does code quality hold when the model must understand a real codebase?
3. Does the model actually use distant context or just attend to recent tokens?

The task: given the contents of this repo's tools/*.py files as context, add a
new --max-tokens N CLI flag to spark_bench.py that overrides the per-model
max_tokens default. The model must:
- Find spark_bench.py in the context
- Understand the existing argparse + per-model config pattern
- Produce a diff that correctly threads the new flag through all the call sites
  (run_throughput_benchmark and run_code_benchmarks)

We bracket spark_bench.py with unrelated files so it's not at the end of the
context (which would make the "long context" trivial — the model would just
attend to recent tokens). The target file is placed based on the context size
being tested.

Scoring is a 6-point rubric on the generated diff — see score_diff() below.

Usage:
    python tools/longctx_bench.py --models qwen122b-bartowski-ik --ctx 32768 65536 131072
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

# Reuse paths from spark_bench.py so we stay in sync with its engine config.
# These are copy-pasted rather than imported to keep this script standalone.
TURBOQUANT_SERVER = os.environ.get(
    "TURBOQUANT_SERVER",
    os.path.expanduser("~/git/TheTom/llama-cpp-turboquant/build/bin/llama-server"),
)
STANDARD_SERVER = os.environ.get(
    "STANDARD_SERVER",
    os.path.expanduser("~/llama.cpp/build/bin/llama-server"),
)
IK_SERVER = os.environ.get(
    "IK_SERVER",
    os.path.expanduser("~/git/ikawrakow/ik_llama.cpp/build/bin/llama-server"),
)
MODELS_DIR = os.path.expanduser("~/.lmstudio/models")
REPO_DIR = Path(__file__).parent.parent

# Models to test. Only the ones we care about for long-context — our current
# S/A-tier set.
MODEL_CONFIGS = {
    "qwen122b-bartowski-ik": {
        "name": "Qwen3.5-122B-A10B Q4_K_M (bartowski) [ik-llama]",
        "path": f"{MODELS_DIR}/bartowski/Qwen3.5-122B-A10B-GGUF/Qwen_Qwen3.5-122B-A10B-Q4_K_M/Qwen_Qwen3.5-122B-A10B-Q4_K_M-00001-of-00002.gguf",
        "server": "ik",
        "reasoning_arg": "--reasoning-budget",
        "reasoning_value": "0",
    },
    "qwen122b-unsloth": {
        "name": "Qwen3.5-122B-A10B Q4_K_M (unsloth)",
        "path": f"{MODELS_DIR}/unsloth/Qwen3.5-122B-A10B-GGUF/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf",
        "server": "standard",
        "reasoning_arg": "-rea",
        "reasoning_value": "off",
    },
    "qwen-coder": {
        "name": "Qwen3-Coder-Next UD-Q4_K_M",
        "path": f"{MODELS_DIR}/unsloth/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-UD-Q4_K_M.gguf",
        "server": "standard",
        "reasoning_arg": "-rea",
        "reasoning_value": "off",
    },
}

DEFAULT_PORT = 8080


# ---------------------------------------------------------------------------
# Context assembly: real files from tools/, with spark_bench.py buried in
# the middle so the model has to use distant context, not just recent tokens.
# ---------------------------------------------------------------------------

# Rough tokens-per-byte for English Python code (empirical: ~0.25 for most
# tokenizers). Used only for target-size estimation when selecting files.
CHARS_PER_TOKEN_EST = 4.0

def read_file(path: Path) -> str:
    return path.read_text()

def wrap_file(path: Path, content: str) -> str:
    """Wrap a file's content with a header showing its relative path."""
    rel = path.relative_to(REPO_DIR)
    return f"=== BEGIN FILE: {rel} ===\n{content}\n=== END FILE: {rel} ===\n"

def select_context_files(target_chars: int) -> list[Path]:
    """Pick files from tools/ to roughly fill target_chars of context.

    We always include spark_bench.py (the file the task asks to modify). We
    pad with other tools/*.py files to reach approximately target_chars,
    preferring smaller files for padding variety. spark_bench.py is placed
    in the *middle* of the returned list so it's far from both ends of the
    context — neither the head nor the recent-attention tail.
    """
    tools_dir = REPO_DIR / "tools"
    spark = tools_dir / "spark_bench.py"
    if not spark.is_file():
        raise SystemExit(f"Missing {spark}")

    all_py = sorted(p for p in tools_dir.glob("*.py") if p.name != "spark_bench.py")
    # Don't include the longctx_bench.py script itself — would be weird.
    all_py = [p for p in all_py if p.name != "longctx_bench.py"]

    spark_chars = len(read_file(spark))
    need_chars = max(0, target_chars - spark_chars - 2000)  # leave room for prompt

    picked: list[Path] = []
    running = 0
    # Prefer small-to-medium files first so we get variety before one giant
    # file dominates.
    all_py.sort(key=lambda p: p.stat().st_size)
    for p in all_py:
        sz = p.stat().st_size
        if running + sz > need_chars:
            continue
        picked.append(p)
        running += sz
        if running >= need_chars * 0.9:
            break

    # Place spark_bench.py in the middle of the picked list.
    mid = len(picked) // 2
    picked.insert(mid, spark)
    return picked


TASK_PROMPT = """I need you to modify one of the files above.

**Task:** Add a new CLI flag `--max-tokens N` to `tools/spark_bench.py` that
lets a user override the per-model `max_tokens` default from the command line.

**Requirements:**
1. Add an `argparse` argument named `--max-tokens` to the `main()` function,
   with `type=int`, `default=None` (so it's optional).
2. When the user provides `--max-tokens N` on the command line, it must
   override the `max_tokens` value from the per-model config dict (the
   `MODEL_CONFIGS` entries currently have a `max_tokens` field for some models
   like `minimax-m25`). The resolution order should be:
     CLI flag > per-model config > hardcoded default of 16384.
3. The resolved value must be passed through to BOTH `run_throughput_benchmark`
   AND `run_code_benchmarks`. Both functions already accept a `max_tokens`
   keyword argument — you just need to wire it up.
4. Do NOT modify any files other than `tools/spark_bench.py`.
5. Produce the complete change as a **unified diff** (format compatible with
   `patch -p1`). Do not paraphrase the changes — emit the actual diff text.

Reply with exactly one fenced `diff` code block containing the unified diff.
No commentary before or after the code block."""


def build_prompt(target_chars: int) -> tuple[str, list[str]]:
    """Build the long-context prompt. Returns (full_prompt, file_list)."""
    files = select_context_files(target_chars)
    parts = []
    for f in files:
        parts.append(wrap_file(f, read_file(f)))
    context = "\n".join(parts)
    file_names = [str(f.relative_to(REPO_DIR)) for f in files]

    prompt = (
        "Below are the contents of several Python files from a code benchmark "
        "repository. Read them carefully — you'll need to understand one of "
        "them to answer the task that follows.\n\n"
        f"{context}\n\n"
        f"{TASK_PROMPT}"
    )
    return prompt, file_names


# ---------------------------------------------------------------------------
# Server management (mirrors spark_bench.py, simplified for this single task)
# ---------------------------------------------------------------------------

def get_server_binary(server_name: str) -> str:
    mapping = {
        "standard": STANDARD_SERVER,
        "ik": IK_SERVER,
        "turboquant": TURBOQUANT_SERVER,
    }
    path = mapping.get(server_name, STANDARD_SERVER)
    if not os.path.isfile(path):
        raise SystemExit(f"Missing server binary: {path}")
    return path

def start_server(bin_path: str, model_path: str, port: int, context_length: int,
                 reasoning_arg: str, reasoning_value: str) -> subprocess.Popen:
    is_ik = "ik_llama" in bin_path
    cmd = [
        bin_path,
        "-m", model_path,
        "--port", str(port),
        "-c", str(context_length),
        "-ngl", "99",
        "-fa", "on",
        "-ctk", "f16",
        "-ctv", "f16",
        "-np", "1",
        "--temp", "0",
        "--no-mmap",
        reasoning_arg, reasoning_value,
    ]
    if not is_ik:
        cmd.append("--jinja")
    print(f"  Server: {os.path.basename(bin_path)}")
    print(f"  Context: {context_length} tokens")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def wait_for_server(port: int, timeout: int = 900) -> bool:
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=2)
            if resp.json().get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(2)
    return False

def stop_server(proc: subprocess.Popen):
    if proc is None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=15)
    except Exception:
        proc.kill()
        proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# Inference: non-streaming, long request timeout since this is a big job
# ---------------------------------------------------------------------------

def run_inference(port: int, prompt: str, max_tokens: int = 8192,
                  request_timeout: int = 2400) -> dict:
    """Streaming inference so we can measure prefill (TTFT) and decode rate
    separately. The blended "completion / total_elapsed" metric is misleading
    on long-context benchmarks because prefill can dominate — a 90K-token
    prompt at 500 tok/s prefill is 180s by itself, before any generation.
    """
    t0 = time.perf_counter()
    first_token_at = None
    content_parts = []
    reasoning_parts = []
    server_usage = None
    finish_reason = "?"

    resp = requests.post(
        f"http://localhost:{port}/v1/chat/completions",
        json={
            "model": "local-model",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        },
        stream=True,
        timeout=request_timeout,
    )
    resp.raise_for_status()

    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8", errors="replace")
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if "usage" in chunk and chunk["usage"]:
            server_usage = chunk["usage"]

        choices = chunk.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        fr = choices[0].get("finish_reason")
        if fr:
            finish_reason = fr

        reasoning = delta.get("reasoning_content", "")
        if reasoning:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            reasoning_parts.append(reasoning)

        content = delta.get("content", "")
        if content:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            content_parts.append(content)

    end = time.perf_counter()
    total_elapsed = end - t0
    prefill_s = (first_token_at - t0) if first_token_at else total_elapsed
    decode_s = (end - first_token_at) if first_token_at else 0.0

    usage = server_usage or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0) or (len(content_parts) + len(reasoning_parts))

    prefill_rate = (prompt_tokens / prefill_s) if prefill_s > 0 and prompt_tokens else 0
    decode_rate = (completion_tokens / decode_s) if decode_s > 0 and completion_tokens else 0
    blended_rate = (completion_tokens / total_elapsed) if total_elapsed > 0 else 0

    return {
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "usage": usage,
        "finish_reason": finish_reason,
        "elapsed_s": round(total_elapsed, 2),
        "prefill_s": round(prefill_s, 2),
        "decode_s": round(decode_s, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prefill_tok_per_sec": round(prefill_rate, 2),
        "decode_tok_per_sec": round(decode_rate, 2),
        "blended_tok_per_sec": round(blended_rate, 2),
        # Keep old name as alias for backward compat with existing scoring code
        "tok_per_sec": round(blended_rate, 2),
    }


# ---------------------------------------------------------------------------
# Scoring: 6-point rubric on the generated diff
# ---------------------------------------------------------------------------

def extract_diff(content: str) -> str | None:
    """Extract the first diff/unified code block from the response."""
    patterns = [
        r"```diff\n(.*?)```",
        r"```patch\n(.*?)```",
        r"```unified-diff\n(.*?)```",
        r"```\n(.*?)```",  # fallback: any fenced block
    ]
    for p in patterns:
        m = re.search(p, content, re.DOTALL)
        if m:
            text = m.group(1)
            if "@@" in text or "---" in text or "+++" in text:
                return text
    return None


def score_diff(diff_text: str | None) -> dict:
    """Score the diff against our 6-point rubric.

    Rubric (each is 0 or 1):
    1. Adds an argparse argument named --max-tokens
    2. Uses default=None (so the CLI doesn't override when not provided)
    3. Threads the value through to run_throughput_benchmark
    4. Threads the value through to run_code_benchmarks
    5. Only modifies tools/spark_bench.py (no other files)
    6. Is a syntactically plausible unified diff (has @@ hunk markers and +/- lines)

    Scoring notes:
    - For #3/#4, the existing code already passes max_tokens to run_code_benchmarks
      (it was wired up when we added per-model max_tokens). A correct diff may
      leave that call unchanged. We therefore check both added lines AND the
      post-application state of the code — if the diff introduces a new flow
      that preserves/uses max_tokens at that call site, we credit it.
    - Regexes use re.DOTALL to match across newlines since argparse calls and
      function calls typically span multiple lines in formatted Python.
    """
    result = {
        "diff_extracted": diff_text is not None,
        "1_argparse_added": 0,
        "2_default_none": 0,
        "3_threads_throughput": 0,
        "4_threads_code": 0,
        "5_only_spark_bench": 0,
        "6_valid_diff_format": 0,
        "total": 0,
    }
    if not diff_text:
        return result

    # 6. Valid diff format
    if re.search(r"@@.*@@", diff_text) and re.search(r"^\+", diff_text, re.MULTILINE):
        result["6_valid_diff_format"] = 1

    # 5. Only spark_bench.py modified (look at diff headers)
    file_headers = re.findall(r"^\+\+\+ [ab]?/(\S+)", diff_text, re.MULTILINE)
    if file_headers and all("spark_bench.py" in f for f in file_headers):
        result["5_only_spark_bench"] = 1
    elif not file_headers:
        # No explicit file headers — check if the diff body mentions only one file
        if "spark_bench" in diff_text and diff_text.count("+++") <= 1:
            result["5_only_spark_bench"] = 1

    # Look only at added lines for the rest
    added = "\n".join(
        line[1:] for line in diff_text.split("\n")
        if line.startswith("+") and not line.startswith("+++")
    )

    # 1. argparse --max-tokens added. Use DOTALL because argparse add_argument
    # calls span multiple lines — "add_argument(" and the flag string are
    # usually on separate lines in formatted Python.
    if re.search(
        r"add_argument\s*\([^)]*?['\"]--max[_-]tokens['\"]",
        added, re.DOTALL
    ):
        result["1_argparse_added"] = 1

    # 2. default=None in the new argument
    if re.search(
        r"add_argument\s*\([^)]*?['\"]--max[_-]tokens['\"][^)]*?default\s*=\s*None",
        added, re.DOTALL
    ):
        result["2_default_none"] = 1

    # 3. run_throughput_benchmark receives max_tokens. Check both added lines
    # (in case the model rewrote the call) and the full diff text (in case
    # the model kept the call site and just added a context line).
    if (re.search(r"run_throughput_benchmark\s*\([^)]*?max_tokens",
                  added, re.DOTALL) or
        re.search(r"run_throughput_benchmark\s*\([^)]*?max_tokens\s*=",
                  diff_text, re.DOTALL)):
        result["3_threads_throughput"] = 1

    # 4. run_code_benchmarks receives max_tokens. The existing code already
    # does this — a correct diff may leave the call unchanged, so check the
    # full diff_text (which includes context lines), not just added lines.
    if re.search(r"run_code_benchmarks\s*\([^)]*?max_tokens\s*=",
                 diff_text, re.DOTALL):
        result["4_threads_code"] = 1

    result["total"] = sum(
        v for k, v in result.items()
        if k not in ("total", "diff_extracted") and isinstance(v, int)
    )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(model_key: str, ctx_size: int, port: int, output_dir: Path) -> dict:
    cfg = MODEL_CONFIGS[model_key]
    bin_path = get_server_binary(cfg["server"])

    # Size the prompt to fit the context window with room for output
    # (reserve ~30% of the window for output + overhead)
    target_chars = int(ctx_size * CHARS_PER_TOKEN_EST * 0.65)
    prompt, file_list = build_prompt(target_chars)
    prompt_chars = len(prompt)

    print(f"\n{'=' * 80}")
    print(f"  Model: {cfg['name']}")
    print(f"  Context size: {ctx_size}")
    print(f"  Target input chars: {target_chars}")
    print(f"  Actual prompt chars: {prompt_chars} (~{prompt_chars // 4} tokens est)")
    print(f"  Input files: {len(file_list)}")
    print(f"{'=' * 80}")

    proc = start_server(
        bin_path, cfg["path"], port, ctx_size,
        cfg["reasoning_arg"], cfg["reasoning_value"],
    )
    print("  Waiting for server...", end="", flush=True)
    if not wait_for_server(port, timeout=900):
        print(" TIMEOUT")
        stop_server(proc)
        return {
            "model": cfg["name"],
            "ctx_size": ctx_size,
            "error": "server start timeout",
        }
    print(" ready")

    print("  Generating...", end="", flush=True)
    try:
        result = run_inference(port, prompt, max_tokens=8192)
    except Exception as e:
        print(f" ERROR: {e}")
        stop_server(proc)
        return {
            "model": cfg["name"],
            "ctx_size": ctx_size,
            "error": str(e),
        }

    print(
        f"\n  prefill: {result['prefill_tok_per_sec']} tok/s ({result['prefill_s']}s)"
        f" | decode: {result['decode_tok_per_sec']} tok/s ({result['decode_s']}s)"
        f" | blended: {result['blended_tok_per_sec']} tok/s"
    )
    print(
        f"  prompt={result['prompt_tokens']} tok, "
        f"completion={result['completion_tokens']} tok | "
        f"total={result['elapsed_s']}s | "
        f"finish={result['finish_reason']}"
    )

    diff = extract_diff(result["content"])
    score = score_diff(diff)
    print(
        f"  Score: {score['total']}/6 | "
        f"argparse={score['1_argparse_added']} "
        f"default=None={score['2_default_none']} "
        f"throughput={score['3_threads_throughput']} "
        f"code={score['4_threads_code']} "
        f"only_one_file={score['5_only_spark_bench']} "
        f"valid_diff={score['6_valid_diff_format']}"
    )

    # Save raw output
    safe_name = f"{cfg['name']}_{ctx_size}".replace(" ", "_").replace("/", "-")
    out_md = output_dir / f"{safe_name}.md"
    out_json = output_dir / f"{safe_name}.json"
    with open(out_md, "w") as f:
        f.write(f"# {cfg['name']} — ctx {ctx_size}\n\n")
        f.write(f"Prompt tokens: {result['prompt_tokens']}\n")
        f.write(f"Completion tokens: {result['completion_tokens']}\n")
        f.write(f"Elapsed: {result['elapsed_s']}s\n")
        f.write(f"Tok/s: {result['tok_per_sec']}\n")
        f.write(f"Score: {score['total']}/6\n\n")
        if result["reasoning"]:
            f.write(f"## Thinking\n\n{result['reasoning']}\n\n")
        f.write(f"## Output\n\n{result['content']}\n")
    with open(out_json, "w") as f:
        json.dump({
            "model": cfg["name"],
            "ctx_size": ctx_size,
            "input_files": file_list,
            "prompt_chars": prompt_chars,
            "metrics": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "elapsed_s": result["elapsed_s"],
                "prefill_s": result["prefill_s"],
                "decode_s": result["decode_s"],
                "prefill_tok_per_sec": result["prefill_tok_per_sec"],
                "decode_tok_per_sec": result["decode_tok_per_sec"],
                "blended_tok_per_sec": result["blended_tok_per_sec"],
                "finish_reason": result["finish_reason"],
            },
            "score": score,
            "content": result["content"],
            "reasoning": result["reasoning"],
        }, f, indent=2)

    stop_server(proc)
    time.sleep(3)
    return {
        "model": cfg["name"],
        "ctx_size": ctx_size,
        "prefill_tok_per_sec": result["prefill_tok_per_sec"],
        "decode_tok_per_sec": result["decode_tok_per_sec"],
        "blended_tok_per_sec": result["blended_tok_per_sec"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "elapsed_s": result["elapsed_s"],
        "finish_reason": result["finish_reason"],
        "score": score["total"],
        "score_detail": score,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=["qwen122b-bartowski-ik"],
        choices=list(MODEL_CONFIGS.keys()),
    )
    parser.add_argument(
        "--ctx", nargs="+", type=int, default=[32768, 65536, 131072],
        help="Context sizes to test",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--output-dir", default="experiments/longctx_bench",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for model_key in args.models:
        for ctx in sorted(args.ctx):
            result = run_benchmark(model_key, ctx, args.port, out_dir)
            all_results.append(result)

    print(f"\n{'=' * 90}")
    print("LONG-CONTEXT BENCHMARK SUMMARY")
    print(f"{'=' * 90}")
    print(
        f"{'Model':42s} | {'Ctx':>6s} | {'Prefill':>11s} | {'Decode':>8s} | "
        f"{'Prompt':>6s} | {'Gen':>5s} | {'Total':>6s} | {'Score':>5s}"
    )
    print("-" * 120)
    for r in all_results:
        if "error" in r:
            name = r["model"][:40]
            print(f"{name:42s} | {r['ctx_size']:>6d} | ERROR: {r['error']}")
        else:
            name = r["model"][:40]
            print(
                f"{name:42s} | {r['ctx_size']:>6d} | "
                f"{r['prefill_tok_per_sec']:>6.0f} tok/s | "
                f"{r['decode_tok_per_sec']:>5.1f} t/s | "
                f"{r['prompt_tokens']:>6d} | {r['completion_tokens']:>5d} | "
                f"{r['elapsed_s']:>5.0f}s | {r['score']:>3d}/6"
            )
    print(f"{'=' * 90}\n")

    outfile = out_dir / "longctx_results.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Raw results saved to {outfile}")


if __name__ == "__main__":
    main()
