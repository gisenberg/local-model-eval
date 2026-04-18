#!/usr/bin/env python3
"""
RULER-style long-context benchmark for Qwen3.6-35B-A3B on RTX Pro 6000.

Goal: measure how far YaRN-extended static rope scaling (factor=2 → 512K,
factor=4 → 1M) trades off vs the native 262K config. Qwen's own guidance
warns that static YaRN (all open backends including llama.cpp, vLLM, SGLang)
degrades short-context quality. This benchmark quantifies that degradation
at the same context lengths under the three configs.

Tasks (4 of the 13 RULER tasks; these are the ones that cleanly gate
yes/no on YaRN correctness):
  1. niah_single  — one magic number in a haystack. Baseline retrieval.
  2. niah_multi   — three magic numbers at different depths. Tests whether
                    attention stays coherent across positions after YaRN.
  3. variable     — chain of variable assignments (a=V0, b=a+1, c=b+1, …).
                    Tests semantic reasoning over long context.
  4. common_words — 3 target words repeated N, N-1, N-2 times in a 200-word
                    bag; list top 3. Tests bag-of-words aggregation / counting.

Configs (each launches its own llama-server, exclusive VRAM use):
  native_262k   — no YaRN. Max usable ctx = 262,144.
  yarn2_512k    — YaRN factor=2, orig=262144. Max ctx = 524,288.
  yarn4_1m      — YaRN factor=4, orig=262144. Max ctx = 1,048,576.

Context points per config (only points ≤ max_ctx are run):
  8192, 32768, 131072, 262144, 524288, 1048576.

The critical comparison is SHORT CONTEXT across configs: if yarn4_1m is
noticeably worse than native_262k at 32K/131K, that's the static-YaRN tax.

Invocation:
  python3 rtxpro6000_ruler_bench.py --config native_262k yarn2_512k yarn4_1m
  python3 rtxpro6000_ruler_bench.py --config yarn4_1m --lengths 524288 1048576

Each (config, length, task) combination runs a small number of trials
(default 3). Writes per-trial JSON and an aggregate table to the
experiments/ directory.

Scoring:
  - niah_single / niah_multi / variable: exact-match on the unique value(s).
    A correct answer contains ALL expected values; partial credit is recorded
    but the primary score is all-or-nothing per trial.
  - common_words: correct if the three most-frequent target words are
    present in the response in any order. Extras are fine.

Runtime warning: a single 1M-token prefill takes ~15-30 min on this hardware.
A full matrix (3 configs × 6 lengths × 4 tasks × 3 trials) is ~216 requests;
expect 6-10 hours end-to-end. Use --lengths and --tasks to narrow.
"""

import argparse
import hashlib
import json
import os
import random
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths and server config
# ---------------------------------------------------------------------------

LLAMA_SERVER = "/home/gisenberg/llama-build/src/build/bin/llama-server"
LLAMA_DIR = os.path.dirname(LLAMA_SERVER)
CUDA_LIB = "/home/gisenberg/.micromamba/envs/cuda/lib"
MODEL_PATH = "/home/gisenberg/models/qwen36-35b-a3b-q8/Qwen3.6-35B-A3B-Q8_0.gguf"
PORT = 18090  # dedicated port for bench (avoid collision with serving)
OUTPUT_ROOT = Path("/home/gisenberg/git/gisenberg/local-model-eval/experiments/ruler_qwen36")


# YaRN configs. `rope_scale` and `yarn_orig_ctx` are the flags llama-server
# needs; `max_ctx` is the usable context window this config supports.
CONFIGS = {
    "native_262k": {
        "name": "Native (no YaRN, 262K)",
        "max_ctx": 262144,
        "rope_flags": [],  # no YaRN
    },
    "yarn2_512k": {
        "name": "YaRN factor=2 (512K)",
        "max_ctx": 524288,
        "rope_flags": [
            "--rope-scaling", "yarn",
            "--rope-scale", "2",
            "--yarn-orig-ctx", "262144",
        ],
    },
    "yarn4_1m": {
        "name": "YaRN factor=4 (1M)",
        "max_ctx": 1048576,
        "rope_flags": [
            "--rope-scaling", "yarn",
            "--rope-scale", "4",
            "--yarn-orig-ctx", "262144",
        ],
    },
}

# Context lengths we'd like to probe. Each config only runs those ≤ its max_ctx.
DEFAULT_LENGTHS = [8192, 32768, 131072, 262144, 524288, 1048576]

# Per-task output budget. max_tokens caps TOTAL output (reasoning + answer),
# so it must exceed REASONING_BUDGET with headroom for the final answer.
# Thinking on this model can run up to ~8K tokens on common_words.
OUTPUT_MAX_TOKENS = 12288
REASONING_BUDGET = 8192

# Measured chars-per-token for Qwen3.6 BPE on this filler prose (~5.1).
# Used to size the filler so we hit ~0.8 * ctx input tokens.
CHARS_PER_TOKEN = 5.0
FILL_FRAC = 0.8


# ---------------------------------------------------------------------------
# Filler generation: deterministic natural-sounding prose
# ---------------------------------------------------------------------------

# A word bank + sentence templates that produce varied, non-repeating prose.
# Keyed off a deterministic RNG so the same (seed, target_chars) always
# produces the same filler — important for reproducibility across configs.

SUBJECTS = [
    "the research team", "a careful observer", "the senior architect",
    "the quiet librarian", "a travelling merchant", "the young apprentice",
    "the visiting professor", "a retired engineer", "the local historian",
    "a determined cyclist", "the patient gardener", "a curious journalist",
    "the veteran sailor", "a meticulous accountant", "the wandering poet",
    "a resourceful mechanic", "the elderly shopkeeper", "a reluctant diplomat",
]
VERBS = [
    "noticed", "described", "considered", "examined", "studied",
    "dismissed", "questioned", "analysed", "recalled", "catalogued",
    "appreciated", "recorded", "collected", "measured", "illustrated",
    "anticipated", "investigated", "rehearsed",
]
OBJECTS = [
    "the faded manuscript", "a brass astrolabe", "the harbour's tide chart",
    "the dusty logbook", "a chipped porcelain teacup", "the old atlas",
    "a cracked ship's bell", "the wooden crate of letters",
    "a worn leather journal", "the iron weathervane", "a folded map",
    "the carved stone lantern", "a silver pocket watch",
    "the wicker reading basket", "a weathered sextant",
    "the bundled seed packets", "a copper kettle", "the lacquered compass",
]
SETTINGS = [
    "in the quiet library before dawn", "near the fog-streaked window",
    "under a pale lantern light", "beside the crackling wood stove",
    "along the mossy stone path", "within the cedar-panelled study",
    "across the rain-slicked courtyard", "amid the scattered papers",
    "behind the curtain of ivy", "atop the creaking wooden stair",
    "through the narrow corridor", "beneath the slate-grey sky",
    "from the crooked attic window", "over the weathered writing desk",
    "past the rows of dusty shelves", "between the brass candlesticks",
]
FACTS = [
    "and took careful notes in a small brown notebook",
    "and set the object down with evident reluctance",
    "and compared it against earlier descriptions in the archive",
    "and remarked that the quality of the binding was unusual",
    "and made a sketch in the margin of the workbook",
    "and added a small annotation to the inventory ledger",
    "and concluded that further inspection would be warranted",
    "and offered a brief summary to the other volunteers present",
    "and recorded the serial number at the back of the catalogue",
    "and left a small paper flag clipped to the relevant page",
    "and resolved to return the following morning with better light",
    "and tucked a copy of the notes into the outgoing post",
]
CONNECTORS = [
    "Later, ", "Soon afterwards, ", "By mid-afternoon, ", "Not long after, ",
    "On the following day, ", "In the hour before sunset, ",
    "Earlier that week, ", "Shortly before supper, ",
    "During the long quiet evening, ", "Just after the rain had stopped, ",
]


def build_filler(n_chars: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    out: list[str] = []
    total = 0
    while total < n_chars:
        subj = rng.choice(SUBJECTS)
        verb = rng.choice(VERBS)
        obj = rng.choice(OBJECTS)
        where = rng.choice(SETTINGS)
        fact = rng.choice(FACTS)
        lead = rng.choice(CONNECTORS) if rng.random() < 0.6 else ""
        sentence = f"{lead}{subj.capitalize() if not lead else subj} {verb} {obj} {where}, {fact}."
        out.append(sentence)
        total += len(sentence) + 1
        if rng.random() < 0.08:
            out.append("")  # paragraph break
    return " ".join(out)[:n_chars]


# ---------------------------------------------------------------------------
# Task builders: each returns (prompt_text, expected_answer, metadata)
# ---------------------------------------------------------------------------


def _insert_at_depth(filler: str, needle: str, depth_frac: float) -> str:
    """Insert a needle sentence at the given fractional depth in the filler."""
    idx = int(len(filler) * depth_frac)
    # Snap to sentence boundary to not mid-word the needle
    while idx < len(filler) and filler[idx] not in (". ", "."):
        idx += 1
    return filler[:idx] + " " + needle + " " + filler[idx:]


def _make_key(rng: random.Random) -> str:
    # Pronounceable fake key, 2 syllables. No dictionary words.
    consonants = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"
    k = (rng.choice(consonants) + rng.choice(vowels) +
         rng.choice(consonants) + rng.choice(vowels) +
         rng.choice(consonants))
    return k


def _make_value(rng: random.Random) -> int:
    return rng.randint(100000, 999999)  # distinctive 6-digit so unlikely in filler


def build_niah_single(target_chars: int, seed: int, depth_frac: float):
    rng = random.Random(seed)
    filler = build_filler(target_chars, seed=seed ^ 0xA5A5)
    key = _make_key(rng)
    value = _make_value(rng)
    needle = f"Internal note to archivists: the magic number for {key} is {value}, and this should be remembered carefully."
    body = _insert_at_depth(filler, needle, depth_frac)
    prompt = (
        "Below is a long archival passage. Read it carefully.\n\n"
        "=== BEGIN PASSAGE ===\n"
        f"{body}\n"
        "=== END PASSAGE ===\n\n"
        f"Question: What is the magic number for {key}? "
        "Answer with just the number, nothing else."
    )
    return {
        "task": "niah_single",
        "prompt": prompt,
        "expected": [str(value)],
        "meta": {"key": key, "value": value, "depth_frac": depth_frac},
    }


def build_niah_multi(target_chars: int, seed: int):
    rng = random.Random(seed)
    filler = build_filler(target_chars, seed=seed ^ 0x5A5A)
    keys = [_make_key(rng) for _ in range(3)]
    vals = [_make_value(rng) for _ in range(3)]
    depths = [0.15, 0.5, 0.85]
    body = filler
    # Insert from last to first so offsets don't shift
    for k, v, d in zip(reversed(keys), reversed(vals), reversed(depths)):
        needle = f"Archivist memo: the magic number for {k} is {v}, recorded on the registry."
        body = _insert_at_depth(body, needle, d)
    prompt = (
        "Below is a long archival passage. Read it carefully.\n\n"
        "=== BEGIN PASSAGE ===\n"
        f"{body}\n"
        "=== END PASSAGE ===\n\n"
        f"Question: What are the magic numbers for {keys[0]}, {keys[1]}, and {keys[2]}? "
        "Answer with three numbers on separate lines, labelled by key, nothing else."
    )
    return {
        "task": "niah_multi",
        "prompt": prompt,
        "expected": [str(v) for v in vals],
        "meta": {"keys": keys, "values": vals, "depths": depths},
    }


def build_variable(target_chars: int, seed: int):
    """Scatter an assignment chain across the context. Ask for the final value.
    Chain: v0 = N; v1 = v0 + 1; v2 = v1 + 1; ...; v7 = v6 + 1.
    Answer is N + 7.
    """
    rng = random.Random(seed)
    filler = build_filler(target_chars, seed=seed ^ 0xC3C3)
    base = _make_value(rng)
    chain_len = 8
    # 8 depths evenly distributed
    depths = [(i + 0.5) / chain_len for i in range(chain_len)]
    body = filler
    sentences = []
    sentences.append(f"Archivist shorthand: v0 := {base}.")
    for i in range(1, chain_len):
        sentences.append(f"Archivist shorthand: v{i} := v{i-1} + 1.")
    # Insert sentences from last to first to keep offsets stable.
    pairs = list(zip(sentences, depths))
    for sent, d in reversed(pairs):
        body = _insert_at_depth(body, sent, d)
    final = base + (chain_len - 1)
    prompt = (
        "Below is a long archival passage. Read it carefully.\n\n"
        "=== BEGIN PASSAGE ===\n"
        f"{body}\n"
        "=== END PASSAGE ===\n\n"
        "Question: Several sentences in the passage define variables of the form "
        "`v0 := N`, `v1 := v0 + 1`, …, `v7 := v6 + 1`. What is the numerical value "
        "of v7? Answer with just the number, nothing else."
    )
    return {
        "task": "variable",
        "prompt": prompt,
        "expected": [str(final)],
        "meta": {"base": base, "chain_len": chain_len, "answer": final},
    }


def build_common_words(target_chars: int, seed: int):
    """Scatter a bag of words; 3 target words appear far more often than noise.

    Because the filler is deterministic prose, we pre-compute the word
    frequencies *without* target words (which are deliberately rare or
    absent in SUBJECTS/VERBS/etc.), then splice in the targets with N,
    N-1, N-2 appearances. We pick targets that don't appear naturally.
    """
    rng = random.Random(seed)
    # Invented words that won't naturally appear in the filler
    targets = ["zephyrine", "quondamly", "verdacious"]
    counts = [24, 19, 15]  # well-separated from noise, unambiguous top-3
    # Small injection cost — scatter ~N_injections one-liners through filler
    lines = []
    for w, c in zip(targets, counts):
        for _ in range(c):
            lines.append(f"A single word was pinned to the margin: {w}.")
    rng.shuffle(lines)
    filler = build_filler(target_chars, seed=seed ^ 0x3333)
    # Splice one injected line after every Kth sentence
    sentences = re.split(r"(?<=[.!?])\s+", filler)
    if len(sentences) <= 2 * len(lines):
        # filler too short — just append
        body = filler + " " + " ".join(lines)
    else:
        step = max(1, len(sentences) // (len(lines) + 1))
        i = step
        for line in lines:
            if i < len(sentences):
                sentences[i] = sentences[i] + " " + line
            i += step
        body = " ".join(sentences)
    prompt = (
        "Below is a long archival passage. Read it carefully.\n\n"
        "=== BEGIN PASSAGE ===\n"
        f"{body}\n"
        "=== END PASSAGE ===\n\n"
        "Question: Three invented words are scattered through the passage. "
        "They appear many more times than any other invented or unusual word. "
        "List the three invented words, ordered by frequency (most frequent first). "
        "Answer with just the three words on separate lines, nothing else."
    )
    return {
        "task": "common_words",
        "prompt": prompt,
        "expected": targets,  # all three must be present
        "meta": {"targets": targets, "counts": counts},
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_output(task: dict, response: str) -> dict:
    resp_lc = response.lower()
    expected = task["expected"]
    hits = [e for e in expected if e.lower() in resp_lc]
    return {
        "task": task["task"],
        "pass": len(hits) == len(expected),
        "partial": len(hits),
        "n_expected": len(expected),
        "hits": hits,
        "missed": [e for e in expected if e not in hits],
    }


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def start_server(config_key: str, ctx: int) -> subprocess.Popen:
    cfg = CONFIGS[config_key]
    # Workaround for llama-server's hard n_ctx_train cap
    # (tools/server/server-context.cpp:764-768 — see llama.cpp issue #17459).
    # When requesting a ctx beyond native 262144 we must rewrite the metadata
    # key the server reads for the cap check. Rope scaling itself is wired
    # through correctly even when the log prints "rope scaling = linear"
    # (that prints the GGUF-side training-time value, not runtime cparams).
    override_kv = []
    if ctx > 262144:
        override_kv = ["--override-kv", f"qwen35moe.context_length=int:{ctx}"]
    # llama.cpp -c is total across slots; with -np 1 it's the per-slot ctx.
    cmd = [
        LLAMA_SERVER,
        "-m", MODEL_PATH,
        "--alias", f"qwen36-ruler-{config_key}",
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "-c", str(ctx),
        "-ngl", "99",
        "-fa", "on",
        "-ctk", "f16", "-ctv", "f16",
        "-np", "1",
        "--no-mmap",
        "--jinja",
        "-rea", "on",
        "--reasoning-budget", str(REASONING_BUDGET),
        # Unsloth Qwen3.6 thinking-mode sampling
        "--temp", "0.6",
        "--top-p", "0.95",
        "--top-k", "20",
        "--min-p", "0.0",
        "--repeat-penalty", "1.0",
    ] + cfg["rope_flags"] + override_kv
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LLAMA_DIR + ":" + CUDA_LIB
    log_path = f"/tmp/ruler-{config_key}-c{ctx}.log"
    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    return proc, log_path


def wait_for_server(timeout: int = 900) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=3) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def stop_server(proc: subprocess.Popen):
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=30)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=10)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def count_tokens(prompt: str) -> int | None:
    body = json.dumps({"content": prompt}).encode()
    req = urllib.request.Request(
        f"http://localhost:{PORT}/tokenize",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            obj = json.loads(r.read())
            toks = obj.get("tokens") or []
            return len(toks)
    except Exception:
        return None


def run_inference(prompt: str, request_timeout: int = 3600) -> dict:
    body = json.dumps({
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": OUTPUT_MAX_TOKENS,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://localhost:{PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    t_send = time.perf_counter()
    first_token_at = None
    last_token_at = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    server_usage = None
    finish = "?"

    with urllib.request.urlopen(req, timeout=request_timeout) as resp:
        buf = b""
        done = False
        while not done:
            chunk = resp.read(2048)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    done = True
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    server_usage = obj["usage"]
                for choice in obj.get("choices", []):
                    delta = choice.get("delta", {})
                    fr = choice.get("finish_reason")
                    if fr:
                        finish = fr
                    r_chunk = delta.get("reasoning_content")
                    if r_chunk:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        last_token_at = time.perf_counter()
                        reasoning_parts.append(r_chunk)
                    c_chunk = delta.get("content")
                    if c_chunk:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        last_token_at = time.perf_counter()
                        content_parts.append(c_chunk)
    t_end = time.perf_counter()

    ttft = (first_token_at - t_send) if first_token_at else None
    decode_elapsed = (last_token_at - first_token_at) if first_token_at and last_token_at and last_token_at > first_token_at else None
    prompt_tok = (server_usage or {}).get("prompt_tokens") or None
    completion_tok = (server_usage or {}).get("completion_tokens") or None
    prefill_tps = (prompt_tok / ttft) if prompt_tok and ttft else None
    decode_tps = (completion_tok / decode_elapsed) if completion_tok and decode_elapsed else None
    return {
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "finish": finish,
        "prompt_tokens": prompt_tok,
        "completion_tokens": completion_tok,
        "ttft_s": round(ttft, 3) if ttft else None,
        "decode_s": round(decode_elapsed, 3) if decode_elapsed else None,
        "total_s": round(t_end - t_send, 3),
        "prefill_tps": round(prefill_tps, 2) if prefill_tps else None,
        "decode_tps": round(decode_tps, 2) if decode_tps else None,
    }


def vram_used_mb() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Task dispatch
# ---------------------------------------------------------------------------


def build_task(task_name: str, target_chars: int, seed: int) -> dict:
    if task_name == "niah_single":
        return build_niah_single(target_chars, seed, depth_frac=0.5)
    if task_name == "niah_single_early":
        return build_niah_single(target_chars, seed, depth_frac=0.1)
    if task_name == "niah_single_late":
        return build_niah_single(target_chars, seed, depth_frac=0.9)
    if task_name == "niah_multi":
        return build_niah_multi(target_chars, seed)
    if task_name == "variable":
        return build_variable(target_chars, seed)
    if task_name == "common_words":
        return build_common_words(target_chars, seed)
    raise ValueError(f"unknown task: {task_name}")


DEFAULT_TASKS = ["niah_single", "niah_multi", "variable", "common_words"]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_one(config_key: str, ctx: int, tasks: list[str], trials: int,
            output_dir: Path) -> list[dict]:
    cfg = CONFIGS[config_key]
    print(f"\n{'#' * 80}")
    print(f"# Config: {cfg['name']}   ctx={ctx:,}")
    print(f"{'#' * 80}")

    proc, log = start_server(config_key, ctx)
    print(f"  starting server (log: {log})", flush=True)
    t0 = time.perf_counter()
    if not wait_for_server(timeout=900):
        print("  SERVER FAILED TO START — tail of log:")
        try:
            with open(log) as f:
                for line in f.readlines()[-40:]:
                    print("   ", line.rstrip())
        except Exception:
            pass
        stop_server(proc)
        return [{
            "config": config_key, "ctx": ctx, "error": "server start timeout",
        }]
    load_s = time.perf_counter() - t0
    vram = vram_used_mb()
    print(f"  ready in {load_s:.1f}s, VRAM={vram} MB")

    results = []
    # Target prompt length: fill the window minus the output budget and a
    # safety margin. At low ctx we end up using most of the window for input;
    # at high ctx there's comfortable headroom.
    safety_tokens = 1024
    target_input_tokens = max(
        512,
        int((ctx - OUTPUT_MAX_TOKENS - safety_tokens) * FILL_FRAC),
    )
    target_chars = int(target_input_tokens * CHARS_PER_TOKEN)

    for task_name in tasks:
        for trial in range(trials):
            seed = hash((config_key, ctx, task_name, trial)) & 0xFFFFFFFF
            task = build_task(task_name, target_chars, seed)
            ntok = count_tokens(task["prompt"])
            headroom = ctx - (ntok or 0) - OUTPUT_MAX_TOKENS - 256
            print(
                f"  [{task_name}#{trial}] prompt_chars={len(task['prompt']):,} "
                f"tokens={ntok} headroom={headroom}"
            )
            if headroom < 0:
                print(f"    SKIP — prompt would exceed ctx")
                results.append({
                    "config": config_key, "ctx": ctx, "task": task_name,
                    "trial": trial, "error": "prompt exceeds ctx",
                    "prompt_tokens_estimated": ntok,
                })
                continue
            try:
                infer = run_inference(task["prompt"])
            except Exception as e:
                print(f"    INFER ERROR: {e}")
                results.append({
                    "config": config_key, "ctx": ctx, "task": task_name,
                    "trial": trial, "error": f"infer: {e}",
                })
                continue
            score = score_output(task, infer["content"])
            print(
                f"    prefill={infer['prefill_tps']} tok/s ({infer['ttft_s']}s) "
                f"decode={infer['decode_tps']} tok/s "
                f"gen={infer['completion_tokens']} → {'PASS' if score['pass'] else 'FAIL'} "
                f"(hits {score['partial']}/{score['n_expected']})"
            )
            result = {
                "config": config_key,
                "config_name": cfg["name"],
                "ctx": ctx,
                "task": task_name,
                "trial": trial,
                "seed": seed,
                "prompt_tokens": infer["prompt_tokens"],
                "completion_tokens": infer["completion_tokens"],
                "ttft_s": infer["ttft_s"],
                "decode_s": infer["decode_s"],
                "total_s": infer["total_s"],
                "prefill_tps": infer["prefill_tps"],
                "decode_tps": infer["decode_tps"],
                "finish": infer["finish"],
                "score": score,
                "meta": task["meta"],
                "vram_mb": vram,
                # Save truncated output for post-hoc inspection
                "content_preview": infer["content"][:2000],
                "reasoning_preview": infer["reasoning"][:2000],
            }
            results.append(result)

            # Stream-save per-trial JSON for crash resilience
            per_trial = output_dir / f"{config_key}_c{ctx}_{task_name}_t{trial}.json"
            per_trial.write_text(json.dumps(result, indent=2))

    stop_server(proc)
    time.sleep(3)
    return results


def summarise(all_results: list[dict]) -> str:
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("RULER QWEN3.6-35B-A3B YARN COMPARISON")
    lines.append("=" * 100)
    lines.append(
        f"{'Config':22s} | {'Ctx':>9s} | {'Task':14s} | {'Pass':>4s} | "
        f"{'Prefill':>9s} | {'Decode':>8s} | {'Gen':>5s}"
    )
    lines.append("-" * 100)
    # Aggregate
    buckets: dict[tuple, list[dict]] = {}
    for r in all_results:
        if "error" in r:
            continue
        key = (r["config"], r["ctx"], r["task"])
        buckets.setdefault(key, []).append(r)
    for key in sorted(buckets.keys()):
        cfg, ctx, task = key
        trials = buckets[key]
        pass_rate = sum(1 for t in trials if t["score"]["pass"]) / len(trials)
        prefill = statistics.mean([t["prefill_tps"] for t in trials if t.get("prefill_tps")]) if any(t.get("prefill_tps") for t in trials) else 0
        decode = statistics.mean([t["decode_tps"] for t in trials if t.get("decode_tps")]) if any(t.get("decode_tps") for t in trials) else 0
        gen = statistics.mean([t["completion_tokens"] for t in trials if t.get("completion_tokens")]) if any(t.get("completion_tokens") for t in trials) else 0
        lines.append(
            f"{cfg:22s} | {ctx:>9,d} | {task:14s} | "
            f"{int(pass_rate * 100):>3d}% | "
            f"{prefill:>6.0f} t/s | "
            f"{decode:>5.1f} t/s | "
            f"{gen:>5.0f}"
        )
    # Errors
    errors = [r for r in all_results if "error" in r]
    if errors:
        lines.append("")
        lines.append("ERRORS:")
        for e in errors:
            lines.append(
                f"  {e.get('config','?')} ctx={e.get('ctx','?')} "
                f"task={e.get('task','?')} trial={e.get('trial','?')} :: {e['error']}"
            )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="+", default=list(CONFIGS.keys()),
                   choices=list(CONFIGS.keys()))
    p.add_argument("--lengths", nargs="+", type=int, default=DEFAULT_LENGTHS,
                   help="Context lengths to test (filtered to each config's max_ctx)")
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS,
                   choices=["niah_single", "niah_single_early", "niah_single_late",
                            "niah_multi", "variable", "common_words"])
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    for cfg in args.config:
        max_ctx = CONFIGS[cfg]["max_ctx"]
        lengths = [L for L in sorted(args.lengths) if L <= max_ctx]
        if not lengths:
            print(f"no lengths within {cfg} max_ctx={max_ctx}; skipping")
            continue
        for ctx in lengths:
            res = run_one(cfg, ctx, args.tasks, args.trials, out_dir)
            all_results.extend(res)
            # Checkpoint the whole roll-up on every (config,ctx) boundary
            rollup = out_dir / "ruler_results.json"
            rollup.write_text(json.dumps(all_results, indent=2))
            print(summarise(all_results))

    # Final
    final = summarise(all_results)
    print(final)
    (out_dir / "ruler_summary.txt").write_text(final)
    (out_dir / "ruler_results.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nAll results → {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nINTERRUPTED — killing any lingering llama-server on port", PORT)
        os.system(f"pkill -f 'port {PORT}'")
        sys.exit(130)
