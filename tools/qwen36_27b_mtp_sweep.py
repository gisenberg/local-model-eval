#!/usr/bin/env python3
"""
Orchestrator for the Qwen3.6-27B FP8 spec-decode sweep.

Per variant: spawn vLLM, poll for ready, run the existing 4-bench coding suite
(`tools/nvfp4_qwen36_27b_bench.py`), scrape /metrics, stop vLLM, write summary.

Variants live in VARIANTS below. Run one at a time:

    python tools/qwen36_27b_mtp_sweep.py baseline
    python tools/qwen36_27b_mtp_sweep.py mtp-1
    ...

Output: experiments/qwen36_27b_mtp_sweep/<variant>/{results.json, metrics.txt, vllm.log, summary.json}
"""
import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path
import requests

REPO = Path(__file__).resolve().parent.parent
OUT_BASE = REPO / "experiments" / "qwen36_27b_mtp_sweep"
PORT = 8090
SERVED_NAME = "qwen36-27b-sweep"
MODEL_PATH = "/mnt/extended/gisenberg/models/qwen36-27b-fp8"
DRAFTER_PATH = "/home/gisenberg/models-vllm/qwen36-27b-dflash-drafter-v2026-04-27"
VENV = "/home/gisenberg/venvs/dflash-pr40898"
CUDA_HOME = f"{VENV}/lib/python3.12/site-packages/nvidia/cu13"

# Common vLLM args matched to the production qwen36-27b llama-swap entry,
# minus the spec-decode block (each variant overrides).
COMMON = [
    "--host", "127.0.0.1",
    "--port", str(PORT),
    "--served-model-name", SERVED_NAME,
    "--max-model-len", "262144",
    # Dropped from 0.92 → 0.88: mtp-2 OOMs at 0.92 during cudagraph_memory
    # profiling ("Tried to allocate 1.56 GiB ... 558.56 MiB free"). 0.88 leaves
    # ~7 GB headroom for spec-decode workspace. Single-stream coding bench
    # never approaches the KV cache ceiling so the smaller pool is invisible.
    "--gpu-memory-utilization", "0.88",
    "--enable-prefix-caching",
    # attention-backend moved out of COMMON: fp8-kv variants need flashinfer
    # because the flash_attn backend rejects kv_cache_dtype=fp8 ("not supported").
    # Default for non-fp8-kv variants: flash_attn (matches production llama-swap).
    "--max-num-batched-tokens", "32768",
    # MTP-1 fails at default max_num_seqs=1024: "max_num_seqs (1024) exceeds
    # available Mamba cache blocks (914)". Spec-decode adds Mamba cache slots
    # per draft step; 256 is comfortably below the floor for k=2 too and our
    # bench is single-stream.
    "--max-num-seqs", "256",
    "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
    "--reasoning-parser", "qwen3",
    # Note: thinking is intentionally LEFT ON to match the existing
    # experiments/dflash_pr40898 reference (22/22, ~9430 tok/run with organic
    # reasoning). The production llama-swap entry sets enable_thinking=false
    # for the agent preset, but coding-bench scores collapse without thinking.
]

FLASH = ["--attention-backend", "flash_attn"]
FLASHINFER = ["--attention-backend", "flashinfer"]

VARIANTS = {
    "baseline": {
        "desc": "FP8 target only, no spec dec (PR-40898 stack baseline)",
        "extra": FLASH,
    },
    "mtp-1": {
        "desc": "qwen3_5_mtp k=1 (model's native MTP head, low-latency preset)",
        "extra": FLASH + ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": 1})],
    },
    "mtp-2": {
        "desc": "qwen3_5_mtp k=2 (model's native MTP head, balanced)",
        "extra": FLASH + ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": 2})],
    },
    "dflash-k7": {
        "desc": "DFlash drafter k=7 (low end of k sweep)",
        "extra": FLASH + ["--speculative-config", json.dumps({"method": "dflash", "model": DRAFTER_PATH, "num_speculative_tokens": 7})],
    },
    "dflash-k11": {
        "desc": "DFlash drafter k=11 (mid k sweep)",
        "extra": FLASH + ["--speculative-config", json.dumps({"method": "dflash", "model": DRAFTER_PATH, "num_speculative_tokens": 11})],
    },
    "dflash-k15-rerun": {
        "desc": "DFlash k=15 re-run on PR-40898 stack (sanity vs experiments/dflash_pr40898)",
        "extra": FLASH + ["--speculative-config", json.dumps({"method": "dflash", "model": DRAFTER_PATH, "num_speculative_tokens": 15})],
    },
    # round 3: fp8-kv variants need flashinfer (flash_attn rejects fp8 kv cache)
    "mtp-1-fp8kv": {
        "desc": "qwen3_5_mtp k=1 + --kv-cache-dtype fp8 (round-1 quality leader + KV halving)",
        "extra": FLASHINFER + [
            "--kv-cache-dtype", "fp8",
            "--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": 1}),
        ],
    },
    "mtp-2-fp8kv": {
        "desc": "qwen3_5_mtp k=2 + --kv-cache-dtype fp8",
        "extra": FLASHINFER + [
            "--kv-cache-dtype", "fp8",
            "--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": 2}),
        ],
    },
    "dflash-k11-fp8kv": {
        "desc": "DFlash k=11 + --kv-cache-dtype fp8",
        "extra": FLASHINFER + [
            "--kv-cache-dtype", "fp8",
            "--speculative-config", json.dumps({"method": "dflash", "model": DRAFTER_PATH, "num_speculative_tokens": 11}),
        ],
    },
}


def build_env():
    env = os.environ.copy()
    env["PATH"] = f"{VENV}/bin:{CUDA_HOME}/bin:/usr/bin:/bin"
    env["CUDA_HOME"] = CUDA_HOME
    # Reduce vllm log spam in the bench's pytest subprocesses
    env.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    return env


def wait_for_ready(timeout=600):
    """Poll /v1/models until the server returns 200 with our model."""
    t0 = time.time()
    last_err = None
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"http://127.0.0.1:{PORT}/v1/models", timeout=2)
            if r.status_code == 200 and SERVED_NAME in r.text:
                return time.time() - t0
        except Exception as e:
            last_err = e
        time.sleep(2)
    raise TimeoutError(f"vLLM did not become ready within {timeout}s (last err: {last_err})")


def scrape_metrics(out_path):
    try:
        r = requests.get(f"http://127.0.0.1:{PORT}/metrics", timeout=5)
        out_path.write_text(r.text)
        # extract spec-decode counters of interest if present
        spec = {}
        for line in r.text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            if "spec_decode" in line or "speculative" in line:
                spec[line.split()[0]] = line.split()[-1]
        return spec
    except Exception as e:
        return {"error": str(e)}


def run_variant(name):
    if name not in VARIANTS:
        print(f"unknown variant: {name}; valid: {list(VARIANTS)}", file=sys.stderr)
        sys.exit(2)
    v = VARIANTS[name]
    out_dir = OUT_BASE / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== variant: {name} ===\n  {v['desc']}")
    cmd = [f"{VENV}/bin/vllm", "serve", MODEL_PATH] + COMMON + v["extra"]
    (out_dir / "serve_cmd.txt").write_text(" ".join(cmd) + "\n")

    log = open(out_dir / "vllm.log", "w")
    proc = subprocess.Popen(
        cmd, env=build_env(), stdout=log, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    summary = {"variant": name, "desc": v["desc"], "pid": proc.pid}
    try:
        print(f"  vllm pid {proc.pid}, waiting for ready...")
        ready_s = wait_for_ready(timeout=600)
        summary["startup_s"] = round(ready_s, 1)
        print(f"  ready in {ready_s:.1f}s, running bench...")

        bench = subprocess.run(
            [
                f"{VENV}/bin/python", str(REPO / "tools" / "nvfp4_qwen36_27b_bench.py"),
                "--port", str(PORT),
                "--served-name", SERVED_NAME,
                "--output-dir", str(out_dir),
            ],
            cwd=str(REPO), env=build_env(),
        )
        summary["bench_returncode"] = bench.returncode

        # scrape metrics after bench so spec-decode counters reflect real workload
        summary["spec_metrics"] = scrape_metrics(out_dir / "metrics.txt")

        # roll up bench results
        rj = out_dir / "results.json"
        if rj.exists():
            res = json.loads(rj.read_text())
            best = sum(min(b.get("best", 0), b.get("expected", 0)) for k, b in res.items() if k != "_meta")
            avg = sum(min(b.get("avg", 0), b.get("expected", 0)) for k, b in res.items() if k != "_meta")
            tps_runs = [r["tok_per_sec"] for k, b in res.items() if k != "_meta" for r in b.get("runs", []) if "tok_per_sec" in r]
            summary["best_of_3"] = best
            summary["avg"] = round(avg, 1)
            summary["mean_tok_per_sec"] = round(sum(tps_runs) / len(tps_runs), 1) if tps_runs else None
            summary["min_tok_per_sec"] = min(tps_runs) if tps_runs else None
            summary["max_tok_per_sec"] = max(tps_runs) if tps_runs else None
    finally:
        print("  stopping vllm...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=20)
        except Exception:
            pass
        # vllm spawns EngineCore + multiprocessing.resource_tracker outside the
        # parent process group, so killpg alone leaves them holding ~86 GB VRAM.
        # Sweep by pattern, then verify VRAM is freed before returning.
        subprocess.run(["pkill", "-KILL", "-f", "VLLM::EngineCore"], check=False)
        subprocess.run(["pkill", "-KILL", "-f", "vllm.*qwen36-27b-sweep"], check=False)
        subprocess.run(["pkill", "-KILL", "-f", "multiprocessing.resource_tracker"], check=False)
        for _ in range(15):
            time.sleep(2)
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            )
            mb = int(r.stdout.strip().splitlines()[0]) if r.returncode == 0 else 99999
            if mb < 1000:
                summary["vram_freed_mb"] = mb
                break
        else:
            summary["vram_freed_mb"] = mb
        log.close()

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  done. {json.dumps({k: summary[k] for k in ('startup_s','best_of_3','avg','mean_tok_per_sec','min_tok_per_sec','max_tok_per_sec') if k in summary}, indent=0)}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=list(VARIANTS) + ["all-round1"])
    args = ap.parse_args()
    if args.variant == "all-round1":
        for name in ["baseline", "mtp-1", "mtp-2", "dflash-k7", "dflash-k11"]:
            run_variant(name)
    else:
        run_variant(args.variant)


if __name__ == "__main__":
    main()
