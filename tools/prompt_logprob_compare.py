#!/usr/bin/env python3
"""Teacher-forced logprob capture and comparison for two server builds.

Capture mode sends fixed texts to an OpenAI-compatible /v1/completions
endpoint with prompt_logprobs and records, for every prompt position, the
logprob of the actual token and the server's top-1 token. Compare mode
reads two captures of the same texts and reports how closely the builds
agree, which separates a numerical regression from sampling noise without
depending on generated text.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path


def capture(args: argparse.Namespace) -> None:
    out = {"base_url": args.base_url, "model": args.model, "texts": []}
    for path in args.text:
        text = Path(path).read_text(encoding="utf-8")[: args.max_chars]
        body = {
            "model": args.model,
            "prompt": text,
            "max_tokens": 1,
            "temperature": 0,
            "prompt_logprobs": 1,
        }
        req = urllib.request.Request(
            args.base_url.rstrip("/") + "/completions",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            data = json.load(resp)
        positions = data.get("prompt_logprobs") or data["choices"][0].get("prompt_logprobs")
        if positions is None:
            sys.exit(f"server returned no prompt_logprobs for {path}")
        rows = []
        for pos in positions:
            if not pos:
                continue
            # vLLM lists the prompt's own token first, then the top-k tokens.
            actual_id = next(iter(pos))
            top1_id = min(pos, key=lambda k: pos[k]["rank"])
            rows.append({"actual": actual_id, "lp": pos[actual_id]["logprob"], "top1": top1_id})
        out["texts"].append({"path": str(path), "positions": rows})
        print(f"{path}: {len(rows)} positions", file=sys.stderr)
    Path(args.output).write_text(json.dumps(out))


def compare(args: argparse.Namespace) -> None:
    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())
    total = agree = 0
    abs_diff = 0.0
    nll_a = nll_b = 0.0
    worst = 0.0
    skipped = []
    for ta, tb in zip(a["texts"], b["texts"], strict=True):
        pa, pb = ta["positions"], tb["positions"]
        # A text edited between captures tokenizes differently; skip it
        # rather than compare logprobs of different tokens.
        if [r["actual"] for r in pa] != [r["actual"] for r in pb]:
            skipped.append(ta["path"])
            continue
        for ra, rb in zip(pa, pb):
            total += 1
            agree += ra["top1"] == rb["top1"]
            d = abs(ra["lp"] - rb["lp"])
            abs_diff += d
            worst = max(worst, d)
            nll_a -= ra["lp"]
            nll_b -= rb["lp"]
    if total == 0:
        sys.exit(f"no text has the same tokens in both captures; skipped {skipped}")
    result = {
        "positions": total,
        "top1_agreement": agree / total,
        "mean_abs_logprob_diff": abs_diff / total,
        "max_abs_logprob_diff": worst,
        "ppl_a": math.exp(nll_a / total),
        "ppl_b": math.exp(nll_b / total),
        "skipped_texts": skipped,
    }
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--base-url", required=True, help="including /v1")
    cap.add_argument("--model", required=True)
    cap.add_argument("--text", action="append", required=True)
    cap.add_argument("--max-chars", type=int, default=40000)
    cap.add_argument("--timeout", type=float, default=600)
    cap.add_argument("--output", required=True)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("a")
    cmp_.add_argument("b")
    args = parser.parse_args()
    capture(args) if args.mode == "capture" else compare(args)


if __name__ == "__main__":
    main()
