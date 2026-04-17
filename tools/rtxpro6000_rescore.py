#!/usr/bin/env python3
"""
Re-score saved coding benchmark responses without re-calling the model.

Reads experiments/rtxpro6000_coding/<model_key>/*.md and runs pytest via
the same extraction + scoring logic as rtxpro6000_coding_bench.py.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rtxpro6000_coding_bench import (
    BENCHMARKS, score_response, MODELS,
)

OUTPUT_ROOT = Path("/home/gisenberg/git/gisenberg/local-model-eval/experiments/rtxpro6000_coding")


def rescore(model_key: str) -> dict:
    artifacts = OUTPUT_ROOT / model_key
    if not artifacts.is_dir():
        raise RuntimeError(f"No artifacts for {model_key}")
    results = {}
    for bench_name, expected, module_name in BENCHMARKS:
        md = artifacts / f"{bench_name}.md"
        if not md.is_file():
            results[bench_name] = {"error": "missing"}
            continue
        resp = md.read_text()
        score = score_response(bench_name, module_name, resp)
        score["expected"] = expected
        results[bench_name] = score
        print(f"  {bench_name}: {score.get('passed', 0)}/{expected} "
              f"(failed={score.get('failed',0)}, errors={score.get('errors',0)}, "
              f"single_file={score.get('single_file')}, status={score.get('status')})")
    total_p = sum(r.get("passed", 0) for r in results.values() if isinstance(r, dict))
    total_e = sum(r.get("expected", 0) for r in results.values() if isinstance(r, dict))
    print(f"TOTAL: {total_p}/{total_e}")
    # Merge into existing summary if present
    summary_path = OUTPUT_ROOT / f"{model_key}.json"
    if summary_path.is_file():
        base = json.loads(summary_path.read_text())
    else:
        base = {"model": MODELS[model_key]["name"], "key": model_key}
    base["total_score"] = f"{total_p}/{total_e}"
    base["benchmarks"] = results
    summary_path.write_text(json.dumps(base, indent=2))
    return base


def main():
    keys = sys.argv[1:] if len(sys.argv) > 1 else [d.name for d in OUTPUT_ROOT.iterdir() if d.is_dir()]
    for k in keys:
        print(f"\n=== {k} ===")
        rescore(k)


if __name__ == "__main__":
    main()
