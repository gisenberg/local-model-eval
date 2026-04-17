#!/usr/bin/env python3
"""Rebuild lever experiment JSON summaries from saved artifact .md files.

The bench had a circular-reference bug that prevented JSON save, but all the
response markdowns were written. This reconstructs the per-experiment summary
by re-running the scoring on each round's .md file.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rtxpro6000_coding_bench import BENCHMARKS, score_response, MODELS

ROOT = Path("/home/gisenberg/git/gisenberg/local-model-eval/experiments/rtxpro6000_levers")


def rescore_experiment(exp_dir: Path):
    """exp_dir like qwen36-35b-a3b-q8__thinking/"""
    name = exp_dir.name
    # Last "__<mode>" is the mode
    key, _, mode = name.rpartition("__")
    if not key:
        return None
    bench_results = {}
    for bench_name, expected, module_name in BENCHMARKS:
        # Look for round0 (or any round - take the last one as final)
        rounds_md = sorted(exp_dir.glob(f"{bench_name}__round*.md"))
        if not rounds_md:
            bench_results[bench_name] = {"error": "missing_artifact", "expected": expected}
            continue
        last_md = rounds_md[-1]
        n_rounds = len(rounds_md)
        resp = last_md.read_text()
        score = score_response(bench_name, module_name, resp)
        score["expected"] = expected
        score["n_rounds"] = n_rounds
        bench_results[bench_name] = score
    total_p = sum(r.get("passed", 0) for r in bench_results.values() if isinstance(r, dict))
    total_e = sum(r.get("expected", 0) for r in bench_results.values() if isinstance(r, dict))
    summary = {
        "model": MODELS.get(key, {}).get("name", key),
        "key": key,
        "mode": mode,
        "total_score": f"{total_p}/{total_e}",
        "benchmarks": bench_results,
    }
    out = ROOT / f"{name}.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    parts = []
    for k, v in bench_results.items():
        if isinstance(v, dict):
            parts.append(f"{k[:3]}:{v.get('passed',0)}/{v.get('expected',0)}")
    per_bench = "  ".join(parts)
    print(f"  {name:55s} {total_p:>2}/{total_e}  ({per_bench})")
    return summary


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    for exp_dir in sorted(ROOT.iterdir()):
        if exp_dir.is_dir():
            rescore_experiment(exp_dir)


if __name__ == "__main__":
    main()
