#!/usr/bin/env python3
"""
Render SVG charts for the RULER-Qwen3.6 YaRN comparison benchmark.

Produces two charts (pure-Python SVG, no matplotlib — stdlib only):
  1. results/ruler_qwen36_pass_rate.svg    — (config × ctx) heatmap, labeled
                                              with pass-rate (0/4, 1/4, ..., 4/4).
  2. results/ruler_qwen36_throughput.svg   — log-scale ctx on X, prefill
                                              tok/s on Y, one line per config.

Input: experiments/ruler_qwen36/ruler_results.json
       (or any set of per-trial *_t0.json files; both are supported).
"""

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/gisenberg/git/gisenberg/local-model-eval")
RESULTS_DIR = ROOT / "experiments" / "ruler_qwen36"
OUT_DIR = ROOT / "results"

CONFIG_ORDER = ["native_262k", "yarn2_512k", "yarn4_1m"]
CONFIG_LABEL = {
    "native_262k": "Native (no YaRN)",
    "yarn2_512k": "YaRN ×2 (512K)",
    "yarn4_1m": "YaRN ×4 (1M)",
}
CONFIG_COLOR = {
    "native_262k": "#1f77b4",   # blue
    "yarn2_512k": "#ff7f0e",    # orange
    "yarn4_1m": "#2ca02c",      # green
}
CONFIG_MAX_CTX = {
    "native_262k": 262144,
    "yarn2_512k": 524288,
    "yarn4_1m": 1048576,
}

TASKS = ["niah_single", "niah_multi", "variable", "common_words"]


def load_trials() -> list[dict]:
    """Load every per-trial JSON, de-duping if ruler_results.json also exists."""
    trials: dict[tuple, dict] = {}
    for p in sorted(RESULTS_DIR.glob("*_t*.json")):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if "config" not in data or "ctx" not in data:
            continue
        key = (data["config"], data["ctx"], data["task"], data.get("trial", 0))
        trials[key] = data
    return list(trials.values())


def aggregate(trials: list[dict]) -> dict[tuple, dict]:
    """(config, ctx) → {pass_count, total, prefill_tps, decode_tps, tasks_seen}."""
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for t in trials:
        if "score" not in t:
            continue
        buckets[(t["config"], t["ctx"])].append(t)
    out = {}
    for key, items in buckets.items():
        passes = sum(1 for t in items if t["score"]["pass"])
        prefill = [t["prefill_tps"] for t in items if t.get("prefill_tps")]
        decode = [t["decode_tps"] for t in items if t.get("decode_tps")]
        out[key] = {
            "pass": passes,
            "total": len(items),
            "prefill_tps": statistics.mean(prefill) if prefill else None,
            "decode_tps": statistics.mean(decode) if decode else None,
            "tasks": sorted({t["task"] for t in items}),
        }
    return out


# ---------------------------------------------------------------------------
# SVG primitives
# ---------------------------------------------------------------------------


def svg_open(width: int, height: int, bg: str = "#ffffff") -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="system-ui, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="{bg}"/>',
    ]


def svg_close() -> str:
    return "</svg>"


def svg_text(x, y, text, size=12, anchor="start", color="#222", weight="normal"):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{color}" font-weight="{weight}">{text}</text>'
    )


def svg_line(x1, y1, x2, y2, color="#888", width=1, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{width}"{d}/>'
    )


def svg_rect(x, y, w, h, fill="#fff", stroke="#222", stroke_width=1, rx=0):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" rx="{rx}"/>'
    )


def svg_circle(cx, cy, r, fill="#1f77b4", stroke="#fff", stroke_width=1):
    return (
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width}"/>'
    )


def svg_polyline(points: list[tuple], color="#1f77b4", width=2):
    pts = " ".join(f"{p[0]},{p[1]}" for p in points)
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-linecap="round" stroke-linejoin="round" points="{pts}"/>'
    )


# ---------------------------------------------------------------------------
# Chart 1: pass-rate heatmap
# ---------------------------------------------------------------------------


def chart_pass_rate(agg: dict) -> str:
    # Collect every observed context point across configs; sort.
    ctxs = sorted({ctx for (_, ctx) in agg.keys()})
    configs = [c for c in CONFIG_ORDER if any(k[0] == c for k in agg.keys())]

    # Layout
    cell_w = 100
    cell_h = 60
    margin_l = 170
    margin_r = 40
    margin_t = 90
    margin_b = 60
    grid_w = cell_w * len(ctxs)
    grid_h = cell_h * len(configs)
    width = margin_l + grid_w + margin_r
    height = margin_t + grid_h + margin_b + 60

    lines = svg_open(width, height)
    lines.append(svg_text(
        width // 2, 28,
        "RULER pass rate — Qwen3.6-35B-A3B Q8 on RTX Pro 6000 llama.cpp CUDA",
        size=18, anchor="middle", weight="600",
    ))
    lines.append(svg_text(
        width // 2, 52,
        "cells: (tasks passed) / (tasks attempted). 4 tasks: niah_single, niah_multi, variable, common_words.",
        size=12, anchor="middle", color="#555",
    ))

    # X axis labels (ctx)
    for i, ctx in enumerate(ctxs):
        x = margin_l + i * cell_w + cell_w / 2
        label = f"{ctx // 1024}K" if ctx >= 1024 else str(ctx)
        lines.append(svg_text(x, margin_t - 12, label, size=13, anchor="middle", weight="600"))

    # Y axis labels (config)
    for j, cfg in enumerate(configs):
        y = margin_t + j * cell_h + cell_h / 2 + 5
        lines.append(svg_text(margin_l - 12, y, CONFIG_LABEL[cfg], size=13, anchor="end", weight="600"))
        dot_cx = margin_l - 150
        lines.append(svg_circle(dot_cx, y - 5, 6, fill=CONFIG_COLOR[cfg], stroke="#fff"))

    # Grid cells
    for j, cfg in enumerate(configs):
        for i, ctx in enumerate(ctxs):
            x = margin_l + i * cell_w
            y = margin_t + j * cell_h
            entry = agg.get((cfg, ctx))
            if entry is None:
                # Not applicable (ctx > max_ctx) OR not yet run
                if ctx > CONFIG_MAX_CTX[cfg]:
                    label_str = "n/a"
                    fill = "#eeeeee"
                    text_color = "#aaa"
                else:
                    label_str = "—"
                    fill = "#f7f7f7"
                    text_color = "#bbb"
            else:
                p = entry["pass"]
                t = entry["total"]
                label_str = f"{p}/{t}"
                # Color: green if all pass, yellow if partial, red if all fail
                ratio = p / t if t else 0
                if ratio == 1:
                    fill = "#a8d8a8"  # soft green
                    text_color = "#0a4a0a"
                elif ratio >= 0.5:
                    fill = "#f5d88a"  # soft amber
                    text_color = "#5a4000"
                else:
                    fill = "#f0a8a8"  # soft red
                    text_color = "#5a0000"
            lines.append(svg_rect(x, y, cell_w, cell_h, fill=fill, stroke="#ccc", stroke_width=1))
            lines.append(svg_text(
                x + cell_w / 2, y + cell_h / 2 + 6,
                label_str, size=18, anchor="middle", weight="700", color=text_color,
            ))
            # Annotate throughput underneath, small
            if entry and entry.get("prefill_tps"):
                sub = f"{entry['prefill_tps']:.0f} t/s pre"
                lines.append(svg_text(
                    x + cell_w / 2, y + cell_h / 2 + 22,
                    sub, size=10, anchor="middle", color=text_color,
                ))

    # Axis titles
    lines.append(svg_text(
        margin_l + grid_w / 2, margin_t + grid_h + 35,
        "context length (tokens)", size=14, anchor="middle", weight="600",
    ))

    # Legend
    legend_y = margin_t + grid_h + 75
    swatches = [
        ("#a8d8a8", "all tasks pass"),
        ("#f5d88a", "≥50% pass"),
        ("#f0a8a8", "<50% pass"),
        ("#eeeeee", "n/a (ctx > config max)"),
        ("#f7f7f7", "not run"),
    ]
    lx = margin_l
    for fill, label in swatches:
        lines.append(svg_rect(lx, legend_y - 12, 18, 14, fill=fill, stroke="#ccc", stroke_width=1))
        lines.append(svg_text(lx + 24, legend_y, label, size=11, color="#444"))
        lx += 150

    lines.append(svg_close())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chart 2: prefill + decode throughput vs context
# ---------------------------------------------------------------------------


def chart_throughput(agg: dict) -> str:
    ctxs = sorted({ctx for (_, ctx) in agg.keys()})
    configs = [c for c in CONFIG_ORDER if any(k[0] == c for k in agg.keys())]
    if not ctxs:
        return "<!-- no data -->"

    # Per-config series
    prefill_series = {}
    decode_series = {}
    for cfg in configs:
        pts_p = []
        pts_d = []
        for ctx in ctxs:
            e = agg.get((cfg, ctx))
            if e and e.get("prefill_tps"):
                pts_p.append((ctx, e["prefill_tps"]))
            if e and e.get("decode_tps"):
                pts_d.append((ctx, e["decode_tps"]))
        prefill_series[cfg] = pts_p
        decode_series[cfg] = pts_d

    # Compute data ranges
    max_prefill = max(
        (v for pts in prefill_series.values() for (_, v) in pts),
        default=1,
    )
    max_decode = max(
        (v for pts in decode_series.values() for (_, v) in pts),
        default=1,
    )
    min_ctx = min(ctxs)
    max_ctx = max(ctxs)

    # Layout: two stacked panels sharing x
    margin_l = 80
    margin_r = 250  # right-side legend
    margin_t = 70
    gap = 50
    panel_w = 720
    panel_h = 260
    width = margin_l + panel_w + margin_r
    height = margin_t + panel_h + gap + panel_h + 80

    def x_for(ctx: int) -> float:
        # log scale on x
        lo = math.log10(min_ctx)
        hi = math.log10(max_ctx)
        t = (math.log10(ctx) - lo) / max(1e-9, hi - lo)
        return margin_l + t * panel_w

    def y_for(val: float, y_top: float, y_range: float, v_max: float) -> float:
        t = val / v_max
        return y_top + (1 - t) * y_range

    lines = svg_open(width, height)
    lines.append(svg_text(
        width // 2, 28,
        "RULER throughput vs context — Qwen3.6-35B-A3B Q8, RTX Pro 6000 llama.cpp",
        size=18, anchor="middle", weight="600",
    ))
    lines.append(svg_text(
        width // 2, 50,
        "prefill (top) and decode (bottom) throughput, averaged across 4 RULER tasks per (config, ctx)",
        size=12, anchor="middle", color="#555",
    ))

    # --- PANEL 1: prefill ---
    p1_top = margin_t
    p1_bot = p1_top + panel_h
    # Frame
    lines.append(svg_rect(margin_l, p1_top, panel_w, panel_h, fill="#fafafa", stroke="#ccc"))
    # Y ticks
    y_max_pre = math.ceil(max_prefill / 1000) * 1000
    for yval in range(0, int(y_max_pre) + 1, 1000):
        y = y_for(yval, p1_top, panel_h, y_max_pre)
        lines.append(svg_line(margin_l, y, margin_l + panel_w, y, color="#ddd"))
        lines.append(svg_text(margin_l - 8, y + 4, f"{yval:,}", size=11, anchor="end", color="#555"))
    # X ticks
    for ctx in ctxs:
        x = x_for(ctx)
        lines.append(svg_line(x, p1_top, x, p1_bot, color="#eee"))
    lines.append(svg_text(
        margin_l - 55, p1_top + panel_h / 2,
        "prefill tok/s", size=12, anchor="middle", color="#444",
        weight="600",
    ).replace("<text ", f'<text transform="rotate(-90 {margin_l - 55} {p1_top + panel_h / 2})" '))

    # Draw lines
    for cfg in configs:
        pts = prefill_series[cfg]
        if not pts:
            continue
        poly_pts = [(x_for(c), y_for(v, p1_top, panel_h, y_max_pre)) for (c, v) in pts]
        lines.append(svg_polyline(poly_pts, color=CONFIG_COLOR[cfg], width=2.5))
        for (cx, cy) in poly_pts:
            lines.append(svg_circle(cx, cy, 4, fill=CONFIG_COLOR[cfg], stroke="#fff", stroke_width=2))

    # --- PANEL 2: decode ---
    p2_top = p1_bot + gap
    p2_bot = p2_top + panel_h
    lines.append(svg_rect(margin_l, p2_top, panel_w, panel_h, fill="#fafafa", stroke="#ccc"))
    y_max_dec = math.ceil(max_decode / 50) * 50
    for yval in range(0, int(y_max_dec) + 1, 50):
        y = y_for(yval, p2_top, panel_h, y_max_dec)
        lines.append(svg_line(margin_l, y, margin_l + panel_w, y, color="#ddd"))
        lines.append(svg_text(margin_l - 8, y + 4, f"{yval}", size=11, anchor="end", color="#555"))
    for ctx in ctxs:
        x = x_for(ctx)
        lines.append(svg_line(x, p2_top, x, p2_bot, color="#eee"))
    lines.append(svg_text(
        margin_l - 55, p2_top + panel_h / 2,
        "decode tok/s", size=12, anchor="middle", color="#444",
        weight="600",
    ).replace("<text ", f'<text transform="rotate(-90 {margin_l - 55} {p2_top + panel_h / 2})" '))

    for cfg in configs:
        pts = decode_series[cfg]
        if not pts:
            continue
        poly_pts = [(x_for(c), y_for(v, p2_top, panel_h, y_max_dec)) for (c, v) in pts]
        lines.append(svg_polyline(poly_pts, color=CONFIG_COLOR[cfg], width=2.5))
        for (cx, cy) in poly_pts:
            lines.append(svg_circle(cx, cy, 4, fill=CONFIG_COLOR[cfg], stroke="#fff", stroke_width=2))

    # X axis labels (below both panels)
    for ctx in ctxs:
        x = x_for(ctx)
        lines.append(svg_line(x, p2_bot, x, p2_bot + 6, color="#888"))
        label = f"{ctx // 1024}K" if ctx >= 1024 else str(ctx)
        lines.append(svg_text(x, p2_bot + 20, label, size=11, anchor="middle", color="#444"))
    lines.append(svg_text(
        margin_l + panel_w / 2, p2_bot + 45,
        "context length (tokens, log scale)", size=13, anchor="middle", weight="600",
    ))

    # Native-ctx boundary marker across both panels
    native_x = x_for(262144)
    lines.append(svg_line(
        native_x, p1_top, native_x, p2_bot,
        color="#c44", width=1.5, dash="6,4",
    ))
    lines.append(svg_text(
        native_x + 6, p1_top + 14,
        "native 262K boundary", size=11, color="#c44", weight="600",
    ))

    # Legend on right side of top panel
    lx = margin_l + panel_w + 40
    ly = p1_top + 20
    lines.append(svg_text(lx, ly, "Config", size=13, weight="600", color="#222"))
    ly += 24
    for cfg in configs:
        lines.append(svg_line(lx, ly - 4, lx + 28, ly - 4, color=CONFIG_COLOR[cfg], width=3))
        lines.append(svg_circle(lx + 14, ly - 4, 4, fill=CONFIG_COLOR[cfg], stroke="#fff", stroke_width=2))
        lines.append(svg_text(lx + 36, ly, CONFIG_LABEL[cfg], size=12, color="#333"))
        ly += 22

    lines.append(svg_close())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    trials = load_trials()
    agg = aggregate(trials)
    print(f"Loaded {len(trials)} trials covering {len(agg)} (config, ctx) cells")
    for k in sorted(agg.keys()):
        e = agg[k]
        print(
            f"  {k[0]:14s} ctx={k[1]:>7,d}  "
            f"pass={e['pass']}/{e['total']}  "
            f"prefill={e['prefill_tps']}  decode={e['decode_tps']}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pass_svg = chart_pass_rate(agg)
    (OUT_DIR / "ruler_qwen36_pass_rate.svg").write_text(pass_svg)
    print(f"\nwrote {OUT_DIR / 'ruler_qwen36_pass_rate.svg'} ({len(pass_svg):,} bytes)")

    tput_svg = chart_throughput(agg)
    (OUT_DIR / "ruler_qwen36_throughput.svg").write_text(tput_svg)
    print(f"wrote {OUT_DIR / 'ruler_qwen36_throughput.svg'} ({len(tput_svg):,} bytes)")


if __name__ == "__main__":
    main()
