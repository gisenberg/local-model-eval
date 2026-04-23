#!/usr/bin/env python3
"""Generate headline charts from the local-model-eval rankings.

Data is hand-curated from results/MODEL_RANKINGS_*.md, HARDWARE_SPECS.md,
and TURBOQUANT.md. Each row represents one measured configuration
(hardware + model + quant + kv_config + thinking), with throughput and
a coding-benchmark score where available.

Output: PNG files in results/charts/
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results" / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Run:
    hardware: str
    model: str
    family: str
    quant: str
    kv: str
    backend: str
    tps: float
    score: float | None           # tests passed
    score_max: float | None       # tests total (17 or 22)
    vram_gb: float | None = None
    thinking: str = "off"
    note: str = ""


# Three-benchmark suite = 17 tests (ExprEval 5 + A* 6 + LRU 6)
# Four-benchmark suite = 22 tests (adds String Processor 5)
# Scores shown as pass %, normalized across suites for the scatter plot.

DATA: list[Run] = [
    # ---- RTX 5090 (rebench-corrected tok/s, scores from MODEL_RANKINGS_5090.md) ----
    Run("RTX 5090", "Gemma 4 26B-A4B", "Gemma",   "Q6_K",   "turbo4", "llama.cpp", 138.9, 17, 17, 26.7),
    Run("RTX 5090", "Gemma 4 26B-A4B", "Gemma",   "Q4_K_M", "turbo4", "llama.cpp", 149.5, 16, 17, 21.2),
    Run("RTX 5090", "Gemma 4 31B-IT",  "Gemma",   "Q4_K_M", "turbo4", "llama.cpp",  50.3, 17, 17, 23.6),
    Run("RTX 5090", "Gemma 31B Opus",  "Gemma",   "Q4_K_M", "turbo4", "llama.cpp",  50.9, 16, 17, 23.6),
    Run("RTX 5090", "Qwopus 27B v3",   "Qwen",    "Q6_K",   "turbo4", "llama.cpp",  49.6, 16, 17, 25.5),
    Run("RTX 5090", "Harmonic 27B",    "Qwen",    "Q4_K_M", "turbo4", "llama.cpp",  61.3, 17, 17, 20.5, thinking="off"),
    Run("RTX 5090", "Qwen 27B Opus",   "Qwen",    "Q4_K_M", "turbo4", "llama.cpp",  60.0, 17, 17, 20.5),
    Run("RTX 5090", "Qwen 27B base",   "Qwen",    "Q6_K",   "turbo4", "llama.cpp",  49.6, 10, 17, 25.6),
    Run("RTX 5090", "Qwen 35B-A3B",    "Qwen",    "Q4_K_M", "turbo4", "llama.cpp", 174.4, 11, 17, 24.8),
    Run("RTX 5090", "Gemma 4 E4B",     "Gemma",   "Q8_0",   "turbo4", "llama.cpp", 131.0,  5, 22, 12.1),
    Run("RTX 5090", "Qwen3-8B",        "Qwen",    "Q4_K_M", "turbo4", "llama.cpp",  85.0,  8, 17,  8.0, note="quality drop"),
    Run("RTX 5090", "Qwen3-8B",        "Qwen",    "Q4_K_M", "f16",    "llama.cpp",  78.0, 12, 17,  8.0),
    # vLLM / NVFP4 variants on 5090
    Run("RTX 5090", "Gemma 4 26B-A4B", "Gemma",   "NVFP4",  "fp8",    "vLLM",      135.0, 22, 22, 31.3),
    Run("RTX 5090", "Gemma 4 31B-IT",  "Gemma",   "NVFP4",  "fp8",    "vLLM",       42.0, 22, 22, 30.3),

    # KV-compression pairs from TURBOQUANT.md (f16 vs turbo4 speed)
    Run("RTX 5090", "Gemma 4 26B-A4B", "Gemma",   "Q6_K",   "f16",    "llama.cpp", 152.0, 17, 17, 26.7, note="f16 baseline"),
    Run("RTX 5090", "Gemma 4 26B-A4B", "Gemma",   "Q4_K_M", "f16",    "llama.cpp", 164.0, 16, 17, 21.2, note="f16 baseline"),

    # ---- RTX Pro 6000 (Blackwell, 96 GB) ----
    Run("RTX Pro 6000", "Gemma 4 31B-IT",   "Gemma", "BF16", "f16", "llama.cpp",  25.1, 22, 22, 82.0),
    Run("RTX Pro 6000", "Gemma 4 31B-IT",   "Gemma", "Q8_0", "f16", "llama.cpp",  43.8, 22, 22, 54.5),
    Run("RTX Pro 6000", "gpt-oss-120b",     "OpenAI","Q8_0", "f16", "llama.cpp", 264.4, 21, 22, 65.8),
    Run("RTX Pro 6000", "Qwen 3.6 35B-A3B", "Qwen",  "BF16", "f16", "llama.cpp", 135.0, 14, 22, 72.1),
    Run("RTX Pro 6000", "Qwen 3.6 35B-A3B", "Qwen",  "Q8_0", "f16", "llama.cpp", 221.0, 15, 22, 41.6),
    Run("RTX Pro 6000", "Gemopus 31B",      "Gemma", "BF16", "f16", "llama.cpp",  25.1, 16, 22, 82.0),
    Run("RTX Pro 6000", "Gemopus 31B",      "Gemma", "Q8_0", "f16", "llama.cpp",  43.8, 15, 22, 54.5),
    Run("RTX Pro 6000", "Qwen3-Coder-Next", "Qwen",  "Q6_K", "f16", "llama.cpp", 196.4, 15, 22, 70.3),
    # Vulkan variants (for engine comparison context, scored same):
    Run("RTX Pro 6000", "Qwen 3.6 35B-A3B", "Qwen",  "BF16", "f16", "llama.cpp-vk",  81.3, 14, 22, 72.1, note="vulkan"),

    # ---- M4 Max 36 GB ----
    Run("M4 Max", "Gemma 4 31B-IT",         "Gemma", "Q4_K_M",     "f16",       "llama.cpp", 15.3, 17, 17, 24.3),
    Run("M4 Max", "Gemma 4 31B-IT",         "Gemma", "Q4_K_M",     "turbo4",    "llama.cpp", 11.8, 17, 17, 21.0, note="long-ctx"),
    Run("M4 Max", "Gemma 4 26B-A4B",        "Gemma", "Q6_K",       "f16",       "llama.cpp", 65.6, 15, 17, 23.0),
    Run("M4 Max", "Gemma 4 26B-A4B",        "Gemma", "Q6_K",       "planar3",   "llama.cpp", 64.5, 15, 17, 23.0),
    Run("M4 Max", "Gemma 4 26B-A4B",        "Gemma", "Q6_K",       "turbo4",    "llama.cpp", 45.0, 16, 17, 20.0),
    Run("M4 Max", "Gemma 4 26B-A4B",        "Gemma", "Q4_K_M",     "f16",       "llama.cpp", 58.9, 11, 17, 17.0),
    Run("M4 Max", "Qwen 27B Opus",          "Qwen",  "Q4_K_M",     "f16",       "llama.cpp", 13.0, 11, 17, 23.0, note="rotorquant baseline"),
    Run("M4 Max", "Qwen 27B Opus",          "Qwen",  "Q4_K_M",     "planar3",   "llama.cpp", 15.5, 11, 17, 23.0),
    Run("M4 Max", "Qwen 27B Opus",          "Qwen",  "MLX-4bit",   "—",         "MLX",       18.5, 13, 17, 14.0),
    Run("M4 Max", "Qwen 3.5 9B",            "Qwen",  "Q4_K_M",     "f16",       "llama.cpp", 35.4,  9, 17,  5.5),
    Run("M4 Max", "Nemotron 3 Nano 4B",     "Other", "Q4_K_M",     "f16",       "llama.cpp", 65.5,  7, 17,  2.8, thinking="on"),
    Run("M4 Max", "Gemma 4 31B-IT",         "Gemma", "Q4_K_M",     "MLX-4bit",  "MLX",        9.9, 17, 17, 20.0),
    Run("M4 Max", "Gemma 4 26B-A4B",        "Gemma", "Q6_K",       "MLX-6bit",  "MLX",       35.5, 12, 17, 20.0),

    # ---- DGX Spark (GB10, 128 GB) ----
    Run("DGX Spark", "Qwen 3.5 122B-A10B", "Qwen",    "Q4_K_M",    "f16", "ik-llama",   26.0, 17, 17, 71.0, note="bartowski"),
    Run("DGX Spark", "Qwen 3.5 122B-A10B", "Qwen",    "Q4_K_M",    "f16", "llama.cpp",  21.0, 18, 17, 72.0, note="unsloth; 18/17 w/ bonus"),
    Run("DGX Spark", "Qwen 3.5 122B-A10B", "Qwen",    "Q4_K_M",    "f16", "llama.cpp",  25.8, 13, 17, 71.0, note="bartowski mainline"),
    Run("DGX Spark", "Qwen 3.5 122B REAP-20","Qwen",  "Q4_K_M",    "f16", "ik-llama",   29.1, 14, 17, 57.0, note="pruned"),
    Run("DGX Spark", "Qwen 3.5 122B-A10B", "Qwen",    "INT4+FP8",  "fp8", "vLLM",       49.0, 16, 17, 75.0, note="vLLM+MTP"),
    Run("DGX Spark", "GLM-4.5-Air",        "Other",   "Q4_K_M",    "f16", "llama.cpp",  21.7, 15, 17, 70.0),
    Run("DGX Spark", "Qwen3-Coder-Next",   "Qwen",    "UD-Q4_K_M", "f16", "llama.cpp",  50.2, 14, 17, 46.0),
    Run("DGX Spark", "Nemotron-3-Super",   "Other",   "Q4_K_M",    "f16", "llama.cpp",  19.7, 11, 17, 87.0, thinking="on"),
    Run("DGX Spark", "Mistral-Small-4",    "Other",   "Q4_K_M",    "f16", "llama.cpp",   8.0,  7, 17, 69.0),
    Run("DGX Spark", "Gemma 4 31B-IT",     "Gemma",   "Q8_0",      "f16", "llama.cpp",   6.7, None, None, 31.0, note="bandwidth-bound"),
    Run("DGX Spark", "MiniMax M2.5",       "Other",   "Q3_K_XL",   "f16", "llama.cpp",  29.6, None, None, 96.0, note="timeouts on A*/LRU"),

    # ---- L40S (Ada Lovelace, 46 GB, vLLM) ----
    Run("L40S", "Ministral-3-14B",   "Other", "BF16", "fp16", "vLLM",   28.0, 13, 17, 33.0),
    Run("L40S", "Qwen 3.5 9B",       "Qwen",  "BF16", "fp16", "vLLM",   44.3,  9, 17, 25.0, note="high variance"),
]


# ---------- styling ----------
HARDWARE_COLORS = {
    "RTX 5090":      "#76b900",   # NVIDIA green
    "RTX Pro 6000":  "#00875a",   # deeper green
    "M4 Max":        "#6b7280",   # graphite
    "DGX Spark":     "#1f77b4",   # blue
    "L40S":          "#ff7f0e",   # orange
}

# marker by quantization tier (precision, not format)
def quant_marker(q: str) -> str:
    q = q.lower()
    if "bf16" in q or q == "f16" or "fp16" in q:
        return "o"
    if "8" in q and "int4" not in q:   # Q8_0, FP8, MLX-8
        return "s"
    if "q6" in q or "6bit" in q:
        return "D"
    if "q4" in q or "4bit" in q or "nvfp4" in q or "ud-q4" in q or "int4" in q:
        return "^"
    if "q3" in q or "iq2" in q:
        return "X"
    return "P"


def score_pct(r: Run) -> float | None:
    if r.score is None or r.score_max is None:
        return None
    # Bonus-test rows (18/17) clip to 100%.
    return min(100.0, 100.0 * r.score / r.score_max)


# ======================================================================
# Chart 1: Quality vs Speed scatter
# ======================================================================
def chart_quality_vs_speed():
    fig, ax = plt.subplots(figsize=(12, 7.5))

    # Draw points per hardware
    for hw, color in HARDWARE_COLORS.items():
        rows = [r for r in DATA if r.hardware == hw and score_pct(r) is not None]
        for r in rows:
            ax.scatter(
                r.tps, score_pct(r),
                color=color, marker=quant_marker(r.quant),
                s=110, alpha=0.85, edgecolor="white", linewidth=0.8,
                zorder=3,
            )

    # Labels for the genuinely headline points (high-score or extreme-speed)
    label_points = [
        ("RTX Pro 6000", "gpt-oss-120b",    "Q8_0"),
        ("RTX 5090",     "Qwen 35B-A3B",    "Q4_K_M"),
        ("RTX 5090",     "Gemma 4 26B-A4B", "Q4_K_M"),   # fastest Gemma
        ("DGX Spark",    "Qwen 3.5 122B-A10B", "Q4_K_M"),
        ("DGX Spark",    "Qwen3-Coder-Next", "UD-Q4_K_M"),
        ("M4 Max",       "Gemma 4 31B-IT",  "Q4_K_M"),
        ("RTX 5090",     "Gemma 4 31B-IT",  "Q4_K_M"),
        ("RTX Pro 6000", "Qwen 3.6 35B-A3B","Q8_0"),
    ]
    labeled = set()
    for hw, m, q in label_points:
        for r in DATA:
            if r.hardware == hw and r.model == m and r.quant == q and r.kv in {"turbo4", "f16"}:
                key = (hw, m, q)
                if key in labeled: continue
                labeled.add(key)
                ax.annotate(
                    f"{r.model}\n{r.quant}",
                    xy=(r.tps, score_pct(r)),
                    xytext=(8, 6), textcoords="offset points",
                    fontsize=8, color="#333",
                )
                break

    ax.set_xscale("log")
    ax.set_xlabel("Decode throughput (tokens/sec, log scale)")
    ax.set_ylabel("Coding benchmark pass rate (%)")
    ax.set_title("Quality vs Speed — every measured (hardware, model, quant) configuration",
                 fontsize=13, pad=12)
    ax.set_ylim(-3, 108)
    ax.grid(True, which="both", alpha=0.25)
    ax.axhline(100, color="#888", linestyle=":", linewidth=0.8, alpha=0.5)

    # Two legends: hardware colors, quant markers
    hw_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=hw,
                   markerfacecolor=c, markeredgecolor="white", markersize=10)
        for hw, c in HARDWARE_COLORS.items()
    ]
    quant_entries = [
        ("BF16 / f16",   "o"),
        ("Q8_0 / FP8",   "s"),
        ("Q6_K",         "D"),
        ("Q4_K_M / NVFP4 / INT4", "^"),
        ("Q3 / IQ2",     "X"),
    ]
    q_handles = [
        plt.Line2D([0], [0], marker=mk, color="w", label=lbl,
                   markerfacecolor="#555", markeredgecolor="white", markersize=9)
        for lbl, mk in quant_entries
    ]
    leg1 = ax.legend(handles=hw_handles, title="Hardware", loc="lower left",
                     fontsize=9, title_fontsize=9, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=q_handles, title="Quantization", loc="lower right",
              fontsize=9, title_fontsize=9, framealpha=0.9)

    plt.tight_layout()
    out = OUT_DIR / "01_quality_vs_speed.png"
    plt.savefig(out, dpi=160)
    plt.close(fig)
    return out


# ======================================================================
# Chart 2: Quantization cliff heatmap
# ======================================================================
def chart_quant_cliff():
    # rows = (hardware, model), cols = ordered quant tiers
    QUANT_ORDER = ["BF16", "Q8_0", "NVFP4", "Q6_K", "Q4_K_M", "Q3_K_XL"]

    def canonical_quant(q: str) -> str | None:
        q = q.replace("UD-", "").strip()
        if q in QUANT_ORDER:
            return q
        if "int4" in q.lower():
            return "Q4_K_M"
        if q.endswith("bit"):
            return None   # MLX-native, skip from this matrix
        return None

    # Pick canonical baseline config: turbo4 on 5090, f16 elsewhere, thinking=off
    def is_baseline(r: Run) -> bool:
        if r.thinking == "on": return False
        if r.backend not in {"llama.cpp", "ik-llama", "vLLM"}: return False
        if r.hardware == "RTX 5090":
            return r.kv in {"turbo4", "fp8"}
        return r.kv in {"f16", "fp8", "fp16"}

    # Build matrix
    rows: list[tuple[str, str]] = []
    row_label: dict[tuple[str, str], str] = {}
    matrix: dict[tuple[str, str], dict[str, float]] = {}
    for r in DATA:
        q = canonical_quant(r.quant)
        if q is None: continue
        s = score_pct(r)
        if s is None: continue
        if not is_baseline(r): continue
        key = (r.hardware, r.model)
        if key not in matrix:
            matrix[key] = {}
            rows.append(key)
            row_label[key] = f"{r.model}   ({r.hardware})"
        # keep best score per cell
        matrix[key][q] = max(matrix[key].get(q, -1), s)

    # Drop rows with only one quant present (no "cliff" to show)
    rows = [k for k in rows if len(matrix[k]) >= 2]

    # Sort rows by hardware then by best-score desc
    hw_order = list(HARDWARE_COLORS.keys())
    rows.sort(key=lambda k: (hw_order.index(k[0]), -max(matrix[k].values())))

    fig, ax = plt.subplots(figsize=(11, max(4, 0.48 * len(rows) + 1.5)))
    grid = np.full((len(rows), len(QUANT_ORDER)), np.nan)
    for i, k in enumerate(rows):
        for j, q in enumerate(QUANT_ORDER):
            if q in matrix[k]:
                grid[i, j] = matrix[k][q]

    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=40, vmax=100,
                   interpolation="nearest")
    ax.set_xticks(range(len(QUANT_ORDER)))
    ax.set_xticklabels(QUANT_ORDER)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row_label[k] for k in rows], fontsize=9)
    ax.set_xlabel("Quantization (←  more precision   |   smaller  →)")
    ax.set_title("Quantization quality cliff — coding benchmark pass rate (%)\n"
                 "Filled cells = measured configuration; gray = not tested",
                 fontsize=12, pad=10)

    # Annotate each cell
    for i in range(len(rows)):
        for j in range(len(QUANT_ORDER)):
            v = grid[i, j]
            if np.isnan(v):
                ax.text(j, i, "·", ha="center", va="center", color="#aaa", fontsize=10)
            else:
                txt_color = "white" if v < 58 or v > 92 else "#111"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color=txt_color, fontsize=9, fontweight="bold")

    # Color each row-label by hardware
    for lbl, k in zip(ax.get_yticklabels(), rows):
        lbl.set_color(HARDWARE_COLORS[k[0]])

    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Pass rate (%)", fontsize=9)

    plt.tight_layout()
    out = OUT_DIR / "02_quant_cliff.png"
    plt.savefig(out, dpi=160)
    plt.close(fig)
    return out


# ======================================================================
# Chart 3: KV compression before/after (per hardware)
# ======================================================================
def chart_kv_compression():
    # Pairs: same hardware + model + quant, different kv config.
    groups: dict[tuple[str, str, str], dict[str, float]] = {}
    for r in DATA:
        key = (r.hardware, r.model, r.quant)
        kv_bucket = {
            "f16": "f16",
            "fp16": "f16",
            "planar3": "planar3 (rotorquant)",
            "turbo4": "turbo4",
            "MLX-4bit": None,  # skip
            "MLX-6bit": None,
            "—": None,
        }.get(r.kv, None)
        if kv_bucket is None:
            continue
        groups.setdefault(key, {})[kv_bucket] = r.tps

    # Keep only groups with ≥2 KV configs measured
    pairs = [(k, v) for k, v in groups.items() if len(v) >= 2]

    # Order: by hardware, then by baseline (f16) throughput desc
    pairs.sort(key=lambda kv: (list(HARDWARE_COLORS).index(kv[0][0]),
                               -kv[1].get("f16", 0)))

    kv_order = ["f16", "planar3 (rotorquant)", "turbo4"]
    kv_colors = {
        "f16":                   "#4c72b0",
        "planar3 (rotorquant)":  "#55a868",
        "turbo4":                "#c44e52",
    }

    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.55 * len(pairs) + 1.5)))
    y = np.arange(len(pairs))
    bar_h = 0.26

    labels = [f"{m} {q}   ({hw})" for (hw, m, q), _ in pairs]

    # One bar per kv config per row
    offsets = {kv: (i - 1) * bar_h for i, kv in enumerate(kv_order)}
    for kv in kv_order:
        vals = [vs.get(kv, 0) for _, vs in pairs]
        ax.barh(y + offsets[kv], vals, height=bar_h,
                color=kv_colors[kv], label=kv,
                edgecolor="white", linewidth=0.5)
        for yi, v in zip(y, vals):
            if v > 0:
                ax.text(v + 1.5, yi + offsets[kv], f"{v:.0f}",
                        va="center", fontsize=8, color="#222")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    for lbl, ((hw, *_), _) in zip(ax.get_yticklabels(), pairs):
        lbl.set_color(HARDWARE_COLORS[hw])

    ax.invert_yaxis()
    ax.set_xlabel("Decode throughput (tokens/sec)")
    ax.set_title("KV cache compression: f16 vs planar3 (rotorquant) vs turbo4\n"
                 "Same hardware + model + weight quant, KV format varied",
                 fontsize=12, pad=10)
    ax.legend(loc="lower right", fontsize=9, title="KV format", title_fontsize=9)
    ax.grid(True, axis="x", alpha=0.25)

    plt.tight_layout()
    out = OUT_DIR / "03_kv_compression.png"
    plt.savefig(out, dpi=160)
    plt.close(fig)
    return out


# ======================================================================
# Chart 4: Hardware head-to-head bars
# ======================================================================
def chart_hardware_bars():
    """For each model that was measured on ≥2 hardware platforms, show
    decode throughput as grouped bars colored by hardware."""
    by_model: dict[str, dict[str, float]] = {}

    # Normalize model names across platforms so same-class configs align
    def canonical(r: Run) -> str | None:
        m = r.model
        q = r.quant
        # Only include "reasonable default" per-platform configs
        # Filter out thinking=on and exotic KV for a clean cross-hw view
        if r.thinking == "on":
            return None
        if r.backend not in {"llama.cpp", "ik-llama"}:
            return None
        if r.hardware == "RTX 5090" and r.kv != "turbo4": return None
        if r.hardware != "RTX 5090" and r.kv not in {"f16", "planar3"}: return None

        aliases = {
            "Qwopus 27B v3":       None,                    # family-dup
            "Gemma 31B Opus":      None,
            "Harmonic 27B":        None,
            "Qwen 27B Opus":       ("Qwen 27B Opus-Distill", q),
            "Qwen 3.6 35B-A3B":    ("Qwen 35B-A3B-class",    q),
            "Qwen 35B-A3B":        ("Qwen 35B-A3B-class",    q),
            "Qwen 3.5 122B-A10B":  ("Qwen 122B-A10B",        q),
            "Qwen3-Coder-Next":    ("Qwen3-Coder-Next",      q),
            "Gemma 4 31B-IT":      ("Gemma 4 31B-IT",        q),
            "Gemma 4 26B-A4B":     ("Gemma 4 26B-A4B",       q),
            "Gemma 31B Opus":      None,
        }
        if m in aliases:
            mapped = aliases[m]
            if mapped is None:
                return None
            m2, q2 = mapped
            return f"{m2}  ({q2})"
        return None

    for r in DATA:
        ck = canonical(r)
        if ck is None: continue
        by_model.setdefault(ck, {})
        # take max tok/s if multiple rows match
        prev = by_model[ck].get(r.hardware, 0)
        by_model[ck][r.hardware] = max(prev, r.tps)

    # Keep only models that appear on ≥2 hardware
    by_model = {k: v for k, v in by_model.items() if len(v) >= 2}

    # Order rows by best throughput desc
    rows = sorted(by_model.items(), key=lambda kv: -max(kv[1].values()))

    hw_order = list(HARDWARE_COLORS.keys())
    n_hw = len(hw_order)
    y = np.arange(len(rows))
    bar_h = 0.78 / n_hw

    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.65 * len(rows) + 1.5)))

    for i, hw in enumerate(hw_order):
        vals = [vs.get(hw, 0) for _, vs in rows]
        offset = (i - (n_hw - 1) / 2) * bar_h
        ax.barh(y + offset, vals, height=bar_h,
                color=HARDWARE_COLORS[hw], label=hw,
                edgecolor="white", linewidth=0.5)
        for yi, v in zip(y, vals):
            if v > 0:
                ax.text(v + max(vals) * 0.01, yi + offset, f"{v:.0f}",
                        va="center", fontsize=8, color="#222")

    ax.set_yticks(y)
    ax.set_yticklabels([k for k, _ in rows], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Decode throughput (tokens/sec)")
    ax.set_title("Same model, different hardware — decode throughput\n"
                 "(llama.cpp / ik-llama defaults per platform; thinking off)",
                 fontsize=12, pad=10)
    ax.legend(loc="lower right", fontsize=9, title="Hardware", title_fontsize=9)
    ax.grid(True, axis="x", alpha=0.25)

    plt.tight_layout()
    out = OUT_DIR / "04_hardware_bars.png"
    plt.savefig(out, dpi=160)
    plt.close(fig)
    return out


# ======================================================================
if __name__ == "__main__":
    outputs = [
        chart_quality_vs_speed(),
        chart_quant_cliff(),
        chart_kv_compression(),
        chart_hardware_bars(),
    ]
    for p in outputs:
        print(f"wrote {p.relative_to(REPO_ROOT)}")
