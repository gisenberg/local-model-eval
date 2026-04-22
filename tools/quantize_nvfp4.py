#!/usr/bin/env python3
"""
Quantize a HF BF16 checkpoint to NVFP4 via NVIDIA Model Optimizer.

Calibration: cnn_dailymail — NVIDIA Model Optimizer's canonical default,
used for every NVFP4 release on their HF org (Qwen3-32B-NVFP4,
Gemma-4-31B-IT-NVFP4, etc.). Keeping the same calibration dataset makes
our quant comparable to NVIDIA's reference checkpoints.

Default 128 samples, 2048 seq_len — matches modelopt's typical LLM PTQ defaults.

Outputs an HF-format checkpoint with `quantization_config.quant_method=modelopt`,
directly loadable by `vllm serve --quantization modelopt_fp4`.

Usage:
  python3 tools/quantize_nvfp4.py \\
    --model ~/models-vllm/qwen36-27b-bf16-hf \\
    --output ~/models-vllm/qwen36-27b-nvfp4 \\
    --num-samples 128 --seq-len 2048
"""

import argparse, copy, fnmatch, json, os, random, sys, time

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint


# Module-name glob patterns to exclude from NVFP4 quantization for VLMs.
# Matches NVIDIA's convention for Gemma-4 / Qwen VLM NVFP4 releases: keep vision
# tower + embed_vision + lm_head in BF16, only quantize the language tower Linears.
VLM_EXCLUDE_DEFAULTS = [
    "*vision_tower*",
    "*visual*",
    "*embed_vision*",
    "*vision_model*",
    "*multi_modal_projector*",
    "*mm_projector*",
]


def build_calib_loader(tokenizer, dataset_name, dataset_config, num_samples, seq_len, batch_size, seed):
    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config, split="train")
    else:
        ds = load_dataset(dataset_name, split="train")
    if len(ds) > num_samples:
        random.seed(seed)
        idx = random.sample(range(len(ds)), num_samples)
        ds = ds.select(idx)

    def fmt(row):
        # Try the common text fields across popular calibration datasets. PTQ calibration just
        # needs representative token-distribution coverage, not task framing.
        for k in ("article", "text", "content", "instruction"):
            v = row.get(k)
            if v:
                return v
        return " ".join(str(v) for v in row.values() if isinstance(v, str))

    texts = [fmt(r) for r in ds]

    batches = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            max_length=seq_len,
            truncation=True,
            padding="max_length",
        )
        batches.append(enc)
    return batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF BF16 checkpoint path or repo id")
    ap.add_argument("--output", required=True, help="Output dir for NVFP4 HF checkpoint")
    ap.add_argument("--calib-dataset", default="abisee/cnn_dailymail")
    ap.add_argument("--calib-config", default="3.0.0", help="dataset config name (cnn_dailymail requires '3.0.0')")
    ap.add_argument("--num-samples", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--quant-cfg",
        default="NVFP4_DEFAULT_CFG",
        help="modelopt quantization config name (e.g. NVFP4_DEFAULT_CFG, NVFP4_AWQ_LITE_CFG)",
    )
    ap.add_argument(
        "--exclude-pattern",
        action="append",
        default=None,
        help="Glob pattern(s) for modules to keep in BF16. Repeatable. Defaults to VLM-safe set.",
    )
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    args = ap.parse_args()

    exclude_patterns = args.exclude_pattern if args.exclude_pattern is not None else VLM_EXCLUDE_DEFAULTS

    os.makedirs(args.output, exist_ok=True)

    print(f"=== NVFP4 Quantization ===")
    print(f"Model       : {args.model}")
    print(f"Output      : {args.output}")
    print(f"Calib       : {args.calib_dataset} ({args.num_samples} samples, seq_len={args.seq_len})")
    print(f"Quant config: {args.quant_cfg}")
    print(f"Exclusions  : {exclude_patterns}")
    print()

    t0 = time.perf_counter()
    print("[1/4] Loading tokenizer + config...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg_obj = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    arch = (getattr(cfg_obj, "architectures", None) or [""])[0]
    is_vlm = "ConditionalGeneration" in arch or "ImageTextToText" in arch or hasattr(cfg_obj, "vision_config")

    print(f"[2/4] Loading model in BF16 (GPU). arch={arch}, vlm={is_vlm}...")
    loader = AutoModelForImageTextToText if is_vlm else AutoModelForCausalLM
    model = loader.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    print(f"      Loaded in {time.perf_counter()-t0:.1f}s")

    print("[3/4] Building calibration batches...")
    batches = build_calib_loader(
        tokenizer,
        args.calib_dataset,
        args.calib_config,
        args.num_samples,
        args.seq_len,
        args.batch_size,
        args.seed,
    )
    device = next(model.parameters()).device

    def forward_loop(m):
        with torch.no_grad():
            for i, batch in enumerate(batches):
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                m(input_ids=input_ids, attention_mask=attn)
                if (i + 1) % 16 == 0:
                    print(f"      calib {i+1}/{len(batches)}")

    cfg = copy.deepcopy(getattr(mtq, args.quant_cfg))
    for pat in exclude_patterns:
        cfg["quant_cfg"][pat] = {"enable": False}

    # Sanity-check which real module names would match the exclusion patterns, so we can log them.
    linear_names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    excluded_hits = sorted({n for n in linear_names for p in exclude_patterns if fnmatch.fnmatch(n, p)})
    print(f"      {len(linear_names)} Linears total, {len(excluded_hits)} excluded by VLM patterns")
    if excluded_hits:
        print(f"      excluded sample: {excluded_hits[:5]}{' ...' if len(excluded_hits) > 5 else ''}")

    print(f"[4/4] Quantizing with {args.quant_cfg}...")
    t1 = time.perf_counter()
    mtq.quantize(model, cfg, forward_loop)
    print(f"      Quantize+calibrate in {time.perf_counter()-t1:.1f}s")

    print(f"      Exporting HF checkpoint to {args.output}...")
    export_hf_checkpoint(model, export_dir=args.output)

    # Ensure tokenizer goes along for the ride.
    tokenizer.save_pretrained(args.output)

    # modelopt's export_hf_checkpoint only writes model + config. For VLMs, vLLM
    # also needs preprocessor_config.json (+ video_preprocessor_config.json) or
    # it'll refuse to load. Copy any sidecar files that exist in the source dir.
    if os.path.isdir(args.model):
        import shutil
        sidecars = [
            "preprocessor_config.json",
            "video_preprocessor_config.json",
            "processor_config.json",
            "vocab.json",
            "merges.txt",
            "chat_template.json",
        ]
        for fn in sidecars:
            src = os.path.join(args.model, fn)
            dst = os.path.join(args.output, fn)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f"      copied sidecar {fn}")

    total_min = (time.perf_counter() - t0) / 60
    print(f"\nDone in {total_min:.1f} min -> {args.output}")


if __name__ == "__main__":
    main()
