#!/usr/bin/env bash
set -euo pipefail

model_root=${MODEL_ROOT:-/mnt/extended/gisenberg/models}

target_repo=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
target_revision=0dcd680e5585c791728c83342b311d0a0026dbeb
target_dir="$model_root/nemotron-3.5-lightning-30b-a3b-nvfp4-0dcd680e"

dspark_repo=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark
dspark_revision=d10c6ff40d6e69d1f92e407e027de3eafdb77645
dspark_dir="$model_root/nemotron-3.5-lightning-30b-a3b-nvfp4-dspark-d10c6ff4"

dflash_repo=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash
dflash_revision=7fc1f1ff4b82b917efbd0710df0872c2bb89caa5
dflash_dir="$model_root/nemotron-3.5-lightning-30b-a3b-nvfp4-dflash-7fc1f1ff"

mkdir -p "$model_root"

hf download "$target_repo" \
    --revision "$target_revision" \
    --local-dir "$target_dir"

hf download "$dspark_repo" \
    --revision "$dspark_revision" \
    --local-dir "$dspark_dir"

hf download "$dflash_repo" \
    --revision "$dflash_revision" \
    --local-dir "$dflash_dir"

du -sh "$target_dir" "$dspark_dir" "$dflash_dir"
