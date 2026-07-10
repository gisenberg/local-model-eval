# Colibri GLM-5.2 Smoke Test

Date: 2026-07-09

## Setup

- Runtime: `JustVugg/colibri` at commit `ed3916b`
- Model: `jlnsrk/GLM-5.2-colibri-int4`
- Host: local homelab Linux box, AMD EPYC 4585PX, 32 logical CPUs, 128 GB RAM
- Model placement for test: `/home/gisenberg/models/glm52-colibri-int4` on the root NVMe volume
- Colibri checkout for test: `/home/gisenberg/git/JustVugg/colibri`

The model was not tested from `/mnt/extended`. Colibri's own `iobench` pattern against that SATA SSD measured only 0.27 GB/s buffered and 0.25 GB/s with `O_DIRECT`. The root NVMe measured 3.11 GB/s buffered and 9.93 GB/s with `O_DIRECT` on the same pattern, so the smoke used the root filesystem despite the large footprint.

## Download And Load

- Download command: `hf download jlnsrk/GLM-5.2-colibri-int4 --local-dir /home/gisenberg/models/glm52-colibri-int4 --max-workers 8`
- Download completed in 54m13s.
- HF reported 379 GB downloaded.
- Local disk footprint was 353G by `du`.
- Local model contained 144 safetensors files: 141 main shards plus 3 MTP shards.
- Colibri `info` reported 144 shard files, 379 GB on disk.
- Model load took 3.62s.
- Dense resident set was 9780.06 MB.
- Runtime reported `layers=78 experts=256`, MTP present, and `DRAFT=0`.
- With `--ram 96`, Colibri projected a 28.0 GB peak and used cache cap 8 experts/layer.

## Smoke Prompt

Command:

```bash
DRAFT=0 python3 /home/gisenberg/git/JustVugg/colibri/c/coli run \
  --model /home/gisenberg/models/glm52-colibri-int4 \
  --ram 96 \
  --temp 0 \
  --ngen 64 \
  "Write a compact Python function add(a, b) that returns the sum, and include one assert."
```

Generated output:

```python
def add(a, b):
    assert isinstance(a, (int, float)) and isinstance(b, (int, float)), "Inputs must be numbers"
    return a + b
```

## Result

- The model worked end to end and produced a correct trivial Python answer.
- Throughput was not practical: 41 generated tokens in 582.55s, or 0.07 tok/s.
- Status checkpoints showed 0.05 tok/s at 16 tokens and 0.07 tok/s at 32 tokens.
- Expert hit rate reached only 18.3%.
- RSS was 21.89 GB.
- Expert loads were very high: 980.5 expert loads/token, or 13.07 per layer across 75 sparse layers.
- MTP was present but disabled for the smoke (`DRAFT=0`), so acceptance was 0/0 and token/forward stayed at 1.00.
- Runtime profile:
  - expert-disk: 97.7s
  - expert-matmul: 347.5s
  - attention: 84.5s
  - other: 52.8s

## Read

Colibri is a real technical path for local GLM-5.2 inference, but on this host it is not useful for our coding benchmarks in its current form. The blocker was not model load or disk capacity; the smoke reached generation correctly. The blocker was cold-path decode speed and CPU-side expert matmul cost. Even with the model on the fast NVMe and 96 GB RAM budget, the first answer ran at 0.07 tok/s.

The experiment was stopped after the smoke. The Colibri checkout, GLM-5.2 model directory, and leftover Hugging Face metadata cache were removed from disk.
