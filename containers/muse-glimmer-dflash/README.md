# Muse Glimmer FP8 with DFlash

This stack serves the Red Hat W8A8 FP8 checkpoint with Meta's Muse Glimmer assistant through vLLM DFlash.

The image is pinned to the dedicated Muse Glimmer vLLM image digest.
The derived layer applies guarded compatibility fixes for native assistant registration, Muse configuration fields, exported assistant weight names, and Muse's direct language-model wrapper.
Each patch fails the image build if the pinned upstream source no longer matches, preventing a silent patch against an incompatible vLLM revision.

Build the image:

```bash
docker build -t local/vllm-muse-glimmer:dflash-native containers/muse-glimmer-dflash
```

Start the server:

```bash
bash containers/muse-glimmer-dflash/run_server.sh
```

The default uses the checkpoint's native 15-token speculative window.
Set `MUSE_DFLASH_TOKENS` to test another window.

```bash
MUSE_DFLASH_TOKENS=8 bash containers/muse-glimmer-dflash/run_server.sh
```

The validated configuration exposes the OpenAI-compatible API on port 8092 with a 131,072-token context limit.
Its lightweight benchmark result is stored in `experiments/muse_glimmer_30b_fp8_dflash15/results.json`.
