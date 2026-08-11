FROM vllm/vllm-openai@sha256:3dd2f182bdfccde57a67b91d79a563f37c870bd3c08dc8033532e4232737519a

COPY patch_muse_glimmer_dflash.py /tmp/patch_muse_glimmer_dflash.py

RUN python3 /tmp/patch_muse_glimmer_dflash.py \
    && rm /tmp/patch_muse_glimmer_dflash.py
