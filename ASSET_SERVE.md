# Asset Serving Contract

This host exposes one consolidated local asset service on Tailscale. Use it only from trusted tailnet clients.

Base URL:

```text
http://rtx6000.tail2fcc57.ts.net:18080
```

Equivalent Tailscale IP:

```text
http://100.107.241.79:18080
```

The old split services on ports `18081`, `18082`, and `18083` are no longer required for normal use. Image, mesh, music, sound effect, and speech-to-text jobs all go through `homelab-server`.

## Preferred Agent Interface

Use the `homelab` CLI from the `homelab-cli` repo when available. It wraps submission, polling, download, prompt refinement, and output paths.

```bash
homelab image "small red cube on a white background" --out-dir ./assets --name red-cube
homelab mesh "a fantasy sword" --out-dir ./assets --name fantasy-sword
homelab music "dark ambient dungeon exploration bed, no vocals" --duration 30 --out-dir ./assets --name dungeon-bed
homelab sfx "short metallic UI confirm click, no music" --duration 1 --out-dir ./assets --name ui-confirm
homelab stt ./voice-note.mp3 --out ./voice-note.txt
```

For mesh generation, `homelab mesh` automatically obtains a reference image if `--image` is not provided. By default it shells out to Codex imagegen for that reference image; use `--image-source server` to use this machine's image endpoint instead.

## Models

| Capability | Endpoint | Default backend | Output |
|---|---|---|---|
| Image | `/v1/image-generations` | FLUX.2-dev NVFP4 via local ComfyUI | PNG |
| Mesh | `/v1/mesh-generations` | Hunyuan3D 2.1 command runner | GLB |
| Music | `/v1/music-generations` | ACE-Step 1.5; Stable Audio 3 available with `backend=stable-audio-3` | WAV |
| Sound effect | `/v1/sound-generations` | Stable Audio 3 Medium | WAV |
| Speech-to-text | `/v1/transcriptions` | Whisper large-v3 via faster-whisper | TXT/JSON |

All heavy GPU work is serialized by `homelab-server` through a single worker and `/tmp/homelab-gpu-generation.lock`, so concurrent CLI requests queue instead of competing for VRAM.

## Raw HTTP Pattern

All endpoints are asynchronous:

1. Submit a job.
2. Poll `/v1/.../{job_id}` until `status` is `completed`.
3. Download `/v1/.../{job_id}/result.<ext>`.

### Image

```bash
BASE=http://rtx6000.tail2fcc57.ts.net:18080

curl -s -X POST "$BASE/v1/image-generations" \
  -H 'content-type: application/json' \
  -d '{
    "prompt": "small red cube on a white background, centered product render",
    "width": 1024,
    "height": 1024,
    "steps": 24,
    "format": "png"
  }' | jq
```

Download:

```bash
curl -L "$BASE/v1/image-generations/$JOB/result.png" -o image.png
```

### Mesh

```bash
BASE=http://rtx6000.tail2fcc57.ts.net:18080
IMAGE=/path/to/object_cutout.png

curl -s -X POST "$BASE/v1/mesh-generations" \
  -F image=@"$IMAGE" \
  -F asset_name=object \
  -F texture=true \
  -F seed=42 \
  -F steps=50 \
  -F guidance_scale=5.0 \
  -F octree_resolution=384 \
  -F num_chunks=8000 \
  -F texture_views=6 \
  -F texture_resolution=512 | jq
```

Download:

```bash
curl -L "$BASE/v1/mesh-generations/$JOB/result.glb" -o object.glb
```

### Music

```bash
BASE=http://rtx6000.tail2fcc57.ts.net:18080

curl -s -X POST "$BASE/v1/music-generations" \
  -H 'content-type: application/json' \
  -d '{
    "prompt": "dark ambient dungeon exploration bed, sparse percussion, no vocals, seamless loop",
    "backend": "ace-step-1.5",
    "duration": 45,
    "format": "wav",
    "seed": 42,
    "loop": true,
    "tags": ["ambient", "game-music"]
  }' | jq
```

Use `"backend": "stable-audio-3"` to compare against Stable Audio 3 Medium.

Download:

```bash
curl -L "$BASE/v1/music-generations/$JOB/result.wav" -o music.wav
```

### Sound Effects

```bash
BASE=http://rtx6000.tail2fcc57.ts.net:18080

curl -s -X POST "$BASE/v1/sound-generations" \
  -H 'content-type: application/json' \
  -d '{
    "prompt": "heavy wooden door creaks open in a stone dungeon, close perspective",
    "duration": 5,
    "format": "wav",
    "seed": 42,
    "negative_prompt": "music, speech, vocals, low quality, clipping"
  }' | jq
```

Download:

```bash
curl -L "$BASE/v1/sound-generations/$JOB/result.wav" -o sfx.wav
```

### Speech To Text

```bash
BASE=http://rtx6000.tail2fcc57.ts.net:18080
AUDIO=/path/to/input.mp3

curl -s -X POST "$BASE/v1/transcriptions" \
  -F audio=@"$AUDIO" \
  -F model=large-v3 \
  -F language=en \
  -F task=transcribe \
  -F beam_size=5 \
  -F vad_filter=true | jq
```

Download:

```bash
curl -L "$BASE/v1/transcriptions/$JOB/result.txt" -o transcript.txt
curl -L "$BASE/v1/transcriptions/$JOB/result.json" -o transcript.json
```

## Python Polling Pattern

```python
from pathlib import Path
import time
import requests

base = "http://rtx6000.tail2fcc57.ts.net:18080"
submit = requests.post(
    f"{base}/v1/sound-generations",
    json={
        "prompt": "short magical UI confirm chime, bright bell and shimmer, no music",
        "duration": 1.5,
        "format": "wav",
        "seed": 42,
    },
    timeout=30,
)
submit.raise_for_status()
job_id = submit.json()["job_id"]

while True:
    status = requests.get(f"{base}/v1/sound-generations/{job_id}", timeout=10)
    status.raise_for_status()
    payload = status.json()
    if payload["status"] == "completed":
        break
    if payload["status"] == "failed":
        raise RuntimeError(payload.get("error", "asset generation failed"))
    time.sleep(3)

result = requests.get(f"{base}/v1/sound-generations/{job_id}/result.wav", timeout=60)
result.raise_for_status()
Path("sfx.wav").write_bytes(result.content)
```

## Operations

Primary repo:

```text
/home/gisenberg/git/gisenberg/homelab-server
```

Service:

```bash
systemctl --user status homelab-server.service
journalctl --user -u homelab-server.service -f
```

Health:

```bash
curl -s http://rtx6000.tail2fcc57.ts.net:18080/health | jq
curl -s http://rtx6000.tail2fcc57.ts.net:18080/v1/models | jq
```
