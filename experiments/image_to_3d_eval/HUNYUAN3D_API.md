# Hunyuan3D 2.1 Textured Mesh API

This host exposes Hunyuan3D 2.1 on the Tailscale network only.

- Base URL: `http://100.107.241.79:18081`
- Model ID: `hunyuan3d-2.1-textured-mesh`
- Output format: GLB (`model/gltf-binary`)
- Queueing: one GPU job runs at a time
- Typical runtime: 70-90 seconds for a simple isolated object with texture
- Measured peak VRAM: about 28.7 GB for shape plus texture

## Health

```bash
curl -s http://100.107.241.79:18081/health | jq
```

## Submit A Textured Mesh Job

Use an isolated object image. Transparent PNG cutouts work best. Avoid busy backgrounds, cropped objects, contact shadows, and multi-object scenes unless you want them reconstructed into the mesh.

```bash
BASE=http://100.107.241.79:18081
IMAGE=/path/to/object_cutout.png

curl -s -X POST "$BASE/v1/textured-meshes" \
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

Response:

```json
{
  "job_id": "9df1...",
  "status": "queued",
  "status_url": "/v1/textured-meshes/9df1...",
  "result_url": "/v1/textured-meshes/9df1.../result.glb"
}
```

## Poll Status

```bash
JOB=the_job_id_from_submit

curl -s "$BASE/v1/textured-meshes/$JOB" | jq
```

Statuses:

- `queued`: waiting behind another job
- `running`: generating shape or texture
- `completed`: result is ready
- `failed`: read `error` in the status response

## Download The GLB

```bash
curl -L "$BASE/v1/textured-meshes/$JOB/result.glb" -o object_textured.glb
```

Optional manifest:

```bash
curl -s "$BASE/v1/textured-meshes/$JOB/manifest" | jq
```

## Python Agent Example

```python
from pathlib import Path
import time
import requests

base = "http://100.107.241.79:18081"
image_path = Path("/path/to/object_cutout.png")

with image_path.open("rb") as f:
    submit = requests.post(
        f"{base}/v1/textured-meshes",
        files={"image": (image_path.name, f, "image/png")},
        data={
            "asset_name": image_path.stem,
            "texture": "true",
            "seed": "42",
            "steps": "50",
            "guidance_scale": "5.0",
            "octree_resolution": "384",
            "num_chunks": "8000",
            "texture_views": "6",
            "texture_resolution": "512",
        },
        timeout=30,
    )
submit.raise_for_status()
job_id = submit.json()["job_id"]

while True:
    status = requests.get(f"{base}/v1/textured-meshes/{job_id}", timeout=10)
    status.raise_for_status()
    payload = status.json()
    if payload["status"] == "completed":
        break
    if payload["status"] == "failed":
        raise RuntimeError(payload.get("error", "generation failed"))
    time.sleep(5)

result = requests.get(f"{base}/v1/textured-meshes/{job_id}/result.glb", timeout=60)
result.raise_for_status()
Path(f"{image_path.stem}_textured.glb").write_bytes(result.content)
```

## Server Command

The current server is intended to be bound to the Tailscale IPv4 only:

```bash
cd /home/gisenberg/git/gisenberg/local-model-eval/experiments/image_to_3d_eval
source .venv-hunyuan/bin/activate
python scripts/hunyuan3d21_api.py --host 100.107.241.79 --port 18081 --preload
```

Do not bind this service to `0.0.0.0` unless it is behind an authenticated reverse proxy.
