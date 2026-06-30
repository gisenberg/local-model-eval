# TRELLIS.2 vs Hunyuan3D 2.1 Image-to-3D Eval

This experiment uses one generated image as the shared input for both image-to-3D systems, exports GLB artifacts, and builds a local browser viewer for visual inspection.

## Input

- `input/robot_figurine.png`: generated with the repo's image generation skill.
- `input/robot_figurine_cutout.png`: transparent-background derivative used as the default model input.
- `input/prompt.txt`: exact generation prompt.

The object is intentionally isolated, fully visible, and material-rich so the two 3D systems are easy to compare. The cutout avoids reconstructing the studio background or contact shadow as geometry.

## Upstream Versions

- TRELLIS.2: `microsoft/TRELLIS.2` at `75fbf0183001ed9876c8dbb35de6b68552ee08bd`
- Hunyuan3D 2.1: `Tencent-Hunyuan/Hunyuan3D-2.1` at `82920d643c0dc2f7bfd7255f45f62d386edfe60c`

## Setup

Clone the pinned upstream repos:

```bash
cd experiments/image_to_3d_eval
./setup.sh --clone
```

Install Hunyuan3D 2.1 into an isolated uv environment:

```bash
./setup.sh --hunyuan
```

TRELLIS.2 uses upstream's Conda-based setup because it builds several CUDA extensions. Install Miniconda/Mambaforge first, make sure `CUDA_HOME` points at the local CUDA toolkit if needed, then run:

```bash
./setup.sh --trellis
```

On this machine, Hunyuan3D 2.1 has been installed and run successfully. TRELLIS.2 is cloned and pinned, but its installer is blocked until Conda is available. `setup.sh --check` prints the local GPU/tooling status.

## Run

Hunyuan3D shape + texture:

```bash
source .venv-hunyuan/bin/activate
python scripts/run_hunyuan3d21.py
```

The Hunyuan runner avoids the upstream `bpy` dependency by saving the paint result as OBJ/MTL first and converting that textured OBJ to GLB with `trimesh`. Use `--skip-texture` for a faster geometry-only run.

TRELLIS.2:

```bash
conda activate trellis2
python scripts/run_trellis2.py
```

Regenerate the visual comparison page:

```bash
python scripts/make_viewer.py
```

Open `output/viewer.html` in a browser. It shows the source image plus any generated GLB files using `<model-viewer>`. TRELLIS.2 also writes a turntable MP4 when rendering succeeds.

## Expected Outputs

- `output/hunyuan3d21/robot_shape.glb`
- `output/hunyuan3d21/robot_textured.glb`
- `output/hunyuan3d21/robot_shape_preview.png`
- `output/trellis2/robot_trellis2.glb`
- `output/trellis2/robot_trellis2.mp4`
- `output/viewer.html`

Large generated outputs, upstream checkouts, Hugging Face caches, and local virtual environments are intentionally ignored by Git.
