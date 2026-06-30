#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TRELLIS.2 image-to-3D on the shared eval input.")
    parser.add_argument("--repo", type=Path, default=ROOT / "third_party" / "TRELLIS.2")
    parser.add_argument("--input", type=Path, default=ROOT / "input" / "robot_figurine_cutout.png")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "output" / "trellis2")
    parser.add_argument("--asset-name", default="robot")
    parser.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--envmap", default="forest", choices=["city", "courtyard", "forest", "interior", "night", "studio", "sunrise", "sunset"])
    parser.add_argument("--pipeline-type", default=None, choices=["512", "1024", "1024_cascade", "1536_cascade"])
    parser.add_argument("--decimation-target", type=int, default=1_000_000)
    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--skip-glb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    input_path = args.input.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not repo.exists():
        raise SystemExit(f"TRELLIS.2 checkout not found: {repo}. Run ./setup.sh --clone first.")
    if not input_path.exists():
        raise SystemExit(f"Input image not found: {input_path}")

    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
    sys.path.insert(0, str(repo))

    import cv2
    import imageio
    import torch
    from PIL import Image
    import trellis2.pipelines.rembg as rembg
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.renderers import EnvMap
    from trellis2.utils import render_utils
    import o_voxel

    if not torch.cuda.is_available():
        raise SystemExit("TRELLIS.2 requires CUDA, but torch.cuda.is_available() is false.")

    started = time.time()
    old_cwd = Path.cwd()
    os.chdir(repo)
    try:
        class NoRembg:
            def to(self, device):
                return self

            def cuda(self):
                return self

            def cpu(self):
                return self

            def __call__(self, image):
                return image

        rembg.NoRembg = NoRembg

        envmap_path = repo / "assets" / "hdri" / f"{args.envmap}.exr"
        envmap = EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread(str(envmap_path), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32,
            device="cuda",
        ))

        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
        pipeline.cuda()

        image = Image.open(input_path)
        mesh = pipeline.run(image, seed=args.seed, pipeline_type=args.pipeline_type)[0]
        mesh.simplify(16_777_216)

        mp4_path = None
        if not args.skip_video:
            mp4_path = out_dir / f"{args.asset_name}_trellis2.mp4"
            video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
            imageio.mimsave(mp4_path, video, fps=15)

        glb_path = None
        if not args.skip_glb:
            glb_path = out_dir / f"{args.asset_name}_trellis2.glb"
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=args.decimation_target,
                texture_size=args.texture_size,
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                verbose=True,
            )
            glb.export(str(glb_path), extension_webp=True)
    finally:
        os.chdir(old_cwd)

    manifest = {
        "model": "TRELLIS.2",
        "upstream": "microsoft/TRELLIS.2",
        "input": str(input_path),
        "asset_name": args.asset_name,
        "model_id": args.model,
        "seed": args.seed,
        "pipeline_type": args.pipeline_type,
        "elapsed_sec": round(time.time() - started, 3),
        "outputs": {
            "glb": str(glb_path) if glb_path else None,
            "mp4": str(mp4_path) if mp4_path else None,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
