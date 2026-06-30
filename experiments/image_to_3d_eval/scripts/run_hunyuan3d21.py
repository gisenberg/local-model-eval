#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
import time
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Hunyuan3D 2.1 on the shared eval input.")
    parser.add_argument("--repo", type=Path, default=ROOT / "third_party" / "Hunyuan3D-2.1")
    parser.add_argument("--input", type=Path, default=ROOT / "input" / "robot_figurine_cutout.png")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "output" / "hunyuan3d21")
    parser.add_argument("--asset-name", default="robot")
    parser.add_argument("--model", default="tencent/Hunyuan3D-2.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--octree-resolution", type=int, default=384)
    parser.add_argument("--num-chunks", type=int, default=8000)
    parser.add_argument("--skip-texture", action="store_true")
    parser.add_argument("--texture-views", type=int, default=6)
    parser.add_argument("--texture-resolution", type=int, default=512, choices=[512, 768])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    input_path = args.input.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not repo.exists():
        raise SystemExit(f"Hunyuan3D checkout not found: {repo}. Run ./setup.sh --clone first.")
    if not input_path.exists():
        raise SystemExit(f"Input image not found: {input_path}")

    os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "hy3dshape"))
    sys.path.insert(0, str(repo / "hy3dpaint"))

    import numpy as np
    import torch
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    try:
        from torchvision_fix import apply_fix
        apply_fix()
    except Exception as exc:
        print(f"Warning: torchvision compatibility fix was not applied: {exc}")

    if not torch.cuda.is_available():
        raise SystemExit("Hunyuan3D 2.1 is configured for CUDA here, but torch.cuda.is_available() is false.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    started = time.time()
    old_cwd = Path.cwd()
    os.chdir(repo)
    try:
        shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(args.model)
        mesh = shape_pipeline(
            image=str(input_path),
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            octree_resolution=args.octree_resolution,
            num_chunks=args.num_chunks,
        )[0]

        shape_path = out_dir / f"{args.asset_name}_shape.glb"
        mesh.export(shape_path)

        textured_path = None
        textured_obj = None
        if not args.skip_texture:
            # Hunyuan's paint module imports bpy only for its OBJ->GLB helper.
            # We avoid that Blender dependency by saving OBJ/MTL and converting
            # the result with trimesh below.
            sys.modules.setdefault("bpy", types.ModuleType("bpy"))
            from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline
            import trimesh

            conf = Hunyuan3DPaintConfig(max_num_view=args.texture_views, resolution=args.texture_resolution)
            conf.realesrgan_ckpt_path = str(repo / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth")
            conf.multiview_cfg_path = str(repo / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml")
            conf.custom_pipeline = str(repo / "hy3dpaint" / "hunyuanpaintpbr")
            paint_pipeline = Hunyuan3DPaintPipeline(conf)

            textured_obj = out_dir / f"{args.asset_name}_textured.obj"
            paint_pipeline(
                mesh_path=str(shape_path),
                image_path=str(input_path),
                output_mesh_path=str(textured_obj),
                save_glb=False,
            )
            textured_path = textured_obj.with_suffix(".glb")
            scene = trimesh.load(str(textured_obj), force="scene", process=False)
            scene.export(str(textured_path))
    finally:
        os.chdir(old_cwd)

    manifest = {
        "model": "Hunyuan3D 2.1",
        "upstream": "Tencent-Hunyuan/Hunyuan3D-2.1",
        "input": str(input_path),
        "asset_name": args.asset_name,
        "model_id": args.model,
        "seed": args.seed,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "elapsed_sec": round(time.time() - started, 3),
        "outputs": {
            "shape_glb": str(shape_path),
            "textured_obj": str(textured_obj) if textured_obj is not None else None,
            "textured_glb": str(textured_path) if textured_path is not None else None,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
