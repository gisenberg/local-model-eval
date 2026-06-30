#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hosted Tripo image-to-3D on an eval input.")
    parser.add_argument("--input", type=Path, default=ROOT / "input" / "robot_figurine_cutout.png")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "output" / "tripo3d_robot")
    parser.add_argument("--asset-name", default="robot")
    parser.add_argument("--model-version", default="v3.0-20250812")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--face-limit", type=int, default=None)
    parser.add_argument("--texture-quality", default="standard", choices=["standard", "detailed"])
    parser.add_argument("--geometry-quality", default="standard", choices=["standard", "detailed"])
    parser.add_argument("--texture-alignment", default="original_image", choices=["original_image", "geometry"])
    parser.add_argument("--orientation", default="default", choices=["default", "align_image"])
    parser.add_argument("--smart-low-poly", action="store_true")
    parser.add_argument("--quad", action="store_true")
    parser.add_argument("--timeout", type=float, default=900)
    return parser.parse_args()


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip().replace("export ", "")
        value = value.strip().strip("\"'")
        os.environ.setdefault(key, value)


def api_key() -> str:
    key = os.environ.get("TRIPO_API_KEY") or os.environ.get("TRIPO_API_TOKEN")
    if not key:
        raise SystemExit("Set TRIPO_API_KEY or TRIPO_API_TOKEN in the environment or .env.")
    return key


async def main_async() -> None:
    args = parse_args()
    load_env(ROOT.parents[1] / ".env")
    input_path = args.input.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise SystemExit(f"Input image not found: {input_path}")

    from tripo3d import TripoClient

    started = time.time()
    downloaded = {}
    task = None
    task_id = None
    glb_path = None

    async with TripoClient(api_key=api_key()) as client:
        balance = await client.get_balance()
        try:
            task_id = await client.image_to_model(
                image=str(input_path),
                model_version=args.model_version,
                texture=True,
                pbr=True,
                model_seed=args.seed,
                texture_seed=args.seed,
                face_limit=args.face_limit,
                texture_quality=args.texture_quality,
                geometry_quality=args.geometry_quality,
                texture_alignment=args.texture_alignment,
                orientation=args.orientation,
                quad=args.quad,
                smart_low_poly=args.smart_low_poly,
            )
        except Exception as exc:
            manifest = {
                "model": "Tripo 3.0",
                "upstream": "Tripo API",
                "input": str(input_path),
                "asset_name": args.asset_name,
                "model_version": args.model_version,
                "seed": args.seed,
                "task_id": None,
                "status": "submit_failed",
                "error": str(exc),
                "balance_at_submit": getattr(balance, "balance", None),
                "frozen_at_submit": getattr(balance, "frozen", None),
                "elapsed_sec": round(time.time() - started, 3),
                "outputs": {"glb": None},
            }
            (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(json.dumps(manifest, indent=2))
            raise SystemExit(f"Tripo task submission failed: {exc}") from exc
        task = await client.wait_for_task(task_id, timeout=args.timeout, verbose=True)
        if str(task.status) != "success" and getattr(task.status, "value", None) != "success":
            manifest = {
                "model": "Tripo 3.0",
                "upstream": "Tripo API",
                "input": str(input_path),
                "asset_name": args.asset_name,
                "model_version": args.model_version,
                "seed": args.seed,
                "task_id": task_id,
                "status": getattr(task.status, "value", str(task.status)),
                "error_code": getattr(task, "error_code", None),
                "error_msg": getattr(task, "error_msg", None),
                "balance_at_submit": getattr(balance, "balance", None),
                "frozen_at_submit": getattr(balance, "frozen", None),
                "elapsed_sec": round(time.time() - started, 3),
                "outputs": {"glb": None},
            }
            (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(json.dumps(manifest, indent=2))
            raise SystemExit(f"Tripo task did not succeed: {manifest['status']} {manifest['error_msg']}")

        downloaded = await client.download_task_models(task, str(out_dir))

    model_file = downloaded.get("pbr_model") or downloaded.get("model") or downloaded.get("base_model")
    if model_file:
        src = Path(model_file)
        glb_path = out_dir / f"{args.asset_name}_tripo3d.glb"
        if src.resolve() != glb_path.resolve():
            shutil.copy2(src, glb_path)

    manifest = {
        "model": "Tripo 3.0",
        "upstream": "Tripo API",
        "input": str(input_path),
        "asset_name": args.asset_name,
        "model_version": args.model_version,
        "seed": args.seed,
        "task_id": task_id,
        "status": getattr(task.status, "value", str(task.status)) if task else None,
        "face_limit": args.face_limit,
        "texture_quality": args.texture_quality,
        "geometry_quality": args.geometry_quality,
        "texture_alignment": args.texture_alignment,
        "orientation": args.orientation,
        "smart_low_poly": args.smart_low_poly,
        "quad": args.quad,
        "elapsed_sec": round(time.time() - started, 3),
        "downloaded": downloaded,
        "outputs": {
            "glb": str(glb_path) if glb_path else None,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
