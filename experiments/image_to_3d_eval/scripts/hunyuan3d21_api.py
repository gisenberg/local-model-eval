#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
import threading
import time
import traceback
import types
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = ROOT / "third_party" / "Hunyuan3D-2.1"
DEFAULT_OUT = ROOT / "output" / "api_jobs"
DEFAULT_MODEL = "tencent/Hunyuan3D-2.1"


def _now() -> float:
    return time.time()


def _jsonable_path(path: Path | None) -> str | None:
    return str(path) if path is not None else None


class JobStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def create(self, params: dict[str, Any]) -> tuple[str, Path]:
        job_id = uuid.uuid4().hex
        job_dir = self.root / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        record = {
            "job_id": job_id,
            "status": "queued",
            "created_at": _now(),
            "updated_at": _now(),
            "params": params,
            "outputs": {},
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = record
        self.write(job_id)
        return job_id, job_dir

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return dict(record) if record is not None else self._load(job_id)

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.update(fields)
            record["updated_at"] = _now()
        self.write(job_id)

    def write(self, job_id: str) -> None:
        record = self.get(job_id)
        if record is None:
            return
        job_dir = self.root / job_id
        (job_dir / "job.json").write_text(json.dumps(record, indent=2) + "\n")

    def _load(self, job_id: str) -> dict[str, Any] | None:
        path = self.root / job_id / "job.json"
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text())
        except Exception:
            return None
        self._jobs[job_id] = record
        return dict(record)


class Hunyuan3DWorker:
    def __init__(self, repo: Path, model: str):
        self.repo = repo.resolve()
        self.model = model
        self._loaded = False
        self._load_lock = threading.Lock()
        self._shape_pipeline = None
        self._paint_pipeline = None
        self._torch = None
        self._np = None

    def preload(self) -> None:
        self._ensure_shape()
        self._ensure_paint(texture_views=6, texture_resolution=512)

    def _prepare_imports(self) -> None:
        if not self.repo.exists():
            raise RuntimeError(f"Hunyuan3D checkout not found: {self.repo}")
        os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
        for path in [self.repo, self.repo / "hy3dshape", self.repo / "hy3dpaint"]:
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)

    def _ensure_shape(self) -> None:
        with self._load_lock:
            if self._shape_pipeline is not None:
                return
            self._prepare_imports()

            import numpy as np
            import torch
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

            try:
                from torchvision_fix import apply_fix
                apply_fix()
            except Exception as exc:
                print(f"Warning: torchvision compatibility fix was not applied: {exc}", flush=True)

            if not torch.cuda.is_available():
                raise RuntimeError("Hunyuan3D 2.1 requires CUDA here, but torch.cuda.is_available() is false.")

            old_cwd = Path.cwd()
            os.chdir(self.repo)
            try:
                self._shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(self.model)
            finally:
                os.chdir(old_cwd)
            self._torch = torch
            self._np = np

    def _ensure_paint(self, texture_views: int, texture_resolution: int) -> None:
        with self._load_lock:
            if self._paint_pipeline is not None:
                return
            self._prepare_imports()
            sys.modules.setdefault("bpy", types.ModuleType("bpy"))
            from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

            conf = Hunyuan3DPaintConfig(max_num_view=texture_views, resolution=texture_resolution)
            conf.realesrgan_ckpt_path = str(self.repo / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth")
            conf.multiview_cfg_path = str(self.repo / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml")
            conf.custom_pipeline = str(self.repo / "hy3dpaint" / "hunyuanpaintpbr")

            old_cwd = Path.cwd()
            os.chdir(self.repo)
            try:
                self._paint_pipeline = Hunyuan3DPaintPipeline(conf)
            finally:
                os.chdir(old_cwd)

    def generate(self, image_path: Path, out_dir: Path, params: dict[str, Any]) -> dict[str, Any]:
        self._ensure_shape()
        torch = self._torch
        np = self._np
        assert torch is not None
        assert np is not None

        seed = int(params["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        generator = torch.Generator(device="cuda").manual_seed(seed)

        asset_name = params["asset_name"]
        started = _now()
        out_dir.mkdir(parents=True, exist_ok=True)

        shape_path = out_dir / f"{asset_name}_shape.glb"
        textured_obj = None
        textured_glb = None

        old_cwd = Path.cwd()
        os.chdir(self.repo)
        try:
            mesh = self._shape_pipeline(
                image=str(image_path),
                num_inference_steps=int(params["steps"]),
                guidance_scale=float(params["guidance_scale"]),
                generator=generator,
                octree_resolution=int(params["octree_resolution"]),
                num_chunks=int(params["num_chunks"]),
            )[0]
            mesh.export(shape_path)

            if bool(params["texture"]):
                self._ensure_paint(
                    texture_views=int(params["texture_views"]),
                    texture_resolution=int(params["texture_resolution"]),
                )
                import trimesh

                textured_obj = out_dir / f"{asset_name}_textured.obj"
                self._paint_pipeline(
                    mesh_path=str(shape_path),
                    image_path=str(image_path),
                    output_mesh_path=str(textured_obj),
                    save_glb=False,
                )
                textured_glb = textured_obj.with_suffix(".glb")
                scene = trimesh.load(str(textured_obj), force="scene", process=False)
                scene.export(str(textured_glb))
        finally:
            os.chdir(old_cwd)

        return {
            "model": "Hunyuan3D 2.1",
            "upstream": "Tencent-Hunyuan/Hunyuan3D-2.1",
            "model_id": self.model,
            "input": str(image_path),
            "asset_name": asset_name,
            "elapsed_sec": round(_now() - started, 3),
            "params": params,
            "outputs": {
                "shape_glb": _jsonable_path(shape_path),
                "textured_obj": _jsonable_path(textured_obj),
                "textured_glb": _jsonable_path(textured_glb),
                "result_glb": _jsonable_path(textured_glb or shape_path),
            },
        }


def build_app(repo: Path, out_dir: Path, model: str, preload: bool) -> FastAPI:
    store = JobStore(out_dir)
    worker = Hunyuan3DWorker(repo=repo, model=model)
    executor = ThreadPoolExecutor(max_workers=1)

    app = FastAPI(
        title="Local Hunyuan3D 2.1 Asset API",
        version="0.1",
        description="Image-to-textured-GLB service for internal Tailscale use.",
    )

    if preload:
        worker.preload()

    def run_job(job_id: str, image_path: Path, job_dir: Path, params: dict[str, Any]) -> None:
        store.update(job_id, status="running", started_at=_now())
        try:
            manifest = worker.generate(image_path=image_path, out_dir=job_dir, params=params)
            (job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            store.update(
                job_id,
                status="completed",
                completed_at=_now(),
                outputs=manifest["outputs"],
                manifest=str(job_dir / "manifest.json"),
            )
        except Exception as exc:
            traceback_text = traceback.format_exc()
            (job_dir / "error.txt").write_text(traceback_text)
            store.update(job_id, status="failed", error=str(exc), traceback=str(job_dir / "error.txt"))

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": "Hunyuan3D 2.1",
            "repo": str(repo),
            "output_dir": str(out_dir),
            "queue": "single-worker",
        }

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "data": [
                {
                    "id": "hunyuan3d-2.1-textured-mesh",
                    "object": "model",
                    "capabilities": ["image-to-3d", "textured-glb"],
                }
            ]
        }

    @app.post("/v1/textured-meshes", status_code=202)
    async def create_textured_mesh(
        image: UploadFile = File(...),
        asset_name: str = Form("asset"),
        seed: int = Form(42),
        steps: int = Form(50),
        guidance_scale: float = Form(5.0),
        octree_resolution: int = Form(384),
        num_chunks: int = Form(8000),
        texture: bool = Form(True),
        texture_views: int = Form(6),
        texture_resolution: int = Form(512),
    ) -> JSONResponse:
        if texture_resolution not in (512, 768):
            raise HTTPException(status_code=400, detail="texture_resolution must be 512 or 768")
        if steps < 1 or steps > 100:
            raise HTTPException(status_code=400, detail="steps must be between 1 and 100")
        if octree_resolution < 128 or octree_resolution > 512:
            raise HTTPException(status_code=400, detail="octree_resolution must be between 128 and 512")

        params = {
            "asset_name": "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in asset_name)[:80] or "asset",
            "seed": seed,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "octree_resolution": octree_resolution,
            "num_chunks": num_chunks,
            "texture": texture,
            "texture_views": texture_views,
            "texture_resolution": texture_resolution,
        }
        job_id, job_dir = store.create(params)
        suffix = Path(image.filename or "input.png").suffix or ".png"
        image_path = job_dir / f"input{suffix}"
        image_path.write_bytes(await image.read())
        store.update(job_id, input=str(image_path))

        executor.submit(run_job, job_id, image_path, job_dir, params)
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": "queued",
                "status_url": f"/v1/textured-meshes/{job_id}",
                "result_url": f"/v1/textured-meshes/{job_id}/result.glb",
            },
        )

    @app.get("/v1/textured-meshes/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        record = store.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="job not found")
        return record

    @app.get("/v1/textured-meshes/{job_id}/manifest")
    def get_manifest(job_id: str) -> dict[str, Any]:
        record = store.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="job not found")
        manifest_path = record.get("manifest")
        if not manifest_path or not Path(manifest_path).exists():
            raise HTTPException(status_code=409, detail=f"job is {record['status']}")
        return json.loads(Path(manifest_path).read_text())

    @app.get("/v1/textured-meshes/{job_id}/result.glb")
    def get_result(job_id: str) -> FileResponse:
        record = store.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="job not found")
        if record["status"] != "completed":
            raise HTTPException(status_code=409, detail=f"job is {record['status']}")
        result = record.get("outputs", {}).get("result_glb")
        if not result or not Path(result).exists():
            raise HTTPException(status_code=500, detail="result file missing")
        return FileResponse(result, media_type="model/gltf-binary", filename=Path(result).name)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Hunyuan3D 2.1 textured mesh generation over FastAPI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--preload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    app = build_app(repo=args.repo, out_dir=args.out_dir, model=args.model, preload=args.preload)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
