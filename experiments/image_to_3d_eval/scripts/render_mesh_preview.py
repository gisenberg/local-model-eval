#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import trimesh


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a simple orthographic PNG preview for a GLB/mesh.")
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=180_000)
    parser.add_argument("--size", type=int, default=768)
    return parser.parse_args()


def rotation_matrix(yaw_deg: float, pitch_deg: float = 0.0) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    ry = np.array([
        [math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)],
    ])
    rx = np.array([
        [1, 0, 0],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch), math.cos(pitch)],
    ])
    return rx @ ry


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise SystemExit(f"No mesh geometry found in {path}")
        return trimesh.util.concatenate(meshes)
    return loaded


def face_base_colors(mesh: trimesh.Trimesh, face_index: np.ndarray) -> np.ndarray:
    faces = mesh.faces[face_index]
    colors = None

    material = getattr(mesh.visual, "material", None)
    texture = getattr(material, "baseColorTexture", None)
    uv = getattr(mesh.visual, "uv", None)
    if texture is not None and uv is not None:
        image = np.asarray(texture.convert("RGB"))
        h, w = image.shape[:2]
        face_uv = uv[faces].mean(axis=1)
        x = np.clip((face_uv[:, 0] * (w - 1)).astype(int), 0, w - 1)
        y = np.clip(((1.0 - face_uv[:, 1]) * (h - 1)).astype(int), 0, h - 1)
        colors = image[y, x]

    if colors is None:
        factor = getattr(material, "baseColorFactor", None)
        if factor is not None:
            factor = np.asarray(factor[:3], dtype=np.float32)
            if factor.max(initial=1.0) <= 1.0:
                factor = factor * 255.0
            colors = np.tile(factor, (len(face_index), 1))

    if colors is None:
        colors = np.full((len(face_index), 3), 210, dtype=np.uint8)

    return np.asarray(colors[:, :3], dtype=np.float32)


def render_view(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    base_colors: np.ndarray,
    yaw: float,
    title: str,
    size: int,
) -> Image.Image:
    rot = rotation_matrix(yaw, -12)
    pts = vertices @ rot.T
    nrm = normals @ rot.T

    xy = pts[:, :2].copy()
    z = pts[:, 2]
    xy -= xy.mean(axis=0)
    extent = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
    scale = (size * 0.78) / max(extent, 1e-6)
    pix = xy * scale + size / 2
    pix[:, 1] = size - pix[:, 1]

    light = np.array([0.35, -0.45, 0.82])
    light = light / np.linalg.norm(light)
    shade = np.clip((nrm @ light) * 0.65 + 0.45, 0.15, 1.0)
    face_z = z[faces].mean(axis=1)
    depth = (face_z - face_z.min()) / max(np.ptp(face_z), 1e-6)
    shade = np.clip(shade * (0.75 + 0.25 * depth), 0.08, 1.0)

    img = Image.new("RGB", (size, size), (246, 247, 249))
    draw = ImageDraw.Draw(img)
    order = np.argsort(face_z)
    tri_pix = pix[faces]

    for idx in order:
        poly = tri_pix[idx]
        if (
            poly[:, 0].max() < 0
            or poly[:, 0].min() >= size
            or poly[:, 1].max() < 0
            or poly[:, 1].min() >= size
        ):
            continue
        rgb = np.clip(base_colors[idx] * (0.35 + 0.75 * shade[idx]), 0, 255).astype(np.uint8)
        draw.polygon([tuple(p) for p in poly], fill=tuple(int(c) for c in rgb))

    draw.rectangle((0, 0, size - 1, size - 1), outline=(210, 216, 224), width=2)
    draw.text((18, 16), title, fill=(34, 38, 45))
    return img


def preview_faces(mesh: trimesh.Trimesh, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    if len(mesh.faces) > count:
        weights = np.asarray(mesh.area_faces, dtype=np.float64)
        weights = weights / weights.sum()
        face_index = np.sort(rng.choice(len(mesh.faces), size=count, replace=False, p=weights))
    else:
        face_index = np.arange(len(mesh.faces))
    faces = mesh.faces[face_index]
    normals = mesh.face_normals[face_index]
    colors = face_base_colors(mesh, face_index)
    return np.asarray(mesh.vertices), faces, normals, colors


def main() -> None:
    args = parse_args()
    mesh_path = args.mesh.resolve()
    out = args.out or mesh_path.with_suffix(".preview.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    mesh = load_mesh(mesh_path)
    vertices, faces, normals, colors = preview_faces(mesh, args.samples)

    views = [
        render_view(vertices, faces, normals, colors, 0, "front", args.size),
        render_view(vertices, faces, normals, colors, 60, "three-quarter", args.size),
        render_view(vertices, faces, normals, colors, 90, "side", args.size),
    ]
    gutter = 18
    sheet = Image.new("RGB", (args.size * 3 + gutter * 2, args.size), (246, 247, 249))
    for i, view in enumerate(views):
        sheet.paste(view, (i * (args.size + gutter), 0))
    sheet.save(out)
    print(out)


if __name__ == "__main__":
    main()
