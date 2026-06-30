#!/usr/bin/env python3
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"


ASSETS = [
    {
        "name": "Robot",
        "input": ROOT / "input" / "robot_figurine_cutout.png",
        "prompt": ROOT / "input" / "prompt.txt",
        "trellis": OUTPUT / "trellis2" / "robot_trellis2_preview.png",
        "hunyuan": OUTPUT / "hunyuan3d21" / "robot_textured_preview.png",
    },
    {
        "name": "Sword",
        "input": ROOT / "input" / "sword_cutout.png",
        "prompt": ROOT / "input" / "sword_prompt.txt",
        "trellis": OUTPUT / "trellis2_sword" / "sword_trellis2_preview.png",
        "hunyuan": OUTPUT / "hunyuan3d21_sword" / "sword_textured_preview.png",
    },
    {
        "name": "Potion",
        "input": ROOT / "input" / "potion_cutout.png",
        "prompt": ROOT / "input" / "potion_prompt.txt",
        "trellis": OUTPUT / "trellis2_potion" / "potion_trellis2_preview.png",
        "hunyuan": OUTPUT / "hunyuan3d21_potion" / "potion_textured_preview.png",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a contact sheet of textured image-to-3D outputs.")
    parser.add_argument("--out", type=Path, default=OUTPUT / "mesh_preview_contact_sheet.png")
    parser.add_argument("--cell-width", type=int, default=520)
    parser.add_argument("--image-height", type=int, default=380)
    return parser.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def cover_image(path: Path, width: int, height: int) -> Image.Image:
    box = Image.new("RGB", (width, height), (248, 249, 251))
    if not path.exists():
        draw = ImageDraw.Draw(box)
        draw.rectangle((0, 0, width - 1, height - 1), outline=(210, 216, 224), width=2)
        draw.text((18, 18), f"Missing:\n{path.name}", fill=(96, 104, 116), font=font(18))
        return box

    image = Image.open(path).convert("RGBA")
    image.thumbnail((width - 28, height - 28), Image.Resampling.LANCZOS)
    checker = Image.new("RGB", image.size, (255, 255, 255))
    if image.mode == "RGBA":
        checker.paste(image, mask=image.getchannel("A"))
        image = checker
    else:
        image = image.convert("RGB")
    box.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    draw = ImageDraw.Draw(box)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(210, 216, 224), width=2)
    return box


def prompt_text(path: Path) -> str:
    if not path.exists():
        return f"Prompt file missing: {path}"
    lines = []
    for line in path.read_text().splitlines():
        if line.startswith(("Primary request:", "Subject:", "Materials/textures:", "Constraints:")):
            lines.append(line)
    return "\n".join(lines) if lines else path.read_text().strip()


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, wrap_chars: int, line_height: int) -> None:
    x, y = xy
    for paragraph in text.splitlines():
        wrapped = textwrap.wrap(paragraph, width=wrap_chars) or [""]
        for line in wrapped:
            draw.text((x, y), line, fill=(46, 52, 60), font=font(18))
            y += line_height
        y += 5


def main() -> None:
    args = parse_args()
    headers = ["Input", "Input Prompt", "TRELLIS.2 Textured", "Hunyuan3D 2.1 Textured"]
    cell_w = args.cell_width
    image_h = args.image_height
    header_h = 86
    row_h = image_h + 124
    margin = 24
    gap = 16
    width = margin * 2 + cell_w * len(headers) + gap * (len(headers) - 1)
    height = margin * 2 + header_h + row_h * len(ASSETS)

    sheet = Image.new("RGB", (width, height), (244, 246, 249))
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 18), "Image-to-3D textured output contact sheet", fill=(18, 22, 28), font=font(26, bold=True))

    x_positions = [margin + i * (cell_w + gap) for i in range(len(headers))]
    for x, header in zip(x_positions, headers):
        draw.text((x, margin + header_h - 24), header, fill=(18, 22, 28), font=font(19, bold=True))

    y = margin + header_h
    for asset in ASSETS:
        draw.rectangle((margin - 10, y - 10, width - margin + 10, y + row_h - 18), fill=(255, 255, 255), outline=(216, 222, 231))
        draw.text((margin, y), asset["name"], fill=(18, 22, 28), font=font(22, bold=True))
        image_top = y + 38
        sheet.paste(cover_image(asset["input"], cell_w, image_h), (x_positions[0], image_top))
        draw_wrapped(draw, (x_positions[1], image_top + 4), prompt_text(asset["prompt"]), 50, 24)
        sheet.paste(cover_image(asset["trellis"], cell_w, image_h), (x_positions[2], image_top))
        sheet.paste(cover_image(asset["hunyuan"], cell_w, image_h), (x_positions[3], image_top))
        y += row_h

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
