"""Generate multi-platform brand assets from `docs/assets/source/banner_master.png`.

Run from the repository root:
    uv run --no-project --with pillow --with numpy python docs/assets/generate_assets.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ASSETS_DIR = Path(__file__).resolve().parent
DEFAULT_MASTER = ASSETS_DIR / "source" / "banner_master.png"
DB_RIGHT_BOUND_X = 485


def make_textured_bg(width: int, height: int, src_strip: np.ndarray) -> Image.Image:
    """Create a seamless mirror-tiled periwinkle paper-textured background."""
    # Build a 2x2 mirror tile so both horizontal and vertical seams match continuously.
    row_pair = np.concatenate([src_strip, src_strip[:, ::-1]], axis=1)
    tile = np.concatenate([row_pair, row_pair[::-1, :]], axis=0)
    tile_h, tile_w = tile.shape[:2]
    rep_y = (height + tile_h - 1) // tile_h
    rep_x = (width + tile_w - 1) // tile_w
    tiled = np.tile(tile, (rep_y, rep_x, 1))[:height, :width]
    return Image.fromarray(tiled.astype(np.uint8), mode="RGB").convert("RGBA")


def extract_transparent(rgb_arr: np.ndarray, bg_rgb: np.ndarray) -> Image.Image:
    """Extract foreground artwork with halo-free alpha and color decontamination."""
    rgb = rgb_arr.astype(np.float32)
    excess = rgb[:, :, 2] - 0.5 * (rgb[:, :, 0] + rgb[:, :, 1])
    bg_excess = float(bg_rgb[2] - 0.5 * (bg_rgb[0] + bg_rgb[1]))

    lo = 12.0
    hi = max(lo + 1.0, bg_excess - 12.0)
    alpha = np.clip((hi - excess) / (hi - lo), 0.0, 1.0)

    out_rgb = rgb.copy()
    semi = (alpha > 0.02) & (alpha < 0.99)
    inv_alpha = alpha[semi][:, None]
    out_rgb[semi] = np.clip(
        (rgb[semi] - (1.0 - inv_alpha) * bg_rgb[None, :]) / inv_alpha,
        0.0,
        255.0,
    )
    out_rgb[alpha <= 0.02] = 0.0
    alpha[alpha <= 0.02] = 0.0

    rgba = np.dstack([out_rgb, alpha * 255.0])
    return Image.fromarray(np.clip(rgba, 0, 255).astype(np.uint8), mode="RGBA")


def tight_crop_rgba(img: Image.Image, alpha_threshold: int = 10) -> Image.Image:
    """Crop an RGBA image tightly around pixels with alpha above `alpha_threshold`."""
    alpha = np.asarray(img)[:, :, 3]
    ys, xs = np.where(alpha > alpha_threshold)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("No foreground pixels found above alpha threshold.")
    return img.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))


def fit_and_paste(
    canvas: Image.Image,
    sprite: Image.Image,
    *,
    max_width: int,
    max_height: int,
    center_x: int | None = None,
    center_y: int | None = None,
) -> Image.Image:
    """Scale `sprite` proportionally within `(max_width, max_height)` and composite it."""
    scale = min(max_width / sprite.width, max_height / sprite.height)
    new_w = max(1, round(sprite.width * scale))
    new_h = max(1, round(sprite.height * scale))
    resized = sprite.resize((new_w, new_h), Image.Resampling.LANCZOS)

    cx = canvas.width // 2 if center_x is None else center_x
    cy = canvas.height // 2 if center_y is None else center_y
    paste_x = cx - new_w // 2
    paste_y = cy - new_h // 2

    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    layer.paste(resized, (paste_x, paste_y), resized)
    return Image.alpha_composite(canvas, layer)


def generate_all(master_path: Path, out_dir: Path) -> list[tuple[str, tuple[int, int], int]]:
    """Generate all multi-platform assets and return `(filename, size, bytes)` entries."""
    out_dir.mkdir(parents=True, exist_ok=True)
    master = Image.open(master_path).convert("RGB")
    master_arr = np.asarray(master)

    bg_rgb = np.median(master_arr[:140, :, :], axis=(0, 1)).astype(np.float32)
    clean_strip = master_arr[16:176, 16:1360].copy()

    full_rgba = tight_crop_rgba(extract_transparent(master_arr, bg_rgb))
    db_rgba = tight_crop_rgba(extract_transparent(master_arr[:, :DB_RIGHT_BOUND_X], bg_rgb))

    written: list[tuple[str, tuple[int, int], int]] = []

    def save_png(filename: str, img: Image.Image) -> Image.Image:
        dest = out_dir / filename
        img.save(dest, format="PNG", optimize=True)
        written.append((filename, img.size, dest.stat().st_size))
        return img

    # 1. GitHub README header banner (1280x420, solid periwinkle and transparent)
    gh_crop = master.crop((0, 160, master.width, 611))
    save_png("banner-github.png", gh_crop.resize((1280, 420), Image.Resampling.LANCZOS))

    gh_trans = fit_and_paste(
        Image.new("RGBA", (1280, 420), (0, 0, 0, 0)),
        full_rgba,
        max_width=1160,
        max_height=348,
    )
    save_png("banner-github-transparent.png", gh_trans)

    # 2. GitHub Repository Social Preview (1280x640, 2:1)
    social_crop = master.crop((0, 40, master.width, 728))
    save_png("social-preview-github.png", social_crop.resize((1280, 640), Image.Resampling.LANCZOS))

    # 3. LinkedIn Personal Cover (1584x396, shifted right to clear bottom-left avatar)
    li_personal_bg = make_textured_bg(1584, 396, clean_strip)
    li_personal = fit_and_paste(
        li_personal_bg,
        full_rgba,
        max_width=1020,
        max_height=276,
        center_x=960,
        center_y=198,
    )
    save_png("banner-linkedin-personal.png", li_personal.convert("RGB"))

    # 4. LinkedIn Company / Page Cover (1128x191)
    li_company_bg = make_textured_bg(1128, 191, clean_strip)
    li_company = fit_and_paste(
        li_company_bg,
        full_rgba,
        max_width=960,
        max_height=140,
        center_x=564,
        center_y=95,
    )
    save_png("banner-linkedin-company.png", li_company.convert("RGB"))

    # 5. Square Brand Icons (512x512 solid & transparent, 256x256 MkDocs logo)
    icon_512_bg = make_textured_bg(512, 512, clean_strip)
    icon_512 = fit_and_paste(icon_512_bg, db_rgba, max_width=410, max_height=410)
    save_png("icon-512.png", icon_512.convert("RGB"))

    icon_512_trans = fit_and_paste(
        Image.new("RGBA", (512, 512), (0, 0, 0, 0)),
        db_rgba,
        max_width=410,
        max_height=410,
    )
    save_png("icon-512-transparent.png", icon_512_trans)

    icon_256_trans = icon_512_trans.resize((256, 256), Image.Resampling.LANCZOS)
    save_png("icon-256-transparent.png", icon_256_trans)

    # 6. Emoji & Favicons (tight padding for crisp small-scale rendering)
    emoji_128 = fit_and_paste(
        Image.new("RGBA", (128, 128), (0, 0, 0, 0)),
        db_rgba,
        max_width=118,
        max_height=118,
    )
    save_png("emoji-128.png", emoji_128)

    mkdocs_fav = fit_and_paste(
        Image.new("RGBA", (64, 64), (0, 0, 0, 0)),
        db_rgba,
        max_width=58,
        max_height=58,
    )
    save_png("mkdocs-favicon-64.png", mkdocs_fav)

    ico_base = fit_and_paste(
        Image.new("RGBA", (256, 256), (0, 0, 0, 0)),
        db_rgba,
        max_width=236,
        max_height=236,
    )
    ico_path = out_dir / "favicon.ico"
    ico_sizes = [(16, 16), (32, 32), (48, 48), (64, 64)]
    ico_base.save(ico_path, format="ICO", sizes=ico_sizes)
    written.append(("favicon.ico", (64, 64), ico_path.stat().st_size))

    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--out-dir", type=Path, default=ASSETS_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    results = generate_all(args.master, args.out_dir)
    for name, (w, h), size_bytes in results:
        print(f"{name:32s} {w:4d}x{h:<4d}  {size_bytes / 1024:6.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
