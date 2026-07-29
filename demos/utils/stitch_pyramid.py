#!/usr/bin/env python3
"""
Stitch a saved pyramid_results directory into a clean pyramid figure.

Levels are laid left→right (or top→bottom with ``--vertical``), sized from the
run's manifest. Each block gets a "z=N" label; ground-truth seed tiles get a red
box (--no_seed_boxes disables).

Usage
-----
    python stitch_pyramid.py <result_dir>
    python stitch_pyramid.py <result_dir> --out /path/to/output.png
    python stitch_pyramid.py <result_dir> --cell_px 96
    python stitch_pyramid.py <result_dir> --vertical --cell_px 48
    python stitch_pyramid.py <result_dir> --no_seed_boxes

Default output: <result_dir>/pyramid_viz.png
Default result location: demos/logs/pyramid_results/<DATETIME>/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# Deep-pyramid figures are 100+ Mpx, which trips Pillow's DecompressionBomb guard.
# These are our own outputs, so lift the cap for any importer of this module.
Image.MAX_IMAGE_PIXELS = None

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PYRAMID_Z_MIN = 14
PYRAMID_Z_MAX = 17

_COL_GAP    = 24    # px gap between zoom-level columns
_PADDING    = 16    # canvas outer padding
_LABEL_H    = 20    # height reserved below each grid for the "z=XX" label

_BG         = (255, 255, 255)           # canvas background — pure white
_CELL_EMPTY = (210, 212, 218)           # flat fill for missing tiles
_LABEL_COL  = ( 60,  62,  70)          # label text colour
_SEED_COL   = (228,  30,  38)           # red outline marking ground-truth seeds


def grid_n(z: int, z_min: int = PYRAMID_Z_MIN) -> int:
    """Tiles per axis at zoom z, relative to the pyramid's lower bound.

    The pyramid is a self-similar 4-level quadtree: its root (z == z_min) is
    always 1×1 regardless of the absolute zoom. ``z_min`` is read from the run's
    manifest (it can slide across the model-trained range), so the grid size must
    be computed relative to it — not a hardcoded base.
    """
    return 1 << (z - z_min)


def _grid_px(n: int, cell_px: int) -> int:
    """Pixel size of an n×n seamless grid — no gaps, no borders."""
    return n * cell_px


def _find_font_path() -> Optional[str]:
    """Locate a scalable TTF on this machine.

    The previous hardcoded list missed common layouts (e.g. this host only ships
    DejaVuSansMono under /usr/share/fonts/dejavu and the proportional DejaVuSans
    only via matplotlib), so font lookup silently fell back to PIL's fixed-size
    bitmap default — which ignores ``size`` and made the z-labels permanently
    tiny. Try explicit paths, then matplotlib's bundled font, then a recursive
    glob, preferring a proportional (non-Mono) sans.
    """
    explicit = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in explicit:
        if os.path.isfile(path):
            return path
    try:                                  # matplotlib always bundles DejaVuSans
        import matplotlib                 # noqa: WPS433
        mpl_font = os.path.join(
            os.path.dirname(matplotlib.__file__),
            "mpl-data", "fonts", "ttf", "DejaVuSans.ttf",
        )
        if os.path.isfile(mpl_font):
            return mpl_font
    except Exception:
        pass
    import glob
    hits = glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
    # Prefer a proportional sans over monospace / specialty scripts.
    hits.sort(key=lambda p: ("Mono" in p, "Sans" not in os.path.basename(p)))
    return hits[0] if hits else None


def _try_font(size: int) -> ImageFont.ImageFont:
    path = _find_font_path()
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ─────────────────────────────────────────────────────────────────────────────
# Core stitching
# ─────────────────────────────────────────────────────────────────────────────

def stitch(
    result_dir: str,
    out_path: Optional[str] = None,
    cell_px: int = 64,
    mark_seeds: bool = True,
    dpi: int = 300,
    label_px: Optional[int] = None,
    vertical: bool = False,
    compress_level: int = 6,
    max_mb: Optional[float] = None,
) -> str:
    manifest_path = os.path.join(result_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"manifest.json not found in {result_dir!r}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    z_min     = manifest.get("pyramid_z_min", PYRAMID_Z_MIN)
    z_max     = manifest.get("pyramid_z_max", PYRAMID_Z_MAX)
    tile_info: Dict[str, dict] = manifest.get("tiles", {})

    levels    = list(range(z_min, z_max + 1))

    # Grid pixel dimensions per level (relative to this run's z_min)
    grid_sizes = {z: _grid_px(grid_n(z, z_min), cell_px) for z in levels}

    # z-label size. Vertical: much larger (≈ cell_px / 1.5) so "z=N" is legible in
    # the giant deep figure; the label sits centred below each block (same as
    # horizontal, just bigger). Horizontal: ≈ cell_px / 3.4. Override via ``label_px``.
    if vertical:
        font_size = label_px if label_px else max(28, round(cell_px / 1.5))
    else:
        font_size = label_px if label_px else max(9, round(cell_px / 3.4))
    label_band = font_size + 8           # room reserved below each grid for "z=XX"
    font_lbl   = _try_font(font_size)

    # Top-left origin of each level block for the chosen orientation.
    #   vertical:   blocks stacked top(root)→bottom(leaf), each horizontally
    #               centred → the classic apex→base pyramid silhouette.
    #   horizontal: blocks left→right, each vertically centred (default).
    # Either way the "z=N" label sits centred just below its block.
    placements: Dict[int, Tuple[int, int]] = {}
    if vertical:
        max_grid_w = grid_sizes[z_max]            # widest row = finest zoom level
        total_w = 2 * _PADDING + max_grid_w
        y = _PADDING
        for z in levels:
            gpx = grid_sizes[z]
            placements[z] = (_PADDING + (max_grid_w - gpx) // 2, y)
            y += gpx + label_band + _COL_GAP      # band below each block for its label
        total_h = y - _COL_GAP + _PADDING         # drop the trailing inter-row gap
    else:
        max_grid_h = grid_sizes[z_max]            # tallest column = finest zoom level
        total_h = 2 * _PADDING + max_grid_h + label_band
        x = _PADDING
        for z in levels:
            gpx = grid_sizes[z]
            placements[z] = (x, _PADDING + (max_grid_h - gpx) // 2)
            x += gpx + _COL_GAP
        total_w = x - _COL_GAP + _PADDING         # drop the trailing inter-col gap

    canvas = Image.new("RGB", (total_w, total_h), _BG)
    draw   = ImageDraw.Draw(canvas)

    for z in levels:
        n     = grid_n(z, z_min)
        gpx   = grid_sizes[z]
        col_x, grid_top = placements[z]

        # Fill the whole block first so missing tiles have a flat background
        draw.rectangle(
            [col_x, grid_top, col_x + gpx - 1, grid_top + gpx - 1],
            fill=_CELL_EMPTY,
        )

        # Paste each tile directly — zero gap, no borders. Collect the cells
        # that were ground-truth seeds so we can outline them on top afterwards
        # (drawn last so the box isn't covered by an adjacent tile's paste).
        seed_cells: list = []
        for gy in range(n):
            for gx in range(n):
                key  = f"{z},{gx},{gy}"
                info = tile_info.get(key, {})

                cx = col_x + gx * cell_px
                cy = grid_top + gy * cell_px

                tile_file = info.get("file")
                if tile_file:
                    tile_path = os.path.join(result_dir, tile_file)
                    if os.path.isfile(tile_path):
                        try:
                            img = Image.open(tile_path).convert("RGB")
                            if img.size != (cell_px, cell_px):
                                img = img.resize((cell_px, cell_px),
                                                 Image.Resampling.LANCZOS)
                            canvas.paste(img, (cx, cy))
                        except Exception:
                            pass

                if mark_seeds and info.get("state") == "seed":
                    seed_cells.append((cx, cy))

        # Red box around each ground-truth seed tile, drawn LAST (on top of all
        # tile pastes) so nothing covers it. Stitch draws no other outlines, so the
        # only box on the figure is this red seed marker. The big vertical figure
        # needs a thick box (a seed is one of ~1024 leaf tiles); the small
        # horizontal figure has far fewer tiles, so a thinner box reads better.
        if seed_cells:
            # vertical (6-level): thick (cell_px//8). horizontal (4-level): the
            # original default (cell_px//24).
            bw = max(2, cell_px // (8 if vertical else 24))
            for cx, cy in seed_cells:
                draw.rectangle(
                    [cx, cy, cx + cell_px - 1, cy + cell_px - 1],
                    outline=_SEED_COL, width=bw,
                )

        # "z=XX" label centred just below the block (both orientations).
        label = f"z={z}"
        try:
            bbox = draw.textbbox((0, 0), label, font=font_lbl)
            lw   = bbox[2] - bbox[0]
        except AttributeError:
            lw = len(label) * (cell_px // 6)
        lx = col_x + gpx // 2 - lw // 2
        ly = grid_top + gpx + 4
        draw.text((lx, ly), label, fill=_LABEL_COL, font=font_lbl)

    # Save
    if out_path is None:
        out_path = os.path.join(result_dir, "pyramid_viz.png")

    # Size-capped path: a deep-pyramid figure is 100+ Mpx, so a lossless PNG is
    # ~125 MB. When ``max_mb`` is set, write a JPEG instead and tune it to fit the
    # budget — keep full resolution and binary-search quality; only downscale if
    # even low quality overflows. Returns the actual (``.jpg``) path written.
    if max_mb is not None:
        return _save_under_size(canvas, out_path, float(max_mb), dpi)

    # Embed DPI so the PDF/LaTeX places the figure at a sensible physical size
    # (e.g. at 300 dpi a 3944px-wide canvas is ~13 in / ~33 cm wide, scale down
    # in the document as needed). PNG is lossless, so this carries full quality.
    # ``compress_level`` (0-9) trades encode time for file size — for the giant
    # deep-pyramid figure (100+ megapixels) a low level (e.g. 1) is far faster to
    # write and still lossless; the default 6 stays for normal-size figures.
    canvas.save(out_path, dpi=(dpi, dpi), compress_level=int(compress_level))
    return out_path


def _save_under_size(img: Image.Image, out_path: str, max_mb: float, dpi: int) -> str:
    """Write ``img`` as a ``.jpg`` no larger than ``max_mb``; return its path.

    Binary-searches JPEG quality at full resolution, stepping down a resolution
    ladder only if even low quality overflows the budget.
    """
    import io
    budget = int(max_mb * 1024 * 1024)
    jpg_path = os.path.splitext(out_path)[0] + ".jpg"

    def _encode(im: Image.Image, q: int) -> bytes:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=q, dpi=(dpi, dpi),
                optimize=True, progressive=True)
        return buf.getvalue()

    best: Optional[bytes] = None
    for scale in (1.0, 0.8, 0.65, 0.5, 0.35, 0.25):
        im = (img if scale == 1.0
              else img.resize((max(1, round(img.width * scale)),
                               max(1, round(img.height * scale))),
                              Image.Resampling.LANCZOS))
        lo, hi, fit = 30, 95, None
        while lo <= hi:                       # largest quality that fits this scale
            q = (lo + hi) // 2
            data = _encode(im, q)
            if len(data) <= budget:
                fit = data; lo = q + 1
            else:
                hi = q - 1
        if fit is not None:                   # found a quality that fits at this scale
            best = fit
            break
        best = _encode(im, 30)                # remember smallest-at-this-scale, step down
    with open(jpg_path, "wb") as f:
        f.write(best)                         # last `best` is the smallest we produced
    return jpg_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stitch Genesis pyramid tiles into a clean pyramid figure PNG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("result_dir",
                        help="Path to a logs/pyramid_results/<DATETIME>/ directory")
    parser.add_argument("--out",     default=None,
                        help="Output PNG path (default: <result_dir>/pyramid_viz.png)")
    parser.add_argument("--cell_px", type=int, default=256,
                        help="Square cell size in pixels (default: 256 = native "
                             "tile resolution, no downscaling = max quality). "
                             "Use 64/96 for smaller preview figures.")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI metadata embedded in the PNG (default: 300, "
                             "print quality). Controls physical size in the PDF, "
                             "not pixel count.")
    parser.add_argument("--no_seed_boxes", action="store_true",
                        help="Disable the red box drawn around ground-truth seed tiles.")
    parser.add_argument("--vertical", action="store_true",
                        help="Stack zoom levels top→bottom (apex→base) instead of "
                             "left→right. Best for deep pyramids (5+ levels).")
    parser.add_argument("--compress_level", type=int, default=6,
                        help="PNG zlib compression 0-9 (default 6). Use 1 for a much "
                             "faster write on giant deep-pyramid figures (still lossless).")
    parser.add_argument("--max_mb", type=float, default=None,
                        help="Cap the output at this many MB by writing a quality-tuned "
                             "JPEG (full resolution; downscales only as a last resort). "
                             "Best for giant deep-pyramid figures. Output uses .jpg.")
    args = parser.parse_args()

    result_dir = os.path.abspath(args.result_dir)
    if not os.path.isdir(result_dir):
        print(f"Error: {result_dir!r} is not a directory.", file=sys.stderr)
        sys.exit(1)

    try:
        out = stitch(result_dir, out_path=args.out, cell_px=args.cell_px,
                     mark_seeds=not args.no_seed_boxes, dpi=args.dpi,
                     vertical=args.vertical, compress_level=args.compress_level,
                     max_mb=args.max_mb)
        print(f"✓ Saved pyramid visualization → {out}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
