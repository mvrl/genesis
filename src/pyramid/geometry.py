"""Subtree-relative tile geometry: level grid size, tile position, level stitching.

Pure layout helpers (no metrics / scoring). Used by the generation strategies to
place and stitch tiles within a subtree level.
"""
from __future__ import annotations

from typing import Dict, Tuple

from PIL import Image

from .dataset import Subtree
from .models import TILE_PX
from .tiles import Tile


def level_grid_n(sub: Subtree, z: int) -> int:
    """Tiles per axis at zoom ``z`` within the subtree (1, 2, 4, …)."""
    return 1 << (z - sub.root[0])


def tile_pos(sub: Subtree, t: Tile) -> Tuple[int, int]:
    """(col, row) of tile ``t`` within its zoom level, relative to the subtree root."""
    z, x, y = t
    n = level_grid_n(sub, z)
    return (x - sub.root[1] * n, y - sub.root[2] * n)


def stitch_level(images: Dict[Tile, Image.Image], sub: Subtree, z: int,
                 tile_px: int = TILE_PX) -> Image.Image:
    """Stitch the resolved tiles at zoom ``z`` into one ``(n·tile_px)²`` mosaic."""
    n = level_grid_n(sub, z)
    canvas = Image.new("RGB", (n * tile_px, n * tile_px), (40, 40, 40))
    for t in sub.at_zoom(z):
        im = images.get(t)
        if im is None:
            continue
        col, row = tile_pos(sub, t)
        canvas.paste(im.convert("RGB").resize((tile_px, tile_px)), (col * tile_px, row * tile_px))
    return canvas
