"""Absolute Web-Mercator tile coordinates, the tile store, and quadtree helpers.

Everything in the full-pyramid framework keys tiles by their **absolute**
slippy-map coordinate ``(z, x, y)`` (zoom, tile_x, tile_y; y origin north).
This matches the dataset and the SR/OP model conditioning, and generalises the
demo's fixed z14→z17 / root-relative grid to any zoom range and any root.

Quadtree relations (standard slippy map):
    parent(z, x, y)        = (z-1, x // 2, y // 2)
    child quadrant (rx,ry) = (z+1, 2x+rx, 2y+ry),  rx, ry in {0, 1}

Quadrant order matches ``src/superresolution/dataloader.py`` and
``src/eval/common.QUAD_ORDER``: UL, UR, LL, LR — i.e. (rx, ry) in
(0,0), (1,0), (0,1), (1,1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image

Tile = Tuple[int, int, int]  # (z, x, y)

# (rx, ry) for UL, UR, LL, LR — the SR mosaic / dataloader order.
QUAD_ORDER: Tuple[Tuple[int, int], ...] = ((0, 0), (1, 0), (0, 1), (1, 1))


def parent_of(t: Tile) -> Tile:
    z, x, y = t
    return (z - 1, x // 2, y // 2)


def children_of(t: Tile) -> List[Tile]:
    z, x, y = t
    return [(z + 1, 2 * x + rx, 2 * y + ry) for (rx, ry) in QUAD_ORDER]


def ancestor_at(t: Tile, z: int) -> Tile:
    """Ancestor of ``t`` at zoom ``z`` (``z`` must be <= t's zoom)."""
    tz, tx, ty = t
    if z > tz:
        raise ValueError(f"ancestor zoom {z} deeper than tile zoom {tz}")
    d = tz - z
    return (z, tx >> d, ty >> d)


def quadrant_in_parent(t: Tile) -> Tuple[int, int]:
    """(rx, ry) position of ``t`` within its parent, each in {0, 1}."""
    _, x, y = t
    return (x & 1, y & 1)


def lca(tiles: Iterable[Tile]) -> Tile:
    """Least common ancestor: the deepest tile that is an ancestor of all.

    The LCA's zoom is at most ``min`` of the input zooms; we reduce every tile
    to a shared zoom and climb until all reductions coincide.
    """
    tl = list(tiles)
    if not tl:
        raise ValueError("lca of empty set")
    if len(tl) == 1:
        return tl[0]
    z = min(t[0] for t in tl)
    while True:
        ancs = {ancestor_at(t, z) for t in tl}
        if len(ancs) == 1:
            return next(iter(ancs))
        z -= 1
        if z < 0:
            raise ValueError("tiles share no common ancestor (different roots?)")


def is_descendant(t: Tile, anc: Tile) -> bool:
    """True if ``t`` is ``anc`` or a descendant of ``anc``."""
    if t[0] < anc[0]:
        return False
    return ancestor_at(t, anc[0]) == anc


def subtree_tiles(root: Tile, depth: int) -> List[Tile]:
    """All tiles of the complete quadtree rooted at ``root`` with ``depth`` levels.

    ``depth=1`` is just the root; ``depth=4`` covers root..root_z+3 (85 tiles).
    Returned in coarse→fine, row-major order.
    """
    out: List[Tile] = [root]
    frontier = [root]
    for _ in range(depth - 1):
        nxt: List[Tile] = []
        for t in frontier:
            nxt.extend(children_of(t))
        out.extend(nxt)
        frontier = nxt
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Tile store
# ─────────────────────────────────────────────────────────────────────────────

class TileState(str, Enum):
    EMPTY      = "empty"       # nothing generated yet
    SEED       = "seed"        # real, known ground-truth tile (never overwritten)
    BOOTSTRAP  = "bootstrap"   # placeholder built by downsampling descendants
    SR         = "sr"          # produced by super-resolution from a parent
    OUTPAINT   = "outpaint"    # produced by the outpainting model
    DOWNSAMPLE = "downsample"  # deterministic ÷2 stitch of 4 known children
    REFINED    = "refined"     # re-generated during a refinement pass
    BICUBIC    = "bicubic"     # non-model bicubic upsample (floor baseline)


# States that hold model-quality (or real) pixels — not to be clobbered by a
# cheap placeholder pass.
REAL_STATES = frozenset({
    TileState.SEED, TileState.SR, TileState.OUTPAINT, TileState.DOWNSAMPLE,
    TileState.REFINED, TileState.BICUBIC,
})


@dataclass
class TileRecord:
    state: TileState = TileState.EMPTY
    image: Optional[Image.Image] = None        # 256x256 RGB
    source: str = ""                            # free-form provenance / step label
    extra: Dict = field(default_factory=dict)   # per-strategy scratch (confidence, zoom_diff, ...)


@dataclass
class TileStore:
    """Map from absolute ``(z, x, y)`` to a ``TileRecord``."""
    tiles: Dict[Tile, TileRecord] = field(default_factory=dict)

    def get(self, t: Tile) -> Optional[TileRecord]:
        return self.tiles.get(t)

    def image(self, t: Tile) -> Optional[Image.Image]:
        r = self.tiles.get(t)
        return r.image if r else None

    def has(self, t: Tile) -> bool:
        """True if ``t`` holds real (non-empty, non-bootstrap) content."""
        r = self.tiles.get(t)
        return r is not None and r.state in REAL_STATES

    def resolved(self, t: Tile) -> bool:
        """True if ``t`` has any image at all (incl. bootstrap placeholder)."""
        r = self.tiles.get(t)
        return r is not None and r.state != TileState.EMPTY and r.image is not None

    def is_seed(self, t: Tile) -> bool:
        r = self.tiles.get(t)
        return r is not None and r.state == TileState.SEED

    def put(self, t: Tile, image: Image.Image, state: TileState, source: str = "", **extra) -> TileRecord:
        rec = TileRecord(state=state, image=image, source=source, extra=dict(extra))
        self.tiles[t] = rec
        return rec

    def put_seed(self, t: Tile, image: Image.Image) -> TileRecord:
        return self.put(t, image, TileState.SEED, source="seed")

    def items(self):
        return self.tiles.items()
