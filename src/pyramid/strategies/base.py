"""Strategy base class, the strategy registry, and the reconcile helper.

A *strategy* turns a set of seed tiles into a fully-resolved pyramid by filling a
``TileStore`` in place:

    @register("my_method")
    class MyStrategy(Strategy):
        def generate(self, store, sub, seeds, ctx): ...
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from PIL import Image

from ..dataset import Subtree
from ..models import ModelBundle, TILE_PX
from ..tiles import Tile, TileState, TileStore, children_of, quadrant_in_parent

HALF = TILE_PX // 2
NEUTRAL = (128, 128, 128)


@dataclass
class GenContext:
    """Everything a strategy needs besides the store + seeds."""
    bundle: ModelBundle
    params: Dict = field(default_factory=dict)
    _log: Optional[Callable[[str], None]] = None

    def log(self, msg: str) -> None:
        if self._log:
            self._log(msg)


class Strategy:
    name: str = "base"

    def generate(self, store: TileStore, sub: Subtree, seeds: List[Tile], ctx: GenContext) -> None:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

_REGISTRY: Dict[str, type] = {}


def register(name: str):
    def _wrap(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return _wrap


def get_strategy(name: str) -> Strategy:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown strategy {name!r}. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def list_strategies() -> List[str]:
    return sorted(_REGISTRY)


# ─────────────────────────────────────────────────────────────────────────────
# Reconcile (shared downsampling pass)
# ─────────────────────────────────────────────────────────────────────────────

def reconcile_up(store: TileStore, sub: Subtree, keep=None) -> int:
    """Replace every non-leaf tile with the exact ÷2 mosaic of its 4 children.

    Makes the pyramid pixel-consistent bottom-up so zooming never "pops". Seeds are
    never overwritten; ``keep(tile) -> bool`` (optional) preserves other tiles too,
    trading exact zoom consistency for keeping their existing content.
    """
    root_z, leaf_z = sub.root[0], sub.leaf_zoom
    fixed = 0
    for z in range(leaf_z - 1, root_z - 1, -1):
        for t in sub.at_zoom(z):
            if store.is_seed(t):
                continue
            if keep is not None and keep(t):
                continue
            kids = children_of(t)
            if not all(store.resolved(c) for c in kids):
                continue
            canvas = Image.new("RGB", (TILE_PX, TILE_PX))
            for c in kids:
                qx, qy = quadrant_in_parent(c)
                thumb = store.image(c).convert("RGB").resize((HALF, HALF), Image.Resampling.BICUBIC)
                canvas.paste(thumb, (qx * HALF, qy * HALF))
            store.put(t, canvas, TileState.DOWNSAMPLE, source="reconcile")
            fixed += 1
    return fixed
