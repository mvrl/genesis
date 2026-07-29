"""
pyramid_core.py — shared data model and demo-side helpers for the pyramid apps.

Imported by pyramid_ui.py, pyramid_gen.py, and the pyramid demo app
(app_pyramid_unified.py). Holds the tile state machine + colour palette,
the PyramidState data model, the zoom-window helpers, and the demo-preview
helpers (bootstrap downsample, seam flagging). Generation itself lives in the
``src/pyramid`` strategy framework; UI / rendering in pyramid_ui.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .genesis_common import inference_device

# ── Constants: pyramid zoom window ────────────────────────────────────────────
# The pyramid is a FIXED 4-level span; only its position slides. The window is
# runtime-settable (set_zoom_window), constrained to the model-trained range.
PYRAMID_LEVELS    = 4         # fixed span (z_min .. z_min+3)
PYRAMID_Z_FLOOR   = 11        # hard lower clamp (model-trained guardrail)
PYRAMID_Z_CEIL    = 18        # hard upper clamp (model-trained guardrail)

# Live window state — module globals so the relative-math helpers (grid_n,
# in_bounds, parent_key, bootstrap, …) read the *current* window at call time;
# set_zoom_window() mutates them.
PYRAMID_Z_MIN = 15
PYRAMID_Z_MAX = PYRAMID_Z_MIN + PYRAMID_LEVELS - 1
TILE_PX       = 256
SEAM_STRIP    = 16      # px either side of a seam used for boundary-diff check
SEAM_THRESH   = 0.25    # mean abs diff above this → flag seam

_DEVICE: torch.device = inference_device()


def z_min() -> int:
    """Current pyramid lower-bound zoom (live)."""
    return PYRAMID_Z_MIN


def z_max() -> int:
    """Current pyramid upper-bound zoom (live)."""
    return PYRAMID_Z_MAX


def valid_lower_bounds() -> List[int]:
    """Lower bounds for which a full 4-level window fits inside the model guardrail."""
    return list(range(PYRAMID_Z_FLOOR, PYRAMID_Z_CEIL - PYRAMID_LEVELS + 2))


def set_zoom_window(lower: int) -> Tuple[int, int]:
    """Slide the fixed 4-level window so its lower bound is ``lower``.

    Clamps so the whole span stays within [PYRAMID_Z_FLOOR, PYRAMID_Z_CEIL].
    Returns the applied (z_min, z_max). Only the absolute zoom labels / GSD
    conditioning / ESRI fetch zoom change; the relative tile layout is unchanged.
    """
    global PYRAMID_Z_MIN, PYRAMID_Z_MAX
    lo = max(PYRAMID_Z_FLOOR, min(int(lower), PYRAMID_Z_CEIL - PYRAMID_LEVELS + 1))
    PYRAMID_Z_MIN = lo
    PYRAMID_Z_MAX = lo + PYRAMID_LEVELS - 1
    return PYRAMID_Z_MIN, PYRAMID_Z_MAX


def grid_n(z: int) -> int:
    """Tiles per axis at zoom z relative to the current window (1, 2, 4, 8)."""
    return 1 << (z - PYRAMID_Z_MIN)




# ─────────────────────────────────────────────────────────────────────────────
# Tile state machine + colour palette
# ─────────────────────────────────────────────────────────────────────────────

class TileState(str, Enum):
    EMPTY        = "empty"
    SEED         = "seed"
    DOWNSAMPLED  = "downsampled"
    SOFT_SR      = "soft_sr"
    HARD_SR      = "hard_sr"
    OUTPAINTED   = "outpainted"
    SEAM_REFINED = "seam_refined"


# Light-theme hex colours for HTML rendering
STATE_BG: Dict[TileState, str] = {
    TileState.EMPTY:        "#dde0ea",
    TileState.SEED:         "#c8dcff",
    TileState.DOWNSAMPLED:  "#dcd8f0",
    TileState.SOFT_SR:      "#c8e8c8",
    TileState.HARD_SR:      "#a0e8b0",
    TileState.OUTPAINTED:   "#f0e0a0",
    TileState.SEAM_REFINED: "#a8e8cc",
}
STATE_BORDER: Dict[TileState, str] = {
    TileState.SEED:         "#2255cc",
    TileState.HARD_SR:      "#228833",
    TileState.SOFT_SR:      "#448844",
    TileState.OUTPAINTED:   "#aa8800",
    TileState.DOWNSAMPLED:  "#5555aa",
    TileState.SEAM_REFINED: "#228866",
}
STATE_FG: Dict[TileState, str] = {
    TileState.EMPTY:        "#888",
    TileState.SEED:         "#1a3a88",
    TileState.DOWNSAMPLED:  "#333377",
    TileState.SOFT_SR:      "#1a5a1a",
    TileState.HARD_SR:      "#0a4a1a",
    TileState.OUTPAINTED:   "#5a4400",
    TileState.SEAM_REFINED: "#0a4433",
}
STATE_LABEL: Dict[TileState, str] = {
    TileState.EMPTY:        "Empty",
    TileState.SEED:         "Seed",
    TileState.DOWNSAMPLED:  "Downsampled",
    TileState.SOFT_SR:      "Soft SR",
    TileState.HARD_SR:      "Hard SR",
    TileState.OUTPAINTED:   "Outpainted",
    TileState.SEAM_REFINED: "Seam refined",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TileRecord:
    state:            TileState             = TileState.EMPTY
    image:            Optional[Image.Image] = None
    self_consistency: float                 = 0.0
    zoom_diff_used:   Optional[int]         = None
    seam_flagged:     bool                  = False


@dataclass
class PyramidState:
    tiles:      Dict[Tuple[int, int, int], TileRecord] = field(default_factory=dict)
    calls_used: int                                    = 0
    selected:   Optional[Tuple[int, int, int]]         = None  # (z, gx, gy)
    show_grid:  bool                                   = True  # grid lines + per-tile boxes

    # ── tile access ──────────────────────────────────────────────────────────

    def record(self, z: int, gx: int, gy: int) -> TileRecord:
        k = (z, gx, gy)
        if k not in self.tiles:
            self.tiles[k] = TileRecord()
        return self.tiles[k]

    def put(self, z: int, gx: int, gy: int, rec: TileRecord) -> None:
        self.tiles[(z, gx, gy)] = rec

    def is_resolved(self, z: int, gx: int, gy: int) -> bool:
        r = self.tiles.get((z, gx, gy))
        return r is not None and r.state != TileState.EMPTY

    def in_bounds(self, z: int, gx: int, gy: int) -> bool:
        # Check the zoom range first: grid_n(z) does 1<<(z-z_min) and would raise
        # on a negative shift for any z below the current window's lower bound.
        if not (PYRAMID_Z_MIN <= z <= PYRAMID_Z_MAX):
            return False
        n = grid_n(z)
        return 0 <= gx < n and 0 <= gy < n

    def confidence(self, z: int, gx: int, gy: int) -> float:
        r = self.tiles.get((z, gx, gy))
        if r is None or r.state == TileState.EMPTY:
            return 0.0
        base = {
            TileState.SEED:         1.00,
            TileState.HARD_SR:      0.85,
            TileState.SEAM_REFINED: 0.80,
            TileState.SOFT_SR:      0.50,
            TileState.OUTPAINTED:   0.40,
            TileState.DOWNSAMPLED:  0.30,
        }.get(r.state, 0.20)
        return min(1.0, base + r.self_consistency * 0.1)

    # ── hierarchy helpers ─────────────────────────────────────────────────────

    def parent_key(self, z: int, gx: int, gy: int) -> Optional[Tuple[int, int, int]]:
        if z <= PYRAMID_Z_MIN:
            return None
        return (z - 1, gx // 2, gy // 2)

    def grandparent_key(self, z: int, gx: int, gy: int) -> Optional[Tuple[int, int, int]]:
        p = self.parent_key(z, gx, gy)
        return None if p is None else self.parent_key(*p)

    def neighbors_4(self, z: int, gx: int, gy: int) -> List[Tuple[int, int, int]]:
        return [
            (z, gx + dx, gy + dy)
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if self.in_bounds(z, gx + dx, gy + dy)
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap — bicubic downsample seeds upward to fill ancestor tiles
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap(pyramid: PyramidState) -> None:
    seeds = [
        (z, gx, gy)
        for (z, gx, gy), r in list(pyramid.tiles.items())
        if r.state == TileState.SEED and r.image is not None
    ]
    for sz, sgx, sgy in seeds:
        img  = pyramid.tiles[(sz, sgx, sgy)].image
        cz, cgx, cgy = sz, sgx, sgy

        while cz > PYRAMID_Z_MIN:
            pz, pgx, pgy = cz - 1, cgx // 2, cgy // 2
            existing = pyramid.tiles.get((pz, pgx, pgy))

            if existing and existing.state in (TileState.SEED, TileState.HARD_SR):
                break

            rx, ry = cgx % 2, cgy % 2
            half   = TILE_PX // 2
            thumb  = img.resize((half, half), Image.Resampling.BICUBIC)

            if not existing or existing.state == TileState.EMPTY:
                canvas = Image.new("RGB", (TILE_PX, TILE_PX), (100, 100, 100))
                canvas.paste(thumb, (rx * half, ry * half))
                pyramid.put(pz, pgx, pgy, TileRecord(
                    state=TileState.DOWNSAMPLED,
                    image=canvas,
                    self_consistency=0.3,
                ))
            elif existing.state == TileState.DOWNSAMPLED and existing.image:
                existing.image.paste(thumb, (rx * half, ry * half))

            img = pyramid.tiles[(pz, pgx, pgy)].image
            cz, cgx, cgy = pz, pgx, pgy



# ─────────────────────────────────────────────────────────────────────────────
# Seam flagging
# ─────────────────────────────────────────────────────────────────────────────

def flag_seams(pyramid: PyramidState, z: int, gx: int, gy: int) -> None:
    """After placing (z,gx,gy), check boundary diff with all resolved neighbours."""
    rec = pyramid.record(z, gx, gy)
    if rec.image is None:
        return
    a = np.array(rec.image.convert("RGB"), np.float32) / 255.0
    S = SEAM_STRIP
    any_flag = False
    for nz, ngx, ngy in pyramid.neighbors_4(z, gx, gy):
        if not pyramid.is_resolved(nz, ngx, ngy):
            continue
        nrec = pyramid.record(nz, ngx, ngy)
        if nrec.image is None:
            continue
        b = np.array(nrec.image.convert("RGB"), np.float32) / 255.0
        if ngx < gx:
            diff = float(np.mean(np.abs(a[:, :S]  - b[:, -S:])))
        elif ngx > gx:
            diff = float(np.mean(np.abs(a[:, -S:] - b[:, :S])))
        elif ngy < gy:
            diff = float(np.mean(np.abs(a[:S, :]  - b[-S:, :])))
        else:
            diff = float(np.mean(np.abs(a[-S:, :] - b[:S, :])))
        if diff > SEAM_THRESH:
            nrec.seam_flagged = True
            any_flag = True
    rec.seam_flagged = any_flag


