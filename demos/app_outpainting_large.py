#!/usr/bin/env python3
"""
Gradio demo: Genesis Outpainting — Large Grid.

A sandbox to watch the quadrant outpainting (OP) model fill a fixed grid at a
single user-chosen zoom level — no super-resolution, just outpainting — growing
outward from seed tiles.

Geometry — seed at the center, shared as corners
------------------------------------------------
The visible grid is 8×8 **cells**, each a 256² tile. Internally every cell is a
2×2 block of **128² quadrants** (so the canvas is a 16×16 quadrant grid). When
you place a 256² seed it is centered on a cell *junction*: its four 128²
quadrants become the inner corners of the four cells meeting at that point. Each
of those cells then has one known corner quadrant; the OP model outpaints the
other three. So the intact seed sits dead-center and generation radiates outward.

A 2×2 window of quadrant cells reassembles exactly one zoom-Z tile, so the OP
runs at full resolution (no down/upsampling) with the GSD scalar = the grid zoom.

The fill sweeps all overlapping 2×2 quadrant windows **most-constrained-first**
(3 known before 1 known) in repeated passes until full, streaming after every
model call so you watch it grow.

Workflow (seed placement mirrors the full-pyramid demo):
  1. Pick the grid zoom level (the GSD tiles are fetched at).
  2. Click a grid cell — the seed will be centered on that cell's lower-right
     junction (shared with its right / lower / lower-right neighbors). Then click
     the ESRI map to assign a satellite tile as the seed.
  3. Repeat for as many seeds as you like, then hit ⚡ Fill grid.

Run: python app_outpainting_large.py  (port: $GRADIO_OP_LARGE_PORT, default 7867).
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
from datetime import datetime
from typing import Dict, Generator, List, Optional, Tuple

_DEMOS = os.path.dirname(os.path.abspath(__file__))
# _DEMOS goes LAST (= highest sys.path priority) so our `utils/` package
# isn't shadowed by `src/outpainting/utils.py`.
sys.path.insert(0, os.path.join(_DEMOS, "..", "src"))
sys.path.insert(0, os.path.join(_DEMOS, "..", "src", "outpainting"))
sys.path.insert(0, _DEMOS)

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from utils.genesis_common import (
    DEFAULT_OP_CKPT,
    OP_KEY_B16_MAIN,
    OP_MODEL_KEYS,
    fetch_tile_at_zoom,
    inference_device,
    inference_device_label,
    load_op_denoiser,
    model_arch,
    run_outpainting,
    run_outpainting_batch,
)
from utils.pyramid_ui import (
    OFFSCREEN_CSS,
    make_cell_apply_js,
    make_leaflet_html,
    make_leaflet_js,
    make_map_apply_js,
    make_pyramid_init_js,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_DEVICE = inference_device()

_DEF_CELLS    = 8               # default 8×8 = 64 display cells (each 256²)
CANVAS_CHOICES = [4, 6, 8, 10]  # selectable canvas size (cells per axis)
CELL    = 128               # one 128² OP quadrant
TILE_PX = 2 * CELL          # 256 — the OP model's tile size (a 2×2 window)
NEUTRAL = (128, 128, 128)   # masked-hole fill colour (matches pyramid backbone)
EMPTY_BG = (221, 224, 234)  # display colour for an unfilled quadrant
SEED_BOX = (228, 30, 38)    # red box drawn around seed footprints in the PNG

CELL_PX  = 64               # display size per 256² cell
_OP_BATCH = 16              # disjoint windows filled per batched OP sampler pass
CELL_GAP = 2

# Grid zoom is user-selectable. A 2×2 quadrant window reassembles one zoom-Z
# tile, so the GSD scalar is Z itself (no Z-1 constraint).
ZOOM_CHOICES = list(range(10, 20))
_DEF_ZOOM    = 16

_DEF_MODEL = OP_KEY_B16_MAIN
_DEF_STEPS = 50
_DEF_CFG   = 1.0

# Off-screen trigger button element ids (CSS pushes them off-screen).
_CELL_APPLY_ID = "oplarge_cell_apply_btn"
_MAP_APPLY_ID  = "oplarge_map_apply_btn"

LEAFLET_HTML = make_leaflet_html("oplarge-mapwrap")
LEAFLET_JS   = make_leaflet_js("oplarge-mapwrap", "__opLargeMapPick", _MAP_APPLY_ID)
MAP_APPLY_JS = make_map_apply_js("__opLargeMapPick", "__opLargeCellClick")
GRID_INIT_JS = make_pyramid_init_js("opLargeCellClick", "__opLargeCellClick",
                                    _CELL_APPLY_ID, "oplarge_cell_trigger")
CELL_APPLY_JS = make_cell_apply_js("__opLargeCellClick")

_LOGS_DIR = os.path.join(_DEMOS, "logs", "outpainting_large")

# The four quadrants of a cell (cx,cy), as (crop box on 256², quadrant coord).
def _cell_quads(cx: int, cy: int) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """Display position (dx,dy in {0,1}) → quadrant coord (qx,qy) for a cell."""
    return {(0, 0): (2 * cx,     2 * cy),
            (1, 0): (2 * cx + 1, 2 * cy),
            (0, 1): (2 * cx,     2 * cy + 1),
            (1, 1): (2 * cx + 1, 2 * cy + 1)}


# ─────────────────────────────────────────────────────────────────────────────
# Grid state
# ─────────────────────────────────────────────────────────────────────────────

class GridState:
    """16×16 grid of 128² quadrants displayed as 8×8 256² cells."""

    def __init__(self, zoom: int = _DEF_ZOOM, cells_n: int = _DEF_CELLS):
        self.zoom: int = int(zoom)
        self.cells_n: int = int(cells_n)          # 256² cells per axis
        self.qn: int = 2 * int(cells_n)           # 128² quadrants per axis
        self.quads: Dict[Tuple[int, int], Image.Image] = {}   # (qx,qy) → 128² PIL
        self.qstate: Dict[Tuple[int, int], str] = {}          # "seed" | "op"
        self.selected: Optional[Tuple[int, int]] = None       # clicked cell (cx,cy)
        self.seeds_placed: int = 0
        # Quadrant-coord bbox (qx0,qy0,qx1,qy1) of each placed seed footprint.
        self.seed_boxes: List[Tuple[int, int, int, int]] = []

    def resolved(self, qx: int, qy: int) -> bool:
        return (qx, qy) in self.quads

    def put_quad(self, qx: int, qy: int, img: Image.Image, state: str) -> None:
        self.quads[(qx, qy)] = img.convert("RGB").resize((CELL, CELL))
        self.qstate[(qx, qy)] = state

    def place_seed(self, cx: int, cy: int, tile256: Image.Image) -> None:
        """Center a 256² seed on cell (cx,cy)'s lower-right junction: its four
        quadrants become the inner corners of the 4 cells meeting there."""
        img = tile256.convert("RGB").resize((TILE_PX, TILE_PX))
        # quadrant targets straddle the junction at quad-coord (2cx+1, 2cy+1).
        targets = {
            (0,    0):    (2 * cx + 1, 2 * cy + 1),   # seed UL → cell (cx,cy)   LR
            (CELL, 0):    (2 * cx + 2, 2 * cy + 1),   # seed UR → cell (cx+1,cy) LL
            (0,    CELL): (2 * cx + 1, 2 * cy + 2),   # seed LL → cell (cx,cy+1) UR
            (CELL, CELL): (2 * cx + 2, 2 * cy + 2),   # seed LR → cell (cx+1,cy+1) UL
        }
        placed: List[Tuple[int, int]] = []
        for (ox, oy), (qx, qy) in targets.items():
            if 0 <= qx < self.qn and 0 <= qy < self.qn:
                self.put_quad(qx, qy, img.crop((ox, oy, ox + CELL, oy + CELL)), "seed")
                placed.append((qx, qy))
        if placed:
            self.seeds_placed += 1
            xs = [q[0] for q in placed]
            ys = [q[1] for q in placed]
            self.seed_boxes.append((min(xs), min(ys), max(xs), max(ys)))

    def n_resolved_quads(self) -> int:
        return len(self.quads)


# ─────────────────────────────────────────────────────────────────────────────
# Model cache
# ─────────────────────────────────────────────────────────────────────────────

_OP_CACHE: dict = {}


def get_model(model_key: str, ckpt_override: str):
    """Load (and cache) the OP denoiser. Auto-pulls the ckpt if not on disk."""
    ckpt = (ckpt_override or "").strip() or DEFAULT_OP_CKPT.get(model_key, "")
    if not ckpt:
        raise FileNotFoundError(
            f"No outpainting checkpoint resolved for {model_key!r}. "
            "Set GENESIS_CKPT_BASE or paste a full path into the override box."
        )
    # dino_weights="" — the trained config is no-DINO (see app_outpainting_quadrant).
    cache_key = (ckpt, model_key, False)
    if cache_key not in _OP_CACHE:
        denoiser, _ = load_op_denoiser(
            ckpt, model_arch(model_key), dino_weights="", device=_DEVICE
        )
        _OP_CACHE[cache_key] = denoiser
    return _OP_CACHE[cache_key]


# ─────────────────────────────────────────────────────────────────────────────
# HTML rendering
# ─────────────────────────────────────────────────────────────────────────────

def _cell_image(grid: GridState, cx: int, cy: int, px: int):
    """Composite a cell's 4 quadrants into one px² image.

    Returns (image, any_known, any_seed, all_known).
    """
    half = px // 2
    canvas = Image.new("RGB", (px, px), EMPTY_BG)
    any_known = any_seed = False
    all_known = True
    for (dx, dy), q in _cell_quads(cx, cy).items():
        if q in grid.quads:
            canvas.paste(grid.quads[q].resize((half, half)), (dx * half, dy * half))
            any_known = True
            if grid.qstate.get(q) == "seed":
                any_seed = True
        else:
            all_known = False
    return canvas, any_known, any_seed, all_known


def _b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=84)
    return base64.b64encode(buf.getvalue()).decode()


def render_grid_html(grid: GridState, *, cell_click_fn: str = "opLargeCellClick",
                     generating: bool = False) -> str:
    """Render the 8×8 cell grid (cells composited from quadrants) + preview."""
    sel = grid.selected
    # The selected seed footprint = the 2×2 cells the seed centers among.
    sel_cells = set()
    if sel is not None:
        scx, scy = sel
        sel_cells = {(scx, scy), (scx + 1, scy), (scx, scy + 1), (scx + 1, scy + 1)}

    n_quads = grid.n_resolved_quads()
    total_q = grid.qn * grid.qn

    if generating:
        banner_cls = "gen-mode"
        banner = (f"⏳ <strong>Outpainting…</strong> &nbsp;·&nbsp; "
                  f"{n_quads}/{total_q} quadrants filled &nbsp;·&nbsp; z={grid.zoom}")
    elif sel is None:
        banner_cls = "none-mode"
        banner = ("🗺 <strong>Step 1:</strong> click a cell — the seed centers on its "
                  "lower-right junction (4 cells share it as corners) &nbsp;·&nbsp; "
                  f"then click the map (fetched at z={grid.zoom})")
    else:
        banner_cls = "sel-mode"
        banner = (f"📍 seed will center among cells <strong>({sel[0]},{sel[1]})…"
                  f"({sel[0]+1},{sel[1]+1})</strong> &nbsp;·&nbsp; click the map to "
                  f"assign a 256² satellite seed at z={grid.zoom}")

    CSS = f"""
<style>
.opl-wrap{{font-family:sans-serif;background:#f7f8fc;border-radius:10px;
          padding:12px;user-select:none;box-sizing:border-box;border:1px solid #dde0ea;}}
.opl-banner{{text-align:center;font-size:0.78em;padding:6px 14px;border-radius:6px;margin-bottom:10px;}}
.opl-banner.none-mode{{background:#f3f4f8;color:#555;border:1px solid #d4d6e0;}}
.opl-banner.sel-mode{{background:#eaeaf8;color:#2f2f9e;border:1px solid #8888cc;}}
.opl-banner.gen-mode{{background:#eaf4fe;color:#0a4a80;border:1px solid #88bbee;}}
.opl-body{{display:flex;gap:16px;align-items:flex-start;}}
.opl-grid{{display:grid;gap:{CELL_GAP}px;background:#c4c6d2;border:1px solid #b8bac8;
          border-radius:3px;padding:{CELL_GAP}px;box-sizing:border-box;flex-shrink:0;
          grid-template-columns:repeat({grid.cells_n},{CELL_PX}px);
          grid-template-rows:repeat({grid.cells_n},{CELL_PX}px);}}
.opl-cell{{cursor:pointer;position:relative;overflow:hidden;box-sizing:border-box;
          width:{CELL_PX}px;height:{CELL_PX}px;transition:filter .1s;}}
.opl-cell:hover{{filter:brightness(0.85);}}
.opl-cell img{{width:100%;height:100%;object-fit:cover;display:block;pointer-events:none;}}
.opl-preview{{flex:1;min-width:0;background:#fff;border:1px solid #dde0ea;border-radius:8px;
             padding:14px;box-sizing:border-box;display:flex;flex-direction:column;gap:10px;}}
.opl-preview h4{{margin:0;font-size:11px;color:#444;font-weight:600;text-transform:uppercase;
                letter-spacing:.06em;border-bottom:1px solid #eee;padding-bottom:6px;}}
#pyr-prev-img{{width:100%;aspect-ratio:1/1;object-fit:cover;border-radius:6px;border:1px solid #d4d6e0;
              background:#e8eaef;display:block;image-rendering:pixelated;max-width:256px;margin:0 auto;}}
.opl-legend{{display:flex;flex-wrap:wrap;gap:5px 12px;margin-top:10px;padding-top:8px;border-top:1px solid #d4d6e0;}}
.opl-leg{{display:flex;align-items:center;gap:4px;color:#555;font-size:9.5px;}}
.opl-dot{{width:10px;height:10px;border-radius:2px;flex-shrink:0;}}
</style>"""

    parts: List[str] = [CSS, '<div class="opl-wrap">']
    parts.append(f'<div class="opl-banner {banner_cls}">{banner}</div>')
    parts.append('<div class="opl-body">')
    parts.append('<div class="opl-grid">')

    for cy in range(grid.cells_n):
        for cx in range(grid.cells_n):
            cell_img, any_known, any_seed, all_known = _cell_image(grid, cx, cy, CELL_PX)
            in_sel = (not generating) and (cx, cy) in sel_cells

            if in_sel:
                outline = "outline:2px solid #2244cc;z-index:5;"
            elif any_seed:
                outline = "outline:2px solid #dd2222;z-index:4;"
            elif all_known:
                outline = "outline:1px solid #228833;"
            elif any_known:
                outline = "outline:1px solid #7ec87e;"
            else:
                outline = "outline:1px solid #c0c3d0;"

            inner = f'<img src="data:image/jpeg;base64,{_b64(cell_img)}" />'
            data_full = f' data-full="data:image/jpeg;base64,{_b64(cell_img.resize((256, 256)))}"'
            title = f"cell ({cx},{cy})"
            parts.append(
                f'<div class="opl-cell" data-z="{grid.zoom}" data-gx="{cx}" data-gy="{cy}"{data_full} '
                f'onclick="{cell_click_fn}(this)" style="{outline}" title="{title}">{inner}</div>'
            )

    parts.append('</div>')  # close grid

    # Current-cell preview (ids must match make_pyramid_init_js).
    if sel is not None:
        cell_img, _, _, _ = _cell_image(grid, sel[0], sel[1], 256)
        preview = (
            f'<img id="pyr-prev-img" src="data:image/jpeg;base64,{_b64(cell_img)}" alt="" />'
            f'<p id="pyr-prev-info" style="font-size:0.78rem;color:#334;margin:6px 0 0;">'
            f'cell ({sel[0]},{sel[1]}) · z={grid.zoom}</p>'
            f'<p id="pyr-prev-hint" style="font-size:0.72rem;color:#888;margin:4px 0 0;">256² cell.</p>'
        )
    else:
        preview = (
            '<div id="pyr-prev-img" style="display:flex;align-items:center;justify-content:center;'
            'background:#f0f1f5;border:1px dashed #bbb;color:#aaa;font-size:0.8rem;">'
            'click any cell<br>to inspect</div>'
            '<p id="pyr-prev-info" style="display:none;"></p>'
            '<p id="pyr-prev-hint" style="display:none;"></p>'
        )
    parts.append(f'<div class="opl-preview"><h4>Current Cell</h4>{preview}</div>')
    parts.append('</div>')  # close body

    # Legend
    parts.append('<div class="opl-legend">')
    parts.append('<div class="opl-leg"><div class="opl-dot" style="outline:2px solid #dd2222;background:#c8dcff;"></div>Has seed corner</div>')
    parts.append('<div class="opl-leg"><div class="opl-dot" style="outline:1px solid #228833;background:#dce8dc;"></div>Outpainted (full)</div>')
    parts.append('<div class="opl-leg"><div class="opl-dot" style="outline:1px solid #7ec87e;background:#eef6ee;"></div>Partial</div>')
    parts.append('<div class="opl-leg"><div class="opl-dot" style="background:#dde0ea;"></div>Empty</div>')
    parts.append('</div>')

    parts.append('</div>')  # close wrap
    return "".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Outpainting fill (full-resolution quadrant windows)
# ─────────────────────────────────────────────────────────────────────────────

def _fill_grid_op(grid: GridState, denoiser, steps: int, cfg: float
                  ) -> Generator[Tuple[str, GridState], None, None]:
    """Grow outward from the seeds via overlapping 2×2 quadrant-window outpainting.

    Most-constrained-first over all overlapping 2×2 quadrant windows; each round
    fills a maximal cell-disjoint (independent) set in one batched OP pass — same
    result as one-at-a-time, far fewer sampler invocations. Yields per committed
    window so the UI streams as the grid fills.
    """
    denoiser.args.num_sampling_steps = int(steps)
    denoiser.args.cfg = float(cfg)
    tile_zoom = grid.zoom   # a 2×2 quadrant window reassembles one zoom-Z tile

    n_windows = 0   # windows filled (compute-fair)
    n_passes  = 0   # actual batched sampler invocations
    while True:
        # All overlapping 2×2 quadrant windows with 1–3 known cells and ≥1 hole.
        candidates: List[Tuple[int, int, int]] = []
        for wy in range(grid.qn - 1):
            for wx in range(grid.qn - 1):
                cells = [(wx + dx, wy + dy) for dy in (0, 1) for dx in (0, 1)]
                k = sum(grid.resolved(cx, cy) for cx, cy in cells)
                if 1 <= k <= 3:
                    candidates.append((k, wx, wy))
        if not candidates:
            break
        candidates.sort(reverse=True)   # most-constrained first

        # Pick a maximal CELL-DISJOINT set (most-constrained first): these windows
        # share no quadrant, so their fills are independent and batch in one OP pass
        # (identical result to one-at-a-time). Contending windows wait for the next
        # round, when these fills have become context.
        used: set = set()
        specs: List[Tuple[int, int, List[Tuple[int, int]]]] = []
        items: List[Tuple[Image.Image, Image.Image]] = []
        for k, wx, wy in candidates:
            cells = [(wx + dx, wy + dy) for dy in (0, 1) for dx in (0, 1)]
            if any(c in used for c in cells):
                continue
            holes = [c for c in cells if not grid.resolved(*c)]
            if not holes or len(holes) == 4:
                continue
            canvas = Image.new("RGB", (TILE_PX, TILE_PX), NEUTRAL)
            mask   = np.full((TILE_PX, TILE_PX), 255, np.uint8)
            for (cqx, cqy) in cells:
                qx, qy = cqx - wx, cqy - wy
                if grid.resolved(cqx, cqy):
                    canvas.paste(grid.quads[(cqx, cqy)], (qx * CELL, qy * CELL))
                    mask[qy * CELL:(qy + 1) * CELL, qx * CELL:(qx + 1) * CELL] = 0
            used.update(cells)
            specs.append((wx, wy, holes))
            items.append((canvas, Image.fromarray(mask, "L")))

        if not items:
            break

        # Batched OP, chunked to bound GPU memory. Each chunk = one sampler pass.
        outs: List[Image.Image] = []
        for i in range(0, len(items), _OP_BATCH):
            chunk = items[i:i + _OP_BATCH]
            outs.extend(run_outpainting_batch(
                denoiser, [c for c, _ in chunk], [m for _, m in chunk],
                _DEVICE, tile_zoom=tile_zoom))
            n_passes += 1
        n_windows += len(items)

        # Commit every window's quadrants, streaming the grid window-by-window.
        for (wx, wy, holes), out in zip(specs, outs):
            for (cqx, cqy) in holes:
                qx, qy = cqx - wx, cqy - wy
                quad = out.crop((qx * CELL, qy * CELL, (qx + 1) * CELL, (qy + 1) * CELL))
                grid.put_quad(cqx, cqy, quad, "op")
            yield (f"  window ({wx},{wy}): filled {len(holes)} quad(s) "
                   f"[{grid.n_resolved_quads()}/{grid.qn * grid.qn} quads, "
                   f"{n_windows} windows / {n_passes} batched passes]", grid)


def _compose_grid_png(grid: GridState) -> str:
    """Composite the full filled canvas into one PNG and return its path.

    Each placed seed's 256² footprint is outlined with a red box.
    """
    os.makedirs(_LOGS_DIR, exist_ok=True)
    side = grid.qn * CELL
    canvas = Image.new("RGB", (side, side), NEUTRAL)
    for (qx, qy), img in grid.quads.items():
        canvas.paste(img.convert("RGB").resize((CELL, CELL)), (qx * CELL, qy * CELL))

    draw = ImageDraw.Draw(canvas)
    bw = max(3, CELL // 24)
    for (x0, y0, x1, y1) in grid.seed_boxes:
        draw.rectangle(
            [x0 * CELL, y0 * CELL, (x1 + 1) * CELL - 1, (y1 + 1) * CELL - 1],
            outline=SEED_BOX, width=bw,
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(_LOGS_DIR, f"op_grid_z{grid.zoom}_{ts}.png")
    canvas.save(out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Event handlers
# ─────────────────────────────────────────────────────────────────────────────

def on_cell_click(click_json: str, zoom: int, grid: GridState):
    """Cell click → select the junction the next seed will center on."""
    try:
        p = json.loads(click_json) if click_json.strip().startswith("{") else {}
    except json.JSONDecodeError:
        p = {}
    if not p or "gx" not in p:
        return render_grid_html(grid), "*Click a grid cell.*", grid
    cx, cy = int(p["gx"]), int(p["gy"])
    grid.selected = (cx, cy)
    msg = (f"📍 Seed will center on the junction of cells ({cx},{cy})…({cx+1},{cy+1}) "
           f"— click the map to assign it (fetched at z={grid.zoom}).")
    return render_grid_html(grid), msg, grid


def on_map_apply(pick_json: str, zoom: int, grid: GridState):
    """Fetch an ESRI tile and center it (as 4 corner quadrants) at the junction."""
    grid.zoom = int(zoom)
    try:
        p = json.loads(pick_json) if pick_json.strip().startswith("{") else {}
    except json.JSONDecodeError:
        p = {}
    if not p or "lat" not in p:
        return None, "*Click the map to pick a location.*", render_grid_html(grid), grid

    lat, lon = float(p["lat"]), float(p["lng"])

    js_gx = p.get("sel_gx")
    if js_gx is not None:
        grid.selected = (int(js_gx), int(p.get("sel_gy", 0)))
    if grid.selected is None:
        return (None, "⚠ **Select a cell first** — click any cell, then the map.",
                render_grid_html(grid), grid)

    try:
        tile = fetch_tile_at_zoom(lon, lat, grid.zoom, TILE_PX)
    except Exception as e:
        return None, f"*Tile fetch error: {e}*", render_grid_html(grid), grid

    img = tile if isinstance(tile, Image.Image) else Image.fromarray(tile)
    cx, cy = grid.selected
    grid.place_seed(cx, cy, img.convert("RGB"))
    msg = (f"✓ Seed centered at cells ({cx},{cy})…({cx+1},{cy+1}) from "
           f"({lat:.4f}, {lon:.4f}) at z={grid.zoom}.  "
           f"{grid.seeds_placed} seed(s) — add more or ⚡ Fill grid.")
    return img.convert("RGB"), msg, render_grid_html(grid), grid


def on_generate(grid: GridState, model_key: str, ckpt_override: str,
                steps: float, cfg: float) -> Generator:
    """Fill the grid with the OP model; stream the grid as it grows."""
    if grid.n_resolved_quads() == 0:
        yield (render_grid_html(grid),
               "*No seeds placed — select a cell and click the map first.*",
               gr.update(visible=False), grid)
        return

    try:
        denoiser = get_model(model_key, ckpt_override)
    except Exception as e:
        yield render_grid_html(grid), f"**Model load error:** {e}", gr.update(visible=False), grid
        return

    yield (render_grid_html(grid, generating=True),
           f"⏳ Filling grid from {grid.seeds_placed} seed(s) at z={grid.zoom}…",
           gr.update(visible=False), grid)

    try:
        for line, updated in _fill_grid_op(grid, denoiser, int(steps), float(cfg)):
            grid = updated
            yield (render_grid_html(grid, generating=True),
                   f"⏳ {line.strip()}", gr.update(visible=False), grid)
    except Exception as e:
        import traceback
        yield (render_grid_html(grid),
               f"**Outpainting error:** {e}\n```\n{traceback.format_exc()}\n```",
               gr.update(visible=False), grid)
        return

    try:
        png = _compose_grid_png(grid)
        dl = gr.update(value=png, visible=True)
        done = f"✓ Done! {grid.n_resolved_quads()}/{grid.qn * grid.qn} quadrants filled at z={grid.zoom}."
    except Exception as e:
        dl = gr.update(visible=False)
        done = f"✓ Filled {grid.n_resolved_quads()} quadrants, but compositing the PNG failed: {e}"

    yield render_grid_html(grid), done, dl, grid


def on_reset_grid(zoom: int, cells_n: int):
    """Zoom or canvas-size change → reset the grid (it clears either way)."""
    fresh = GridState(zoom=int(zoom), cells_n=int(cells_n))
    msg = (f"📏 Canvas **{int(cells_n)}×{int(cells_n)} cells** at **z{int(zoom)}**. "
           "Grid cleared — place seeds and fill.")
    return render_grid_html(fresh), msg, gr.update(visible=False), fresh


def on_clear(zoom: int, cells_n: int):
    fresh = GridState(zoom=int(zoom), cells_n=int(cells_n))
    return render_grid_html(fresh), "*Grid cleared.*", gr.update(visible=False), fresh


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="Genesis Outpainting — Large Grid") as demo:
        gr.HTML(f"<style>{OFFSCREEN_CSS}</style>")   # hide off-screen trigger button
        gr.Markdown(
            "## Genesis Outpainting — Large Grid\n"
            "Watch the **quadrant outpainting model** fill a **user-sized N×N** grid of "
            "256² cells at a single zoom level — outpainting only, no super-resolution. "
            "Each 256² seed is **centered on a cell junction**, so its four quadrants "
            "become the inner corners of the four surrounding cells; the model then "
            "outpaints the rest, growing outward at full resolution, most-constrained-first. "
            "Seed footprints are boxed in red in the downloadable image."
        )

        grid_state = gr.State(GridState(_DEF_ZOOM, _DEF_CELLS))
        cell_click_raw = gr.Textbox(value="{}", visible=False)
        map_click_raw  = gr.Textbox(value="{}", visible=False)

        with gr.Row():
            zoom_dd = gr.Dropdown(
                ZOOM_CHOICES, value=_DEF_ZOOM, label="Grid zoom level",
                info=("Esri tiles are fetched at this zoom; a 2×2 quadrant window is one "
                      "tile at this zoom. Changing it clears the grid."),
            )
            canvas_dd = gr.Dropdown(
                CANVAS_CHOICES, value=_DEF_CELLS, label="Canvas size (cells per axis)",
                info="N×N grid of 256² cells. Larger = more outpaint windows (slower). "
                     "Changing it clears the grid.",
            )

        with gr.Row(equal_height=False):
            with gr.Column(scale=4, min_width=320):
                gr.Markdown("### 1 · Pick seed tiles from the map")
                gr.HTML(value=LEAFLET_HTML, html_template="${value}",
                        js_on_load=LEAFLET_JS, container=True, min_height=340)
                map_apply_btn = gr.Button("Refresh from last map click",
                                          variant="secondary", elem_id=_MAP_APPLY_ID)
                esri_preview = gr.Image(type="pil", label="Last placed seed tile (256²)", height=160)

            with gr.Column(scale=5, min_width=320):
                gr.Markdown("### 2 · Model settings")
                model_key = gr.Dropdown(OP_MODEL_KEYS, value=_DEF_MODEL, label="Architecture",
                                        info="Quadrant-trained OP model.")
                with gr.Row():
                    steps_sl = gr.Slider(10, 100, value=_DEF_STEPS, step=5, label="Sampling steps")
                    cfg_sl   = gr.Slider(1.0, 4.0, value=_DEF_CFG, step=0.5, label="CFG scale")
                ckpt_override = gr.Textbox(
                    label="Optional checkpoint path override",
                    placeholder="Default: pulled from Google Drive (auto)",
                )
                with gr.Row():
                    generate_btn = gr.Button("⚡ Fill grid", variant="primary", scale=2)
                    clear_btn    = gr.Button("🗑 Clear", variant="stop", scale=1)
                status_md = gr.Markdown("")

        gr.Markdown("### 3 · Grid — click a cell (seed centers on its lower-right junction), then click the map")

        # Off-screen trigger button (Gradio must register its click events).
        cell_apply_btn = gr.Button("oplarge_cell_trigger", elem_id=_CELL_APPLY_ID,
                                   elem_classes=["pyr-offscreen-btn"])
        grid_html = gr.HTML(value=render_grid_html(GridState(_DEF_ZOOM, _DEF_CELLS)),
                            js_on_load=GRID_INIT_JS)

        download_btn = gr.DownloadButton(label="⬇ Download grid image", visible=False,
                                         variant="secondary")

        # ── Event wiring ──────────────────────────────────────────────────────
        zoom_dd.change(
            fn=on_reset_grid, inputs=[zoom_dd, canvas_dd],
            outputs=[grid_html, status_md, download_btn, grid_state],
        )
        canvas_dd.change(
            fn=on_reset_grid, inputs=[zoom_dd, canvas_dd],
            outputs=[grid_html, status_md, download_btn, grid_state],
        )
        map_apply_btn.click(
            fn=on_map_apply, inputs=[map_click_raw, zoom_dd, grid_state],
            outputs=[esri_preview, status_md, grid_html, grid_state], js=MAP_APPLY_JS,
        )
        cell_apply_btn.click(
            fn=on_cell_click, inputs=[cell_click_raw, zoom_dd, grid_state],
            outputs=[grid_html, status_md, grid_state], js=CELL_APPLY_JS,
        )
        generate_btn.click(
            fn=on_generate, inputs=[grid_state, model_key, ckpt_override, steps_sl, cfg_sl],
            outputs=[grid_html, status_md, download_btn, grid_state],
        )
        clear_btn.click(
            fn=on_clear, inputs=[zoom_dd, canvas_dd],
            outputs=[grid_html, status_md, download_btn, grid_state],
        )

    return demo


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_OP_LARGE_PORT", 7867))
    print(f"[Genesis OP Large Grid] Device: {inference_device_label(_DEVICE)}")
    build_ui().launch(server_name="0.0.0.0", server_port=port, share=True)
