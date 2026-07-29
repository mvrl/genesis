#!/usr/bin/env python3
"""
Gradio demo: Genesis Pyramid Builder — `unified` LARGE (6-level / deep).

Like ``app_pyramid_unified.py`` (same `unified` strategy and workflow) but builds
a 6-level quadtree (1365 tiles, leaf 32×32; window defaults z13→z18, selectable
over z11–z18). Cells render tiny at this depth by design; the payoff is the
download, stitched vertically (apex→base) as a tall column.

⚠ Cost: at depth 6 the D-1 greedy fill spans a 16×16 level and the leaf SR
upsamples an 8192² canvas — far heavier than the 4-level demo. Use a big GPU.

Port: GRADIO_FULL_LARGE_PORT (default 7868)
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
from datetime import datetime
from typing import Dict, Generator, List, Tuple

_DEMOS = os.path.dirname(os.path.abspath(__file__))
# Order matters: each `sys.path.insert(0, X)` ends up BELOW the next insert, so
# put _DEMOS LAST to keep our `utils/` package ahead of the flat `utils.py`
# modules under src/outpainting/ and src/superresolution/. `src` is on the path
# so the `pyramid` framework package imports cleanly.
sys.path.insert(0, os.path.join(_DEMOS, "..", "src"))
sys.path.insert(0, os.path.join(_DEMOS, "..", "src", "superresolution"))
sys.path.insert(0, os.path.join(_DEMOS, "..", "src", "outpainting"))
sys.path.insert(0, _DEMOS)

import gradio as gr
from PIL import Image

from utils.genesis_common import fetch_tile_at_zoom, inference_device_label

# ── Deepen the pyramid to 6 levels (this is the ONLY substantive divergence from
# app_pyramid_unified.py). pyramid_core exposes the span as a module global read
# at call time by set_zoom_window / valid_lower_bounds / grid_n, so overriding it
# *before* the name-imports below makes the whole demo + framework subtree 6-deep.
import utils.pyramid_core as _pc
_pc.PYRAMID_LEVELS = 6
_pc.set_zoom_window(13)            # initial window z13 → z18 (used by first render)

from utils.pyramid_core import (
    PYRAMID_LEVELS,
    PYRAMID_Z_FLOOR,
    PYRAMID_Z_CEIL,
    TILE_PX,
    TileState,
    TileRecord,
    PyramidState,
    STATE_LABEL,
    bootstrap,
    flag_seams,
    grid_n,
    z_min,
    z_max,
    set_zoom_window,
    valid_lower_bounds,
    _DEVICE,
)
from utils.pyramid_ui import (
    OFFSCREEN_CSS,
    render_pyramid_html,
    render_seed_table_html,
    make_leaflet_html,
    make_leaflet_js,
    make_map_apply_js,
    make_pyramid_init_js,
    make_cell_apply_js,
)
from utils.pyramid_gen import save_pyramid
from utils.stitch_pyramid import stitch

# ── New-framework generation engine (src/pyramid) ────────────────────────────
from pyramid.models import ModelBundle
from pyramid.dataset import Subtree
from pyramid.strategies import GenContext, get_strategy
from pyramid.tiles import TileStore, TileState as FwTileState, subtree_tiles


# ─────────────────────────────────────────────────────────────────────────────
# Element IDs & JS strings (unique to this demo)
# ─────────────────────────────────────────────────────────────────────────────

_CELL_APPLY_ID = "uni_cell_apply_btn"
_MAP_APPLY_ID  = "uni_map_apply_btn"

LEAFLET_HTML    = make_leaflet_html("uni-mapwrap")
LEAFLET_JS      = make_leaflet_js("uni-mapwrap", "__uniMapPick", _MAP_APPLY_ID)
MAP_APPLY_JS    = make_map_apply_js("__uniMapPick", "__uniCellClick")
PYRAMID_INIT_JS = make_pyramid_init_js("uniCellClick", "__uniCellClick",
                                       _CELL_APPLY_ID, "uni_cell_trigger")
CELL_APPLY_JS   = make_cell_apply_js("__uniCellClick")

# Window is a fixed PYRAMID_LEVELS span; only its lower bound slides
# (set_zoom_window), constrained to [PYRAMID_Z_FLOOR, PYRAMID_Z_CEIL].
_DEPTH = PYRAMID_LEVELS   # 6 → 1365 tiles (leaf 32×32)

# (label, lower_bound) choices for the zoom-window dropdown, e.g. "z13 → z18".
_ZOOM_WINDOW_CHOICES: List[Tuple[str, int]] = [
    (f"z{lo} → z{lo + PYRAMID_LEVELS - 1}", lo) for lo in valid_lower_bounds()
]
_DEF_LOWER = 13   # default window z13 → z18

# Cell size for the stitched download figure. Use FULL native 256² tiles for max
# quality — at depth 6 the leaf is 32×32 tiles, so the figure is giant (leaf row
# = 32·256 = 8192px wide, column ~17k px tall). That is intentional; downscale the
# downloaded PNG afterwards as needed. (This only affects the download, not the
# on-screen grid, which auto-shrinks to fit the page.)
_VIZ_CELL_PX = TILE_PX   # 256
# Cap the downloadable figure: the full-res lossless PNG is ~125 MB, so the figure
# is saved as a quality-tuned JPEG (full resolution) that stays under this budget.
_MAX_FIG_MB = 30

# Default unified config.
_DEF_ARCH    = "JiT-H/16"
_DEF_CFG     = 2.5
_DEF_OVERLAP = 32
_DEF_T       = 0.40
_DEF_STEPS   = 50

# Cache one ModelBundle per architecture (model load is expensive).
_BUNDLE_CACHE: Dict[str, ModelBundle] = {}

# Map framework tile-states → the demo's display states / palette.
_FW2UI_STATE: Dict[FwTileState, TileState] = {
    FwTileState.SEED:       TileState.SEED,
    FwTileState.SR:         TileState.HARD_SR,
    FwTileState.OUTPAINT:   TileState.OUTPAINTED,
    FwTileState.DOWNSAMPLE: TileState.DOWNSAMPLED,
    FwTileState.REFINED:    TileState.SEAM_REFINED,
    FwTileState.BICUBIC:    TileState.DOWNSAMPLED,
    FwTileState.BOOTSTRAP:  TileState.DOWNSAMPLED,
}


def _get_bundle(arch: str, steps: int, cfg: float) -> ModelBundle:
    """Cached ModelBundle for ``arch``; sampler knobs refreshed each call."""
    b = _BUNDLE_CACHE.get(arch)
    if b is None:
        b = ModelBundle(arch=arch, steps=steps, cfg=cfg)
        _BUNDLE_CACHE[arch] = b
    # Refresh sampler knobs (unified SR reads bundle.steps/bundle.cfg per call;
    # OP reads op.args, so push onto loaded models too).
    b.steps, b.cfg = steps, cfg
    if b._sr is not None:
        b._sr.args.num_sampling_steps = steps
        b._sr.args.cfg = cfg
    if b._op is not None:
        b._op.args.num_sampling_steps = steps
        b._op.args.cfg = cfg
    return b


# ─────────────────────────────────────────────────────────────────────────────
# Bridge: PyramidState (UI) ←→ TileStore / Subtree (framework)
# ─────────────────────────────────────────────────────────────────────────────

def _make_subtree() -> Subtree:
    # Root at the current window's lower bound; the framework conditions the
    # SR/OP models on these real absolute zooms (correct GSD per level).
    root: Tuple[int, int, int] = (z_min(), 0, 0)
    tiles = subtree_tiles(root, _DEPTH)
    return Subtree(site_id="demo", tag="demo", root=root, depth=_DEPTH,
                   tiles=tiles, row_index={})


def _seeds_to_store(pyramid: PyramidState) -> Tuple[TileStore, List[Tuple[int, int, int]]]:
    """Build a fresh TileStore holding only the user's SEED tiles (abs coords)."""
    store = TileStore()
    seeds: List[Tuple[int, int, int]] = []
    for (z, gx, gy), rec in pyramid.tiles.items():
        if rec.state == TileState.SEED and rec.image is not None:
            t = (z, gx, gy)
            store.put_seed(t, rec.image.convert("RGB"))
            seeds.append(t)
    seeds.sort()
    return store, seeds


def _sync_store_to_pyramid(store: TileStore, pyramid: PyramidState) -> None:
    """Snapshot the framework store back into the UI PyramidState for rendering."""
    for t, rec in list(store.tiles.items()):
        if rec.image is None or rec.state == FwTileState.EMPTY:
            continue
        ui_state = (TileState.SEED if rec.state == FwTileState.SEED
                    else _FW2UI_STATE.get(rec.state, TileState.HARD_SR))
        pyramid.put(t[0], t[1], t[2], TileRecord(
            state=ui_state, image=rec.image, self_consistency=1.0,
        ))


# Cooperative cancel for the (long, GPU-heavy) deep generation. Each run gets a
# token (_RUN_ID); the worker's log callback raises if the token is superseded, so
# the strategy unwinds at its next ctx.log() — actually freeing the GPU instead of
# letting a detached thread run to completion. Stop/Reset and any new run both bump
# _RUN_ID, which invalidates the previous worker (avoids a stale worker surviving a
# quick reset→regenerate and double-occupying the GPU). No strategy change needed.
_RUN_ID = 0


class _Cancelled(Exception):
    """Raised inside the worker (via ctx.log) to abort a superseded/cancelled run."""


def generate_pyramid(
    pyramid: PyramidState,
    bundle: ModelBundle,
    params: Dict,
    strategy_name: str = "unified",
) -> Generator[Tuple[str, PyramidState], None, None]:
    """Run the selected strategy in a worker thread, streaming log + grid.

    The strategy fills its TileStore in place; its log callback is captured into
    a queue and the (shared) store is snapshotted into ``pyramid`` on each log
    line so the UI grid updates live. Generation aborts promptly if ``_CANCEL``
    is set (Stop/Reset): the log callback raises ``_Cancelled`` in the worker.
    """
    sub = _make_subtree()
    store, seeds = _seeds_to_store(pyramid)

    global _RUN_ID
    _RUN_ID += 1
    my_id = _RUN_ID                       # this run's token; superseded by reset / new run
    log_q: "queue.Queue[str | None]" = queue.Queue()

    def _log_cb(msg: str) -> None:
        if _RUN_ID != my_id:              # reset or a newer run → abort in the worker
            raise _Cancelled()
        log_q.put(msg)

    ctx = GenContext(bundle=bundle, params=params, _log=_log_cb)
    strat = get_strategy(strategy_name)

    err: Dict[str, BaseException] = {}

    def _run() -> None:
        try:
            strat.generate(store, sub, seeds, ctx)
        except BaseException as e:        # noqa: BLE001 - surfaced to the UI
            err["e"] = e
        finally:
            log_q.put(None)               # sentinel: generation finished

    th = threading.Thread(target=_run, daemon=True)
    th.start()

    # Throttle live snapshots. The deep pyramid streams ~1.3k tiles one-by-one;
    # re-rendering the whole grid (and base64-encoding every tile) on every single
    # per-tile log line is O(tiles²) and dominates wall-clock — the server-side
    # loop must drain before save_pyramid runs, so a naive stream can hang for many
    # minutes after the grid already looks full. Coalesce per-tile updates: render
    # at most every SNAPSHOT_EVERY tiles. Phase-level lines (no leading space)
    # always render, and the final state is always rendered after the loop.
    SNAPSHOT_EVERY = 32
    tile_since = 0
    while True:
        line = log_q.get()
        if line is None or _RUN_ID != my_id:
            break
        if line.startswith(" "):                 # per-tile progress line
            tile_since += 1
            if tile_since < SNAPSHOT_EVERY:
                continue                         # skip the heavy re-render
            tile_since = 0
        _sync_store_to_pyramid(store, pyramid)
        yield line, pyramid

    th.join()
    if "e" in err and not isinstance(err["e"], _Cancelled):
        raise err["e"]
    if _RUN_ID != my_id or isinstance(err.get("e"), _Cancelled):
        yield "⏹ Generation stopped.", pyramid
        return

    _sync_store_to_pyramid(store, pyramid)
    yield "✓ Generation complete.", pyramid


# ─────────────────────────────────────────────────────────────────────────────
# Event handlers
# ─────────────────────────────────────────────────────────────────────────────

def on_map_apply(pick_json: str, lower: int, pyramid: PyramidState):
    """Fetch ESRI tile and place it as a seed at the currently selected cell."""
    set_zoom_window(int(lower))
    try:
        p = json.loads(pick_json) if pick_json.strip().startswith("{") else {}
    except json.JSONDecodeError:
        p = {}
    if not p or "lat" not in p:
        return (gr.update(), "*Click the map to pick a location.*",
                "", "", "",
                render_pyramid_html(pyramid, cell_click_fn="uniCellClick"),
                render_seed_table_html(pyramid), pyramid)

    lat, lon, mz = float(p["lat"]), float(p["lng"]), float(p["zoom"])

    # Prefer the (z, gx, gy) from the JS payload over pyramid.selected to dodge
    # the cell-then-map click race (the map click can land before the cell-select
    # round-trips through gr.State, so trust the payload's own selection).
    js_z = p.get("sel_z")
    if js_z is not None:
        sel_z  = int(js_z)
        sel_gx = int(p.get("sel_gx", 0))
        sel_gy = int(p.get("sel_gy", 0))
        pyramid.selected = (sel_z, sel_gx, sel_gy)
    elif pyramid.selected is not None:
        sel_z, sel_gx, sel_gy = pyramid.selected
    else:
        return (
            gr.update(),
            "⚠ **Select a pyramid cell first** — click any cell, then click the map.",
            f"{lat:.5f}", f"{lon:.5f}", f"map:{mz:.0f}",
            render_pyramid_html(pyramid, cell_click_fn="uniCellClick"),
            render_seed_table_html(pyramid), pyramid,
        )

    try:
        tile = fetch_tile_at_zoom(lon, lat, sel_z, TILE_PX)
    except Exception as e:
        return (
            gr.update(), f"*Tile fetch error: {e}*",
            f"{lat:.5f}", f"{lon:.5f}", f"map:{mz:.0f} → cell:z{sel_z}",
            render_pyramid_html(pyramid, cell_click_fn="uniCellClick"),
            render_seed_table_html(pyramid), pyramid,
        )

    img = tile if isinstance(tile, Image.Image) else Image.fromarray(tile)
    img = img.convert("RGB").resize((TILE_PX, TILE_PX), Image.Resampling.BICUBIC)
    was_seed = pyramid.tiles.get((sel_z, sel_gx, sel_gy), TileRecord()).state == TileState.SEED
    pyramid.put(sel_z, sel_gx, sel_gy, TileRecord(
        state=TileState.SEED, image=img, self_consistency=1.0,
    ))
    bootstrap(pyramid)
    flag_seams(pyramid, sel_z, sel_gx, sel_gy)

    n_seeds = sum(1 for r in pyramid.tiles.values() if r.state == TileState.SEED)
    verb    = "Updated" if was_seed else f"Seed #{n_seeds} placed"
    info    = (f"✓ {verb} at z={sel_z} ({sel_gx},{sel_gy}) from ({lat:.4f}, {lon:.4f}).  "
               f"Add more seeds or ⚡ Generate Full Pyramid!")
    return (
        img, info,
        f"{lat:.5f}", f"{lon:.5f}", f"z={sel_z} (cell)",
        render_pyramid_html(pyramid, cell_click_fn="uniCellClick"),
        render_seed_table_html(pyramid), pyramid,
    )


def on_cell_click(click_json: str, lower: int, pyramid: PyramidState):
    """Cell click → select the cell (for seed placement or inspection)."""
    set_zoom_window(int(lower))
    try:
        p = json.loads(click_json) if click_json.strip().startswith("{") else {}
    except json.JSONDecodeError:
        p = {}
    if not p or "z" not in p:
        return (render_pyramid_html(pyramid, cell_click_fn="uniCellClick"),
                render_seed_table_html(pyramid), "*Click a pyramid cell.*",
                pyramid, gr.update())

    z, gx, gy = int(p["z"]), int(p["gx"]), int(p["gy"])
    pyramid.selected = (z, gx, gy)
    rec   = pyramid.record(z, gx, gy)
    state = STATE_LABEL.get(rec.state, rec.state.value)
    msg   = (f"📍 Selected z={z} ({gx},{gy}) [{state}] — "
             f"click map to assign a satellite seed (fetched at z={z})")
    return (render_pyramid_html(pyramid, cell_click_fn="uniCellClick"),
            render_seed_table_html(pyramid), msg, pyramid, gr.update())


def on_generate(
    pyramid: PyramidState,
    lower: int,
    arch: str,
    steps_n: float,
    cfg: float,
    overlap: float,
    t_start: float,
    recon_skip: bool,
    out_base: str,
) -> Generator:
    """Fill the entire pyramid with the `unified` strategy; stream log, save tiles."""
    set_zoom_window(int(lower))
    strategy = "unified"
    n_seeds = sum(1 for r in pyramid.tiles.values() if r.state == TileState.SEED)
    if n_seeds == 0:
        yield (render_pyramid_html(pyramid, cell_click_fn="uniCellClick"),
               render_seed_table_html(pyramid),
               "*No seeds placed — add at least one seed first.*", "",
               gr.update(), pyramid)
        return

    cached   = arch in _BUNDLE_CACHE and _BUNDLE_CACHE[arch]._sr is not None
    load_msg = ("✅ Model already loaded." if cached
                else f"⏳ Loading {arch} models (first time — will be cached)…")
    log: List[str] = [load_msg]
    yield (render_pyramid_html(pyramid, cell_click_fn="uniCellClick", generating=True),
           render_seed_table_html(pyramid),
           load_msg, "\n".join(log[-60:]), gr.update(), pyramid)

    try:
        bundle = _get_bundle(arch, int(steps_n), float(cfg))
    except Exception as e:
        yield (render_pyramid_html(pyramid, cell_click_fn="uniCellClick"),
               render_seed_table_html(pyramid),
               f"**Model load error:** {e}", "\n".join(log), gr.update(), pyramid)
        return

    # One params dict shared by all strategies. Each strategy reads only the
    # keys it knows via ctx.params.get(..., default), so unknown keys are
    # harmless and missing keys fall back to that strategy's own defaults.
    # unified reads uni_overlap, uni_t (+ arch/cfg/steps via bundle); its seam-fix
    # uses built-in defaults (seam_thresh/band/max). reconcile_skip_unblended is
    # an experimental toggle (see the checkbox).
    params = {
        "uni_overlap": int(overlap),
        "uni_t": float(t_start),
        "reconcile_skip_unblended": bool(recon_skip),
    }

    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.expanduser(out_base), run_ts)
    os.makedirs(out_dir, exist_ok=True)
    log.append(f"Strategy: {strategy} | {arch} | cfg={cfg} overlap={int(overlap)}px "
               f"t={t_start} steps={int(steps_n)}")
    log.append(f"Zoom window: z{z_min()} → z{z_max()} ({_DEPTH} levels)")
    log.append(f"Seeds: {n_seeds} | Output: {out_dir}")
    log.append(f"Device: {inference_device_label(_DEVICE)}")

    sr0, op0 = bundle.sr_calls, bundle.op_calls
    try:
        for log_line, updated in generate_pyramid(pyramid, bundle, params, strategy):
            pyramid = updated
            pyramid.calls_used = (bundle.sr_calls - sr0) + (bundle.op_calls - op0)
            if not log_line.startswith(" "):     # phase-level lines only
                log.append(log_line)
            status = (log_line.lstrip() if log_line.lstrip().startswith("✓")
                      else f"⏳ {log_line.lstrip()}")
            yield (render_pyramid_html(pyramid, cell_click_fn="uniCellClick", generating=True),
                   render_seed_table_html(pyramid),
                   status, "\n".join(log[-60:]), gr.update(), pyramid)
    except Exception as e:
        log.append(f"✗ Generation error: {e}")
        yield (render_pyramid_html(pyramid, cell_click_fn="uniCellClick"),
               render_seed_table_html(pyramid),
               f"**Generation error:** {e}", "\n".join(log[-60:]), gr.update(), pyramid)
        return

    n_saved  = save_pyramid(pyramid, out_dir)
    sr_used, op_used = bundle.sr_calls - sr0, bundle.op_calls - op0
    log.append(f"✓ Saved {n_saved} tiles → {out_dir}")
    log.append(f"  SR windows: {sr_used} | OP calls: {op_used}")

    # Always stitch the saved tiles into a single pyramid figure for download.
    # Deep pyramid → stitch VERTICALLY (apex→base) at FULL native 256² tile size.
    # The lossless PNG would be ~125 MB, so cap it at _MAX_FIG_MB: stitch writes a
    # quality-tuned JPEG (full resolution) that fits the budget. Seeds boxed in red.
    try:
        viz_path = stitch(out_dir, cell_px=_VIZ_CELL_PX, mark_seeds=True, vertical=True,
                          max_mb=_MAX_FIG_MB)
        log.append(f"  Pyramid figure: {viz_path}")
        download_update = gr.update(value=viz_path, visible=True)
        done_msg = f"✓ Done!  {n_saved} tiles saved → {out_dir}"
    except Exception as e:
        log.append(f"  ⚠ Stitch failed: {e}")
        download_update = gr.update(visible=False)
        done_msg = f"✓ Done ({n_saved} tiles) — but stitching the pyramid failed: {e}"

    yield (
        render_pyramid_html(pyramid, cell_click_fn="uniCellClick"),
        render_seed_table_html(pyramid),
        done_msg,
        "\n".join(log[-60:]),
        download_update,
        pyramid,
    )


def on_toggle_grid(show_grid: bool, pyramid: PyramidState):
    """Toggle grid lines / per-tile boxes on the pyramid visualization.

    With grid lines off only ground-truth seed tiles are boxed (in red); every
    other tile is shown bare across all zoom levels.
    """
    pyramid.show_grid = bool(show_grid)
    return render_pyramid_html(pyramid, cell_click_fn="uniCellClick"), pyramid


def on_clear(lower: int, show_grid: bool):
    set_zoom_window(int(lower))
    fresh = PyramidState(show_grid=bool(show_grid))
    return (render_pyramid_html(fresh, cell_click_fn="uniCellClick"),
            render_seed_table_html(fresh),
            "*Pyramid cleared.*", "", gr.update(visible=False), fresh)


def on_reset(lower: int, show_grid: bool):
    """Stop & Reset: abort any in-flight generation and clear the whole pyramid.

    Bumps the run token so the worker thread unwinds at its next log (freeing the
    GPU); the button's ``cancels=`` also stops the streaming UI event. Returns a
    fresh, empty pyramid so the user can start over (pick cell → seed → generate).
    """
    global _RUN_ID
    _RUN_ID += 1                          # invalidate any in-flight run
    set_zoom_window(int(lower))
    fresh = PyramidState(show_grid=bool(show_grid))
    return (render_pyramid_html(fresh, cell_click_fn="uniCellClick"),
            render_seed_table_html(fresh),
            "⏹ Stopped & reset — pick a cell, place a seed, and generate again.",
            "", gr.update(visible=False), fresh)


def on_window_change(lower: int, show_grid: bool):
    """User picked a new zoom window → reset and re-render at the new zooms.

    Seeds placed under the previous window live at zooms that no longer exist in
    the grid, so changing the window clears the pyramid (same as Clear All).
    """
    z0, z1 = set_zoom_window(int(lower))
    fresh = PyramidState(show_grid=bool(show_grid))
    msg = (f"📏 Zoom window set to **z{z0} → z{z1}** ({_DEPTH} levels). "
           "Pyramid cleared — place seeds and generate.")
    return (render_pyramid_html(fresh, cell_click_fn="uniCellClick"),
            render_seed_table_html(fresh),
            msg, "", gr.update(visible=False), fresh)


def _on_arch_change(arch: str, steps: float, cfg: float):
    """Suggest H/16-vs-B/16 friendly defaults when the user switches arch."""
    return gr.update()  # knobs are arch-agnostic; nothing to remap


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

def build_ui():
    # NB: in Gradio 6.0 `theme`/`css` moved from the Blocks constructor to
    # launch() — they're passed there (see __main__) so the offscreen-button CSS
    # actually applies.
    with gr.Blocks(title="Genesis Pyramid — Unified Large") as demo:

        gr.Markdown(
            "## Genesis Pyramid — Unified Large  🛰️\n"
            f"Fill a complete **{PYRAMID_LEVELS}-level** quadtree (deep variant) from seed "
            "tiles with the **`unified`** strategy: bring seeds to the penultimate (D-1) "
            "level, greedily outpaint it, super-resolve once to the leaf, repair seams with "
            "a free-floating outpaint pass, then reconcile. See "
            "[`demos/README.md`](README.md) for the algorithm.\n\n"
            f"⚠ Deep pyramid (leaf is {1 << (PYRAMID_LEVELS - 1)}×{1 << (PYRAMID_LEVELS - 1)} "
            "tiles): cells render tiny and generation is **much heavier/slower** than the "
            "4-level demo. The downloadable figure is stitched **vertically** (apex→base).\n\n"
            "**Workflow:** Choose the zoom window → click a pyramid cell → click the map to "
            "place a satellite seed (any cell, any level). Repeat. Then click "
            "**⚡ Generate Full Pyramid**, streaming live."
        )

        pyramid_state  = gr.State(PyramidState())
        cell_click_raw = gr.Textbox(value="{}", visible=False)
        map_click_raw  = gr.Textbox(value="{}", visible=False)

        # ── Zoom window (sets the absolute zoom range of the whole pyramid) ──────
        # Changing it resets the grid (seeds live at zooms that no longer exist).
        with gr.Row():
            zoom_window_dd = gr.Dropdown(
                _ZOOM_WINDOW_CHOICES, value=_DEF_LOWER,
                label=f"Zoom window  ({PYRAMID_LEVELS} levels)",
                info=(f"Absolute Web-Mercator zoom range to synthesize. Span is fixed at "
                      f"{PYRAMID_LEVELS} levels; bounded to z{PYRAMID_Z_FLOOR}–z{PYRAMID_Z_CEIL} "
                      "(what the SR/OP models were trained on). Changing this clears the grid."),
                scale=1,
            )

        # ── Top row: map (left) | seed table + controls (right) ──────────────
        with gr.Row(equal_height=False):

            with gr.Column(scale=4, min_width=320):
                gr.Markdown("### 1 · Pick seed tiles from the map")
                gr.HTML(value=LEAFLET_HTML, html_template="${value}",
                        js_on_load=LEAFLET_JS, container=True, min_height=340)
                with gr.Row():
                    lat_tb  = gr.Textbox(label="Lat",  interactive=False, scale=1)
                    lon_tb  = gr.Textbox(label="Lon",  interactive=False, scale=1)
                    zoom_tb = gr.Textbox(label="Zoom", interactive=False, scale=1)
                map_apply_btn = gr.Button("Refresh from last map click",
                                          variant="secondary", elem_id=_MAP_APPLY_ID)
                esri_preview  = gr.Image(type="pil", label="Last placed seed tile (256²)", height=160)
                map_info_md   = gr.Markdown(
                    "*Click a pyramid cell first, then click the map to assign a seed.*"
                )

            with gr.Column(scale=5, min_width=320):
                gr.Markdown("### 2 · Seeds assigned")
                seed_table_html = gr.HTML(value=render_seed_table_html(PyramidState()))

                gr.Markdown("### 3 · SR settings")
                with gr.Row():
                    arch_dd = gr.Dropdown(["JiT-H/16", "JiT-B/16"], value=_DEF_ARCH,
                                          label="Architecture",
                                          info="H/16 = best quality (recommended); B/16 = faster")
                    cfg_sl  = gr.Slider(1.0, 4.0, value=_DEF_CFG, step=0.5,
                                        label="CFG scale",
                                        info="Dominant sharpness lever for the SR passes")
                with gr.Row():
                    overlap_sl = gr.Slider(8, 64, value=_DEF_OVERLAP, step=8,
                                           label="SR window overlap (px)",
                                           info="Tunes the SR passes (D-1→leaf, seed SR-up): "
                                                "seam↔sharpness band, lower = sharper/seamier. "
                                                "Does not affect the outpainting steps.")
                    t_sl       = gr.Slider(0.20, 0.80, value=_DEF_T, step=0.05,
                                           label="SR warm-start (t_start)",
                                           info="SDEdit warm-start for the SR passes; "
                                                "lower = more model freedom / detail.")

                with gr.Row():
                    generate_btn = gr.Button("⚡ Generate Full Pyramid",
                                             variant="primary", size="lg")
                    reset_btn    = gr.Button("⏹ Stop & Reset", variant="stop")
                    clear_btn    = gr.Button("🗑 Clear All", variant="secondary")

                status_md = gr.Markdown("")

                with gr.Accordion("Advanced & output settings", open=False):
                    steps_sl   = gr.Slider(10, 100, value=_DEF_STEPS, step=5,
                                           label="Sampling steps")
                    recon_skip_cb = gr.Checkbox(
                        value=True, label="Reconcile: keep unblended D-1 tiles",
                        info="Default ON for the deep pyramid. Keeps the original "
                             "penultimate-level (z=leaf−1) tiles whose leaf children were "
                             "NOT seam-fixed, instead of overwriting them with ÷2(leaf) — so "
                             "their generated content isn't needlessly downsampled. The other "
                             "coarse levels (≤ leaf−2) have no content of their own, so they "
                             "are always reconciled (÷2). Untick for an exactly zoom-consistent "
                             "pyramid at z=leaf−1.")
                    out_folder = gr.Textbox(
                        value=os.path.join(_DEMOS, "logs", "pyramid_unified_large_results"),
                        label="Output base folder (timestamped sub-folder per run)",
                    )

        # ── Bottom: full-width pyramid visualization ──────────────────────────
        gr.Markdown("### 4 · Pyramid — click cells to assign seeds or inspect "
                    "*(zoom labels follow the window above)*")

        grid_toggle = gr.Checkbox(
            value=True, label="Show grid lines",
            info="Off → seamless mosaic with a red box only on seed tiles "
                 "(no boxes on any other tiles across zoom levels).",
        )

        gr.HTML(f'<style>.full-offscreen-btn{{{OFFSCREEN_CSS.split("{")[1].split("}")[0]}}}</style>')
        cell_apply_btn = gr.Button("uni_cell_trigger",
                                   elem_id=_CELL_APPLY_ID,
                                   elem_classes=["pyr-offscreen-btn"])
        cell_info_md = gr.Markdown("*Click a pyramid cell to select it.*")

        pyramid_html = gr.HTML(
            value=render_pyramid_html(PyramidState(), cell_click_fn="uniCellClick"),
            js_on_load=PYRAMID_INIT_JS,
        )

        log_tb = gr.Textbox(label="Generation log", lines=12, max_lines=20,
                            interactive=False)
        download_btn = gr.DownloadButton(
            label="⬇ Download pyramid", visible=False, variant="secondary",
        )

        # ── Event wiring ──────────────────────────────────────────────────────

        _map_outs = [esri_preview, map_info_md, lat_tb, lon_tb, zoom_tb,
                     pyramid_html, seed_table_html, pyramid_state]

        # Changing the zoom window resets the grid and re-renders at the new zooms.
        zoom_window_dd.change(
            fn=on_window_change,
            inputs=[zoom_window_dd, grid_toggle],
            outputs=[pyramid_html, seed_table_html, cell_info_md,
                     log_tb, download_btn, pyramid_state],
        )

        grid_toggle.change(
            fn=on_toggle_grid,
            inputs=[grid_toggle, pyramid_state],
            outputs=[pyramid_html, pyramid_state],
        )

        map_apply_btn.click(
            fn=on_map_apply,
            inputs=[map_click_raw, zoom_window_dd, pyramid_state],
            outputs=_map_outs,
            js=MAP_APPLY_JS,
        )

        cell_apply_btn.click(
            fn=on_cell_click,
            inputs=[cell_click_raw, zoom_window_dd, pyramid_state],
            outputs=[pyramid_html, seed_table_html, cell_info_md, pyramid_state, esri_preview],
            js=CELL_APPLY_JS,
        )

        gen_evt = generate_btn.click(
            fn=on_generate,
            inputs=[pyramid_state, zoom_window_dd, arch_dd, steps_sl, cfg_sl,
                    overlap_sl, t_sl, recon_skip_cb, out_folder],
            outputs=[pyramid_html, seed_table_html, status_md, log_tb,
                     download_btn, pyramid_state],
        )

        # Stop & Reset: cancel the running generation event (stops the UI stream)
        # AND set the cancel flag (unwinds the GPU worker), then clear the grid so
        # the user can restart from scratch (pick cell → seed → generate).
        reset_btn.click(
            fn=on_reset,
            inputs=[zoom_window_dd, grid_toggle],
            outputs=[pyramid_html, seed_table_html, cell_info_md,
                     log_tb, download_btn, pyramid_state],
            cancels=[gen_evt],
        )

        clear_btn.click(
            fn=on_clear,
            inputs=[zoom_window_dd, grid_toggle],
            outputs=[pyramid_html, seed_table_html, cell_info_md,
                     log_tb, download_btn, pyramid_state],
        )

    return demo


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_FULL_LARGE_PORT", 7868))
    print(f"[Genesis Pyramid · Unified Large] Device: {inference_device_label(_DEVICE)}")
    if os.environ.get("GENESIS_PRELOAD", "1").lower() not in ("0", "false", "no"):
        try:
            print(f"[Genesis Pyramid · Unified Large] Pre-loading {_DEF_ARCH} SR+OP…", flush=True)
            b = _get_bundle(_DEF_ARCH, _DEF_STEPS, _DEF_CFG)
            b._ensure_sr()
            b._ensure_op()
            print("[Genesis Pyramid · Unified Large] ✓ Models ready.", flush=True)
        except Exception as _e:
            print(f"[Genesis Pyramid · Unified Large] Pre-load skipped ({_e}).", flush=True)
    build_ui().launch(server_name="0.0.0.0", server_port=port, share=True,
                      theme=gr.themes.Soft(), css=OFFSCREEN_CSS)
