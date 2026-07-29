"""
pyramid_ui.py — HTML/CSS/JS rendering for the pyramid demos.

Provides:
  render_pyramid_html()    — the full S2-grid pyramid widget
  render_seed_table_html() — seed assignment table
  make_leaflet_html()      — ESRI map HTML stub
  make_leaflet_js()        — Leaflet initialisation JS
  make_map_apply_js()      — JS preprocessor that reads the map pick global
  make_pyramid_init_js()   — cell-click handler registration JS
  make_cell_apply_js()     — JS preprocessor that reads the cell click global
  OFFSCREEN_CSS            — CSS for hiding the off-screen trigger button
"""
from __future__ import annotations

import base64
import io
from typing import Optional

from PIL import Image

from .pyramid_core import (
    TileState,
    STATE_BG,
    STATE_BORDER,
    STATE_FG,
    STATE_LABEL,
    PyramidState,
    grid_n,
    z_min,
    z_max,
)

# Cell display constants

CELL_PX  = 44   # each pyramid cell is this many px square
CELL_GAP =  2   # gap between cells inside a grid
THUMB_PX = CELL_PX

# CSS injected into gr.Blocks(css=) to hide the off-screen trigger button.
# The button must be visible=True so Gradio registers its click events; we just
# push it off-screen so users never see it.
OFFSCREEN_CSS = """
.pyr-offscreen-btn {
    position: fixed !important;
    top: -9999px !important;
    left: -9999px !important;
    width: 1px !important;
    height: 1px !important;
    overflow: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}
"""


# Helpers

def _b64(img: Image.Image, w: int, h: int) -> str:
    """Encode a PIL image as a base64 JPEG string at the given dimensions."""
    thumb = img.resize((max(1, w), max(1, h)), Image.Resampling.BICUBIC)
    buf   = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


# Pyramid HTML widget

def render_pyramid_html(
    pyramid: PyramidState,
    *,
    cell_click_fn: str = "pyrCellClick",
    generating: bool = False,
) -> str:
    """
    Render the full S2-grid pyramid widget as an HTML string.

    Parameters
    ----------
    pyramid        : current pyramid state
    cell_click_fn  : name of the JS function called on cell click
                     (e.g. "fullCellClick" for the full-pyramid demo)
    generating     : if True, show the "generating…" banner with progress
    """
    sel = pyramid.selected
    show_grid = getattr(pyramid, "show_grid", True)

    # Adaptive cell size. Levels are laid left→right, so the total row spans
    # Σ grid_n = 2^depth − 1 cells. A deep pyramid (e.g. the 6-level "large"
    # variant → 63 cells wide) overflows the page at the full 44px, so shrink the
    # cell to fit a width budget. Capped at CELL_PX and never enlarged, so shallow
    # pyramids (≤4 levels: ≤15 cells) keep the exact original 44px rendering.
    _n_levels   = z_max() - z_min() + 1
    _cells_wide = (1 << _n_levels) - 1
    cell_px  = max(10, min(CELL_PX, 900 // max(1, _cells_wide)))
    thumb_px = cell_px

    # The "Current Cell" preview panel sits beside the grid and competes for
    # width. A deep pyramid (>4 levels, e.g. the 6-level "large" variant) needs
    # the full width, so drop the panel there and let the grid use it all. The
    # cell-click JS guards on these elements being absent, so it safely no-ops.
    show_preview = _n_levels <= 4

    # Deep pyramids also get a horizontal-scroll safety net so the grid can never
    # blow out the page; shallow (≤4-level) pyramids keep the original
    # ``flex-shrink:0`` so their rendering is byte-for-byte unchanged.
    cols_overflow = "flex-shrink:0;" if _n_levels <= 4 else "min-width:0;overflow-x:auto;"

    # Banner
    if generating:
        banner_cls = "gen-mode"
        resolved = sum(1 for r in pyramid.tiles.values() if r.state != TileState.EMPTY)
        total    = sum(grid_n(z) ** 2 for z in range(z_min(), z_max() + 1))
        banner_txt = (f"⏳ <strong>Generating…</strong> &nbsp;·&nbsp; "
                      f"{resolved}/{total} tiles resolved &nbsp;·&nbsp; "
                      f"{pyramid.calls_used} model calls used")
    elif sel is None:
        banner_cls = "none-mode"
        banner_txt = ("🗺 <strong>Step 1:</strong> Click a pyramid cell to select it "
                      "&nbsp;·&nbsp; then click the map to assign a satellite seed "
                      "at that zoom level, or click ⚡&nbsp;Generate to fill it")
    else:
        sz, sgx, sgy = sel
        rec   = pyramid.tiles.get(sel)
        state = STATE_LABEL.get(rec.state, "empty") if rec else "empty"
        has_seed = any(r.state == TileState.SEED for r in pyramid.tiles.values())
        banner_cls = "sel-mode"
        banner_txt = (
            f"📍 <strong>z={sz} ({sgx},{sgy})</strong> [{state}] selected "
            f"&nbsp;·&nbsp; <em>Map will fetch at zoom&nbsp;{sz}</em> — "
            f"navigate &amp; click map to assign satellite seed"
            + (" &nbsp;·&nbsp; or ⚡&nbsp;Generate to fill with the model"
               if has_seed else "")
        )

    # Grid lines ON → grey gutters between cells + cell-state outlines.
    # Grid lines OFF → seamless mosaic (no gaps/background) and the only box drawn
    # is a red outline on ground-truth seed tiles; all other tiles are bare.
    grid_gap    = CELL_GAP if show_grid else 0
    grid_bg     = "#c4c6d2" if show_grid else "transparent"
    grid_border = "1px solid #b8bac8" if show_grid else "none"
    grid_pad    = CELL_GAP if show_grid else 0

    CSS = f"""
<style>
.pyr3-wrap{{font-family:sans-serif;background:#f7f8fc;border-radius:10px;
           padding:12px;user-select:none;box-sizing:border-box;
           border:1px solid #dde0ea;}}
.pyr3-banner{{text-align:center;font-size:0.78em;padding:6px 14px;border-radius:6px;
             margin-bottom:10px;}}
.pyr3-banner.none-mode{{background:#f3f4f8;color:#555;border:1px solid #d4d6e0;}}
.pyr3-banner.sel-mode{{background:#eaeaf8;color:#2f2f9e;border:1px solid #8888cc;}}
.pyr3-banner.seed-placed{{background:#eaf6ea;color:#256625;border:1px solid #7ec87e;}}
.pyr3-banner.gen-mode{{background:#eaf4fe;color:#0a4a80;border:1px solid #88bbee;}}
.pyr3-body{{display:flex;gap:16px;align-items:flex-start;width:100%;box-sizing:border-box;}}
.pyr3-cols{{display:flex;gap:10px;align-items:flex-start;{cols_overflow}}}
.pyr3-col{{display:flex;flex-direction:column;gap:0;align-items:center;flex-shrink:0;}}
.pyr3-zlabel{{color:#666;font-size:10px;text-align:center;line-height:1.4;
             padding:0 2px 4px;}}
.pyr3-zlabel b{{color:#222;font-size:11px;display:block;}}
.pyr3-grid{{display:grid;gap:{grid_gap}px;background:{grid_bg};
           border:{grid_border};border-radius:3px;padding:{grid_pad}px;
           box-sizing:border-box;}}
.pcell{{cursor:pointer;position:relative;overflow:hidden;box-sizing:border-box;
       width:{cell_px}px;height:{cell_px}px;transition:filter .1s;}}
.pcell:hover{{filter:brightness(0.85);}}
.pcell img{{width:100%;height:100%;object-fit:cover;display:block;pointer-events:none;}}
.pyr3-preview{{flex:1;min-width:0;background:#fff;border:1px solid #dde0ea;
              border-radius:8px;padding:14px;box-sizing:border-box;
              display:flex;flex-direction:column;gap:10px;align-self:stretch;}}
.pyr3-preview h4{{margin:0;font-size:11px;color:#444;font-weight:600;
                  text-transform:uppercase;letter-spacing:.06em;
                  border-bottom:1px solid #eee;padding-bottom:6px;}}
#pyr-prev-img{{width:100%;aspect-ratio:1/1;object-fit:cover;border-radius:6px;
               border:1px solid #d4d6e0;background:#e8eaef;display:block;
               image-rendering:pixelated;max-width:256px;margin:0 auto;}}
#pyr-prev-info{{font-size:11px;color:#444;line-height:1.8;margin:0;
               background:#f7f8fc;border-radius:6px;padding:8px 10px;}}
#pyr-prev-hint{{font-size:10px;color:#999;font-style:italic;margin:0;}}
.pyr3-legend{{display:flex;flex-wrap:wrap;gap:5px 10px;margin-top:10px;
             padding-top:8px;border-top:1px solid #d4d6e0;}}
.pleg{{display:flex;align-items:center;gap:3px;color:#555;font-size:9.5px;}}
.pleg-dot{{width:9px;height:9px;border-radius:2px;flex-shrink:0;}}
</style>"""

    parts = [CSS, '<div class="pyr3-wrap">']
    parts.append(f'<div class="pyr3-banner {banner_cls}">{banner_txt}</div>')
    parts.append('<div class="pyr3-body">')
    parts.append('<div class="pyr3-cols">')

    for z in range(z_min(), z_max() + 1):
        n = grid_n(z)
        parts.append('<div class="pyr3-col">')
        parts.append(
            f'<div class="pyr3-zlabel"><b>z={z}</b>{n}&times;{n}</div>'
        )
        parts.append(
            f'<div class="pyr3-grid" '
            f'style="grid-template-columns:repeat({n},{cell_px}px);'
            f'grid-template-rows:repeat({n},{cell_px}px);">'
        )

        for gy in range(n):
            for gx in range(n):
                rec   = pyramid.tiles.get((z, gx, gy))
                state = rec.state if rec else TileState.EMPTY
                bg    = STATE_BG.get(state, "#dde0ea")
                is_sel = (not generating) and (sel == (z, gx, gy))

                if not show_grid:
                    # Clean mosaic: only seed tiles get a red box, nothing else.
                    if state == TileState.SEED:
                        outline = "outline:2px solid #dd2222;z-index:5;"
                    else:
                        outline = ""
                else:
                    # Grid mode: seeds ALWAYS keep red outline.
                    # If a seed is also selected, add an inner blue ring so both
                    # states are visible (selected takes only the inner ring, not
                    # the outer red — so the red box never disappears).
                    if state == TileState.SEED:
                        if is_sel:
                            outline = ("outline:3px solid #dd2222;"
                                       "box-shadow:inset 0 0 0 2px #2244cc;z-index:5;")
                        else:
                            outline = "outline:2px solid #dd2222;z-index:5;"
                    elif is_sel:
                        outline = "outline:2px solid #2244cc;z-index:5;"
                    elif state != TileState.EMPTY:
                        bc = STATE_BORDER.get(state, "#888")
                        outline = f"outline:1px solid {bc};"
                    else:
                        outline = "outline:1px solid #c0c3d0;"

                inner, data_full = "", ""
                if rec and rec.image:
                    try:
                        b64_thumb = _b64(rec.image, thumb_px, thumb_px)
                        inner     = f'<img src="data:image/jpeg;base64,{b64_thumb}" />'
                        # The 512² ``data-full`` feeds the hover/click preview panel.
                        # It's by far the most expensive part of a render (and the
                        # whole payload), so only emit it when that panel exists.
                        # Deep pyramids drop the panel (show_preview=False) → skipping
                        # this turns an O(tiles²) streaming cost into O(tiles).
                        if show_preview:
                            b64_full = _b64(rec.image, 512, 512)
                            data_full = f' data-full="data:image/jpeg;base64,{b64_full}"'
                    except Exception:
                        pass

                title = (f"z={z} ({gx},{gy}) | {STATE_LABEL.get(state, state.value)}"
                         + (f" | sc={rec.self_consistency:.2f}"
                            if rec and rec.self_consistency > 0 else ""))
                parts.append(
                    f'<div class="pcell" '
                    f'data-z="{z}" data-gx="{gx}" data-gy="{gy}"{data_full} '
                    f'onclick="{cell_click_fn}(this)" '
                    f'style="background:{bg};{outline}" '
                    f'title="{title}">'
                    f'{inner}</div>'
                )

        parts.append('</div></div>')  # close grid + col

    parts.append('</div>')  # close cols

    # Current cell preview panel
    if sel:
        sel_rec  = pyramid.tiles.get(sel)
        state    = sel_rec.state if sel_rec else TileState.EMPTY
        lbl      = STATE_LABEL.get(state, state.value)
        sc_str   = (f" | sc={sel_rec.self_consistency:.2f}"
                    if sel_rec and sel_rec.self_consistency > 0 else "")
        info_txt = f"z={sel[0]} ({sel[1]},{sel[2]}) | {lbl}{sc_str}"
        bg_col   = STATE_BG.get(state, "#dde0ea")
        fg_col   = STATE_FG.get(state, "#333")
        border   = STATE_BORDER.get(state, "#aaa")

        if sel_rec and sel_rec.image:
            try:
                b64_prev = _b64(sel_rec.image, 512, 512)
                preview_html = (
                    f'<img id="pyr-prev-img" src="data:image/jpeg;base64,{b64_prev}" '
                    f'alt="" style="width:100%;aspect-ratio:1/1;max-width:256px;'
                    f'margin:0 auto;display:block;image-rendering:pixelated;'
                    f'border-radius:4px;" />'
                    f'<p id="pyr-prev-info" style="font-size:0.78rem;color:#334;'
                    f'margin:6px 0 0;word-break:break-all;">{info_txt}</p>'
                    f'<p id="pyr-prev-hint" style="font-size:0.72rem;color:#667;'
                    f'margin:4px 0 0;">256&times;256 image shown above.</p>'
                )
            except Exception:
                sel_rec = None

        if not (sel_rec and sel_rec.image):
            preview_html = (
                f'<div id="pyr-prev-img" style="width:100%;aspect-ratio:1/1;'
                f'max-width:256px;margin:0 auto;display:flex;align-items:center;'
                f'justify-content:center;background:{bg_col};border:2px solid {border};'
                f'border-radius:4px;font-size:0.9rem;font-weight:600;'
                f'color:{fg_col};">{lbl}</div>'
                f'<p id="pyr-prev-info" style="font-size:0.78rem;color:#334;'
                f'margin:6px 0 0;word-break:break-all;">{info_txt}</p>'
                f'<p id="pyr-prev-hint" style="font-size:0.72rem;color:#888;'
                f'margin:4px 0 0;">No image yet — generate or assign a seed.</p>'
            )
    else:
        preview_html = (
            '<div id="pyr-prev-img" style="width:100%;aspect-ratio:1/1;max-width:256px;'
            'margin:0 auto;display:flex;align-items:center;justify-content:center;'
            'background:#f0f1f5;border:1px dashed #bbb;border-radius:4px;'
            'font-size:0.8rem;color:#aaa;">click any cell<br>to inspect</div>'
            '<p id="pyr-prev-info" style="display:none;"></p>'
            '<p id="pyr-prev-hint" style="display:none;"></p>'
        )

    if show_preview:
        parts.append(f'<div class="pyr3-preview"><h4>Current Cell</h4>{preview_html}</div>')
    parts.append('</div>')  # close body

    # Legend — only the four tile kinds that actually appear: Seed (red box,
    # with or without grid lines), Superresolved, Outpainted, Downsampled.
    legend_items = [
        ("Seed",          'outline:2px solid #dd2222;background:#eee;'),
        ("Superresolved", f'background:{STATE_BG[TileState.HARD_SR]};'),
        ("Outpainted",    f'background:{STATE_BG[TileState.OUTPAINTED]};'),
        ("Downsampled",   f'background:{STATE_BG[TileState.DOWNSAMPLED]};'),
    ]
    parts.append('<div class="pyr3-legend">')
    for label, swatch in legend_items:
        parts.append(
            f'<div class="pleg">'
            f'<div class="pleg-dot" style="{swatch}"></div>'
            f'{label}</div>'
        )
    parts.append('</div>')
    parts.append('</div>')  # close pyr3-wrap
    return "".join(parts)


# Seed table HTML

def render_seed_table_html(pyramid: PyramidState) -> str:
    """Return an HTML table of all placed seeds, or a placeholder message."""
    seeds = sorted(
        [(z, gx, gy) for (z, gx, gy), r in pyramid.tiles.items()
         if r.state == TileState.SEED],
        key=lambda t: (t[0], t[1], t[2]),
    )
    if not seeds:
        return (
            '<p style="color:#666;font-size:0.82em;margin:6px 0 0;font-family:sans-serif;">'
            'No seeds yet — click a pyramid cell, then click the map to assign a satellite seed.</p>'
        )
    rows = "".join(
        f'<tr style="background:{"#f0f7f0" if i % 2 == 1 else "#e8f4e8"};">'
        f'<td style="padding:5px 10px;color:#267326;font-weight:600;">🌱 {i}</td>'
        f'<td style="padding:5px 10px;color:#333;">z={z}</td>'
        f'<td style="padding:5px 10px;color:#555;">({gx},&thinsp;{gy})</td>'
        f'</tr>'
        for i, (z, gx, gy) in enumerate(seeds, 1)
    )
    return (
        '<table style="width:100%;border-collapse:collapse;font-size:0.8em;'
        'font-family:sans-serif;border-radius:6px;overflow:hidden;border:1px solid #d4dcd4;">'
        '<thead><tr style="background:#d4e8d4;">'
        '<th style="padding:5px 10px;text-align:left;color:#444;font-weight:600;">Seed</th>'
        '<th style="padding:5px 10px;text-align:left;color:#444;font-weight:600;">Zoom</th>'
        '<th style="padding:5px 10px;text-align:left;color:#444;font-weight:600;">Grid (gx,&thinsp;gy)</th>'
        '</tr></thead>'
        f'<tbody>{rows}</tbody>'
        '</table>'
    )


# Leaflet map HTML + JS factories

def make_leaflet_html(map_wrap_class: str = "pyr-mapwrap") -> str:
    return (
        f'<div class="{map_wrap_class}" style="height:320px;width:100%;border-radius:8px;'
        f'overflow:hidden;border:1px solid #ccc;"></div>'
        f'<p style="margin:4px 0 0;color:#666;font-size:0.82em;">'
        f'<strong>Step 1:</strong> Click a pyramid cell &nbsp;→&nbsp; '
        f'<strong>Step 2:</strong> Navigate here and click a location to assign '
        f'that satellite image as seed. '
        f'The tile is always fetched at the <em>selected cell\'s zoom level</em>.</p>'
    )


def make_leaflet_js(
    map_wrap_class: str,
    map_pick_var: str,
    map_btn_id: str,
) -> str:
    """
    Leaflet initialisation JS injected via gr.HTML js_on_load.

    map_wrap_class : CSS class of the map div (e.g. "pyr-mapwrap")
    map_pick_var   : window global that stores the last click (e.g. "__pyrMapPick")
    map_btn_id     : elem_id of the off-screen map_apply_btn
    """
    return f"""
(function () {{
  const root = element.querySelector('.{map_wrap_class}') || element;
  root.innerHTML = '';
  window.{map_pick_var} = null;
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
  document.head.appendChild(link);
  const scr = document.createElement('script');
  scr.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
  scr.onload = function () {{
    const map = L.map(root, {{worldCopyJump: true}}).setView([20, 0], 2);
    L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
      {{ maxZoom: 19, attribution: '&copy; Esri' }}
    ).addTo(map);
    let mk = null;
    map.on('click', function (e) {{
      if (mk) map.removeLayer(mk);
      mk = L.circleMarker(e.latlng, {{radius:6, color:'#44aaff', fillOpacity:0.9}}).addTo(map);
      var rawLng = e.latlng.lng;
      var lng = ((rawLng + 180) % 360 + 360) % 360 - 180;
      window.{map_pick_var} = {{lat: e.latlng.lat, lng: lng, zoom: map.getZoom()}};
      function clickApply() {{
        var id = '{map_btn_id}';
        var el = document.getElementById(id);
        var btn = el && el.tagName==='BUTTON' ? el : (el ? el.querySelector('button') : null);
        if (!btn) btn = document.querySelector('#'+id+' button');
        if (btn) btn.click();
      }}
      setTimeout(clickApply, 200);
      setTimeout(clickApply, 600);
    }});
  }};
  document.head.appendChild(scr);
}})();
"""


def make_map_apply_js(map_pick_var: str, cell_pick_var: str | None = None) -> str:
    """
    JS preprocessor for map_apply_btn.click(): reads the map pick global and
    injects lat/lng/zoom as JSON.

    If `cell_pick_var` is given, the most recently clicked cell is also added as
    sel_z/sel_gx/sel_gy. Python-side `on_map_apply` should prefer these over
    `pyramid.selected` to avoid a state-sync race when the user clicks a new cell
    then the map before the cell-click handler has flushed `pyramid_state`.
    """
    cell_read = ""
    if cell_pick_var:
        cell_read = f"""
    var cell = window.{cell_pick_var};
    if (cell && typeof cell.z === 'number') {{
      payload.sel_z  = cell.z;
      payload.sel_gx = cell.gx;
      payload.sel_gy = cell.gy;
    }}"""
    return f"""
function(_unused_raw, pyramid_state_val) {{
  var p = window.{map_pick_var};
  if (p && typeof p.lat === 'number') {{
    var payload = {{lat: p.lat, lng: p.lng, zoom: p.zoom}};{cell_read}
    return [JSON.stringify(payload), pyramid_state_val];
  }}
  return ['{{}}', pyramid_state_val];
}}
"""


# Cell-click JS factories

def make_pyramid_init_js(
    click_fn_name: str,
    cell_pick_var: str,
    cell_btn_id: str,
    trigger_text: str,
) -> str:
    """
    One-time JS injected via gr.HTML js_on_load.
    Registers the global cell-click handler that:
      1. Updates the preview panel instantly in the browser
      2. Stores the click coordinates and fires the off-screen trigger button
    """
    return f"""
(function () {{
  window.{click_fn_name} = function(cell) {{
    var fullSrc = cell.getAttribute('data-full');
    var info    = cell.getAttribute('title') || '';

    var prevImg  = document.getElementById('pyr-prev-img');
    var prevInfo = document.getElementById('pyr-prev-info');
    var prevHint = document.getElementById('pyr-prev-hint');
    if (prevImg && fullSrc && prevImg.tagName === 'IMG') {{
      prevImg.src = fullSrc;
      prevImg.style.display = 'block';
      if (prevInfo) {{ prevInfo.innerHTML = info.replace(/\\s*\\|\\s*/g, '<br>'); prevInfo.style.display = 'block'; }}
      if (prevHint) prevHint.style.display = 'block';
    }} else if (prevInfo) {{
      prevInfo.innerHTML = info.replace(/\\s*\\|\\s*/g, '<br>');
      prevInfo.style.display = 'block';
    }}

    window.{cell_pick_var} = {{
      z:  parseInt(cell.dataset.z),
      gx: parseInt(cell.dataset.gx),
      gy: parseInt(cell.dataset.gy)
    }};

    function tryClick() {{
      var id  = '{cell_btn_id}';
      var el  = document.getElementById(id);
      var btn = el && el.tagName === 'BUTTON' ? el
               : (el ? el.querySelector('button') : null);
      if (!btn) btn = document.querySelector('#' + id + ' button');
      if (btn) {{ btn.click(); return; }}
      var nodes = document.querySelectorAll('button');
      for (var i = 0; i < nodes.length; i++) {{
        var t = (nodes[i].innerText || nodes[i].textContent || '');
        if (t.indexOf('{trigger_text}') >= 0) {{ nodes[i].click(); return; }}
      }}
    }}
    setTimeout(tryClick, 150);
    setTimeout(tryClick, 500);
  }};
}})();
"""


def make_cell_apply_js(cell_pick_var: str) -> str:
    """
    JS preprocessor for cell_apply_btn.click().
    Reads the cell click global and injects z/gx/gy as JSON.
    """
    return f"""
function(_unused_raw, pyramid_state_val) {{
  var p = window.{cell_pick_var};
  var j = '{{}}';
  if (p && typeof p.z === 'number')
    j = JSON.stringify({{z:p.z, gx:p.gx, gy:p.gy}});
  return [j, pyramid_state_val];
}}
"""
