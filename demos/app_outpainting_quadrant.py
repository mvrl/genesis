#!/usr/bin/env python3
"""
Gradio demo: Genesis outpainting — quadrant context (1–3 known quadrants).

Click the satellite map to fetch a 256×256 tile, select 1–3 quadrants as known
context (the rest become the hole), then Generate to outpaint the hole.

Run: python app_outpainting_quadrant.py  (port: $GRADIO_OP_PORT, default 7862).
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import sys
from datetime import datetime

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
    model_arch,
    esri_viewer_url,
    fetch_tile_at_zoom,
    inference_device,
    inference_device_label,
    load_op_denoiser,
    nearest_training_zoom,
    outpaint_context_white_holes_preview,
    run_outpainting,
)

_DEVICE   = inference_device()
_OP_CACHE: dict = {}
_LOGS_DIR = os.path.join(_DEMOS, "logs", "op_quadrant_results")

_APPLY_BTN_ID = "genesis_op2_apply_pick"

# ---------------------------------------------------------------------------
# Quadrant definitions  (pixel coords for 256×256 tile)
# ---------------------------------------------------------------------------

HALF = 128  # half of 256

QUADRANT_COORDS = {
    "UL": (0,    0,    HALF, HALF),
    "UR": (HALF, 0,    256,  HALF),
    "LL": (0,    HALF, HALF, 256),
    "LR": (HALF, HALF, 256,  256),
}

QUADRANT_CHOICES = [
    ("↖ UL — upper-left",  "UL"),
    ("↗ UR — upper-right", "UR"),
    ("↙ LL — lower-left",  "LL"),
    ("↘ LR — lower-right", "LR"),
]

# Center of each quadrant in display pixels (for labels; display is 400×400)
_LABEL_POS_NORM = {
    "UL": (0.25, 0.25),
    "UR": (0.75, 0.25),
    "LL": (0.25, 0.75),
    "LR": (0.75, 0.75),
}

# ---------------------------------------------------------------------------
# Map HTML / JS
# ---------------------------------------------------------------------------

LEAFLET_HTML = """
<div class="genesis-op2-mapwrap" style="height:500px;width:100%;border-radius:8px;overflow:hidden;border:1px solid #444;"></div>
<p style="margin:8px 0 0;color:#666;font-size:0.9em;">
  <strong>Click</strong> the map to set location. Tile fetches automatically; or press <strong>Refresh</strong> manually.
</p>
"""

LEAFLET_JS = r"""
(function () {
  const root = element.querySelector('.genesis-op2-mapwrap') || element;
  root.innerHTML = '';
  window.__genesisOP2Pick = null;
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
  document.head.appendChild(link);
  const scr = document.createElement('script');
  scr.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
  scr.onload = function () {
    const map = L.map(root).setView([20, 0], 2);
    L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 19, attribution: '&copy; Esri' }
    ).addTo(map);
    let marker = null;
    map.on('click', function (e) {
      if (marker) { map.removeLayer(marker); }
      marker = L.circleMarker(e.latlng, { radius: 6, color: '#44aaff', fillOpacity: 0.9 }).addTo(map);
      window.__genesisOP2Pick = { lat: e.latlng.lat, lng: e.latlng.lng, zoom: map.getZoom() };
      function clickApply() {
        var id = 'APPLY_PICK_BTN_ID';
        var wrap = document.getElementById(id);
        var btn = wrap && wrap.tagName === 'BUTTON' ? wrap : (wrap ? wrap.querySelector('button') : null);
        if (!btn) btn = document.querySelector('#' + id + ' button');
        if (!btn) {
          var nodes = document.querySelectorAll('button');
          for (var i = 0; i < nodes.length; i++) {
            var t = (nodes[i].innerText || nodes[i].textContent || '');
            if (t.indexOf('Refresh') >= 0) { nodes[i].click(); return; }
          }
        }
        if (btn) btn.click();
      }
      setTimeout(clickApply, 200);
      setTimeout(clickApply, 600);
    });
  };
  document.head.appendChild(scr);
})();
""".replace("APPLY_PICK_BTN_ID", _APPLY_BTN_ID)

APPLY_JS = """
function (_dummy) {
  var p = window.__genesisOP2Pick;
  return [p && typeof p.lat === 'number'
    ? JSON.stringify({ lat: p.lat, lng: p.lng, zoom: p.zoom })
    : '{}'];
}
"""

# ---------------------------------------------------------------------------
# Mask / preview helpers
# ---------------------------------------------------------------------------

def build_quadrant_mask_pil(known: list[str], size: int = 256) -> Image.Image:
    """
    L-mode mask: 0 = known context, 255 = hole to synthesize.
    Quadrants in `known` are context; all others are holes.
    """
    half = size // 2
    quads = {
        "UL": (0,    0,    half, half),
        "UR": (half, 0,    size, half),
        "LL": (0,    half, half, size),
        "LR": (half, half, size, size),
    }
    mask = Image.new("L", (size, size), 255)   # start all-hole
    draw = ImageDraw.Draw(mask)
    for q in known:
        if q in quads:
            x1, y1, x2, y2 = quads[q]
            draw.rectangle([x1, y1, x2 - 1, y2 - 1], fill=0)
    return mask


def make_quadrant_preview(tile: Image.Image, known: list[str], size: int = 256) -> Image.Image:
    """
    Tile with known quadrants shown in colour, hole quadrants in white.
    Draws the quadrant grid and labels.
    """
    tile_r = tile.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    mask   = build_quadrant_mask_pil(known, size)
    img    = outpaint_context_white_holes_preview(tile_r, mask, size=size)

    draw = ImageDraw.Draw(img)
    half = size // 2

    # Grid lines
    draw.line([(half, 0), (half, size - 1)], fill=(90, 90, 90), width=1)
    draw.line([(0, half), (size - 1, half)], fill=(90, 90, 90), width=1)

    # Quadrant labels
    centres = {
        "UL": (half // 2,        half // 2),
        "UR": (half + half // 2, half // 2),
        "LL": (half // 2,        half + half // 2),
        "LR": (half + half // 2, half + half // 2),
    }
    for q, (cx, cy) in centres.items():
        is_known = q in known
        color    = (40, 200, 80) if is_known else (220, 70, 70)
        symbol   = "✓" if is_known else "?"
        draw.text((cx - 12, cy - 8), f"{q} {symbol}", fill=color)

    return img


def _tile_to_html(tile: Image.Image) -> str:
    """Encode PIL tile as a hidden <img> data-URL (used for the canvas fallback in OP1)."""
    buf = io.BytesIO()
    tile.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img id="genesis-tile-data" src="data:image/png;base64,{b64}" style="display:none">'


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------

def get_model(model_key: str, ckpt_override: str):
    # Local existence is NOT pre-checked here — load_op_denoiser() auto-pulls
    # the ckpt from Google Drive if it isn't already on disk.
    ckpt = (ckpt_override or "").strip() or DEFAULT_OP_CKPT.get(model_key, "")
    if not ckpt:
        raise FileNotFoundError(
            f"No outpainting checkpoint path resolved for {model_key!r}. "
            f"Set GENESIS_CKPT_BASE or paste a full path into the 'Optional checkpoint path' textbox."
        )
    # Force dino_weights="" to skip the DINO branch: the trained config is
    # no-DINO and its DINO-branch weights were never updated, so feeding real
    # DINO features at inference would contaminate the output.
    cache_key = (ckpt, model_key, False)
    if cache_key not in _OP_CACHE:
        denoiser, _ = load_op_denoiser(
            ckpt, model_arch(model_key), dino_weights="", device=_DEVICE
        )
        _OP_CACHE[cache_key] = denoiser
    return _OP_CACHE[cache_key]


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def compute_op_metrics(out: Image.Image, base: Image.Image, mask: Image.Image) -> dict:
    """
    Outpainting quality metrics. PSNR/SSIM/boundary-diff/sharpness are computed
    on the hole region only; LPIPS/DISTS run on the full 256×256 image (context
    pixels match out vs base, so the score is still hole-driven).
    """
    from scipy.ndimage import convolve, binary_erosion
    import torch

    hole  = np.array(mask.convert("L")) > 127
    out_a = np.array(out.convert("RGB"),  dtype=np.float64)
    base_a = np.array(base.convert("RGB"), dtype=np.float64)

    boundary  = hole & ~binary_erosion(hole, iterations=4)
    boundary3 = boundary[:, :, None].repeat(3, axis=2)
    diff_bnd  = np.abs(out_a - base_a)[boundary3].reshape(-1, 3)
    boundary_diff = float(diff_bnd.mean()) if boundary3.any() else 0.0

    kern = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    gray = np.array(out.convert("L"), dtype=np.float64)
    sharpness = float(np.var(convolve(gray, kern)[hole]))

    metrics = {"boundary_diff": boundary_diff, "sharpness": sharpness}

    # PSNR and SSIM on hole pixels
    hole3 = hole[:, :, None]
    sr_h = out_a[hole3.repeat(3, axis=2)].reshape(-1, 3).flatten()
    gt_h = base_a[hole3.repeat(3, axis=2)].reshape(-1, 3).flatten()
    mse  = float(np.mean((sr_h - gt_h) ** 2))
    metrics["psnr"] = 10 * math.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")
    mu_s, mu_g = sr_h.mean(), gt_h.mean()
    sig_s, sig_g = sr_h.std(), gt_h.std()
    sig_xy = float(np.mean((sr_h - mu_s) * (gt_h - mu_g)))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    metrics["ssim"] = float(
        ((2 * mu_s * mu_g + c1) * (2 * sig_xy + c2))
        / ((mu_s ** 2 + mu_g ** 2 + c1) * (sig_s ** 2 + sig_g ** 2 + c2))
    )

    def _to_tensor(pil: Image.Image) -> torch.Tensor:
        arr = np.array(pil.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    out_t  = _to_tensor(out)
    base_t = _to_tensor(base)

    try:
        import lpips
        if not hasattr(compute_op_metrics, "_lpips_fn"):
            compute_op_metrics._lpips_fn = lpips.LPIPS(net="alex", verbose=False)
        with torch.no_grad():
            metrics["lpips"] = float(compute_op_metrics._lpips_fn(
                out_t * 2 - 1, base_t * 2 - 1).item())
    except Exception:
        pass

    try:
        from DISTS_pytorch import DISTS
        if not hasattr(compute_op_metrics, "_dists_fn"):
            compute_op_metrics._dists_fn = DISTS()
        with torch.no_grad():
            metrics["dists"] = float(compute_op_metrics._dists_fn(out_t, base_t).item())
    except Exception:
        pass

    return metrics


def format_op_metrics(m: dict) -> str:
    parts = []
    if "psnr" in m:
        parts.append(f"**PSNR ↑** (hole): `{m['psnr']:.2f} dB`")
    if "ssim" in m:
        parts.append(f"**SSIM ↑** (hole): `{m['ssim']:.4f}`")
    if "lpips" in m:
        parts.append(f"**LPIPS ↓** (vs GT): `{m['lpips']:.4f}`")
    if "dists" in m:
        parts.append(f"**DISTS ↓** (vs GT): `{m['dists']:.4f}`")
    parts.append(f"**Boundary diff ↓**: `{m['boundary_diff']:.2f}`")
    parts.append(f"**Sharpness ↑** (hole): `{m['sharpness']:.1f}`")
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

_DEFAULT_KNOWN = ["UL"]   # sensible default shown on tile load


def apply_from_map(pick_json: str):
    """Fetch tile on map click / Refresh. Returns tile state + initial preview."""
    try:
        o = json.loads(pick_json) if pick_json else {}
    except json.JSONDecodeError:
        o = {}
    if not o or "lat" not in o:
        return gr.update(), gr.update(), gr.update(), "*Click the map to set a location.*", "", "", ""
    try:
        lat      = float(o["lat"])
        lon      = float(o["lng"])
        map_zoom = float(o["zoom"])
    except (KeyError, TypeError, ValueError):
        return gr.update(), gr.update(), gr.update(), "*Invalid map click.*", "", "", ""

    z = nearest_training_zoom(int(round(map_zoom)))
    try:
        tile = fetch_tile_at_zoom(lon, lat, z, 256)
    except Exception as e:
        return gr.update(), gr.update(), gr.update(), f"*Tile error:* {e}", "", "", ""

    pick    = {"lat": lat, "lon": lon, "map_zoom": map_zoom, "z": z}
    link    = esri_viewer_url(lon, lat, z)
    info    = (
        f"**Tile z={z}** · [ArcGIS viewer]({link})\n\n"
        "Tile loaded. **Select known quadrants** below (green ✓ = context, red ? = hole), then **Generate**."
    )
    preview = make_quadrant_preview(tile, _DEFAULT_KNOWN)
    return tile, pick, preview, info, f"{lat:.6f}", f"{lon:.6f}", f"{map_zoom:.2f}"


def update_preview(known: list[str], tile: Image.Image | None, pick: dict | None):
    """Live preview update when the user changes quadrant selection."""
    if tile is None or pick is None:
        return gr.update(), gr.update()
    n_known = len(known)
    n_hole  = 4 - n_known
    if n_known == 0:
        return gr.update(), "**Select at least 1 quadrant** as context."
    if n_hole == 0:
        return gr.update(), "**Uncheck at least 1 quadrant** to create a hole."
    prev = make_quadrant_preview(tile, known)
    info = (
        f"**{n_known} known** quadrant(s) · **{n_hole} hole(s)** to generate. "
        f"Green ✓ = context · Red ? = will be synthesized."
    )
    return prev, info


def generate_op(
    pick: dict | None,
    tile: Image.Image | None,
    known: list[str],
    model_key: str,
    ckpt_override: str,
    steps: float,
    cfg: float,
):
    _hide = gr.update(visible=False)
    if not pick or "z" not in pick:
        return None, "**Click the map** to fetch a tile first.", _hide
    if tile is None:
        return None, "**Tile not loaded** — click the map and wait for the fetch.", _hide
    if not known:
        return None, "**Select at least 1 quadrant** as context.", _hide
    if len(known) == 4:
        return None, "**Uncheck at least 1 quadrant** to create a hole.", _hide

    try:
        denoiser = get_model(model_key, ckpt_override)
    except Exception as e:
        return None, f"**Load error:** {e}", _hide

    mask = build_quadrant_mask_pil(known, size=256)

    # Engine reads self.steps / self.cfg_scale (snapshotted at __init__) — set
    # the attributes; mutating args.* after load has no effect.
    denoiser.steps = int(steps)
    denoiser.cfg_scale = float(cfg)

    try:
        out = run_outpainting(denoiser, tile.convert("RGB"), mask, _DEVICE, tile_zoom=pick.get("z"))
    except Exception as e:
        import traceback
        return None, f"**Inference error:** {e}\n```\n{traceback.format_exc()}\n```", _hide

    # Output strip is generated | ground truth (256 each).
    gap       = 24
    gap_color = (238, 238, 242)
    blend     = Image.new("RGB", (256 * 2 + gap, 256), gap_color)
    blend.paste(out.convert("RGB"),                     (0, 0))
    blend.paste(tile.convert("RGB").resize((256, 256)), (256 + gap, 0))

    metrics  = compute_op_metrics(out.convert("RGB"), tile.convert("RGB").resize((256, 256)), mask)
    n_hole   = 4 - len(known)
    hole_str = ", ".join(q for q in ["UL", "UR", "LL", "LR"] if q not in known)
    msg = (
        f"**Done.** Known: {', '.join(known)} · Generated: {hole_str} "
        f"({n_hole} quadrant{'s' if n_hole > 1 else ''}) · z={pick['z']}.\n\n"
        f"Left: **generated** · Right: **ground truth**.\n\n"
        f"{format_op_metrics(metrics)}"
    )

    # Save the result strip for download (generated | ground truth).
    try:
        os.makedirs(_LOGS_DIR, exist_ok=True)
        out_path = os.path.join(_LOGS_DIR, f"op_quadrant_z{pick['z']}_{datetime.now():%Y%m%d_%H%M%S}.png")
        blend.save(out_path)
        dl = gr.update(value=out_path, visible=True)
    except Exception:
        dl = _hide
    return blend, msg, dl


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="Genesis Outpainting — Quadrant Context") as demo:
        gr.Markdown(
            "## Genesis Outpainting — Quadrant Context\n"
            "The tile is split into four **128×128 quadrants**. "
            "Select **1–3 quadrants as context** (green ✓); the rest will be outpainted.\n\n"
            "**1.** Click the map → tile fetches. "
            "**2.** Check known quadrants. "
            "**3.** Hit **Generate**."
        )

        pick_state = gr.State(value=None)
        tile_state = gr.State(value=None)   # PIL Image, lives in server memory

        _dummy_tb = gr.Textbox(value="{}", visible=False)

        # --- map ---
        gr.HTML(
            value=LEAFLET_HTML,
            html_template="${value}",
            js_on_load=LEAFLET_JS,
            container=True,
            min_height=520,
        )

        with gr.Row():
            lat_tb  = gr.Textbox(label="Latitude",          interactive=False, scale=1)
            lon_tb  = gr.Textbox(label="Longitude",         interactive=False, scale=1)
            zoom_tb = gr.Textbox(label="Map zoom at click", interactive=False, scale=1)

        use_loc = gr.Button(
            "Refresh from last map click", variant="secondary", elem_id=_APPLY_BTN_ID
        )
        info = gr.Markdown("*Click the map…*")

        # --- quadrant selector + preview side by side ---
        with gr.Row():
            with gr.Column(scale=1):
                known_qgrp = gr.CheckboxGroup(
                    choices=QUADRANT_CHOICES,
                    value=_DEFAULT_KNOWN,
                    label="Known quadrants (context)",
                    info="Select 1–3 quadrants. Unchecked quadrants will be synthesized.",
                )
                gr.Markdown(
                    "<small style='color:#888;'>"
                    "Green ✓ = model sees these pixels &nbsp;|&nbsp; "
                    "Red ? = model must generate these pixels"
                    "</small>"
                )
            with gr.Column(scale=1):
                preview = gr.Image(
                    type="pil",
                    label="Context (coloured) + hole (white)",
                    interactive=False,
                )

        # --- model settings ---
        with gr.Accordion("Model settings", open=False):
            model_key     = gr.Dropdown(
                OP_MODEL_KEYS, value=OP_KEY_B16_MAIN, label="Architecture",
                info="Main run, locked config (no DINO, no time weighting). "
                     "Trained on the quadrant mask curriculum. Pick B16 for speed, H16 for capacity.",
            )
            ckpt_override = gr.Textbox(
                label="Optional checkpoint path override",
                placeholder="Default: op-new-stage3-step0800000.ckpt from Google Drive (auto-pulled)",
            )
            steps    = gr.Slider(10, 100, value=50, step=5,   label="Sampling steps")
            cfg      = gr.Slider(1.0, 4.0, value=1.0, step=0.5, label="CFG scale")

        gen_btn = gr.Button("Generate", variant="primary")
        out_img = gr.Image(type="pil", label="Generated | Ground truth")
        out_msg = gr.Markdown("")
        download_btn = gr.DownloadButton(
            label="⬇ Download result", visible=False, variant="secondary",
        )

        # ----------------------------------------------------------------
        # Event wiring
        # ----------------------------------------------------------------
        map_outs = [tile_state, pick_state, preview, info, lat_tb, lon_tb, zoom_tb]

        use_loc.click(
            apply_from_map,
            inputs=[_dummy_tb],
            outputs=map_outs,
            js=APPLY_JS,
        )

        known_qgrp.change(
            update_preview,
            inputs=[known_qgrp, tile_state, pick_state],
            outputs=[preview, info],
        )

        gen_btn.click(
            generate_op,
            inputs=[pick_state, tile_state, known_qgrp,
                    model_key, ckpt_override, steps, cfg],
            outputs=[out_img, out_msg, download_btn],
        )

    return demo


if __name__ == "__main__":
    _port = int(os.environ.get("GRADIO_OP_PORT", 7862))
    print(f"[Genesis outpainting quadrant] Inference device: {inference_device_label(_DEVICE)}")
    build_ui().launch(server_name="0.0.0.0", server_port=_port, share=True)
