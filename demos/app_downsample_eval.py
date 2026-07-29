#!/usr/bin/env python3
"""
Downsample evaluation demo: which interpolation method best represents a
high-res child tile when downsampled to its parent zoom level?

For a map-picked location it fetches the child tile at z_child (source) and the
real Esri tile at z_child - d (ground truth), crops the parent to the sub-region
matching the child, then downsamples the child with each method and scores it
against the GT crop. Metrics per method: PSNR, SSIM, LPIPS (AlexNet), DISTS.

Port: GRADIO_DS_PORT (default 7863)
"""
from __future__ import annotations

import json
import math
import os
import sys

_DEMOS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DEMOS)  # makes utils/ importable
sys.path.insert(0, os.path.join(_DEMOS, "..", "src"))

import gradio as gr
import numpy as np
import torch
from PIL import Image

from utils.genesis_common import (
    fetch_tile_pil,
    lonlat_to_tile_xy,
    nearest_training_zoom,
)

# --------------------------------------------------------------------------- #
# Lazy LPIPS singleton (AlexNet, fast, good for perceptual quality)
# --------------------------------------------------------------------------- #
_LPIPS_FN = None


def _get_lpips():
    global _LPIPS_FN
    if _LPIPS_FN is None:
        try:
            import lpips
            _LPIPS_FN = lpips.LPIPS(net="alex", verbose=False)
            _LPIPS_FN.eval()
        except ImportError:
            _LPIPS_FN = "unavailable"
    return _LPIPS_FN


# --------------------------------------------------------------------------- #
# Interpolation methods under evaluation
# --------------------------------------------------------------------------- #
METHODS = {
    "Nearest":  Image.Resampling.NEAREST,
    "Bilinear": Image.Resampling.BILINEAR,
    "Bicubic":  Image.Resampling.BICUBIC,
    "Lanczos":  Image.Resampling.LANCZOS,
}

# --------------------------------------------------------------------------- #
# Map widget
# --------------------------------------------------------------------------- #
LEAFLET_HTML = """
<div class="ds-eval-map" style="height:420px;width:100%;border-radius:8px;overflow:hidden;border:1px solid #555;"></div>
<p style="margin:6px 0 0;color:#666;font-size:0.85em;">Click to pick location. The child tile at the chosen zoom will be fetched as the high-res source.</p>
"""

_APPLY_BTN_ID = "ds_eval_apply_pick"
LEAFLET_JS = r"""
(function () {
  const root = element.querySelector('.ds-eval-map') || element;
  root.innerHTML = '';
  window.__dsEvalPick = null;
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
    let mk = null;
    map.on('click', function (e) {
      if (mk) map.removeLayer(mk);
      mk = L.circleMarker(e.latlng, { radius: 6, color: '#44ddff', fillOpacity: 0.9 }).addTo(map);
      window.__dsEvalPick = { lat: e.latlng.lat, lng: e.latlng.lng, zoom: map.getZoom() };
      function tryClick() {
        var id = 'DS_APPLY_ID';
        var el = document.getElementById(id);
        var btn = el && el.tagName === 'BUTTON' ? el : (el ? el.querySelector('button') : null);
        if (!btn) btn = document.querySelector('#' + id + ' button');
        if (btn) btn.click();
      }
      setTimeout(tryClick, 150);
      setTimeout(tryClick, 500);
    });
  };
  document.head.appendChild(scr);
})();
""".replace("DS_APPLY_ID", _APPLY_BTN_ID)

APPLY_JS = """
function(_raw, z_cur) {
  var p = window.__dsEvalPick;
  if (p && typeof p.lat === 'number') {
    var zc = Math.max(11, Math.min(19, Math.round(p.zoom)));
    return [JSON.stringify({lat: p.lat, lng: p.lng, zoom: zc}), zc];
  }
  return [_raw, z_cur];
}
"""


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _to_np(im: Image.Image) -> np.ndarray:
    return np.array(im.convert("RGB"), dtype=np.float32) / 255.0


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def ssim_simple(a: np.ndarray, b: np.ndarray) -> float:
    """Single-scale mean SSIM per channel."""
    c1, c2 = (0.01 ** 2), (0.03 ** 2)
    scores = []
    for c in range(3):
        x, y = a[..., c], b[..., c]
        mx, my = x.mean(), y.mean()
        sx = float(x.std()) ** 2
        sy = float(y.std()) ** 2
        sxy = float(np.mean((x - mx) * (y - my)))
        num = (2 * mx * my + c1) * (2 * sxy + c2)
        den = (mx ** 2 + my ** 2 + c1) * (sx + sy + c2)
        scores.append(num / (den + 1e-8))
    return float(np.mean(scores))


def lpips_score(gt: Image.Image, pred: Image.Image) -> str:
    fn = _get_lpips()
    if fn == "unavailable" or fn is None:
        return "N/A"
    if gt.size != pred.size:
        pred = pred.resize(gt.size, Image.Resampling.BILINEAR)
    a = torch.from_numpy(_to_np(gt)).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    b = torch.from_numpy(_to_np(pred)).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    with torch.no_grad():
        v = fn(a, b).item()
    return f"{v:.4f}"


def dists_score(gt: Image.Image, pred: Image.Image) -> str:
    try:
        from DISTS_pytorch import DISTS
        if not hasattr(dists_score, "_fn"):
            dists_score._fn = DISTS()
        if gt.size != pred.size:
            pred = pred.resize(gt.size, Image.Resampling.BILINEAR)
        a = torch.from_numpy(_to_np(gt)).permute(2, 0, 1).unsqueeze(0)
        b = torch.from_numpy(_to_np(pred)).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            return f"{dists_score._fn(a, b).item():.4f}"
    except Exception:
        return "N/A"


# --------------------------------------------------------------------------- #
# Core evaluation
# --------------------------------------------------------------------------- #
def run_eval(pick_json: str, zc_raw: int, d: int) -> tuple:
    """Score each downsampling method (child at zc → parent_size = 256/2^d)
    against the real Esri parent crop at zp = zc - d as ground truth."""
    zc = int(max(11, min(19, round(float(zc_raw)))))
    d  = int(max(1, min(4, d)))
    zp = max(10, zc - d)
    actual_d = zc - zp

    try:
        p = json.loads(pick_json) if pick_json else {}
    except json.JSONDecodeError:
        p = {}
    if not p or "lat" not in p:
        return (None,) * (2 + len(METHODS)) + ("*Click map first, then hit Run.*",)

    lat, lon = float(p["lat"]), float(p["lng"])
    xc, yc = lonlat_to_tile_xy(lon, lat, zc)

    try:
        child_src = fetch_tile_pil(zc, xc, yc).resize((256, 256), Image.Resampling.LANCZOS)
    except Exception as e:
        return (None,) * (2 + len(METHODS)) + (f"Child tile fetch error: {e}",)

    xp, yp = xc >> actual_d, yc >> actual_d
    try:
        parent_full = fetch_tile_pil(zp, xp, yp).resize((256, 256), Image.Resampling.LANCZOS)
    except Exception as e:
        return (None,) * (2 + len(METHODS)) + (f"Parent tile fetch error: {e}",)

    # Crop the Esri parent to the sub-region that matches the child tile.
    # child (xc, yc) at zc occupies quadrant (xc % 2^d, yc % 2^d) within the ancestor.
    parent_size = max(16, 256 // (2 ** actual_d))
    qx = xc % (2 ** actual_d)
    qy = yc % (2 ** actual_d)
    cx0 = qx * parent_size
    cy0 = qy * parent_size
    parent_crop = parent_full.crop((cx0, cy0, cx0 + parent_size, cy0 + parent_size))
    gt_np = _to_np(parent_crop)

    # Upscale GT crop to 256x256 with nearest for display (no extra blurring)
    parent_crop_disp = parent_crop.resize((256, 256), Image.Resampling.NEAREST)

    # Downsample child with each method, measure vs GT crop
    results = {}
    ds_imgs = {}
    for name, resamp in METHODS.items():
        ds = child_src.resize((parent_size, parent_size), resamp)
        arr = _to_np(ds)
        results[name] = {
            "psnr":  psnr(gt_np, arr),
            "ssim":  ssim_simple(gt_np, arr),
            "lpips": lpips_score(parent_crop, ds),
            "dists": dists_score(parent_crop, ds),
        }
        # Show at 256x256 with nearest so we see actual downsampled pixels
        ds_imgs[name] = ds.resize((256, 256), Image.Resampling.NEAREST)

    # Side-by-side strips: downsampled child (left) | Esri GT crop (right)
    gap = 8
    rows = []
    for name in METHODS:
        row = Image.new("RGB", (256 + gap + 256, 256), (30, 30, 34))
        row.paste(ds_imgs[name], (0, 0))
        row.paste(parent_crop_disp, (256 + gap, 0))
        rows.append(row)

    # Metrics table
    lines = [
        f"**Source: child z={zc}**  \u2192  downsample to z={zp} "
        f"(d={actual_d}, target size={parent_size}\u00b2)  |  GT = Esri parent crop\n",
        "| Method | PSNR \u2191 | SSIM \u2191 | LPIPS \u2193 | DISTS \u2193 |",
        "|--------|--------|--------|---------|---------|",
    ]
    for name, m in sorted(results.items(), key=lambda x: -x[1]["psnr"]):
        lines.append(
            f"| {name} | {m['psnr']:.2f} dB | {m['ssim']:.4f} | {m['lpips']} | {m['dists']} |"
        )
    lines.append(
        "\n*Left = child tile downsampled by each method. "
        "Right = real Esri tile at parent zoom (GT). "
        "Both shown at 256\u00b2 with nearest-neighbour to preserve pixel accuracy. "
        "Higher PSNR/SSIM = better. Lower LPIPS/DISTS = better perceptually.*"
    )

    return (
        child_src,
        parent_full,
        rows[0],   # Nearest
        rows[1],   # Bilinear
        rows[2],   # Bicubic
        rows[3],   # Lanczos
        "\n".join(lines),
    )


# --------------------------------------------------------------------------- #
# Gradio UI
# --------------------------------------------------------------------------- #
def _apply(raw: str, z):
    try:
        p = json.loads(raw) if raw.strip().startswith("{") else {}
    except Exception:
        p = {}
    if not p or "lat" not in p:
        return raw, z, "*Click map first.*"
    lat, lon = float(p["lat"]), float(p["lng"])
    z2 = int(max(11, min(19, round(float(p.get("zoom", z))))))
    return raw, z2, f"Picked lat={lat:.5f} lon={lon:.5f} z={z2} \u2014 ready."


def build_ui():
    with gr.Blocks(title="Downsample evaluation") as demo:
        gr.Markdown(
            "## Interpolation downsampling evaluation\n"
            "**Question:** which interpolation method most faithfully represents a "
            "high-res tile when downsampled to a lower zoom level?\n\n"
            "**Method:** fetch child tile at `z_child` (source), fetch real Esri tile "
            "at `z_child \u2212 d` (ground truth). Crop the Esri parent to the sub-region "
            "matching the child, then downsample the child with each method and compare "
            "to that GT crop at native parent resolution."
        )
        pick_raw  = gr.Textbox(value="{}", visible=False)
        gr.HTML(
            value=LEAFLET_HTML,
            html_template="${value}",
            js_on_load=LEAFLET_JS,
            container=True,
            min_height=440,
        )
        with gr.Row():
            zc = gr.Slider(11, 19, value=16, step=1, label="Child zoom z_child")
            d  = gr.Slider(1, 4, value=1, step=1,
                           label="Zoom steps down d  (parent = z_child \u2212 d)")
        pick_msg  = gr.Markdown("*Click map, then run.*")
        apply_btn = gr.Button("Refresh from last map click", elem_id=_APPLY_BTN_ID)
        run_btn   = gr.Button("Run evaluation", variant="primary")

        apply_btn.click(
            fn=_apply,
            inputs=[pick_raw, zc],
            outputs=[pick_raw, zc, pick_msg],
            js=APPLY_JS,
        )

        with gr.Row():
            src_img    = gr.Image(type="pil", label="Source: child tile at z_child (256\u00b2)")
            parent_img = gr.Image(type="pil", label="Esri parent tile at z_child\u2212d (full, for reference)")
        gr.Markdown(
            "### Side-by-side: downsampled child (left) | Esri parent crop GT (right)\n"
            "Both shown at 256\u00b2 with nearest-neighbour \u2014 no extra blurring."
        )
        with gr.Row():
            nn_img = gr.Image(type="pil", label="Nearest \u2193")
            bl_img = gr.Image(type="pil", label="Bilinear \u2193")
        with gr.Row():
            bc_img = gr.Image(type="pil", label="Bicubic \u2193")
            lz_img = gr.Image(type="pil", label="Lanczos \u2193 \u2b50")
        metrics_md = gr.Markdown("")

        run_btn.click(
            run_eval,
            inputs=[pick_raw, zc, d],
            outputs=[src_img, parent_img, nn_img, bl_img, bc_img, lz_img, metrics_md],
        )

    return demo


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_DS_PORT", 7863))
    build_ui().launch(server_name="0.0.0.0", server_port=port, share=True)
