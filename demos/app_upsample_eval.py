#!/usr/bin/env python3
"""
Upsample evaluation demo — bicubic baseline vs Esri GT across zoom levels.

For a chosen parent tile (256×256 at z_parent), shows source plus three
[bicubic | GT] side-by-side panels at d=1/2/3 (512/1024/2048 px), scored
with PSNR / SSIM / LPIPS. Gradio panel heights grow with the level.

Run: python app_upsample_eval.py  (port: GRADIO_US_PORT, default 7864).
"""
from __future__ import annotations

import json
import math
import os
import sys

_DEMOS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DEMOS)
sys.path.insert(0, os.path.join(_DEMOS, "..", "src"))

import gradio as gr
import numpy as np
import torch
from PIL import Image

from utils.genesis_common import fetch_tile_pil, lonlat_to_tile_xy

# Lazy LPIPS singleton
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


# Metrics
def _to_np(im: Image.Image) -> np.ndarray:
    return np.array(im.convert("RGB"), dtype=np.float32) / 255.0


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 100.0 if mse < 1e-10 else 10.0 * math.log10(1.0 / mse)


def ssim_simple(a: np.ndarray, b: np.ndarray) -> float:
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    scores = []
    for c in range(3):
        x, y = a[..., c], b[..., c]
        mx, my = x.mean(), y.mean()
        sx, sy = float(x.std()) ** 2, float(y.std()) ** 2
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
        return f"{fn(a, b).item():.4f}"


# GT child mosaic
def fetch_child_mosaic(zp: int, xp: int, yp: int, d: int) -> Image.Image:
    """Stitch 2^d × 2^d Esri child tiles into a (256·2^d)² mosaic."""
    n     = 2 ** d
    tw    = 256
    mosaic = Image.new("RGB", (tw * n, tw * n))
    for dy in range(n):
        for dx in range(n):
            tile = fetch_tile_pil(zp + d, xp * n + dx, yp * n + dy)
            mosaic.paste(tile.resize((tw, tw), Image.Resampling.LANCZOS),
                         (dx * tw, dy * tw))
    return mosaic


# Side-by-side composite builder
def make_side_by_side(left: Image.Image, right: Image.Image,
                      gap: int = 12) -> Image.Image:
    """Two same-size squares side by side (bicubic left, GT right) with a thin
    separator. Result is (2·w + gap) × h."""
    w, h = left.size
    out = Image.new("RGB", (w + gap + w, h), (220, 220, 220))
    out.paste(left,  (0,       0))
    out.paste(right, (w + gap, 0))
    return out


# Map widget
LEAFLET_HTML = """
<div class="us-eval-map" style="height:420px;width:100%;border-radius:8px;
     overflow:hidden;border:1px solid #ccc;"></div>
<p style="margin:6px 0 0;color:#666;font-size:0.85em;">
  Click a location. The parent tile at the chosen zoom will be the low-res source.
</p>
"""

_APPLY_BTN_ID = "us_eval_apply_pick"
LEAFLET_JS = r"""
(function () {
  const root = element.querySelector('.us-eval-map') || element;
  root.innerHTML = '';
  window.__usEvalPick = null;
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
  document.head.appendChild(link);
  const scr = document.createElement('script');
  scr.src  = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
  scr.onload = function () {
    const map = L.map(root).setView([20, 0], 2);
    L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 19, attribution: '&copy; Esri' }
    ).addTo(map);
    let mk = null;
    map.on('click', function (e) {
      if (mk) map.removeLayer(mk);
      mk = L.circleMarker(e.latlng,
           { radius: 7, color: '#ff8800', fillOpacity: 0.9 }).addTo(map);
      window.__usEvalPick = { lat: e.latlng.lat, lng: e.latlng.lng,
                              zoom: map.getZoom() };
      function tryClick() {
        var id = 'US_APPLY_ID';
        var el  = document.getElementById(id);
        var btn = el && el.tagName === 'BUTTON' ? el
                : (el ? el.querySelector('button') : null);
        if (!btn) btn = document.querySelector('#' + id + ' button');
        if (btn) btn.click();
      }
      setTimeout(tryClick, 150);
      setTimeout(tryClick, 500);
    });
  };
  document.head.appendChild(scr);
})();
""".replace("US_APPLY_ID", _APPLY_BTN_ID)

APPLY_JS = """
function(_raw, z_cur) {
  var p = window.__usEvalPick;
  if (p && typeof p.lat === 'number') {
    var zp = Math.max(10, Math.min(17, Math.round(p.zoom)));
    return [JSON.stringify({lat: p.lat, lng: p.lng, zoom: zp}), zp];
  }
  return [_raw, z_cur];
}
"""


# Core evaluation — bicubic vs GT at 3 depth levels
def run_eval(pick_json: str, zp_raw: int) -> tuple:
    """Returns (source_256, panel_d1, panel_d2, panel_d3, metrics_md).

    Each panel_dN is a [bicubic | GT] side-by-side PIL image at native
    upsampled resolution.
    """
    zp = int(max(10, min(17, round(float(zp_raw)))))

    try:
        p = json.loads(pick_json) if pick_json else {}
    except json.JSONDecodeError:
        p = {}
    if not p or "lat" not in p:
        return (None, None, None, None, "*Click map first, then hit Run.*")

    lat, lon = float(p["lat"]), float(p["lng"])
    xp, yp   = lonlat_to_tile_xy(lon, lat, zp)

    # Source parent
    try:
        parent = fetch_tile_pil(zp, xp, yp).resize((256, 256), Image.Resampling.LANCZOS)
    except Exception as e:
        return (None, None, None, None, f"Parent fetch error: {e}")

    # One panel per depth d ∈ {1, 2, 3}
    panels  = []
    metrics = []

    for d in (1, 2, 3):
        child_sz = 256 * (2 ** d)

        # Bicubic upsample from parent
        bicubic = parent.resize((child_sz, child_sz), Image.Resampling.BICUBIC)

        # GT: stitch Esri child tiles
        try:
            gt = fetch_child_mosaic(zp, xp, yp, d)
        except Exception as e:
            panels.append(None)
            metrics.append(f"d={d}: fetch error — {e}")
            continue

        # Metrics at native child resolution
        gt_np  = _to_np(gt)
        bc_np  = _to_np(bicubic)
        m_psnr = psnr(gt_np, bc_np)
        m_ssim = ssim_simple(gt_np, bc_np)
        m_lp   = lpips_score(gt, bicubic)

        metrics.append(
            f"| {child_sz}×{child_sz} (d={d}) "
            f"| {m_psnr:.2f} dB | {m_ssim:.4f} | {m_lp} |"
        )

        # Full-resolution side-by-side panel
        panels.append(make_side_by_side(bicubic, gt))

    # Metrics table
    z_child_max = min(zp + 3, 19)
    header = [
        f"**Source:** parent z={zp}  ({xp}, {yp})  →  "
        f"bicubic vs Esri GT at z={zp+1} / {zp+2} / {z_child_max}\n",
        "| Size (d) | PSNR ↑ | SSIM ↑ | LPIPS ↓ |",
        "|----------|--------|--------|---------|",
    ]
    footer = (
        "\n*Left panel = bicubic upsampled parent. "
        "Right panel = real Esri child mosaic (GT). "
        "Panels returned at full native resolution; "
        "Gradio display height doubles with each level.*"
    )
    metrics_md = "\n".join(header + metrics) + footer

    return (parent, panels[0], panels[1], panels[2], metrics_md)


# Gradio UI
def _apply(raw: str, z):
    try:
        p = json.loads(raw) if raw.strip().startswith("{") else {}
    except Exception:
        p = {}
    if not p or "lat" not in p:
        return raw, z, "*Click map first.*"
    lat, lon = float(p["lat"]), float(p["lng"])
    z2 = int(max(10, min(17, round(float(p.get("zoom", z))))))
    return raw, z2, f"Picked  lat={lat:.5f}  lon={lon:.5f}  z={z2} — ready."


def build_ui():
    with gr.Blocks(title="Bicubic upsample evaluation") as demo:
        gr.Markdown(
            "## Bicubic upsample evaluation — 3 zoom levels\n"
            "Pick any location on the map, choose the parent zoom, then run.\n"
            "Each panel shows **bicubic upsampled parent (left)** vs "
            "**real Esri child mosaic (right)** at the native upsampled resolution.\n"
            "Panel display height doubles at each level (256 → 512 → 1024 px) "
            "so the visual size is proportional to the actual tile size."
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
            zp        = gr.Slider(10, 17, value=15, step=1,
                                  label="Parent zoom z_parent")
            pick_msg  = gr.Markdown("*Click map, then run.*")

        apply_btn = gr.Button("Refresh from last map click",
                              elem_id=_APPLY_BTN_ID, size="sm")
        run_btn   = gr.Button("▶ Run evaluation", variant="primary")

        apply_btn.click(
            fn=_apply, inputs=[pick_raw, zp], outputs=[pick_raw, zp, pick_msg],
            js=APPLY_JS,
        )

        # Output panels — heights double with each level
        src_img = gr.Image(type="pil",
                           label="Source — parent tile 256×256",
                           height=256)

        gr.Markdown("### d=1 — 512×512  |  bicubic (left)  vs  GT (right)")
        img_d1  = gr.Image(type="pil",
                           label="d=1 → 512×512",
                           height=256)

        gr.Markdown("### d=2 — 1024×1024  |  bicubic (left)  vs  GT (right)")
        img_d2  = gr.Image(type="pil",
                           label="d=2 → 1024×1024",
                           height=512)

        gr.Markdown("### d=3 — 2048×2048  |  bicubic (left)  vs  GT (right)")
        img_d3  = gr.Image(type="pil",
                           label="d=3 → 2048×2048",
                           height=1024)

        metrics_md = gr.Markdown("")

        run_btn.click(
            fn=run_eval,
            inputs=[pick_raw, zp],
            outputs=[src_img, img_d1, img_d2, img_d3, metrics_md],
        )

    return demo


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_US_PORT", 7864))
    build_ui().launch(server_name="0.0.0.0", server_port=port, share=True)
