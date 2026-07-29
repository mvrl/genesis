#!/usr/bin/env python3
"""
Gradio demo: Genesis super-resolution (full-tile, ZL → ZL+1).

Click the map to fetch a 256×256 parent tile, then Generate: the model produces
all four ZL+1 children as a stitched 512×512 mosaic in one forward pass (strictly
×2). Compared against bicubic upsampling and the real ZL+1 children from Esri.

Run: python app_superresolution.py  (port: $GRADIO_SR_PORT, default 7860).
"""
from __future__ import annotations

import json
import math
import os
import sys

_DEMOS = os.path.dirname(os.path.abspath(__file__))
# Order matters: each `sys.path.insert(0, X)` ends up BELOW the next insert, so
# put _DEMOS LAST to keep our `utils/` package ahead of
# `src/superresolution/utils.py`, which would otherwise shadow the import.
sys.path.insert(0, os.path.join(_DEMOS, "..", "src"))
sys.path.insert(0, os.path.join(_DEMOS, "..", "src", "superresolution"))
sys.path.insert(0, _DEMOS)

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.genesis_common import (
    DEFAULT_SR_CKPT,
    SR_MODEL_KEYS,
    model_arch,
    fetch_tile_at_zoom,
    fetch_tile_pil,
    inference_device,
    inference_device_label,
    load_sr_denoiser,
    lonlat_to_tile_xy,
    nearest_training_zoom,
    resolved_dino_weights_path,
    run_superresolution,
)

_DEVICE = inference_device()
_SR_CACHE: dict = {}

_APPLY_BTN_ID = "genesis_sr_apply_pick"

LEAFLET_HTML = """
<div class="genesis-sr-mapwrap" style="height:500px;width:100%;border-radius:8px;overflow:hidden;border:1px solid #444;"></div>
<p style="margin:8px 0 0;color:#666;font-size:0.9em;"><strong>Click</strong> the map — latitude, longitude, parent tile preview, and zoom info fill in automatically. Then press <strong>Generate</strong>.</p>
"""

LEAFLET_JS = r"""
(function () {
  const root = element.querySelector('.genesis-sr-mapwrap') || element;
  root.innerHTML = '';
  window.__genesisSRPick = null;
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
      marker = L.circleMarker(e.latlng, { radius: 6, color: '#ff4444', fillOpacity: 0.9 }).addTo(map);
      window.__genesisSRPick = { lat: e.latlng.lat, lng: e.latlng.lng, zoom: map.getZoom() };
      function clickApply() {
        var id = 'APPLY_PICK_BTN_ID';
        var wrap = document.getElementById(id);
        var btn = wrap && wrap.tagName === 'BUTTON' ? wrap : (wrap ? wrap.querySelector('button') : null);
        if (!btn) btn = document.querySelector('#' + id + ' button');
        if (!btn) {
          var nodes = document.querySelectorAll('button');
          for (var i = 0; i < nodes.length; i++) {
            var t = (nodes[i].innerText || nodes[i].textContent || '');
            if (t.indexOf('Refresh from last') >= 0) { nodes[i].click(); return; }
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


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------

def get_model(model_key: str, ckpt_override: str):
    # load_sr_denoiser() auto-pulls the ckpt from Google Drive if absent; an
    # empty path is the only configuration error we surface up-front.
    ckpt = (ckpt_override or "").strip() or DEFAULT_SR_CKPT.get(model_key, "")
    if not ckpt:
        raise FileNotFoundError(
            f"No SR checkpoint path resolved for {model_key!r}. "
            f"Set GENESIS_CKPT_BASE or paste a full path into the 'Optional checkpoint path' textbox."
        )
    dino = resolved_dino_weights_path()   # auto-pulls from Google Drive when missing
    cache_key = (ckpt, model_key)
    if cache_key not in _SR_CACHE:
        denoiser, _ = load_sr_denoiser(
            ckpt, model_arch(model_key), dino_weights=dino, device=_DEVICE,
        )
        _SR_CACHE[cache_key] = denoiser
    return _SR_CACHE[cache_key]


# ---------------------------------------------------------------------------
# Tile / mosaic helpers
# ---------------------------------------------------------------------------

# QUAD_ORDER from src/superresolution/dataloader.py: UL, UR, LL, LR (qy*2+qx).
# Child (ZL+1, 2*xp+qx, 2*yp+qy)  with  qx ∈ {0,1}, qy ∈ {0,1}.
QUAD_OFFSETS = [(0, 0), (1, 0), (0, 1), (1, 1)]   # (qx, qy)


def stitch_quad_mosaic(children: list[Image.Image | None], tile_size: int = 256,
                       missing_color: tuple[int, int, int] = (200, 200, 200)) -> Image.Image:
    """Stitch [UL, UR, LL, LR] PIL tiles into a 2*tile_size square. None → grey block."""
    s = tile_size
    out = Image.new("RGB", (2 * s, 2 * s), missing_color)
    placements = [(0, 0), (s, 0), (0, s), (s, s)]   # matches QUAD_OFFSETS
    for child, (px, py) in zip(children, placements):
        if child is None:
            continue
        c = child.convert("RGB")
        if c.size != (s, s):
            c = c.resize((s, s), Image.Resampling.BILINEAR)
        out.paste(c, (px, py))
    return out


def fetch_gt_mosaic(zp: int, xp: int, yp: int, tile_size: int = 256
                    ) -> tuple[Image.Image, int]:
    """Fetch the four ZL+1 child tiles of parent (zp,xp,yp); stitch into 512×512.
    Returns (mosaic, n_present)."""
    children: list[Image.Image | None] = [None, None, None, None]
    n = 0
    for i, (qx, qy) in enumerate(QUAD_OFFSETS):
        try:
            c = fetch_tile_pil(zp + 1, 2 * xp + qx, 2 * yp + qy)
            children[i] = c
            n += 1
        except Exception:
            children[i] = None
    return stitch_quad_mosaic(children, tile_size=tile_size), n


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _laplacian_sharpness(img: Image.Image) -> float:
    """Variance of the discrete Laplacian on grayscale — higher = crisper edges."""
    from scipy.ndimage import convolve
    gray = np.array(img.convert("L"), dtype=np.float64)
    kern = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    return float(np.var(convolve(gray, kern)))


def compute_image_metrics(img: Image.Image, gt: Image.Image | None) -> dict:
    """PSNR / SSIM / LPIPS / DISTS / Sharpness on the full image vs GT."""
    img_a = np.array(img, dtype=np.float64)
    metrics: dict = {"sharpness": _laplacian_sharpness(img)}

    if gt is not None:
        if gt.size != img.size:
            gt = gt.resize(img.size, Image.Resampling.BILINEAR)
        gt_a = np.array(gt, dtype=np.float64)

        mse = float(np.mean((img_a - gt_a) ** 2))
        metrics["psnr"] = 10 * math.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")

        mu_i, mu_g = img_a.mean(), gt_a.mean()
        sig_i, sig_g = img_a.std(), gt_a.std()
        sig_xy = float(np.mean((img_a - mu_i) * (gt_a - mu_g)))
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        metrics["ssim"] = float(
            ((2 * mu_i * mu_g + c1) * (2 * sig_xy + c2))
            / ((mu_i ** 2 + mu_g ** 2 + c1) * (sig_i ** 2 + sig_g ** 2 + c2))
        )

        import torch

        def _to_tensor(pil: Image.Image) -> torch.Tensor:
            arr = np.array(pil.convert("RGB"), dtype=np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

        img_t = _to_tensor(img)
        gt_t = _to_tensor(gt)

        try:
            import lpips
            if not hasattr(compute_image_metrics, "_lpips_fn"):
                compute_image_metrics._lpips_fn = lpips.LPIPS(net="alex", verbose=False)
            with torch.no_grad():
                metrics["lpips"] = float(
                    compute_image_metrics._lpips_fn(img_t * 2 - 1, gt_t * 2 - 1).item())
        except Exception:
            pass

        try:
            from DISTS_pytorch import DISTS
            if not hasattr(compute_image_metrics, "_dists_fn"):
                compute_image_metrics._dists_fn = DISTS()
            with torch.no_grad():
                metrics["dists"] = float(
                    compute_image_metrics._dists_fn(img_t, gt_t).item())
        except Exception:
            pass

    return metrics


def _cmp(a: float, b: float, higher_is_better: bool = True) -> str:
    if abs(a - b) < 1e-6:
        return "—"
    if higher_is_better:
        return "**SR ↑**" if b > a else "Bicubic ↑"
    return "**SR ↓**" if b < a else "Bicubic ↓"


def format_comparison_table(bm: dict, sm: dict) -> str:
    rows = [
        "| Metric | Bicubic | Genesis SR | Better |",
        "|:-------|:-------:|:---------:|:------:|",
    ]
    if "psnr" in bm and "psnr" in sm:
        rows.append(f"| PSNR ↑ (vs GT, dB) | `{bm['psnr']:.2f}` | `{sm['psnr']:.2f}` | {_cmp(bm['psnr'], sm['psnr'])} |")
    if "ssim" in bm and "ssim" in sm:
        rows.append(f"| SSIM ↑ (vs GT) | `{bm['ssim']:.4f}` | `{sm['ssim']:.4f}` | {_cmp(bm['ssim'], sm['ssim'])} |")
    if "lpips" in bm and "lpips" in sm:
        rows.append(f"| LPIPS ↓ (perceptual, vs GT) | `{bm['lpips']:.4f}` | `{sm['lpips']:.4f}` | {_cmp(bm['lpips'], sm['lpips'], higher_is_better=False)} |")
    if "dists" in bm and "dists" in sm:
        rows.append(f"| DISTS ↓ (structure+texture, vs GT) | `{bm['dists']:.4f}` | `{sm['dists']:.4f}` | {_cmp(bm['dists'], sm['dists'], higher_is_better=False)} |")
    rows.append(f"| Sharpness ↑ (Laplacian var) | `{bm['sharpness']:.1f}` | `{sm['sharpness']:.1f}` | {_cmp(bm['sharpness'], sm['sharpness'])} |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Panel layout
# ---------------------------------------------------------------------------

_LABEL_H = 22
_LABEL_BG = (30, 30, 35)
_LABEL_FG = (220, 220, 220)
_GAP = 6
_GAP_COLOR = (238, 238, 242)


def _make_labeled_panel(img: Image.Image, label: str) -> Image.Image:
    """Caption strip on top of a square image. Image is shown at its native size."""
    w, h = img.size
    panel = Image.new("RGB", (w, h + _LABEL_H), _LABEL_BG)
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    tx = max(0, (w - (bbox[2] - bbox[0])) // 2)
    ty = max(0, (_LABEL_H - (bbox[3] - bbox[1])) // 2)
    draw.text((tx, ty), label, fill=_LABEL_FG, font=font)
    panel.paste(img, (0, _LABEL_H))
    return panel


def compose_strip(panels: list[Image.Image]) -> Image.Image:
    """Side-by-side strip with _GAP px between same-height panels."""
    h = panels[0].height
    total_w = sum(p.width for p in panels) + _GAP * (len(panels) - 1)
    strip = Image.new("RGB", (total_w, h), _GAP_COLOR)
    x = 0
    for p in panels:
        strip.paste(p, (x, 0))
        x += p.width + _GAP
    return strip


# ---------------------------------------------------------------------------
# Map → tile
# ---------------------------------------------------------------------------

def build_parent(lat: float, lon: float, map_zoom: float):
    """Snap to a training-distribution parent zoom, fetch the 256×256 parent tile."""
    zp = nearest_training_zoom(int(round(map_zoom)))
    # Constrain so child zoom (zp+1) stays ≤ 19 (Esri max).
    zp = min(zp, 18)
    zp = max(zp, 10)
    xp, yp = lonlat_to_tile_xy(float(lon), float(lat), zp)
    parent = fetch_tile_at_zoom(float(lon), float(lat), zp, 256)
    return parent, zp, xp, yp


def refresh_preview(pick: dict | None):
    if not pick or "map_zoom" not in pick:
        return gr.update(), "*Click the map to select a location.*", gr.update(), "", "", ""
    lat, lon, mz = pick["lat"], pick["lon"], pick["map_zoom"]
    parent, zp, xp, yp = build_parent(lat, lon, mz)
    pick2 = {**pick, "zp": zp, "xp": xp, "yp": yp}
    info = (
        f"**Parent zoom:** z={zp} ({xp},{yp})  →  "
        f"**Output children zoom:** z={zp+1} (2×2 tiles → stitched 512×512)."
    )
    return parent, info, pick2, f"{lat:.6f}", f"{lon:.6f}", f"{mz:.2f}"


def apply_from_map(pick_json_str: str):
    try:
        o = json.loads(pick_json_str) if pick_json_str else {}
    except json.JSONDecodeError:
        o = {}
    if not o or "lat" not in o:
        return gr.update(), "*Click the map on the satellite layer.*", gr.update(), "", "", ""
    try:
        pick = {
            "lat": float(o["lat"]),
            "lon": float(o["lng"]),
            "map_zoom": float(o["zoom"]),
        }
    except (KeyError, TypeError, ValueError):
        return gr.update(), "*Could not read map click — try again.*", gr.update(), "", "", ""
    return refresh_preview(pick)


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------

def generate_sr(pick: dict | None, model_key: str, ckpt_override: str,
                steps: float, cfg: float, bicubic_init: bool, bicubic_t_start: float):
    if not pick or "zp" not in pick:
        return None, "**Click the map** first (wait for lat/lon and preview to appear)."
    try:
        denoiser = get_model(model_key, ckpt_override)
    except Exception as e:
        return None, f"**Load error:** {e}"

    parent = fetch_tile_at_zoom(pick["lon"], pick["lat"], pick["zp"], 256)
    zp, xp, yp = pick["zp"], pick["xp"], pick["yp"]

    denoiser.args.num_sampling_steps = int(steps)
    denoiser.args.cfg = float(cfg)
    denoiser.args.bicubic_init = bool(bicubic_init)
    denoiser.args.bicubic_init_t_start = float(bicubic_t_start)

    try:
        sr_out = run_superresolution(denoiser, parent, _DEVICE, target_zoom=zp + 1)
    except Exception as e:
        import traceback
        return None, f"**Inference error:** {e}\n```\n{traceback.format_exc()}\n```"

    # Bicubic baseline: 256 parent → 512 (no model)
    bicubic_out = parent.resize((512, 512), Image.Resampling.BICUBIC)

    # GT mosaic — fetch the 4 children at zp+1
    gt_mosaic, n_gt = fetch_gt_mosaic(zp, xp, yp, tile_size=256)
    gt = gt_mosaic if n_gt > 0 else None

    bm = compute_image_metrics(bicubic_out, gt)
    sm = compute_image_metrics(sr_out, gt)

    panels = [
        # Parent is omitted from the strip on purpose: it's 256 while the rest
        # are 512, and it's already shown in the parent-conditioning panel above.
        _make_labeled_panel(bicubic_out, "Bicubic 512"),
        _make_labeled_panel(sr_out, "Genesis SR 512"),
    ]
    if gt is not None:
        gt_label = f"GT mosaic z={zp+1} ({n_gt}/4)" if n_gt < 4 else f"GT mosaic z={zp+1}"
        panels.append(_make_labeled_panel(gt, gt_label))

    strip = compose_strip(panels)

    init_label = "bicubic init" if bicubic_init else "random noise z₀"
    gt_note = f"{n_gt}/4 children fetched" if gt is not None else "unavailable"
    msg = (
        f"**Done** — Genesis SR ({init_label}) · GT: {gt_note} at z={zp+1}\n\n"
        + format_comparison_table(bm, sm)
    )
    return strip, msg


APPLY_JS = """
function(_unused) {
  var p = window.__genesisSRPick;
  var j = '{}';
  if (p && typeof p.lat === 'number') {
    j = JSON.stringify({lat: p.lat, lng: p.lng, zoom: p.zoom});
  }
  return [j];
}
"""


def build_ui():
    with gr.Blocks(title="Genesis Super-Resolution") as demo:
        gr.Markdown(
            "## Super-resolution — full-tile (×2, ZL → ZL+1)\n"
            "Click the map → **latitude, longitude, map zoom**, and **256×256 parent tile** appear. "
            "Press **Generate** to run the model in a single forward pass; the output is a "
            "**512×512 mosaic of the four ZL+1 children**. Compared against bicubic upsampling and "
            "the real ZL+1 children fetched from Esri."
        )
        pick_state = gr.State(value=None)
        pick_raw = gr.Textbox(value="{}", visible=False, elem_id="genesis_sr_pick_raw")

        gr.HTML(
            value=LEAFLET_HTML,
            html_template="${value}",
            js_on_load=LEAFLET_JS,
            container=True,
            min_height=520,
        )

        with gr.Row():
            lat_tb = gr.Textbox(label="Latitude", interactive=False, scale=1)
            lon_tb = gr.Textbox(label="Longitude", interactive=False, scale=1)
            zoom_tb = gr.Textbox(label="Map zoom (at click)", interactive=False, scale=1)
        preview = gr.Image(type="pil", label="Parent conditioning at click (256²)")
        info = gr.Markdown("*Click the map…*")
        use_loc = gr.Button("Refresh from last map click", variant="secondary",
                            elem_id=_APPLY_BTN_ID)

        with gr.Row():
            model_key = gr.Dropdown(SR_MODEL_KEYS, value=SR_MODEL_KEYS[0], label="Model")
            ckpt_override = gr.Textbox(
                label="Optional checkpoint path",
                placeholder="Default: $GENESIS_CKPT_BASE/ablations/superresolution/abl_dino_lpips/...",
            )
        with gr.Row():
            steps = gr.Slider(10, 100, value=50, step=5, label="Sampling steps")
            cfg = gr.Slider(1.0, 4.0, value=1.0, step=0.5, label="CFG scale")
        bicubic_init = gr.Checkbox(
            value=False,
            label="Bicubic init (SDEdit warm start)",
            info="Start diffusion from a bicubic-upsampled parent rather than random noise.",
        )
        bicubic_t_start = gr.Slider(
            0.01, 0.9, value=0.5, step=0.01,
            label="Bicubic init t_start",
            info="How far into the trajectory to inject the bicubic prior. Higher = more structure preserved.",
            visible=False,
        )
        gen_btn = gr.Button("Generate", variant="primary")
        out_img = gr.Image(type="pil", label="Bicubic 512 | Genesis SR 512 | GT 512")
        out_msg = gr.Markdown("")

        outs6 = [preview, info, pick_state, lat_tb, lon_tb, zoom_tb]

        bicubic_init.change(
            fn=lambda v: gr.update(visible=v),
            inputs=bicubic_init,
            outputs=bicubic_t_start,
        )

        use_loc.click(
            apply_from_map,
            inputs=[pick_raw],
            outputs=outs6,
            js=APPLY_JS,
        )
        gen_btn.click(
            generate_sr,
            inputs=[pick_state, model_key, ckpt_override, steps, cfg, bicubic_init, bicubic_t_start],
            outputs=[out_img, out_msg],
        )

    return demo


if __name__ == "__main__":
    _port = int(os.environ.get("GRADIO_SR_PORT", 7860))
    print(f"[Genesis SR] Inference device: {inference_device_label(_DEVICE)}")
    build_ui().launch(server_name="0.0.0.0", server_port=_port, share=True)
