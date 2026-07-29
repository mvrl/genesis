#!/usr/bin/env python3
"""
Upsample comparison demo — bicubic baseline vs Genesis SR vs Esri GT.

For a chosen parent tile (256×256 at z_parent), shows source plus three
[bicubic | Genesis SR | GT] panels at d=1/2/3 (512/1024/2048 px). Genesis SR
is applied iteratively (one ×2 step per depth) and both bicubic and SR are
scored against the Esri GT mosaic (PSNR / SSIM / LPIPS / DISTS).

Run: python app_upsample_comp.py  (port: GRADIO_US_COMP_PORT, default 7866).

NB: SR was trained for single-step ×2 only; d=2/d=3 are iterative cascades that
compound error — visual demo, not a fair quality benchmark.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

_DEMOS = os.path.dirname(os.path.abspath(__file__))
# Order matters: each insert(0, …) ends up BELOW the next, so put _DEMOS LAST
# to keep our `utils/` package ahead of flat `utils.py` modules under src/.
sys.path.insert(0, os.path.join(_DEMOS, "..", "src"))
sys.path.insert(0, os.path.join(_DEMOS, "..", "src", "superresolution"))
sys.path.insert(0, _DEMOS)

import gradio as gr
import numpy as np
import torch
from PIL import Image

from utils.genesis_common import (
    DEFAULT_SR_CKPT,
    SR_KEY_B16_MAIN,
    SR_MODEL_KEYS,
    fetch_tile_pil,
    inference_device,
    inference_device_label,
    load_sr_denoiser,
    lonlat_to_tile_xy,
    model_arch,
    resolved_dino_weights_path,
    run_superresolution,
)

_DEVICE   = inference_device()
_SR_CACHE: dict = {}
_APPLY_BTN_ID = "us_comp_apply_pick"


# Lazy LPIPS + DISTS singletons
_LPIPS_FN = None
_DISTS_FN = None


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


def _get_dists():
    global _DISTS_FN
    if _DISTS_FN is None:
        try:
            from DISTS_pytorch import DISTS
            _DISTS_FN = DISTS()
            _DISTS_FN.eval()
        except ImportError:
            _DISTS_FN = "unavailable"
    return _DISTS_FN


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
    """LPIPS (AlexNet backbone). Inputs in [-1, 1]. Lower is better."""
    fn = _get_lpips()
    if fn == "unavailable" or fn is None:
        return "N/A"
    if gt.size != pred.size:
        pred = pred.resize(gt.size, Image.Resampling.BILINEAR)
    a = torch.from_numpy(_to_np(gt)).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    b = torch.from_numpy(_to_np(pred)).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    with torch.no_grad():
        return f"{fn(a, b).item():.4f}"


def dists_score(gt: Image.Image, pred: Image.Image) -> str:
    """DISTS (Deep Image Structure and Texture Similarity).
    Inputs in [0, 1]. Lower is better."""
    fn = _get_dists()
    if fn == "unavailable" or fn is None:
        return "N/A"
    if gt.size != pred.size:
        pred = pred.resize(gt.size, Image.Resampling.BILINEAR)
    a = torch.from_numpy(_to_np(gt)).permute(2, 0, 1).unsqueeze(0)
    b = torch.from_numpy(_to_np(pred)).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        try:
            return f"{fn(a, b).item():.4f}"
        except Exception:
            return "N/A"


# GT child mosaic + 3-way composite
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


def make_three_way(bicubic: Image.Image, sr: Image.Image, gt: Image.Image,
                   gap: int = 12) -> Image.Image:
    """[bicubic | Genesis SR | GT] side-by-side with thin separators."""
    w, h = bicubic.size
    out  = Image.new("RGB", (3 * w + 2 * gap, h), (220, 220, 220))
    out.paste(bicubic, (0, 0))
    out.paste(sr,      (w + gap, 0))
    out.paste(gt,      (2 * (w + gap), 0))
    return out


# SR model cache
def get_sr_model(model_key: str, ckpt_override: str = ""):
    ckpt = (ckpt_override or "").strip() or DEFAULT_SR_CKPT.get(model_key, "")
    if not ckpt:
        raise FileNotFoundError(
            f"No SR checkpoint path resolved for {model_key!r}. "
            f"Set GENESIS_CKPT_BASE or paste a full path into the ckpt override."
        )
    dino = resolved_dino_weights_path()
    key  = (ckpt, model_key)
    if key not in _SR_CACHE:
        denoiser, _ = load_sr_denoiser(
            ckpt, model_arch(model_key), dino_weights=dino, device=_DEVICE,
        )
        _SR_CACHE[key] = denoiser
    return _SR_CACHE[key]


# Iterative Genesis SR
@torch.no_grad()
def iterative_sr(parent: Image.Image, denoiser, zp: int, d: int) -> Image.Image:
    """Apply Genesis SR d times. parent (256²) → (256·2^d)² stitched mosaic.

    Each SR call upsamples one 256² tile to a 512² mosaic of its 4 children
    (raster order: UL, UR, LL, LR). For d > 1 each child is SR'd in turn,
    building a 2^d × 2^d grid stitched at the end.
    """
    if d <= 0:
        return parent
    grid: list[list[Image.Image]] = [[parent]]   # 1×1
    for step in range(d):
        target_zoom = zp + step + 1
        rows = len(grid)
        cols = len(grid[0])
        new_grid: list[list[Image.Image]] = [[None] * (cols * 2)
                                             for _ in range(rows * 2)]
        for r in range(rows):
            for c in range(cols):
                tile = grid[r][c]
                mosaic = run_superresolution(denoiser, tile, _DEVICE,
                                             target_zoom=target_zoom)
                # 4-child split — assume raster layout
                new_grid[2*r    ][2*c    ] = mosaic.crop((0,   0,   256, 256))
                new_grid[2*r    ][2*c + 1] = mosaic.crop((256, 0,   512, 256))
                new_grid[2*r + 1][2*c    ] = mosaic.crop((0,   256, 256, 512))
                new_grid[2*r + 1][2*c + 1] = mosaic.crop((256, 256, 512, 512))
        grid = new_grid

    rows = len(grid)
    cols = len(grid[0])
    out  = Image.new("RGB", (cols * 256, rows * 256))
    for r in range(rows):
        for c in range(cols):
            out.paste(grid[r][c], (c * 256, r * 256))
    return out


# Map widget
LEAFLET_HTML = """
<div class="us-comp-map" style="height:420px;width:100%;border-radius:8px;
     overflow:hidden;border:1px solid #ccc;"></div>
<p style="margin:6px 0 0;color:#666;font-size:0.85em;">
  Click a location. The parent tile at the chosen zoom will be the low-res source.
</p>
"""

LEAFLET_JS = r"""
(function () {
  const root = element.querySelector('.us-comp-map') || element;
  root.innerHTML = '';
  window.__usCompPick = null;
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
           { radius: 7, color: '#22aa55', fillOpacity: 0.9 }).addTo(map);
      window.__usCompPick = { lat: e.latlng.lat, lng: e.latlng.lng,
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
  var p = window.__usCompPick;
  if (p && typeof p.lat === 'number') {
    var zp = Math.max(10, Math.min(17, Math.round(p.zoom)));
    return [JSON.stringify({lat: p.lat, lng: p.lng, zoom: zp}), zp];
  }
  return [_raw, z_cur];
}
"""


# Core comparison
def run_eval(pick_json: str, zp_raw: int, model_key: str, ckpt_override: str,
             steps: float, cfg: float, depths_csv: str,
             bicubic_init: bool, bicubic_t_start: float):
    """Returns (source_256, panel_d1, panel_d2, panel_d3, metrics_md).

    Each panel_dN is a [bicubic | Genesis SR | GT] side-by-side. `depths_csv`
    is a comma-separated subset of "1,2,3"; omitted depths skip the SR cascade
    (still show bicubic + GT) to save compute.
    """
    zp = int(max(10, min(17, round(float(zp_raw)))))

    try:
        p = json.loads(pick_json) if pick_json else {}
    except json.JSONDecodeError:
        p = {}
    if not p or "lat" not in p:
        return (None, None, None, None, "*Click map first, then hit Run.*")

    requested_depths = {int(d.strip())
                        for d in depths_csv.split(",")
                        if d.strip().isdigit() and int(d.strip()) in (1, 2, 3)}
    if not requested_depths:
        requested_depths = {1, 2, 3}

    lat, lon = float(p["lat"]), float(p["lng"])
    xp, yp   = lonlat_to_tile_xy(lon, lat, zp)

    # Source parent
    try:
        parent = fetch_tile_pil(zp, xp, yp).resize((256, 256), Image.Resampling.LANCZOS)
    except Exception as e:
        return (None, None, None, None, f"Parent fetch error: {e}")

    # Load SR model (cached)
    try:
        denoiser = get_sr_model(model_key, ckpt_override)
        denoiser.steps    = int(steps)
        denoiser.cfg_scale = float(cfg)
        # Bicubic-init SDEdit warm start
        denoiser.args.bicubic_init         = bool(bicubic_init)
        denoiser.args.bicubic_init_t_start = float(bicubic_t_start)
    except Exception as e:
        return (parent, None, None, None, f"**Model load error:** {e}")

    # For each depth d ∈ {1,2,3}: bicubic, iterative SR, GT, metrics
    panels  = [None, None, None]
    metrics_rows = []

    for d in (1, 2, 3):
        child_sz = 256 * (2 ** d)

        # Bicubic baseline
        bicubic = parent.resize((child_sz, child_sz), Image.Resampling.BICUBIC)

        # Esri GT mosaic
        try:
            gt = fetch_child_mosaic(zp, xp, yp, d)
        except Exception as e:
            metrics_rows.append(f"| d={d}, {child_sz}² | — | — | — | "
                                f"fetch error: {e} |")
            continue

        # Genesis SR (skipped if depth not requested)
        if d in requested_depths:
            t0 = time.time()
            try:
                sr_img = iterative_sr(parent, denoiser, zp, d)
            except Exception as e:
                sr_img = None
                sr_err = str(e)
            else:
                sr_err = None
            sr_dt = time.time() - t0
        else:
            sr_img = None
            sr_err = "skipped"
            sr_dt  = 0.0

        # Composite — fall back to bicubic+GT-only when SR missing
        if sr_img is None:
            placeholder = Image.new("RGB", bicubic.size, (60, 60, 60))
            panels[d - 1] = make_three_way(bicubic, placeholder, gt)
        else:
            panels[d - 1] = make_three_way(bicubic, sr_img, gt)

        # Metrics — bicubic vs GT, then SR vs GT
        gt_np  = _to_np(gt)
        bc_np  = _to_np(bicubic)
        m_psnr_bc = psnr(gt_np, bc_np)
        m_ssim_bc = ssim_simple(gt_np, bc_np)
        m_lp_bc   = lpips_score(gt, bicubic)
        m_ds_bc   = dists_score(gt, bicubic)
        metrics_rows.append(
            f"| d={d}, {child_sz}² | **bicubic** | {m_psnr_bc:.2f} | "
            f"{m_ssim_bc:.4f} | {m_lp_bc} | {m_ds_bc} | — |"
        )

        if sr_img is not None:
            sr_np = _to_np(sr_img)
            m_psnr_sr = psnr(gt_np, sr_np)
            m_ssim_sr = ssim_simple(gt_np, sr_np)
            m_lp_sr   = lpips_score(gt, sr_img)
            m_ds_sr   = dists_score(gt, sr_img)
            metrics_rows.append(
                f"| d={d}, {child_sz}² | **Genesis SR** | {m_psnr_sr:.2f} | "
                f"{m_ssim_sr:.4f} | {m_lp_sr} | {m_ds_sr} | {sr_dt:.1f} s |"
            )
        else:
            metrics_rows.append(
                f"| d={d}, {child_sz}² | Genesis SR | — | — | — | — | {sr_err} |"
            )

    # Metrics table
    z_child_max = min(zp + 3, 19)
    init_label  = (f"bicubic init @ t_start={bicubic_t_start:.2f}"
                   if bicubic_init else "random noise z₀")
    header = [
        f"**Source:** parent z={zp}  ({xp}, {yp}) — ({lat:.5f}, {lon:.5f})\n"
        f"**Model:** `{model_key}` · sampling steps {int(steps)} · cfg {cfg:.1f} · init: {init_label}\n"
        f"**Comparison:** bicubic upsample vs Genesis SR (iterative) vs Esri GT, "
        f"at z={zp+1} / {zp+2} / {z_child_max}\n",
        "| Level (size) | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | Note |",
        "|--------------|--------|--------|--------|---------|---------|------|",
    ]
    footer = (
        "\n*Each panel = [bicubic upsample | Genesis SR | Esri GT child mosaic]. "
        "SR was trained for **single-step ×2 only** (d=1); d=2 and d=3 are "
        "iterative cascades (compounding error) — useful for visual demo but "
        "the metrics drop is expected.*"
    )
    metrics_md = "\n".join(header + metrics_rows) + footer
    return (parent, panels[0], panels[1], panels[2], metrics_md)


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


# Gradio UI
def build_ui():
    with gr.Blocks(title="Upsample comparison — bicubic vs Genesis SR vs GT") as demo:
        gr.Markdown(
            "## Upsample comparison — bicubic vs Genesis SR vs Esri GT\n"
            f"**Device:** {inference_device_label(_DEVICE)}  ·  "
            "Click any location on the map, pick a parent zoom, choose a "
            "model arch, then **Run**.\n\n"
            "Each panel shows **bicubic upsampled parent** (left) | **Genesis "
            "SR iterative output** (middle) | **real Esri child mosaic** "
            "(right). Both bicubic and SR are quantified against the GT in "
            "the table below.\n\n"
            "> SR was trained for a single ×2 step (zoom_diff=1). d=2 and d=3 "
            "are **iterative cascades** that compound error — included here "
            "for visual demonstration, not as a fair quality measure."
        )

        pick_raw = gr.Textbox(value="{}", visible=False)
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

        with gr.Row():
            model_key = gr.Dropdown(SR_MODEL_KEYS, value=SR_KEY_B16_MAIN,
                                    label="SR architecture",
                                    info="B16 is faster; H16 is the larger main run.")
            depths    = gr.CheckboxGroup(
                choices=["1", "2", "3"], value=["1", "2", "3"],
                label="Depths to run SR (×2, ×4, ×8)",
                info="Uncheck to skip a level — bicubic+GT still shown.",
            )

        with gr.Row():
            bicubic_init = gr.Checkbox(
                value=True,
                label="Bicubic init (SDEdit warm start)",
                info="Start diffusion from the bicubic-upsampled parent instead of pure noise — "
                     "preserves coarse structure from the parent zoom; default ON.",
            )
            bicubic_t_start = gr.Slider(
                0.01, 0.9, value=0.5, step=0.01,
                label="Bicubic init t_start",
                info="How far into the diffusion trajectory to inject the bicubic prior. "
                     "Higher = more structure preserved from the parent.",
            )

        with gr.Accordion("Advanced", open=False):
            ckpt_override = gr.Textbox(
                label="Optional SR checkpoint path override", placeholder="auto"
            )
            steps_sl = gr.Slider(10, 100, value=50, step=5,
                                 label="Diffusion sampling steps")
            cfg_sl   = gr.Slider(1.0, 4.0, value=1.0, step=0.5,
                                 label="CFG scale")

        apply_btn = gr.Button("Refresh from last map click",
                              elem_id=_APPLY_BTN_ID, size="sm")
        run_btn   = gr.Button("▶ Run comparison", variant="primary")

        apply_btn.click(
            fn=_apply, inputs=[pick_raw, zp], outputs=[pick_raw, zp, pick_msg],
            js=APPLY_JS,
        )

        src_img = gr.Image(type="pil",
                           label="Source — parent tile 256×256",
                           height=256)

        gr.Markdown("### d=1 — 512×512  |  bicubic  ·  Genesis SR  ·  GT")
        img_d1  = gr.Image(type="pil", label="d=1 → 512×512", height=256)

        gr.Markdown("### d=2 — 1024×1024  |  bicubic  ·  Genesis SR (cascade ×4)  ·  GT")
        img_d2  = gr.Image(type="pil", label="d=2 → 1024×1024", height=512)

        gr.Markdown("### d=3 — 2048×2048  |  bicubic  ·  Genesis SR (cascade ×8)  ·  GT")
        img_d3  = gr.Image(type="pil", label="d=3 → 2048×2048", height=1024)

        metrics_md = gr.Markdown("")

        run_btn.click(
            fn=lambda raw, zp_v, mk, ckpt, st, cfg, dlist, bi, bts: run_eval(
                raw, zp_v, mk, ckpt, st, cfg, ",".join(dlist), bi, bts,
            ),
            inputs=[pick_raw, zp, model_key, ckpt_override, steps_sl, cfg_sl,
                    depths, bicubic_init, bicubic_t_start],
            outputs=[src_img, img_d1, img_d2, img_d3, metrics_md],
        )

    return demo


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_US_COMP_PORT", 7866))
    print(f"[Genesis upsample-comp] Device: {inference_device_label(_DEVICE)}",
          flush=True)
    build_ui().launch(server_name="0.0.0.0", server_port=port, share=True)
