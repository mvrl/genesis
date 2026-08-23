#!/usr/bin/env python3
"""
Genesis quick start — super-resolution + outpainting on a live Esri tile.

Downloads the main JiT-H/16 checkpoints from https://huggingface.co/MVRL/genesis,
fetches one real Esri World Imagery tile (a Bavarian village amid fields and
forest, zoom 16), then:

  1. SR: the 256x256 parent tile (z16) -> 512x512 mosaic of its four z17
     children in a single forward pass
  2. OP: the tile viewed as a 2x2 quadrant mosaic (each quadrant is the
     footprint of one real z17 child); keep the upper-left quadrant real,
     outpaint the other three

All inputs/outputs are saved to logs/example_sr_op/ (git-ignored):

    01_original_tile.png    the fetched Esri tile (input to both tasks)
    02_sr_input.png         SR input  = the original 256x256 parent tile
    03_sr_bicubic_512.png   baseline  = plain 512x512 bicubic upsample of the
                            input (what SR starts from — compare against 04)
    04_sr_output.png        SR output = 512x512 mosaic of its 4 children
    05_op_masked_input.png  OP input  = 1 real quadrant, 3 white holes
    06_op_output.png        OP output = the 3 masked quadrants synthesized

Run from the root of a https://github.com/mvrl/genesis checkout,
with its environment active (`uv sync && source .venv/bin/activate`):

    python demos/example_sr_op.py

SR uses the showcase configuration validated by the paper's pyramid engine
(50 steps, guidance 2.5, SDEdit warm start t=0.4 from the bicubic upsample);
OP uses the paper eval setting (50 steps, cfg = 1.0). EMA weights are loaded
automatically.

Imagery: Esri World Imagery — Source: Esri, Maxar, Earthstar Geographics,
and the GIS User Community.
"""
import os
import sys

# Repo root: works from demos/ (where the repo ships it), the repo root, or a
# standalone download run from inside a checkout.
_HERE = os.path.dirname(os.path.abspath(__file__))
for REPO in (_HERE, os.path.dirname(_HERE), os.getcwd()):
    if os.path.isdir(os.path.join(REPO, "src")):
        break
else:
    raise SystemExit("Run this script from a genesis checkout (github.com/mvrl/genesis).")

# Same sys.path recipe as demos/*.py — demos/ goes LAST (highest priority)
# so its `utils/` package is not shadowed by src/*/utils.py.
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "demos"))

import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from utils.genesis_common import (  # demos/utils/genesis_common.py
    fetch_tile_at_zoom,
    inference_device,
    load_op_denoiser,
    load_sr_denoiser,
    outpaint_context_white_holes_preview,
    run_outpainting,
    run_superresolution,
)

HF_REPO = "MVRL/genesis"
ARCH = "JiT-H/16"  # main H/16 checkpoints (~49 GB download; set "JiT-B/16" for the 7 GB variant)

# A Bavarian village amid fields and forest — rural content the models were trained on.
LAT, LON, ZOOM = 48.6529, 12.9062, 16  # training zoom levels are 10..19


def main() -> None:
    device = inference_device()  # cuda if available, else cpu
    OUT = os.path.join(REPO, "logs", "example_sr_op")
    os.makedirs(OUT, exist_ok=True)
    print(f"Device: {device} | outputs -> {OUT}")

    # 1. Checkpoints from the HF Hub (cached under ~/.cache/huggingface).
    #    The main SR model is DINOv3-conditioned; the main OP model is no-DINO.
    sr_ckpt = hf_hub_download(HF_REPO, "main/superresolution/main_H16/sr-full-tile-stage3-step0800000.ckpt")
    op_ckpt = hf_hub_download(HF_REPO, "main/outpainting/main_H16/op-new-stage3-step0800000.ckpt")
    dino = hf_hub_download(HF_REPO, "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")

    # 2. One real 256x256 Esri World Imagery tile — the input to BOTH tasks.
    tile = fetch_tile_at_zoom(LON, LAT, ZOOM)  # PIL RGB, 256x256
    tile.save(f"{OUT}/01_original_tile.png")

    # 3. Super-resolution: parent at --zoom -> 512x512 mosaic of its 4 children.
    #    target_zoom is the CHILD zoom level (feeds the model's GSD embedder).
    tile.save(f"{OUT}/02_sr_input.png")
    # Bicubic baseline: SR technically starts from this upsample (the SDEdit
    # warm start below anchors on it) — save it so 03 vs 04 shows exactly what
    # the model adds beyond plain interpolation.
    tile.resize((512, 512), Image.BICUBIC).save(f"{OUT}/03_sr_bicubic_512.png")
    sr_model, _ = load_sr_denoiser(sr_ckpt, ARCH, dino_weights=dino, device=device)
    # Showcase settings — the SR configuration the paper's pyramid engine runs
    # (guidance + warm start anchor structure/palette to the parent tile). The
    # engine reads `cfg_scale` (snapshotted at load), so set the attribute.
    sr_model.cfg_scale = 2.5
    sr_model.args.bicubic_init = True             # read live by the sampler
    sr_model.args.bicubic_init_t_start = 0.4
    sr_out = run_superresolution(sr_model, tile, device, target_zoom=ZOOM + 1)
    sr_out.save(f"{OUT}/04_sr_output.png")
    print("SR done: 02_sr_input.png (256 parent) -> 04_sr_output.png (512 mosaic); "
          "compare with 03_sr_bicubic_512.png (plain bicubic baseline)")

    del sr_model  # free memory before loading the OP model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 4. Outpainting: 2x2 quadrant mosaic — 1 real quadrant, 3 masked holes.
    #    Mask convention: white (255) = hole to generate, black (0) = known.
    mask = Image.new("L", (256, 256), 255)  # everything is a hole ...
    mask.paste(0, (0, 0, 128, 128))         # ... except the upper-left quadrant
    # (easier variant — keep the left half known: mask.paste(0, (0, 0, 128, 256)))

    op_model, _ = load_op_denoiser(op_ckpt, ARCH, dino_weights="", device=device)
    outpaint_context_white_holes_preview(tile, mask).save(f"{OUT}/05_op_masked_input.png")
    op_out = run_outpainting(op_model, tile, mask, device, tile_zoom=ZOOM)
    op_out.save(f"{OUT}/06_op_output.png")
    print("OP done: 05_op_masked_input.png (1 real quadrant) -> 06_op_output.png (outpainted)")


if __name__ == "__main__":
    main()
