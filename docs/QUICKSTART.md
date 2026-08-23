# Quick start — inference

One SR ×2 pass and one quadrant-outpainting pass on a live Esri World Imagery tile, with the
main JiT-H/16 checkpoints pulled from the [🤗 release](https://huggingface.co/MVRL/genesis)
(~49 GB the first time; swap `H16`→`B16` in the two checkpoint paths for a lighter 7 GB variant):

```python
import os, sys
sys.path.insert(0, "src"); sys.path.insert(0, "demos")

from huggingface_hub import hf_hub_download
from PIL import Image
from utils.genesis_common import (fetch_tile_at_zoom, inference_device,
                                  load_sr_denoiser, load_op_denoiser,
                                  outpaint_context_white_holes_preview,
                                  run_superresolution, run_outpainting)

device = inference_device()                       # cuda if available, else cpu
out = "logs/example_sr_op"; os.makedirs(out, exist_ok=True)

sr_ckpt = hf_hub_download("MVRL/genesis", "main/superresolution/main_H16/sr-full-tile-stage3-step0800000.ckpt")
op_ckpt = hf_hub_download("MVRL/genesis", "main/outpainting/main_H16/op-new-stage3-step0800000.ckpt")
dino    = hf_hub_download("MVRL/genesis", "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")

tile = fetch_tile_at_zoom(lon=12.9062, lat=48.6529, zoom=16)   # a real 256×256 tile (Bavarian village)
tile.save(f"{out}/01_original_tile.png")

# SR: z16 parent → 512×512 mosaic of its four z17 children.
# Showcase config = the paper's pyramid-engine SR settings (cfg 2.5 + bicubic
# warm start); the engine reads `cfg_scale`, so set the attribute (not args.cfg).
tile.save(f"{out}/02_sr_input.png")
tile.resize((512, 512), Image.BICUBIC).save(f"{out}/03_sr_bicubic_512.png")  # baseline SR starts from
sr_model, _ = load_sr_denoiser(sr_ckpt, "JiT-H/16", dino_weights=dino, device=device)
sr_model.cfg_scale = 2.5
sr_model.args.bicubic_init = True
sr_model.args.bicubic_init_t_start = 0.4
run_superresolution(sr_model, tile, device, target_zoom=17).save(f"{out}/04_sr_output.png")

# OP: keep the upper-left quadrant real, outpaint the other three (white = hole)
mask = Image.new("L", (256, 256), 255); mask.paste(0, (0, 0, 128, 128))
op_model, _ = load_op_denoiser(op_ckpt, "JiT-H/16", dino_weights="", device=device)
outpaint_context_white_holes_preview(tile, mask).save(f"{out}/05_op_masked_input.png")
run_outpainting(op_model, tile, mask, device, tile_zoom=16).save(f"{out}/06_op_output.png")
```

Every input and output lands in `logs/example_sr_op/`, numbered in pipeline order — original
tile, SR input / bicubic baseline / SR output, masked OP input, outpainted OP output. The SR
pass technically starts from the bicubic upsample (the warm start anchors on it), so comparing
`03_sr_bicubic_512.png` against `04_sr_output.png` shows exactly what the model adds beyond
plain interpolation.

Or run the same as one file — [`demos/example_sr_op.py`](../demos/example_sr_op.py):

```bash
python demos/example_sr_op.py
```

> Live imagery: Esri World Imagery — Source: Esri, Maxar, Earthstar Geographics, and the GIS
> User Community.
