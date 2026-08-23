<h1 align="center">Genesis: A Generative Engine for Hierarchical Satellite Image Synthesis</h1>

<p align="center">
  <a href="https://subash-khanal.github.io/genesis"><img src="https://img.shields.io/badge/Project-Page-2ea44f" alt="Project Page"></a>
  <img src="https://img.shields.io/badge/arXiv-coming%20soon-b31b1b" alt="arXiv coming soon">
  <a href="https://huggingface.co/MVRL/genesis"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow" alt="Hugging Face Models"></a>
</p>

<p align="center">
  <a href="https://subash-khanal.github.io">Subash Khanal</a>,
  <a href="https://www.linkedin.com/in/yangzhi-cui/">Yangzhi Cui</a>,
  <a href="https://dcher95.github.io">Daniel Cher</a>,
  <a href="https://ericx003.github.io">Eric Xing</a>,
  <a href="https://brian-j-wei.github.io">Brian Wei</a>,
  <a href="https://vishu26.github.io">Srikumar Sastry</a>,
  <a href="https://jacobsn.github.io">Nathan Jacobs</a>
</p>

<p align="center"><b>Washington University in St. Louis</b></p>

<p align="center"><b>ACM SIGSPATIAL 2026</b> — <b><i>Oral Presentation</i></b></p>


![Genesis: sparse seed tiles → full multi-scale quadtree](assets/task.png)

Genesis completes a sparse set of seed satellite tiles into a full, zoomable Web-Mercator
**quadtree** — filling every scale and location — with two flow-matching *JiT* operators
(parent-conditional super-resolution + mask-based outpainting) composed into one pyramid engine.
From the input seeds, the engine (1) takes the seed tiles, (2) brings them to a common working
level, (3) greedily out-paints to complete that level, (4–5) super-resolves down to the leaf
tiles, and (6) fills the coarser levels by downsampling — yielding a pyramid that is consistent
across scales and honors every seed.

![The Genesis pyramid-completion pipeline](assets/pipeline.png)

## Install

Uses [uv](https://docs.astral.sh/uv/). From the repo root:
```bash
uv sync                     # creates .venv, installs the pinned deps (Python 3.13, torch 2.8 cu128)
source .venv/bin/activate   # or prefix commands with `uv run`
```
For CPU-only, drop the `pytorch-cu128` index in `pyproject.toml` first.

## Checkpoints & data

Everything auto-downloads on first use — no authentication, no setup, nothing on a private path.

| Artifact | Hosted on | How it's obtained |
|----------|-----------|-------------------|
| Checkpoints (main **and** ablation) | 🤗 [`MVRL/genesis`](https://huggingface.co/MVRL/genesis) | auto-download to `$GENESIS_CKPT_BASE` (`--exp main_b16 / main_h16 / abl_*`) |
| DINOv3-sat encoder weights | 🤗 [`MVRL/genesis`](https://huggingface.co/MVRL/genesis) | auto-download; override with `$GENESIS_DINO_WEIGHTS` |
| Split meta + hierarchy index | 🤗 [`MVRL/genesis`](https://huggingface.co/MVRL/genesis) `shared/` | auto-download to `$GENESIS_CKPT_BASE/shared/` |
| Git-10M tiles | 🤗 [`lcybuaa/Git-10M`](https://huggingface.co/datasets/lcybuaa/Git-10M) | `datasets.load_dataset(...)` (override cache with `$HF_DATASETS_CACHE`) |
| dense500 benchmark | 🤗 [`dcher95/dense500`](https://huggingface.co/datasets/dcher95/dense500) | auto-download via `pyramid.dataset.GlobalDense` |

Files cache under `$GENESIS_CKPT_BASE` (default `<repo>/logs/ckpts/`). To pre-download the full
release (~70 GB): `bash src/eval/download_ckpts.sh`.

## Quick start — inference

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

Or run the same as one file — [`demos/example_sr_op.py`](demos/example_sr_op.py):

```bash
python demos/example_sr_op.py
```

> Live imagery: Esri World Imagery — Source: Esri, Maxar, Earthstar Geographics, and the GIS
> User Community.

## Demos

The main demo is the **full-pyramid generator**: drop sparse seed tiles on a map and Genesis
completes the entire multi-scale quadtree, streaming tile-by-tile.

```bash
python demos/app_pyramid_unified.py   # opens a Gradio app; prints a local + shareable URL
```

Click the map to drop seeds → **Generate**; checkpoints auto-download on first use. Other apps
(per-tile SR, single-tile outpainting, lateral-outpainting grid, deep 6-level pyramid, no-model
interpolation baselines) are in [`demos/README.md`](demos/README.md).

## Evaluate

One command per paper table, with the paper-exact defaults baked in — see
[`docs/EVALUATION.md`](docs/EVALUATION.md):

```bash
bash src/eval/run_all_sr.sh          # super-resolution
bash src/eval/run_all_op.sh          # outpainting
bash src/eval/run_all_ablation.sh    # SR ablation
bash src/eval/run_all_dense500.sh    # dense500 pyramid completion
```

## Train

Train the two operators from scratch — see [`docs/TRAINING.md`](docs/TRAINING.md):

```bash
cd src/superresolution && ./run_sr.sh    # parent-conditional SR
cd src/outpainting     && ./run_op.sh    # mask-based outpainting
```

## Repository

```
src/
├── superresolution/   # Parent-conditional 256 → 512 SR (ZL → ZL+1)
├── outpainting/       # Mask-based 256×256 image completion
├── pyramid/           # Full-pyramid generation engine (multi-scale tile completion)
├── eval/              # Per-sample + pyramid evaluation
├── dataprep/          # One-shot scripts to build the shared meta + hierarchy
└── util/              # Shared building blocks (RoPE, RMSNorm, sin-cos pos embed)
```

| Guide | What |
|-------|------|
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | evaluate the released models on the paper's benchmarks |
| [`docs/TRAINING.md`](docs/TRAINING.md) | train the SR & OP operators |
| [`docs/HYPERPARAMETERS.md`](docs/HYPERPARAMETERS.md) | every training/eval hyperparameter + sample count (paper vs. code) |
| [`docs/METRICS.md`](docs/METRICS.md) · [`docs/METRICS_DENSE500.md`](docs/METRICS_DENSE500.md) | metric definitions (tile · pyramid) |
| [`demos/README.md`](demos/README.md) | interactive Gradio demos |

## Citation

```bibtex
@inproceedings{khanal2026genesis,
  title     = {{Genesis}: A Generative Engine for Hierarchical Satellite Image Synthesis},
  author    = {Khanal, Subash and Cui, Yangzhi and Cher, Daniel and Xing, Eric and
               Wei, Brian and Sastry, Srikumar and Jacobs, Nathan},
  booktitle = {ACM SIGSPATIAL International Conference on Advances in
               Geographic Information Systems (SIGSPATIAL)},
  year      = {2026},
}
```

## License & acknowledgments

Released under the MIT License — see [`LICENSE`](LICENSE). Both Genesis operators build on
[JiT](https://github.com/LTH14/JiT) — we thank the authors for releasing their codebase. The
DINOv3-sat encoder weights are
from Meta AI's [DINOv3](https://github.com/facebookresearch/dinov3) release (governed by Meta's
DINOv3 license). Training data comes from the
[Git-10M](https://huggingface.co/datasets/lcybuaa/Git-10M) corpus; demo imagery is fetched live
from Esri World Imagery.
