# Genesis Demos

Gradio apps for the current SR and outpainting models in
[`src/superresolution/`](../src/superresolution/) and
[`src/outpainting/`](../src/outpainting/), plus the full-pyramid generator in
[`src/pyramid/`](../src/pyramid/).

The eight apps fall into four groups: **super-resolution**, **outpainting**,
the **full-pyramid builder**, and **no-model interpolation baselines**.

| App | Group | What it does | Port |
|-----|-------|--------------|:---:|
| `app_superresolution.py` | SR | 256→512 full-tile ×2 SR; shows parent · bicubic · SR mosaic · GT mosaic. | 7860 |
| `app_upsample_comp.py` | SR | 1→2→3-level upsample comparison: bicubic · **Genesis SR (iterative, optional bicubic-init SDEdit warm start)** · Esri GT, all quantified (PSNR / SSIM / LPIPS / DISTS). | 7866 |
| `app_outpainting_quadrant.py` | OP | Single 256² tile: mark 1–3 quadrants as known context → outpaint the rest. The model the whole pyramid is built on, in isolation. | 7862 |
| `app_outpainting_large.py` | OP | Sandbox for the **lateral** OP mechanism: pick a zoom and canvas size (N×N cells, default 8), place seed tiles (a seed centers on a cell junction → its 4 quadrants seed 4 neighbors), then watch the grid fill outward, full-resolution, at one zoom; download the composite (seeds boxed in red). This is exactly the engine behind the `unified` strategy. | 7867 |
| `app_pyramid_unified.py` | Pyramid | Place seeds → fill a complete 4-level quadtree (default z15→z18) with the **[`unified`](#pyramid-algorithm--unified)** strategy (D-1 greedy outpainting → SR → free-floating seam-fix → reconcile); streams live tile-by-tile, stitched-PNG download. | 7865 |
| `app_pyramid_unified_large.py` | Pyramid | Same `unified` strategy, **deep 6-level** quadtree (1365 tiles, leaf 32×32; default z13→z18). Cells render tiny by design; the download is stitched **vertically** (apex→base) for a deep-zoom figure. Much heavier/slower — large-memory GPU. | 7868 |
| `app_downsample_eval.py` / `app_upsample_eval.py` | Baseline | Pure interpolation baselines (no model) for down-/up-sampling. | 7863 / 7864 |

For a non-interactive quick start (one SR pass + one outpainting pass on a live Esri tile,
inputs/outputs saved to `logs/example_sr_op/`), run `python demos/example_sr_op.py`.

SR is strictly ×2 (ZL → ZL+1) per call. For iterative ×4 / ×8 / ×16 see the
`--scale` flag documented in [`src/eval/README.md`](../src/eval/README.md).

The pyramid strategy is described as paper-ready pseudocode in
[Pyramid algorithm — `unified`](#pyramid-algorithm--unified) below.

---

## Setup

```bash
source .venv/bin/activate          # from repo root, after: uv sync
```

That's it. The demos auto-fetch missing files from the public Hugging
Face model repo
(<https://huggingface.co/MVRL/genesis>) — no authentication or setup —
on the first **Generate** click and cache them under `<repo>/logs/ckpts/`
(git-ignored). Defaults are the **main 800k-step runs** — SR uses
DINO + LPIPS (weight=0.5); OP is locked at no-DINO + no-time-weight
under the quadrant mask curriculum (`quad3` → `quad2_{adj,diag}` →
`quad1` → uniform mix). Both B16 and H16 are available as dropdown
options:

| File | Hugging Face source (path in [`MVRL/genesis`](https://huggingface.co/MVRL/genesis)) |
|------|--------------------------------------------------------------------------------------|
| SR B16 | `main/superresolution/main_B16/sr-full-tile-stage3-step0800000.ckpt` |
| SR H16 | `main/superresolution/main_H16/sr-full-tile-stage3-step0800000.ckpt` |
| OP B16 | `main/outpainting/main_B16/op-new-stage3-step0800000.ckpt` |
| OP H16 | `main/outpainting/main_H16/op-new-stage3-step0800000.ckpt` |
| DINOv3 | `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` |

To stage everything upfront (offline use, etc.):

```bash
bash demos/sync_ckpts.sh           # main ckpts + DINOv3 weights → $GENESIS_CKPT_BASE
```

Full Hugging Face layout in the root [`README.md`](../README.md) (see *Data & weights*).
For a one-file inference example, see [`example_sr_op.py`](example_sr_op.py) in this folder.

---

## Launch

```bash
python demos/app_superresolution.py        # SR — 256→512 full-tile
python demos/app_upsample_comp.py          # SR — bicubic vs Genesis SR vs GT (d=1..3)
python demos/app_outpainting_quadrant.py   # OP — quadrant context (single tile)
python demos/app_outpainting_large.py      # OP — lateral grid fill (unified engine)
python demos/app_pyramid_unified.py        # Fill the whole pyramid — unified (4 levels)
python demos/app_pyramid_unified_large.py  # Same, deep 6-level pyramid (vertical figure)
```

Each app prints a local URL and a public `share=True` link.

**Usage** is uniform: click the map → tile auto-fetches → adjust whatever the
app exposes (quadrants, sampling steps) → **Generate**. The OP demo overwrites
known pixels with GT, so visible diffs are confined to the hole.

### `app_upsample_comp.py` — specifics

Compares **bicubic** vs **Genesis SR** vs **Esri GT** at three upsampling
depths from one parent tile. Both bicubic and SR are quantified against GT
with **PSNR / SSIM / LPIPS / DISTS** in the metrics table; SR runtime per
depth is also reported.

| Knob | Default | Effect |
|------|---------|--------|
| **Parent zoom** | 15 | Zoom level of the source parent tile (10–17) |
| **SR architecture** | `JiT-B/16` | B16 is faster; H16 is the larger main run |
| **Depths to run SR** | 1, 2, 3 | Uncheck to skip cascades (bicubic+GT still shown) |
| **Bicubic init (SDEdit warm start)** | **ON** | Start diffusion from the bicubic-upsampled parent instead of pure noise — preserves coarse structure from the parent zoom. Toggle off to start from random `z₀`. |
| **Bicubic init t_start** | 0.5 | How far into the trajectory to inject the bicubic prior (higher = more parent structure preserved) |
| **Sampling steps** | 50 | Euler steps per SR call |
| **CFG scale** | 1.0 | Inference CFG (1.0 = no guidance) |

SR was trained for **single-step ×2** (zoom_diff=1). The d=2 / d=3 panels
apply SR **iteratively** (1 + 4 + 16 = 21 SR calls for d=3) — useful for
visualising cascaded super-resolution but the metrics drop is expected
because the cascade is out-of-distribution.

---

## Configuration

Before running any demo, set `GENESIS_CKPT_BASE` in your shell:

```bash
export GENESIS_CKPT_BASE=/path/to/checkpoints/project_genesis/ckpt_cache
# Add to ~/.bashrc to persist across sessions
```

Checkpoints are auto-downloaded from Hugging Face on first **Generate** — no
authentication needed. To pull everything upfront, run
`bash demos/sync_ckpts.sh`.

| Variable | Default | Effect |
|----------|---------|--------|
| `GENESIS_CKPT_BASE` | `<repo>/logs/ckpts` | Where auto-downloaded ckpts live (mirrors the Hugging Face repo layout) |
| `GENESIS_HF_REPO` | `MVRL/genesis` | Override the Hugging Face model repo to pull from |
| `GENESIS_DINO_WEIGHTS` | `$GENESIS_CKPT_BASE/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` | Skip the auto-download and use a pre-existing DINO weights file |
| `GENESIS_DEVICE` | auto | `cuda` or `cpu` |
| `GENESIS_CKPT_WEIGHTS` | `ema` | `online` to skip EMA-into-live copy |
| `GENESIS_SR_B16_CKPT` / `GENESIS_OP_B16_CKPT` (and `_H16_`) | _(none)_ | Point a dropdown entry at an arbitrary path |
| `GRADIO_SR_PORT` / `GRADIO_OP_PORT` / `GRADIO_DS_PORT` | 7860 / 7862 / 7863 | SR / OP-quadrant / downsample-eval ports |
| `GRADIO_US_PORT` / `GRADIO_US_COMP_PORT` | 7864 / 7866 | Upsample eval / comparison ports |
| `GRADIO_OP_LARGE_PORT` | 7867 | Large-grid lateral-outpainting port |
| `GRADIO_FULL_PORT` | 7865 | Pyramid builder — `unified` port |
| `GRADIO_FULL_LARGE_PORT` | 7868 | Pyramid builder — `unified` deep 6-level port |

Per-run override: paste a full ckpt path into each app's "Optional checkpoint
path" textbox.

---

## Notes

- **Auto-download**: missing files are fetched via `huggingface_hub` from
  [`MVRL/genesis`](https://huggingface.co/MVRL/genesis) on first
  **Generate**. If the Hugging Face source doesn't exist, the app fails with
  a clear `FileNotFoundError` listing both the expected local path and the
  Hugging Face source.
- `torch.hub` also fetches the DINOv3 *Python code* (not the .pth weights) on
  first launch; `lpips`/`DISTS_pytorch` fetch their AlexNet/VGG weights
  (~6 MB) on first metric eval.
- First **Generate** is slow (loads ckpt into in-memory cache); restart
  clears it.
- Loaders use `strict=False`, so an arch-mismatched ckpt loads silently with
  many tensors at random init — eyeball the GT panel to confirm a real load.
- **Pyramid strategy**: `app_pyramid_unified.py` runs the **`unified`**
  strategy — see [Pyramid algorithm — `unified`](#pyramid-algorithm--unified)
  below. Outputs (tiles + stitched `pyramid_viz.png`) land under
  `demos/logs/pyramid_unified_results/` (gitignored).

---

## Pyramid algorithm — `unified`

`app_pyramid_unified.py` fills a complete 4-level quadtree from seed tiles with
the **`unified`** strategy ([`../src/pyramid/strategies/unified.py`](../src/pyramid/strategies/unified.py)).
It concentrates the work at the **penultimate (D-1) level**: bring every seed
there, greedily outpaint that level, super-resolve once to the leaf, repair seams
with a **free-floating** outpaint pass, then reconcile. (Step-by-step notation and
pseudocode follow below; the evaluation protocol is in
[`evaluate_pyramid.md`](evaluate_pyramid.md).)

**Notation.** Subtree 𝒯 rooted at *r*=(z_r,·,·), leaf zoom z_L; tile *t*=(z,x,y)
with image *I(t)*; seeds 𝒮 (ground truth, any zoom — never overwritten). Models:
`OP_φ(c, m; z)` outpaints the hole (mask *m*) of a 256² canvas *c*, known pixels
frozen, in-distribution masks = unions of 128² quadrants; `SR_θ(c; z, t)` is
**single-canvas multi-window MultiDiffusion** ×2 super-resolution (SDEdit
warm-start *t*, window overlap *o*) — overlapping 256² windows whose per-step
velocities are fused over a plateau partition-of-unity, so it upscales a whole
G·256² mosaic seam-free in one trajectory. **Every** SR in the algorithm goes
through `SR_θ`: both the coarse-seed upsample (step 2) and the D-1→leaf pass
(step 4). The lateral OP engine treats a 2×2 window of 128² quadrants as one 256²
tile at the current zoom (the [`app_outpainting_large.py`](app_outpainting_large.py) mechanism).

```
Algorithm  unified
Input: seeds 𝒮, subtree 𝒯, models {OP_φ, SR_θ}, overlap o, warm-start t
 1: z_{D-1} ← z_leaf − 1                                ▷ penultimate level (a 2N×2N quadrant grid)
 2: for each seed s ∈ 𝒮:  place onto the D-1 quadrant grid
        zoom(s) < z_{D-1} → SR_θ up to z_{D-1} (footprint stitched → one MultiDiffusion pass per level)
        zoom(s) = z_{D-1} → as-is;   zoom(s) = z_leaf → bicubic ÷2 (one quadrant)
 3: LateralOutpaint(level z_{D-1}, o)                   ▷ greedy-fill the rest of D-1, then assemble its tiles
 4: c ← stitch(level z_{D-1});  M ← SR_θ(c ; z_leaf, t) with leaf seeds anchored;  write leaf tiles
 5: F ← SeamFix(leaf)                                   ▷ free-floating outpaint; F = leaf tiles it modified
 6: Reconcile(keep)                                     ▷ coarser levels = bicubic ÷2 of children; seeds kept
        keep(t) = z(t)=z_{D-1} ∧ resolved(t) ∧ children(t) ∩ F = ∅   ▷ default: keep pre-SR D-1 tile (skip_unblended=True)
 7: return { I(t) : t ∈ 𝒯 }

Procedure LateralOutpaint(level z, overlap o):          ▷ same-zoom, full-resolution, no downsampling
 1: lay a 2n × 2n grid of 128² quadrants (n = tiles/axis); seed it with the known quadrants
 2: repeat
 3:     W ← all overlapping 2×2 quadrant windows with 1–3 known cells   ▷ a window = one 256² z-tile
 4:     for w ∈ W, most-constrained (most-known) first do
 5:         o′ ← OP_φ(assemble(w), hole_mask(w); z); write the filled 128² quadrants back
 6: until no window has a fillable hole;  write each completed tile to the store as it finishes

Procedure SeamFix(leaf):                                ▷ not confined to the tile grid
 1: M ← stitch(leaf);  score every internal tile boundary by mean |Δ| across it
 2: for each boundary with score > τ (worst-first, ≤ max_fix):
 3:     crop a 256² window CENTERED on the seam (straddling the two tiles)
 4:     mask a band of width b over the seam;  o′ ← OP_φ(window, band; z_leaf)
 5:     feather o′ back into M over the band (known pixels unchanged)
 6: re-split M into leaf tiles (seeds preserved)
```

<details>
<summary>LaTeX source (paste-ready <code>algorithmic</code> floats)</summary>

```latex
\begin{algorithm}\caption{\textsc{unified} pyramid fill}\label{alg:unified}
\begin{algorithmic}[1]
\Require seeds $\mathcal{S}$, subtree $\mathcal{T}$ (leaf zoom $z_L$), models $\mathrm{OP}_\phi,\mathrm{SR}_\theta$, overlap $o$, warm-start $t$, flag \textsc{skip} (default: \texttt{True})
\State $z_{D\text{-}1}\gets z_L-1$ \Comment{penultimate level $=2N{\times}2N$ quadrant grid}
\For{each seed $s\in\mathcal{S}$} \Comment{place onto the D-1 quadrant grid}
  \If{$\mathrm{zoom}(s)<z_{D\text{-}1}$} stitch its footprint and $\mathrm{SR}_\theta$ up to $z_{D\text{-}1}$ (one MultiDiffusion pass / level)
  \ElsIf{$\mathrm{zoom}(s)=z_{D\text{-}1}$} use as-is
  \Else{ $\;\mathrm{zoom}(s)=z_L$:} bicubic $\div2$ into one quadrant \EndIf
\EndFor
\State $\textsc{LateralOutpaint}(z_{D\text{-}1},o)$ \Comment{greedy-fill D-1, assemble tiles}
\State $c\gets\mathrm{stitch}(z_{D\text{-}1})$;\ $M\gets\mathrm{SR}_\theta(c;z_L,t)$ (leaf seeds anchored);\ write leaf
\State $F\gets\textsc{SeamFix}(\text{leaf})$ \Comment{$F=$ leaf tiles modified}
\State $\textsc{Reconcile}(\mathit{keep})$,\quad $\mathit{keep}(t)=\textsc{skip}\wedge z(t){=}z_{D\text{-}1}\wedge \mathrm{children}(t)\cap F=\varnothing$
\State \Return $\{I(t):t\in\mathcal{T}\}$
\end{algorithmic}\end{algorithm}

\begin{algorithm}\caption{\textsc{LateralOutpaint}(level $z$, overlap $o$)}
\begin{algorithmic}[1]
\State lay a $2n{\times}2n$ grid of $128^2$ quadrants; seed it with the known quadrants
\Repeat
  \State $W\gets$ overlapping $2{\times}2$ quadrant windows with $1\text{–}3$ known cells \Comment{window $=$ one $256^2$ tile at $z$}
  \For{$w\in W$, most-constrained first}
    \State $o'\gets\mathrm{OP}_\phi(\mathrm{assemble}(w),\ \mathrm{holes}(w);\ z)$;\ write filled $128^2$ quadrants back
  \EndFor
\Until{no window has a fillable hole} \Comment{write each completed tile live}
\end{algorithmic}\end{algorithm}

\begin{algorithm}\caption{\textsc{SeamFix}(leaf) $\to$ modified tiles $F$}\label{alg:seamfix}
\begin{algorithmic}[1]
\State $M\gets\mathrm{stitch}(\text{leaf})$;\ score every internal tile boundary by mean $|\Delta|$
\For{each boundary with score $>\tau$ (worst-first, $\le \texttt{seam\_max}$)}
  \State crop a $256^2$ window \emph{centered on the seam};\ mask a band of width $b$
  \State $o'\gets\mathrm{OP}_\phi(\text{window},\text{band};z_L)$;\ feather $o'$ back over the band
\EndFor
\State re-split $M$ into leaf tiles (seeds preserved);\ \Return $F$
\end{algorithmic}\end{algorithm}
```
</details>

**Cost** scales with the D-1 size: on the z15→z18 default, D-1 = z17 (a 4×4 level)
⇒ ≤ 49 OP windows for the lateral fill, plus one SR (D-1→leaf) and the seam-fix
(≤ `seam_max` OP calls, only where a real seam exceeds the threshold).

**Knobs** (app parameter sliders → `ctx.params`): `arch` / `cfg` / `steps` (the bundle's
model size, CFG, diffusion steps); **SR window overlap** = `uni_overlap` and **SR
warm-start** = `uni_t`, which tune the SR passes only (the greedy outpainting and
seam-fix use `op_fill`, which has no overlap/t knob). The seam-fix has built-in
defaults (not exposed): `seam_thresh` (boundary-discontinuity threshold, 14),
`seam_band` (repaint band width px, 64), `seam_max` (max regions fixed, 64).

**`reconcile_skip_unblended`** (checkbox, **default on**): in step 6, keep the
original penultimate-level (z=leaf−1) tiles whose leaf children were *not*
seam-fixed, instead of overwriting them with ÷2(leaf). Preserves the pre-SR D-1
content and generally produces better visual quality; uncheck to fall back to
full reconcile-all (exact zoom consistency at every level).
