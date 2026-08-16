# Full-Pyramid Generation — Framework

Unifies the Genesis super-resolution (SR) and outpainting (OP) models into one
framework that synthesizes a **complete quadtree pyramid** of 256×256 satellite
tiles from an **arbitrary set of multi-scale seed tiles**.

```
src/pyramid/
├── tiles.py        # absolute (z,x,y) coords, TileStore, quadtree + LCA helpers
├── dataset.py      # Subtree carving, global-dense loader, seed protocols
├── geometry.py     # subtree-relative layout: level_grid_n, tile_pos, stitch_level
├── models.py       # thin wrapper over demos/utils/genesis_common (quad-SR, quadrant-OP)
├── unified_sr.py   # single-canvas MultiDiffusion ×2 SR (sr_level_unified)
├── multidiffusion.py  # MultiDiffusion SR primitive used by models.py
├── verify_data.py  # data-integrity / seed-protocol sanity CLI
└── strategies/
    ├── base.py         # Strategy ABC, registry, reconcile_up
    └── unified.py # the one generation strategy (see algorithm doc below)
```

The single generation **strategy** is `unified`: bring seeds to the
penultimate (D-1) level, greedily outpaint that level, super-resolve once to the
leaf, repair seams with a free-floating outpaint pass, then reconcile. Its
paper-ready pseudocode lives in
[`../../demos/README.md`](../../demos/README.md) (Pyramid algorithm).

## Run

The strategy is driven by the Gradio demo (see [`../../demos/README.md`](../../demos/README.md)):

```bash
python demos/app_pyramid_unified.py    # place seeds on a map → fill the pyramid
```

Data-integrity / protocol sanity check (no GPU):

```bash
PYTHONPATH=src python -m pyramid.verify_data
```

## Checkpoints

Checkpoints resolve **local-first, with Hugging Face auto-download as fallback**.
The wrapper probes `$GENESIS_CKPT_BASE` (and `logs/ckpts`); if a checkpoint (or
the DINOv3 weights) is missing it is fetched from the public model repo
[`MVRL/genesis`](https://huggingface.co/MVRL/genesis) via `huggingface_hub`
into `logs/ckpts/` and reused — no authentication needed. DINOv3 weights use
`$GENESIS_DINO_WEIGHTS` if set, else auto-download. See
[`../../demos/README.md`](../../demos/README.md) for the full checkpoint table
and env vars.

Set `GENESIS_CKPT_BASE` before running (add to `~/.bashrc` to persist):

```bash
export GENESIS_CKPT_BASE=/path/to/checkpoints/project_genesis/ckpt_cache
```

## Adding a strategy

Create `strategies/<name>.py` with an `@register("<name>")` class implementing
`generate(self, store, sub, seeds, ctx)` (fill `store` in place; seeds are
pre-placed — preserve them), and import it in `strategies/__init__.py`. Reusable
helpers: `reconcile_up` (base.py), `level_grid_n`/`tile_pos`/`stitch_level`
(geometry.py), `lca` (tiles.py), and the model calls on `ctx.bundle`
(`op_fill`, `sr_level_unified`).
