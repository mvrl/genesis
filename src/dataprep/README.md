# Data Preparation — Quad-Aware Split Builder

This document covers `build_quad_splits_hf.py`, which creates the benchmark splits for the
super-resolution (SR) and outpainting (OP) tasks, and `visualize_splits.py`, which lets you
inspect those splits interactively on an ESRI satellite basemap.

---

## Overview

The Git-10M dataset (`lcybuaa/Git-10M` on HuggingFace) contains **10,503,567** satellite
images organised as Web Mercator tiles at zoom levels 10–18.  For the SR task the model
must predict a 2×2 mosaic of ZL+1 children given a ZL parent; this requires parent→child
relationships to be preserved *within* each split.  A naïve random per-tile split would put
parent and child into different splits, making SR evaluation impossible.

`build_quad_splits_hf.py` assigns splits at the **complete quad level**: every evaluation
quad moves its parent tile *and* all four children together into the same eval split.

---

## Split Construction (`build_quad_splits_hf.py`)

### Inputs

| File | Default location | Description |
|------|------------------|-------------|
| `hierarchy_index.pkl` | `data/` | Quadtree parent→child relationships |
| `location_to_indices.pkl` | `data/` | Tile location (`z_x_y`) → HF dataset row index |
| `World_Urban_Areas.shp` | `data/shapefiles/` | ESRI urban area polygons (EPSG:3857), used for spatial stratification |

### Algorithm (7 steps)

**Step 1 — Find all complete quads.**  
Iterate every tile location; keep only those where the tile *and all 4 ZL+1 children* exist
in the dataset.  This yields **452,080** complete quads across ZL 10–17.

**Step 2 — Identify urban / non-urban quads.**  
Each quad's parent-tile centroid is spatially joined against the 3,598 urban polygons
(ESRI World Urban Areas, EPSG:3857).  Result: 279,645 urban (61.9 %) / 172,435 non-urban.

**Step 3 — Assign spatial test** (10,000 quads, 50 % urban / 50 % non-urban).  
Parent tile + 4 children → `spatial_test`.  **49,825 tiles total.**

**Step 4 — Assign spatial val** (5,000 quads, 50 % urban / 50 % non-urban).  
From remaining quads (excluding spatial-test overlaps).  **24,958 tiles total.**

**Step 5 — Assign random test** (10,000 quads, no spatial constraint).  
Shuffled draw from remaining quads.  **49,826 tiles total.**

**Step 6 — Assign random val** (5,000 quads).  
From remaining quads after random-test exclusion.  **24,969 tiles total.**

**Step 7 — Assign train + eval-tree exclusion.**  
All remaining geo-tagged tiles go to `train` (6,568,890 tiles).  
Then a BFS descends from every eval tile: any tile reachable via parent→child edges is an
*eval-tree descendant* — if left in training, it would form an SR pair where the parent
conditioning image comes from an eval split (data leakage).  These **177,510 tiles** are
labeled `train_excl` and excluded from `splits['train']`.

The 3,607,589 non-geo (Million-AID) tiles with `location=None` are assigned `split=train`
in `index_to_meta` and included in `splits['train']`; they are valid for outpainting but
never form SR pairs (no tile location → no parent lookup).

### Outputs

**`git10m_quad_meta.json`**

```json
{
  "index_to_meta": {
    "<hf_index>": {"location": "z_x_y", "satellite": "google",
                   "split": "train", "lat": 37.7, "lon": -122.4,
                   "gsd": 0.9, "zoom": 17}
  },
  "splits": {
    "train":        [10176479 HF indices, seeded-shuffled, ZL-balanced],
    "val_random":   [24969 indices],
    "val_spatial":  [24958 indices],
    "random_test":  [49826 indices],
    "spatial_test": [49825 indices]
  }
}
```

The `splits` lists are shuffled once with `seed=42` so any prefix slice is ZL-balanced
(prevents the all-ZL=10-roots problem that arises from the HF dataset's ZL-contiguous ordering).

**`hierarchy_index_quad.pkl`**  
The hierarchy object with `splits` dict embedded (same structure as above).

### Final Split Summary (seed=42)

| Split | Geo tiles | Total in `splits[*]` | Complete quads | Purpose |
|-------|-----------|----------------------|----------------|---------|
| `train` | 6,568,890 | **10,176,479** (+ 3.6M non-geo) | 362,572 | SR + OP training |
| `train_excl` | 177,510 | — (not in any splits list) | — | Eval-tree descendants; excluded from SR |
| `val_random` | 24,969 | 24,969 | **5,000** | IID validation; drives ablation gating |
| `val_spatial` | 24,958 | 24,958 | **5,000** (50% urban) | Spatial generalization gap |
| `random_test` | 49,826 | 49,826 | **10,000** | Final IID benchmark |
| `spatial_test` | 49,825 | 49,825 | **10,000** (50% urban) | Final spatial benchmark |

**SR training pairs** (parent tiles with ≥1 child in train, same-split parents only):
892,593 total — 362,572 complete (4/4 children) + 530,021 partial (1–3/4 children).

### Data Leakage Prevention

Two mechanisms ensure no eval image can appear as a training conditioning input:

1. **Eval-tree BFS** (build script): tiles that are ZL+1, ZL+2, … descendants of eval
   tiles are labeled `train_excl` and removed from `splits['train']`.
2. **Same-split parent guard** (dataloader): `HFParentChildDataset` also enforces
   `parent_idx ∈ split_idx_set`, so even if a future build script change misses a tile,
   no cross-split pair can be formed at runtime.

### Split integrity verification

The build script prints a verification pass at the end of its run (the `VERIFICATION` block,
shown in the sample output `build_quad_splits_hf.txt`) confirming:

| Check | Result | Meaning |
|-------|--------|---------|
| Train tiles with eval SR-parent | **0** | No training pair uses an eval conditioning image |
| Eval parent→child split mismatches | **387** | Multi-level boundary edge cases (<0.1% of quads, benign — these pairs are unused by both train and eval dataloaders) |
| Duplicate HF indices in metadata | **0** | No tile in two splits |
| `train_excl` tiles without eval ancestor | **0** | All excluded tiles properly descend from eval |

The 387 edge cases arise when a tile is added to one eval split as a *child* (via a ZL=N→N+1 quad) and to a different eval split as a *parent* (via a ZL=N→N+1 quad at a higher zoom). The SR dataloaders skip them because `require_complete_quad` finds no single split containing all 4 siblings.

### Running the build script

```bash
cd src/dataprep
source .venv/bin/activate
python build_quad_splits_hf.py
# outputs saved under data/ (override with $GENESIS_GIT10M_META / $GENESIS_HINDEX)
```

Rather than rebuild, you can download the prebuilt files from the public Google
Drive release folder under `shared/`. Training/eval look for them under
`$GENESIS_CKPT_BASE/shared/` (and eval auto-pulls them there on first use), so
mirror that layout:

```bash
GDRIVE_FOLDER_ID=1OKe2YRJFkwwFD3GWXVJMEr6qKGJz0qyA
DEST="${GENESIS_CKPT_BASE:-logs/ckpts}/shared"
rclone copy gdrive:shared/git10m_quad_meta.json    "$DEST/" \
    --drive-root-folder-id "$GDRIVE_FOLDER_ID" --progress
rclone copy gdrive:shared/hierarchy_index_quad.pkl "$DEST/" \
    --drive-root-folder-id "$GDRIVE_FOLDER_ID" --progress
```

---

## Split Visualizer (`visualize_splits.py`)

An interactive Gradio app that renders each geo-tagged tile as a colored dot at its
lat/lon center on an ESRI World Imagery satellite basemap.  Val/test splits are always
shown in full; the train split is randomly sampled (default 50k) so the map stays
responsive.  Clicking **Resample train tiles** draws a fresh random subset, giving broad
coverage over repeated refreshes.

Non-geo tiles (Million-AID, `location=None`) have no coordinates and are not shown.

### Color coding

| Color | Split |
|-------|-------|
| 🟦 Blue | `train` (50k random sample per view) |
| 🔷 Slate | `train_excl` (eval-tree descendants) |
| 🟩 Green | `val_random` (always shown fully) |
| 🟧 Orange | `val_spatial` (always shown fully) |
| 🟥 Red | `random_test` (always shown fully) |
| 🟪 Purple | `spatial_test` (always shown fully) |

### Controls

| Interaction | Effect |
|-------------|--------|
| Pan / zoom | Navigate the map (ESRI satellite, unlimited zoom) |
| **Visible splits** checkboxes | Toggle individual splits on/off instantly (client-side, no reload) |
| Hover a dot | Shows split label and tile ID (`z_x_y`) |
| **🔄 Resample train tiles** button | Picks a new random train sample and redraws (zoom/pan preserved) |

### Usage

```bash
source .venv/bin/activate
cd src/dataprep

# default: all val/test in full, 50k random train tiles
python visualize_splits.py

# custom meta path
python visualize_splits.py --meta data/git10m_quad_meta.json

# denser train coverage per resample (slower render)
python visualize_splits.py --n_train 100000

# generate a public share link (useful on compute nodes without port forwarding)
python visualize_splits.py --port 7861 --share
```

Then open `http://localhost:7860` in your browser.

> **Note:** Loading the 1.7 GB `git10m_quad_meta.json` takes ~30 s on first launch.
> After that, each resample and redraw takes a few seconds.
> The `train` split has 6.6M geo-tagged tiles — keep `--n_train` ≤ 100k for smooth rendering.
