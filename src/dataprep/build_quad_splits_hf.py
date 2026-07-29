"""
build_quad_splits_hf.py — Quad-aware split builder for SR + OP tasks.

Requires:
  - hierarchy_index.pkl        data/hierarchy_index.pkl
  - location_to_indices.pkl    data/location_to_indices.pkl
  - World_Urban_Areas.shp      data/shapefiles/World_Urban_Areas.shp

Difference from build_splits_hf.py
------------------------------------
The OP / SR tasks require parent → child relationships.  This script therefore:
  • Assigns eval splits (val_random, val_spatial, random_test, spatial_test) only to
    COMPLETE 2×2 quads — every parent AND all 4 children go into the same eval split.
  • Computes eval-tree descendants via BFS (any tile reachable by following parent→child
    edges from an eval tile) and labels them 'train_excl'.  Without this, a train tile
    could form an SR pair where the conditioning parent image comes from an eval split.
  • Assigns all remaining geo-tagged tiles to train.  A training parent needs at least
    1 child in the training set; missing children (1–3) are genuinely absent from Git-10M
    (never withheld for eval), so the partial-quad loss mask is meaningful.

Splits written to index_to_meta:
  train         : geo-tagged tiles not in any eval split or train_excl
  train_excl    : eval-tree descendants — excluded from splits['train'], labeled for tracing
  val_random    : 5k  complete quads, randomly selected
  val_spatial   : 5k  complete quads, 50% urban / 50% non-urban
  random_test   : 10k complete quads, randomly selected
  spatial_test  : 10k complete quads, 50% urban / 50% non-urban
  (non-geo Million-AID tiles: split=train in index_to_meta, included in splits['train'])

Output:
  git10m_quad_meta.json
    ├── index_to_meta  — per-sample metadata (split, lat, lon, gsd, zoom, location)
    └── splits         — seeded-shuffled (seed=42) HF index lists per split; ZL-balanced
                         so any prefix slice contains a representative mix of zoom levels.
  hierarchy_index_quad.pkl    — hierarchy with split assignments embedded
"""

import json
import math
import os
import pickle
import random
import sys
from collections import defaultdict

import geopandas as gpd
import numpy as np
from shapely.geometry import Point
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hierarchical_index import HierarchyIndex


# ── helpers (same as build_splits_hf.py) ─────────────────────────────────────

def tile_to_latlon(zoom, x, y):
    n = 2.0 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    return math.degrees(lat_rad), lon


def calculate_gsd(lat, zoom):
    R = 6378137
    initial_resolution = 2 * math.pi * R / 256
    resolution = initial_resolution / (2 ** zoom)
    return resolution * math.cos(math.radians(lat))


def parse_location(location):
    """Parse location string like '15_1234_5678' into (zoom, x, y)."""
    if location is None:
        return None
    parts = str(location).split('_')
    if len(parts) != 3:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except (ValueError, IndexError):
        return None


# ── quad-specific helper ──────────────────────────────────────────────────────

def quad_children(loc):
    """Return the 4 expected child location strings of a parent."""
    p = parse_location(loc)
    if p is None:
        return []
    zp, xp, yp = p
    return [
        f"{zp+1}_{2*xp}_{2*yp}",
        f"{zp+1}_{2*xp+1}_{2*yp}",
        f"{zp+1}_{2*xp}_{2*yp+1}",
        f"{zp+1}_{2*xp+1}_{2*yp+1}",
    ]


# ── full dataset metadata (mirrors build_splits_hf.py's create_full_dataset_meta) ──

def create_full_dataset_meta(location_to_indices, valid_location_to_indices,
                             train_locs, val_random_locs, val_spatial_locs,
                             random_test_locs, spatial_test_locs,
                             output_path, seed=42, excl_locs=None):
    """
    Write git10m_quad_meta.json.

    Output structure
    ----------------
    {
      "index_to_meta": { "<hf_index>": {location, satellite, split, lat, lon, gsd, zoom}, ... },
      "splits": {
        "train":        [list of HF indices, seeded-shuffled],
        "val_random":   [...],
        "val_spatial":  [...],
        "random_test":  [...],
        "spatial_test": [...]
      }
    }

    Tiles in excl_locs are labeled 'train_excl' in index_to_meta (traceable) but
    are NOT included in splits['train'].  These are eval-tree descendants: tiles
    whose SR parent image would come from an eval split, making them unsuitable
    for training without cross-split conditioning leakage.

    The "splits" lists are shuffled once with a fixed seed so that downstream
    code can consume them directly without re-sorting.  This breaks the
    ZL-contiguous ordering of the raw HF dataset (ZL=10 tiles occupy indices
    0–93k, ZL=11 tiles 93k–181k, …) and ensures any prefix slice of
    splits["train"] contains a representative mix of zoom levels.  These
    lists are the canonical, shareable benchmark splits.

    Mirrors create_full_dataset_meta() in build_splits_hf.py.
    Samples whose location is not in any named split default to 'train'.
    """
    print("\nCreating full dataset metadata...")

    location_to_split = {}
    for loc in train_locs:        location_to_split[loc] = 'train'
    for loc in val_random_locs:   location_to_split[loc] = 'val_random'
    for loc in val_spatial_locs:  location_to_split[loc] = 'val_spatial'
    for loc in random_test_locs:  location_to_split[loc] = 'random_test'
    for loc in spatial_test_locs: location_to_split[loc] = 'spatial_test'
    if excl_locs:
        for loc in excl_locs:     location_to_split[loc] = 'train_excl'

    # Pre-compute lat/lon/GSD for all valid locations
    print("Computing lat/lon/GSD for all locations...")
    location_metadata = {}
    for loc in valid_location_to_indices.keys():
        parsed = parse_location(loc)
        if parsed is not None:
            zoom, x, y = parsed
            lat, lon = tile_to_latlon(zoom, x, y)
            location_metadata[loc] = {
                'lat': lat, 'lon': lon,
                'gsd': calculate_gsd(lat, zoom), 'zoom': zoom
            }

    # Build index → metadata and collect per-split index lists simultaneously
    index_to_meta    = {}
    satellite_counts = {}
    split_counts     = {}
    split_indices    = {sp: [] for sp in
                        ['train', 'val_random', 'val_spatial', 'random_test', 'spatial_test']}

    for loc, idx_list in valid_location_to_indices.items():
        split    = location_to_split.get(loc, 'train')
        loc_meta = location_metadata.get(loc)
        if loc_meta is None:
            continue

        for item in idx_list:
            if isinstance(item, dict):
                sample_idx = item['index']
                satellite  = item.get('satellite', 'google')
            else:
                sample_idx = item
                satellite  = 'google'

            index_to_meta[sample_idx] = {
                'location':  loc,
                'satellite': satellite,
                'split':     split,
                'lat':       loc_meta['lat'],
                'lon':       loc_meta['lon'],
                'gsd':       loc_meta['gsd'],
                'zoom':      loc_meta['zoom'],
            }
            satellite_counts[satellite] = satellite_counts.get(satellite, 0) + 1
            split_counts[split]         = split_counts.get(split, 0) + 1
            if split in split_indices:
                split_indices[split].append(sample_idx)

    # Samples with None location (e.g. Million-AID) → train
    if None in location_to_indices:
        for item in location_to_indices[None]:
            if isinstance(item, dict):
                sample_idx = item['index'];  satellite = item.get('satellite', 'other')
            else:
                sample_idx = item;           satellite = 'other'
            index_to_meta[sample_idx] = {
                'location': None, 'satellite': satellite, 'split': 'train',
                'lat': 0.0, 'lon': 0.0, 'gsd': 0.0, 'zoom': 0,
            }
            satellite_counts[satellite] = satellite_counts.get(satellite, 0) + 1
            split_counts['train']       = split_counts.get('train', 0) + 1
            split_indices['train'].append(sample_idx)

    # Shuffle each split's index list with the fixed seed so the ordering is
    # ZL-balanced and deterministic.  The raw HF dataset is ZL-contiguous
    # (ZL=10: 0–93k, ZL=11: 93k–181k, …, Million-AID: last 3.6M), so without
    # shuffling any sorted prefix slice of train would be 100% ZL=10 root tiles.
    rng = random.Random(seed)
    for sp in split_indices:
        rng.shuffle(split_indices[sp])
    print(f"Shuffled split index lists with seed={seed}.")

    total = len(index_to_meta)
    print(f"\nTotal samples in metadata: {total:,}")
    for sp in ['train', 'val_random', 'val_spatial', 'random_test', 'spatial_test']:
        c = split_counts.get(sp, 0)
        print(f"  {sp:15s}: {c:>8,}  ({100*c/total:.2f}%)")

    print("\nSatellite distribution:")
    for sat, c in sorted(satellite_counts.items()):
        print(f"  {sat}: {c:,} ({100*c/total:.2f}%)")

    gsd_vals = [m['gsd'] for m in index_to_meta.values() if m['gsd'] > 0]
    if gsd_vals:
        print(f"\nGSD stats (excl. {total - len(gsd_vals):,} samples with gsd=0):")
        print(f"  min={min(gsd_vals):.2f}  max={max(gsd_vals):.2f}  "
              f"mean={np.mean(gsd_vals):.2f}  median={np.median(gsd_vals):.2f}  m/px")

    with open(output_path, 'w') as f:
        json.dump({'index_to_meta': index_to_meta, 'splits': split_indices}, f)
    print(f"\n✅ Saved full dataset metadata to {output_path}")


# ── distribution printer ──────────────────────────────────────────────────────

def print_split_distribution(train_locs, val_random_locs, val_spatial_locs,
                             random_test_locs, spatial_test_locs):
    """
    Print parent → children completeness distribution per zoom level for every split.

    For EVERY tile in the split (not just assigned parents), computes its 4 expected
    ZL+1 children and counts how many are also in the split.  This is a broader check
    than "were these assigned as complete eval quads", so two categories of rows appear:

    • 4-child rows  — the assigned complete evaluation quads; 10,000 for test splits,
                      5,000 for val splits.  These are always intact by construction.

    • 1-/2-child rows in eval splits (rare, ~37–50 total) — INCIDENTAL cross-zoom
                      relationships between tiles that landed in the same eval split
                      for independent reasons.  Example: a ZL=13 tile T13 is in
                      spatial_test as the *child* of an assigned ZL=12→13 quad;
                      one of T13's ZL=14 children C1 also ended up in spatial_test
                      as the *parent* of an independent ZL=14→15 quad; the other
                      3 ZL=14 siblings of C1 are in train → T13 shows as "1 child".
                      These rows are NOT assigned eval quads and are filtered out by
                      require_complete_quad=True in the dataloader and probe_dataset.py.

    • Training      : genuine mix of 1-/2-/3-/4-child pairs — the 1-/2-/3-child rows
                      reflect tiles whose ZL+1 children are genuinely absent from
                      Git-10M (never withheld for eval, always in train).
    """
    print("\n" + "=" * 80)
    print("SPLIT DISTRIBUTION: parent → child completeness by zoom level")
    print("=" * 80)

    splits = [
        ("train",        train_locs),
        ("val_random",   val_random_locs),
        ("val_spatial",  val_spatial_locs),
        ("random_test",  random_test_locs),
        ("spatial_test", spatial_test_locs),
    ]

    hdr = (f"  {'ZL':>4} | {'1 child':>8} | {'2 children':>10} | "
           f"{'3 children':>10} | {'4 children':>10} | {'Total':>8}")
    sep = "  " + "-" * 4 + "-+-" + "-" * 8 + "-+-" + "-" * 10 + "-+-" + \
          "-" * 10 + "-+-" + "-" * 10 + "-+-" + "-" * 8

    for split_name, split_locs in splits:
        split_set = set(split_locs)

        # zoom → n_children → count
        zoom_dist = defaultdict(lambda: defaultdict(int))

        for loc in split_set:
            parsed = parse_location(loc)
            if parsed is None:
                continue
            zp = parsed[0]
            children = quad_children(loc)
            if not children:
                continue
            n_present = sum(1 for c in children if c in split_set)
            if n_present > 0:
                zoom_dist[zp][n_present] += 1

        total_parents = sum(sum(v.values()) for v in zoom_dist.values())
        print(f"\n  Split: {split_name}  ({total_parents:,} parent tiles with ≥1 child in split)")
        if not zoom_dist:
            print("    (no parent-child pairs found)")
            continue

        print(hdr)
        print(sep)

        grand = defaultdict(int)
        for zp in sorted(zoom_dist.keys()):
            d = zoom_dist[zp]
            row_total = sum(d.values())
            print(f"  {zp:>4} | {d.get(1,0):>8,} | {d.get(2,0):>10,} | "
                  f"{d.get(3,0):>10,} | {d.get(4,0):>10,} | {row_total:>8,}")
            for n, c in d.items():
                grand[n] += c

        g_total = sum(grand.values())
        print(sep)
        print(f"  {'ALL':>4} | {grand.get(1,0):>8,} | {grand.get(2,0):>10,} | "
              f"{grand.get(3,0):>10,} | {grand.get(4,0):>10,} | {g_total:>8,}")

    print()


# ── main split builder ────────────────────────────────────────────────────────

def create_splits(hierarchy_pkl, location_to_indices_file,
                  shapefile_path,
                  output_meta_json, output_hierarchy_pkl,
                  val_random_quads   = 5_000,
                  val_spatial_quads  = 5_000,
                  test_random_quads  = 10_000,
                  test_spatial_quads = 10_000,
                  seed=42):
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    # Load hierarchy
    with open(hierarchy_pkl, 'rb') as f:
        hierarchy = pickle.load(f)
    print(f"Hierarchy locations: {len(hierarchy.all_locations):,}")

    # Load location → HF indices mapping
    print(f"\nLoading location_to_indices from {location_to_indices_file}...")
    with open(location_to_indices_file, 'rb') as f:
        loc_data = pickle.load(f)
    location_to_indices = loc_data['location_to_indices']
    print(f"Total locations in pkl: {len(location_to_indices):,}")

    # Filter out None / unparseable locations
    valid_location_to_indices = {
        loc: idxs for loc, idxs in location_to_indices.items()
        if loc is not None and parse_location(loc) is not None
    }
    print(f"Valid locations after filtering: {len(valid_location_to_indices):,}")

    # Intersect: need BOTH hierarchy structure AND actual HF images
    all_locs = hierarchy.all_locations & set(valid_location_to_indices.keys())
    print(f"Locations with hierarchy + HF indices: {len(all_locs):,}")

    # Compute lat/lon for all intersected locations
    print("\nConverting tile coordinates to lat/lon...")
    location_to_coords = {}
    for loc in all_locs:
        parsed = parse_location(loc)
        if parsed is not None:
            zoom, x, y = parsed
            location_to_coords[loc] = tile_to_latlon(zoom, x, y)
    all_locs = set(location_to_coords.keys())   # drop any that couldn't be parsed
    print(f"Locations with valid coordinates: {len(all_locs):,}")

    total_samples = sum(len(valid_location_to_indices[loc]) for loc in all_locs)
    print(f"Total HF samples: {total_samples:,}")

    # ── Find all complete 2×2 quads ───────────────────────────────────────────
    print("\nFinding all complete 2×2 quads (parent + all 4 children in dataset)...")
    all_quads = []
    for loc in all_locs:
        children = quad_children(loc)
        if children and all(c in all_locs for c in children):
            all_quads.append((loc, children))
    print(f"Complete quads available (all zoom levels): {len(all_quads):,}")

    # ── Identify urban / non-urban via shapefile ──────────────────────────────
    print(f"\nIdentifying urban areas using {shapefile_path}...")
    cities = gpd.read_file(shapefile_path)
    print(f"Loaded {len(cities):,} urban area polygons (CRS: {cities.crs})")

    data, geoms = [], []
    for p, _ in all_quads:
        lat, lon = location_to_coords[p]
        data.append(p)
        geoms.append(Point(lon, lat))

    points_gdf = gpd.GeoDataFrame({'location': data}, geometry=geoms, crs='EPSG:4326')
    points_gdf = points_gdf.to_crs(cities.crs)
    print("Performing spatial join...")
    urban_set       = set(gpd.sjoin(points_gdf, cities, how='inner',
                                    predicate='within')['location'].values)
    urban_quads     = [(p, ch) for p, ch in all_quads if p in urban_set]
    non_urban_quads = [(p, ch) for p, ch in all_quads if p not in urban_set]
    print(f"  Urban quads    : {len(urban_quads):,}  "
          f"({100*len(urban_quads)/len(all_quads):.1f}%)")
    print(f"  Non-urban quads: {len(non_urban_quads):,}  "
          f"({100*len(non_urban_quads)/len(all_quads):.1f}%)")

    # ── STEP 1: Spatial test ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"STEP 1: SPATIAL TEST ({test_spatial_quads:,} quads, 50% urban / 50% non-urban)")
    print("=" * 60)

    n_st         = min(test_spatial_quads // 2, len(urban_quads), len(non_urban_quads))
    st_urban     = random.sample(urban_quads, n_st)
    st_non_urban = random.sample(non_urban_quads, n_st)
    spatial_test_quads = st_urban + st_non_urban
    spatial_test_locs  = set()
    for p, ch in spatial_test_quads:
        spatial_test_locs.add(p)
        spatial_test_locs.update(ch)

    st_samples = sum(len(valid_location_to_indices[loc])
                     for loc in spatial_test_locs if loc in valid_location_to_indices)
    print(f"Spatial test quads    : {len(spatial_test_quads):,}")
    print(f"Spatial test locations: {len(spatial_test_locs):,}")
    print(f"Spatial test samples  : {st_samples:,} ({100*st_samples/total_samples:.2f}%)")
    print(f"  Urban/non-urban     : {len(st_urban):,} / {len(st_non_urban):,}")

    # ── STEP 2: Spatial val ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"STEP 2: SPATIAL VAL ({val_spatial_quads:,} quads, 50% urban / 50% non-urban)")
    print("=" * 60)

    rem_urban = [(p, ch) for p, ch in urban_quads
                 if p not in spatial_test_locs
                 and not any(c in spatial_test_locs for c in ch)]
    rem_non_urban = [(p, ch) for p, ch in non_urban_quads
                     if p not in spatial_test_locs
                     and not any(c in spatial_test_locs for c in ch)]
    n_vs              = min(val_spatial_quads // 2, len(rem_urban), len(rem_non_urban))
    vs_urban          = random.sample(rem_urban, n_vs)
    vs_non_urban      = random.sample(rem_non_urban, n_vs)
    val_spatial_quads_list = vs_urban + vs_non_urban
    val_spatial_locs  = set()
    for p, ch in val_spatial_quads_list:
        val_spatial_locs.add(p)
        val_spatial_locs.update(ch)

    vs_samples = sum(len(valid_location_to_indices[loc])
                     for loc in val_spatial_locs if loc in valid_location_to_indices)
    print(f"Val spatial quads    : {len(val_spatial_quads_list):,}")
    print(f"Val spatial locations: {len(val_spatial_locs):,}")
    print(f"Val spatial samples  : {vs_samples:,} ({100*vs_samples/total_samples:.2f}%)")
    print(f"  Urban/non-urban    : {len(vs_urban):,} / {len(vs_non_urban):,}")

    # ── STEP 3: Random test ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"STEP 3: RANDOM TEST ({test_random_quads:,} quads)")
    print("=" * 60)

    all_spatial_locs = spatial_test_locs | val_spatial_locs
    remaining_quads  = [
        (p, ch) for p, ch in all_quads
        if p not in all_spatial_locs
        and not any(c in all_spatial_locs for c in ch)
    ]
    random.shuffle(remaining_quads)

    random_test_quads_list = remaining_quads[:test_random_quads]
    random_test_locs       = set()
    for p, ch in random_test_quads_list:
        random_test_locs.add(p)
        random_test_locs.update(ch)

    rt_samples = sum(len(valid_location_to_indices[loc])
                     for loc in random_test_locs if loc in valid_location_to_indices)
    print(f"Random test quads    : {len(random_test_quads_list):,}")
    print(f"Random test locations: {len(random_test_locs):,}")
    print(f"Random test samples  : {rt_samples:,} ({100*rt_samples/total_samples:.2f}%)")

    # ── STEP 4: Random val ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"STEP 4: RANDOM VAL ({val_random_quads:,} quads)")
    print("=" * 60)

    val_random_quads_list = [
        (p, ch) for p, ch in remaining_quads[test_random_quads:]
        if p not in random_test_locs
        and not any(c in random_test_locs for c in ch)
    ][:val_random_quads]
    val_random_locs = set()
    for p, ch in val_random_quads_list:
        val_random_locs.add(p)
        val_random_locs.update(ch)

    vr_samples = sum(len(valid_location_to_indices[loc])
                     for loc in val_random_locs if loc in valid_location_to_indices)
    print(f"Val random quads    : {len(val_random_quads_list):,}")
    print(f"Val random locations: {len(val_random_locs):,}")
    print(f"Val random samples  : {vr_samples:,} ({100*vr_samples/total_samples:.2f}%)")

    # ── STEP 5: Train ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: TRAIN (all remaining locations; partial quads allowed)")
    print("=" * 60)

    all_eval_locs = spatial_test_locs | val_spatial_locs | random_test_locs | val_random_locs

    # BFS downward from every eval tile: any tile reachable by following
    # parent→child edges is an "eval-tree descendant".  If left in train,
    # such a tile would form an SR conditioning pair where the parent image
    # comes from an eval split — a data-leakage path.  We exclude the full
    # subtree from train so the split is clean by construction.
    print("Computing eval-tree descendants to exclude from train "
          "(prevents cross-split conditioning leakage)...")
    eval_tree_excl: set = set()
    frontier = list(all_eval_locs)
    while frontier:
        next_frontier = []
        for loc in frontier:
            for child_loc in quad_children(loc):
                if (child_loc in all_locs
                        and child_loc not in all_eval_locs
                        and child_loc not in eval_tree_excl):
                    eval_tree_excl.add(child_loc)
                    next_frontier.append(child_loc)
        frontier = next_frontier
    excl_samples = sum(len(valid_location_to_indices[loc])
                       for loc in eval_tree_excl if loc in valid_location_to_indices)
    print(f"  {len(eval_tree_excl):,} eval-tree descendants excluded "
          f"({excl_samples:,} samples, labeled 'train_excl' in metadata)")

    train_locs    = [loc for loc in all_locs if loc not in all_eval_locs and loc not in eval_tree_excl]
    train_set     = set(train_locs)
    train_samples = sum(len(valid_location_to_indices[loc])
                        for loc in train_locs if loc in valid_location_to_indices)

    train_complete = sum(
        1 for p, ch in all_quads
        if p in train_set and all(c in train_set for c in ch)
    )
    print(f"Train locations     : {len(train_locs):,}")
    print(f"Train samples       : {train_samples:,} ({100*train_samples/total_samples:.2f}%)")
    print(f"Complete train quads: {train_complete:,} (partial pairs discovered at runtime)")

    # ── Verification ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    assert not (val_random_locs  & val_spatial_locs),   "val_random / val_spatial overlap!"
    assert not (val_random_locs  & random_test_locs),   "val_random / random_test overlap!"
    assert not (val_random_locs  & spatial_test_locs),  "val_random / spatial_test overlap!"
    assert not (val_random_locs  & train_set),          "val_random / train overlap!"
    assert not (val_spatial_locs & random_test_locs),   "val_spatial / random_test overlap!"
    assert not (val_spatial_locs & spatial_test_locs),  "val_spatial / spatial_test overlap!"
    assert not (val_spatial_locs & train_set),          "val_spatial / train overlap!"
    assert not (random_test_locs & spatial_test_locs),  "random_test / spatial_test overlap!"
    assert not (random_test_locs & train_set),          "random_test / train overlap!"
    assert not (spatial_test_locs & train_set),         "spatial_test / train overlap!"

    total_check = (len(train_locs) + len(val_random_locs) + len(val_spatial_locs) +
                   len(random_test_locs) + len(spatial_test_locs) + len(eval_tree_excl))
    assert total_check == len(all_locs), \
        f"Location count mismatch: {total_check} vs {len(all_locs)}"

    samples_check = train_samples + vr_samples + vs_samples + rt_samples + st_samples + excl_samples
    assert samples_check == total_samples, \
        f"Sample count mismatch: {samples_check} vs {total_samples}"

    print("✅ No overlaps between splits")
    print(f"✅ All {total_check:,} locations accounted for "
          f"({len(eval_tree_excl):,} labeled 'train_excl', excluded from splits['train'])")
    print(f"✅ All {samples_check:,} samples accounted for")

    print("\nSplit summary:")
    print(f"  {'train':15s}: {len(train_locs):>8,} locs  {train_samples:>10,} samples  "
          f"({100*train_samples/total_samples:.2f}%)")
    print(f"  {'train_excl':15s}: {len(eval_tree_excl):>8,} locs  {excl_samples:>10,} samples  "
          f"({100*excl_samples/total_samples:.2f}%)  [eval-tree descendants, excluded from SR training]")
    print(f"  {'val_random':15s}: {len(val_random_locs):>8,} locs  {vr_samples:>10,} samples  "
          f"({100*vr_samples/total_samples:.2f}%)  [{len(val_random_quads_list):,} complete quads]")
    print(f"  {'val_spatial':15s}: {len(val_spatial_locs):>8,} locs  {vs_samples:>10,} samples  "
          f"({100*vs_samples/total_samples:.2f}%)  [{len(val_spatial_quads_list):,} complete quads]")
    print(f"  {'random_test':15s}: {len(random_test_locs):>8,} locs  {rt_samples:>10,} samples  "
          f"({100*rt_samples/total_samples:.2f}%)  [{len(random_test_quads_list):,} complete quads]")
    print(f"  {'spatial_test':15s}: {len(spatial_test_locs):>8,} locs  {st_samples:>10,} samples  "
          f"({100*st_samples/total_samples:.2f}%)  [{len(spatial_test_quads):,} complete quads]")

    # ── Save hierarchy ─────────────────────────────────────────────────────────
    hierarchy.splits = {
        'train':        train_locs,
        'train_excl':   list(eval_tree_excl),
        'val_random':   list(val_random_locs),
        'val_spatial':  list(val_spatial_locs),
        'random_test':  list(random_test_locs),
        'spatial_test': list(spatial_test_locs),
        'metadata': {
            'seed':               seed,
            'val_random_quads':   len(val_random_quads_list),
            'val_spatial_quads':  len(val_spatial_quads_list),
            'random_test_quads':  len(random_test_quads_list),
            'spatial_test_quads': len(spatial_test_quads),
            'train_locs':         len(train_locs),
            'train_excl_locs':    len(eval_tree_excl),
        },
    }
    with open(output_hierarchy_pkl, 'wb') as f:
        pickle.dump(hierarchy, f)
    print(f"\n✅ Saved hierarchy to {output_hierarchy_pkl}")

    # ── STEP 6: Full dataset metadata ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: CREATE FULL DATASET METADATA FILE")
    print("=" * 60)

    create_full_dataset_meta(
        location_to_indices, valid_location_to_indices,
        train_locs,
        list(val_random_locs), list(val_spatial_locs),
        list(random_test_locs), list(spatial_test_locs),
        output_meta_json,
        seed=seed,
        excl_locs=eval_tree_excl,
    )

    # ── STEP 7: Distribution report ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: PARENT → CHILDREN DISTRIBUTION BY ZOOM LEVEL")
    print("=" * 60)

    print_split_distribution(
        train_locs,
        list(val_random_locs), list(val_spatial_locs),
        list(random_test_locs), list(spatial_test_locs),
    )

    return hierarchy


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    create_splits(
        hierarchy_pkl           = 'data/hierarchy_index.pkl',
        location_to_indices_file = 'data/location_to_indices.pkl',
        shapefile_path          = 'data/shapefiles/World_Urban_Areas.shp',
        output_meta_json        = os.environ.get("GENESIS_GIT10M_META", "data/git10m_quad_meta.json"),
        output_hierarchy_pkl    = os.environ.get("GENESIS_HINDEX", "data/hierarchy_index_quad.pkl"),
        val_random_quads        = 5_000,
        val_spatial_quads       = 5_000,
        test_random_quads       = 10_000,
        test_spatial_quads      = 10_000,
        seed                    = 42,
    )
