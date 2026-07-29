"""
pyramid_gen.py — disk I/O for the pyramid demo.

Provides:
  save_pyramid()  — write all non-empty tiles + manifest.json to disk
  make_zip()      — ZIP all tiles + manifest for download

(Generation now lives in the ``src/pyramid`` strategy framework; the old
coarse-to-fine demo generator has been removed.)
"""
from __future__ import annotations

import json
import os
import zipfile
from typing import Dict

from .pyramid_core import PyramidState, TileState, z_min, z_max


def save_pyramid(pyramid: PyramidState, out_dir: str) -> int:
    """Save all non-empty tiles as PNG + manifest.json.  Returns number of tiles saved."""
    os.makedirs(out_dir, exist_ok=True)
    manifest: Dict = {
        "pyramid_z_min": z_min(),
        "pyramid_z_max": z_max(),
        "calls_used":    pyramid.calls_used,
        "tiles":         {},
    }
    saved = 0
    for (z, gx, gy), rec in pyramid.tiles.items():
        if rec.state == TileState.EMPTY or rec.image is None:
            continue
        fname = f"z{z}_x{gx}_y{gy}.png"
        rec.image.save(os.path.join(out_dir, fname))
        manifest["tiles"][f"{z},{gx},{gy}"] = {
            "file":             fname,
            "state":            rec.state.value,
            "self_consistency": round(rec.self_consistency, 4),
            "zoom_diff_used":   rec.zoom_diff_used,
            "seam_flagged":     rec.seam_flagged,
        }
        saved += 1
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return saved


def make_zip(out_dir: str) -> str:
    """ZIP all PNGs + manifest.json from out_dir; return path to the .zip file."""
    zip_path = out_dir.rstrip("/") + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(out_dir):
            if fname.endswith(".png") or fname == "manifest.json":
                zf.write(os.path.join(out_dir, fname), fname)
    return zip_path
