#!/usr/bin/env python3
"""
visualize_splits.py — Scatter plot of Git-10M quad splits on ESRI satellite imagery.

Architecture (mirrors Genesis demo pattern):
  - gr.HTML map container — initialized ONCE via js_on_load, never replaced
  - hidden gr.Textbox data store — updated on every resample
  - data_box.change(js=UPDATE_JS) — calls window.__updateSplitMap(newData) client-side

Colors / split counts (seed=42):
    blue   — train         (6,568,890 geo-tagged SR-clean tiles)
    slate  — train_excl    (177,510 eval-tree descendants, excluded from SR training)
    green  — val_random    (24,969 tiles, 5,000 complete quads)
    orange — val_spatial   (24,958 tiles, 5,000 complete quads, 50% urban)
    red    — random_test   (49,826 tiles, 10,000 complete quads)
    purple — spatial_test  (49,825 tiles, 10,000 complete quads, 50% urban)

Usage:
    python visualize_splits.py
    python visualize_splits.py --meta data/git10m_quad_meta.json
    python visualize_splits.py --n_train 100000 --port 7861 --share
"""

import argparse
import json
import os
import random

import gradio as gr


# ── config ────────────────────────────────────────────────────────────────────

SPLITS = {
    'val_random':   '#4CAF50',
    'val_spatial':  '#FF9800',
    'random_test':  '#F44336',
    'spatial_test': '#9C27B0',
    'train_excl':   '#78909C',
    'train':        '#2196F3',
}
SPLIT_ORDER = list(SPLITS.keys())
SPLIT_IDX   = {s: i for i, s in enumerate(SPLIT_ORDER)}

# Static container HTML — rendered once, never replaced.
MAP_HTML = """
<div class="split-mapwrap"
     style="height:700px;width:100%;border-radius:8px;overflow:hidden;border:1px solid #333;">
</div>
"""

# Runs once when the map component loads.
# Initializes Leaflet and registers window.__updateSplitMap for later calls.
LEAFLET_INIT_JS = """
(function () {
  const root = element.querySelector('.split-mapwrap') || element;
  root.innerHTML = '';

  const COLORS = """ + json.dumps([SPLITS[s] for s in SPLIT_ORDER]) + """;
  const NAMES  = """ + json.dumps(SPLIT_ORDER) + """;

  function init() {
    const map = L.map(root, { preferCanvas: true }).setView([20, 0], 2);
    L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 23, maxNativeZoom: 19, attribution: '&copy; Esri' }
    ).addTo(map);

    const layer = L.layerGroup().addTo(map);

    window.__updateSplitMap = function(pts, visibleSplits) {
      window.__splitPts = pts;   // cache for filter-only updates
      layer.clearLayers();
      // null  → no filter, show all splits
      // []    → all unchecked, show nothing
      // [...]  → show only the named splits
      var visSet = null;
      if (visibleSplits !== null && visibleSplits !== undefined) {
        visSet = {};
        visibleSplits.forEach(function(s) { visSet[NAMES.indexOf(s)] = true; });
      }
      for (let i = 0; i < pts.length; i += 3) {
        if (visSet !== null && !visSet[pts[i+2]]) continue;
        L.circleMarker([pts[i], pts[i+1]], {
          radius:      3,
          color:       COLORS[pts[i+2]],
          fillColor:   COLORS[pts[i+2]],
          fillOpacity: 0.75,
          weight:      0
        }).bindTooltip(NAMES[pts[i+2]]).addTo(layer);
      }
    };

    // Draw initial data if already available
    if (window.__splitDataPending) {
      window.__updateSplitMap(window.__splitDataPending, null);
      window.__splitDataPending = null;
    }
  }

  if (window.L) {
    init();
  } else {
    const css = document.createElement('link');
    css.rel  = 'stylesheet';
    css.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
    document.head.appendChild(css);
    const scr = document.createElement('script');
    scr.src    = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
    scr.onload = init;
    document.head.appendChild(scr);
  }
})();
"""

# Runs client-side when the hidden data textbox changes.
# Parses the JSON and calls window.__updateSplitMap to refresh markers.
UPDATE_JS = """
function(data_json) {
  try {
    var pts = JSON.parse(data_json);
    if (window.__updateSplitMap) {
      window.__updateSplitMap(pts, window.__splitVisible || null);
    } else {
      window.__splitDataPending = pts;
    }
  } catch(e) {}
  return data_json;
}
"""

# Runs client-side when the hidden filter textbox changes.
# Parses the JSON array of selected split names and calls window.__updateSplitMap.
FILTER_JS = """
function(filter_json) {
  try {
    var visible = JSON.parse(filter_json);
    window.__splitVisible = visible;
    if (window.__updateSplitMap && window.__splitPts) {
      window.__updateSplitMap(window.__splitPts, visible);
    }
  } catch(e) {}
  return filter_json;
}
"""


# ── data loading ──────────────────────────────────────────────────────────────

def load_all(meta_path: str) -> dict:
    print(f"Loading {meta_path}  (may take ~30 s)…")
    with open(meta_path) as f:
        meta = json.load(f)
    by_split: dict = {k: [] for k in SPLITS}
    for _, m in meta['index_to_meta'].items():
        if not m.get('location'):
            continue
        split = m.get('split', 'train')
        if split not in by_split:
            continue
        by_split[split].append((round(m['lat'], 5), round(m['lon'], 5)))
    for s in SPLIT_ORDER:
        if by_split.get(s):
            print(f"  {s:15s}: {len(by_split[s]):>8,}")
    return by_split


def make_data_json(by_split: dict) -> str:
    """Flat array [lat, lon, split_idx, ...] as JSON string."""
    pts = []
    for split, rows in by_split.items():
        idx = SPLIT_IDX[split]
        for lat, lon in rows:
            pts += [lat, lon, idx]
    return json.dumps(pts)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser('Git-10M Split Visualizer')
    parser.add_argument('--meta',
                        default=os.environ.get("GENESIS_GIT10M_META") or "data/git10m_quad_meta.json",
                        help="Path to git10m_quad_meta.json (built by build_quad_splits_hf.py).")
    parser.add_argument('--n_train', type=int, default=50_000,
                        help='Train dots per view (default 50k; val/test always shown fully)')
    parser.add_argument('--port',  type=int, default=7860)
    parser.add_argument('--share', action='store_true')
    args = parser.parse_args()

    all_data = load_all(args.meta)
    n_train  = args.n_train

    # Legend HTML (static)
    legend_rows = ''.join(
        f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:12px">'
        f'<span style="width:10px;height:10px;border-radius:50%;background:{SPLITS[s]};'
        f'display:inline-block"></span>{s} ({len(all_data.get(s, [])):,})</span>'
        for s in SPLIT_ORDER if all_data.get(s)
    )

    def resample() -> str:
        by_split = {}
        for split, rows in all_data.items():
            if split == 'train' and len(rows) > n_train:
                by_split[split] = random.sample(rows, n_train)
            else:
                by_split[split] = rows
        return make_data_json(by_split)

    initial_data = resample()
    print('Launching Gradio…')

    with gr.Blocks(title='Git-10M Split Visualizer') as demo:
        with gr.Row():
            gr.HTML(
                f'<div style="font-size:14px;padding:6px 0">'
                f'<b>Git-10M Split Visualizer</b> &nbsp;·&nbsp; {legend_rows}</div>'
            )
            btn = gr.Button('🔄  Resample train tiles', scale=0, variant='primary')

        # Split visibility toggle — Python serialises selection, JS reads it
        split_toggle = gr.CheckboxGroup(
            choices=SPLIT_ORDER,
            value=SPLIT_ORDER,
            label='Visible splits',
            interactive=True,
        )

        # Map container — rendered ONCE, never replaced
        gr.HTML(value=MAP_HTML, js_on_load=LEAFLET_INIT_JS)

        # Hidden data store — updated on resample
        data_box   = gr.Textbox(value=initial_data,                  visible=False)
        # Hidden filter store — updated on checkbox change
        filter_box = gr.Textbox(value=json.dumps(SPLIT_ORDER),       visible=False)

        # Python: serialise checkbox selection → filter_box
        split_toggle.change(
            fn=lambda v: json.dumps(v),
            inputs=[split_toggle],
            outputs=[filter_box],
        )
        # JS: parse filter_box and re-render markers client-side
        filter_box.change(fn=None, js=FILTER_JS, inputs=[filter_box], outputs=[filter_box])

        # JS: parse new data and re-render with current filter
        data_box.change(fn=None, js=UPDATE_JS, inputs=[data_box], outputs=[data_box])

        btn.click(fn=resample, inputs=[], outputs=[data_box])

    demo.launch(server_port=args.port, share=args.share)


if __name__ == '__main__':
    main()
