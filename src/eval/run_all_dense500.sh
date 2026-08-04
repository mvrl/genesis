#!/usr/bin/env bash
# dense500 pyramid results — generate (both archs × protocol A/B/C × window 1–5, with the
# paper-exact run_pyramid_eval defaults: 50 steps, cfg 2.5, overlap 32, τ=0.40, 100 sites/window
# → all 500 sites) then score image-quality + semantic + consistency + caption.
# Metrics land in outputs/dense500_{B16,H16}/_metrics*.
#   bash src/eval/run_all_dense500.sh
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"   # repo/src (for `import pyramid`)
for TAG in B16 H16; do
  ARCH=$([ "$TAG" = H16 ] && echo "JiT-H/16" || echo "JiT-B/16")
  for P in A B C; do
    for W in 1 2 3 4 5; do
      python -m pyramid.run_pyramid_eval --protocol "$P" --window "$W" --arch "$ARCH" \
        --out_root "outputs/dense500_$TAG"
    done
  done
  python eval_pyramid.py  --gen_root "outputs/dense500_$TAG" --device cuda
  python eval_ext.py      --gen_root "outputs/dense500_$TAG" --families consistency,semantic --device cuda
  python caption_agree.py --gen_root "outputs/dense500_$TAG" --sites_per_cell 10 --device cuda
done
echo "[run_all_dense500] done — metrics in src/eval/outputs/dense500_{B16,H16}/_metrics*"
