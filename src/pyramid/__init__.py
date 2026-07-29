"""Full-pyramid generation framework: unify SR + outpainting to synthesize a
complete quadtree of satellite tiles from arbitrary multi-scale seeds.

Run modules with the repo's ``src`` dir on PYTHONPATH, e.g.:
    PYTHONPATH=src python -m pyramid.verify_data

The generation strategy lives in ``strategies/unified.py``; the demo apps
in ``demos/`` drive it.
"""
