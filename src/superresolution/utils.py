"""Curriculum boundary utilities and checkpoint helpers for super-resolution training."""

import os


def compute_stage_boundaries(max_steps: int) -> tuple[int, int, int]:
    """Compute SR zoom-level curriculum boundaries from total training steps.

    Easy tiles  : parent ZL ≤ zl_easy_max (default 15)  (coarse geography, low-texture)
    Hard tiles  : parent ZL > zl_easy_max               (fine urban/vegetation detail)

    Stage 0  (0           → boundary1): 100% easy ZL (parent ZL ≤ zl_easy_max; up to target ZL 16)
    Stage 1  (boundary1   → boundary2): 50% easy / 50% hard
    Stage 2  (boundary2   → boundary3): Uniform ZL (∼72% hard, all pairs)
    Stage 3  (boundary3   → max_steps): Uniform ZL, complete quads only (all 4 children present)

    Boundaries at 20% / 40% / 60% of max_steps, rounded to nearest 5k.
    For small max_steps the 5k rounding is skipped to keep boundaries valid.
    """
    def _round5k(x: int) -> int:
        return max(5000, round(x / 5000) * 5000)

    if max_steps >= 20_000:
        boundary1 = _round5k(int(max_steps * 0.20))
        boundary2 = _round5k(int(max_steps * 0.40))
        boundary3 = _round5k(int(max_steps * 0.60))
    else:
        # Small run (smoke test / debug): use fractional splits without 5k rounding
        boundary1 = max(1, int(max_steps * 0.20))
        boundary2 = max(boundary1 + 1, int(max_steps * 0.40))
        boundary3 = max(boundary2 + 1, int(max_steps * 0.60))

    # Clamp so boundaries are strictly ordered and never reach max_steps
    boundary1 = min(boundary1, max_steps - 3)
    boundary2 = min(max(boundary2, boundary1 + 1), max_steps - 2)
    boundary3 = min(max(boundary3, boundary2 + 1), max_steps - 1)
    return boundary1, boundary2, boundary3


def compute_checkpoint_steps(max_steps: int, stage_steps: tuple[int, int, int]) -> dict[int, str]:
    """Return {step: label} for all steps at which a checkpoint should be saved.

    Combines two sources:
      - 10 uniformly-spaced steps: max_steps * i // 10 for i = 1..10
      - Stage-boundary steps: end of each of the 4 stages
        (b1 = end stage0, b2 = end stage1, b3 = end stage2, max_steps = end stage3)

    Stage-boundary labels take priority when a step falls in both sets.
    """
    b1, b2, b3 = stage_steps
    stage_boundary_map = {b1: "stage0", b2: "stage1", b3: "stage2", max_steps: "stage3"}

    uniform_steps = {max_steps * i // 10 for i in range(1, 11)}

    save_steps: dict[int, str] = {}
    for step in uniform_steps:
        save_steps[step] = stage_boundary_map.get(step, "uniform")
    for step, label in stage_boundary_map.items():
        if step not in save_steps:
            save_steps[step] = label
    return save_steps


try:
    import pytorch_lightning as pl

    class StepCheckpointCallback(pl.Callback):
        """Saves checkpoints at 10 uniform steps + end of each curriculum stage.

        Checkpoint filenames:
          {prefix}-uniform-step{N:07d}.ckpt      (uniform-only step)
          {prefix}-stage{k}-step{N:07d}.ckpt     (stage-boundary step; takes priority)
        """

        def __init__(
            self,
            max_steps: int,
            stage_steps: tuple[int, int, int],
            output_dir: str,
            fname_prefix: str = "",
        ):
            super().__init__()
            self.output_dir = output_dir
            self.fname_prefix = fname_prefix
            self.save_steps = compute_checkpoint_steps(max_steps, stage_steps)
            self._saved: set[int] = set()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            step = trainer.global_step
            if step in self.save_steps and step not in self._saved:
                self._saved.add(step)
                label = self.save_steps[step]
                prefix = f"{self.fname_prefix}-" if self.fname_prefix else ""
                fname = f"{prefix}{label}-step{step:07d}.ckpt"
                path = os.path.join(self.output_dir, fname)
                trainer.save_checkpoint(path)

except ImportError:
    pass  # pytorch_lightning not available (e.g. during data-prep scripts)
