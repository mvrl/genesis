"""Curriculum boundary utilities and checkpoint helpers for outpainting training."""

import os


def compute_stage_boundaries(max_steps: int) -> tuple[int, int, int]:
    """Compute outpainting curriculum stage boundaries from total training steps.

    The quadrant-aligned mask curriculum has 4 stages (3 boundaries):
        Stage 0  (0           → boundary1): quad3                            (3 known, 25% hole)
        Stage 1  (boundary1   → boundary2): 50/50 quad2_adj / quad2_diag     (2 known, 50% hole)
        Stage 2  (boundary2   → boundary3): quad1                            (1 known, 75% hole)
        Stage 3  (boundary3   → max_steps): uniform mix of all 4 regimes     (1–3 known, 25–75% hole)

    Boundaries at 20% / 40% / 60% of max_steps, rounded to nearest 5k.
    For small max_steps the 5k rounding is skipped to keep boundaries valid.
    """
    def _round5k(x: int) -> int:
        return max(5000, round(x / 5000) * 5000)

    if max_steps >= 20_000:
        s1 = _round5k(int(max_steps * 0.20))
        s2 = _round5k(int(max_steps * 0.40))
        s3 = _round5k(int(max_steps * 0.60))
    else:
        # Small run (smoke test / debug): use fractional splits without 5k rounding
        s1 = max(1, int(max_steps * 0.20))
        s2 = max(s1 + 1, int(max_steps * 0.40))
        s3 = max(s2 + 1, int(max_steps * 0.60))

    # Clamp so boundaries are strictly ordered and never reach max_steps
    s1 = min(s1, max_steps - 3)
    s2 = min(max(s2, s1 + 1), max_steps - 2)
    s3 = min(max(s3, s2 + 1), max_steps - 1)
    return s1, s2, s3


def compute_checkpoint_steps(max_steps: int, stage_steps: tuple[int, int, int]) -> dict[int, str]:
    """Return {step: label} for all steps at which a checkpoint should be saved.

    Combines two sources:
      - 10 uniformly-spaced steps: max_steps * i // 10 for i = 1..10
      - Stage-boundary steps: end of each of the 4 stages
        (s1 = end stage0, s2 = end stage1, s3 = end stage2, max_steps = end stage3)

    Stage-boundary labels take priority when a step falls in both sets.
    """
    s1, s2, s3 = stage_steps
    stage_boundary_map = {s1: "stage0", s2: "stage1", s3: "stage2", max_steps: "stage3"}

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
