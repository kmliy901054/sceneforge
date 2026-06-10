"""sceneforge/augment — appearance randomization for REAL robot data (v2 feature A).

Evidence base: the VLA probe (experiments/vla_probe/REPORT.md + COMPARISON.md)
measured background appearance leaking into VLA action heads on two
architecturally different models — restyling only the far background (robot /
objects / workspace pixel-identical) moves the commanded translation ~8-10×
above the JPEG floor. ``restyle.restyle_frames`` is the productized version of
that probe's arm-S pipeline: a ROSIE-style background-appearance randomizer
whose near-pixel composite is bitwise exact.
"""
from sceneforge.augment.restyle import restyle_frames  # noqa: F401


def augment_dataset(*args, **kwargs):
    """LeRobot v2.x dataset augmenter — lazy import so ``sceneforge.augment``
    stays importable without pyarrow (see lerobot_io.py for the contract)."""
    from sceneforge.augment.lerobot_io import augment_dataset as _impl
    return _impl(*args, **kwargs)
