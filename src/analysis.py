"""Example library module: put the reusable code of the paper here (models, solvers, loaders).

Scripts in run/ import from src/ directly, e.g. `from analysis import damped_oscillation`.
Delete this file once the project has its own code.
"""
import numpy as np


def damped_oscillation(t, amplitude=1.0, decay=0.3, frequency=1.0):
    """x(t) = A exp(-gamma t) cos(2 pi f t)."""
    t = np.asarray(t, dtype=float)
    return amplitude * np.exp(-decay * t) * np.cos(2 * np.pi * frequency * t)


def add_noise(x, level=0.05, seed=0):
    """Gaussian noise with standard deviation `level` (reproducible through `seed`)."""
    rng = np.random.default_rng(seed)
    return x + level * rng.standard_normal(np.shape(x))
