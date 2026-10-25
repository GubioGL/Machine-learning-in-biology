"""Example experiment: generate data, save it in data/, and draw the figure straight into the paper.

    python run/example_figure.py

Writes data/example/damped_oscillation.csv and paper/figures/example.pdf (+ a local .png preview).
Copy this file as the starting point of your own scripts, then delete it.
"""

# --- repository layout: src/ and run/ go on the import path, and relative paths
# --- such as "data/..." are resolved from the repository root. Keep this block in every script.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "src").is_dir() and (p / "run").is_dir())
for _p in (_ROOT / "run", _ROOT / "src", _HERE):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
_os.chdir(_ROOT)
# ---------------------------------------------------------------------------------

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import add_noise, damped_oscillation
from paths import DATA, save_figure

# ---- experiment ------------------------------------------------------------------
t = np.linspace(0, 10, 400)
x_true = damped_oscillation(t)
t_obs = np.linspace(0, 10, 40)
x_obs = add_noise(damped_oscillation(t_obs), level=0.05, seed=0)

out = DATA / "example"
out.mkdir(parents=True, exist_ok=True)
np.savetxt(out / "damped_oscillation.csv", np.column_stack([t_obs, x_obs]),
           delimiter=",", header="t,x_measured", comments="")

# ---- figure ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.4, 2.4))           # one column of a two-column journal
ax.plot(t, x_true, color="#2A78D6", lw=1.5, label="model")
ax.plot(t_obs, x_obs, "o", ms=3, color="#EB6834", label="measurement")
ax.set_xlabel("time $t$")
ax.set_ylabel("$x(t)$")
ax.legend(frameon=False, fontsize=8)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

path = save_figure(fig, "example")
print(f"saved {Path(path).relative_to(_ROOT)}")
