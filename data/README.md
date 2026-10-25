# data/

Inputs and results that the scripts in `run/` read or write, and that the paper's numbers come from.

- Keep everything the figures need here, so `run/` can regenerate the paper from a fresh clone.
- GitHub rejects files over 100 MB and slows down above ~1 GB per repository. Very large raw data
  (checkpoints, per-epoch histories) belongs in an archive such as Zenodo, which also gives a DOI for
  the data-availability statement. Commit a small summary (CSV/JSON) here and a script that downloads
  the rest.
- `example/` is written by `run/example_figure.py`; delete it with the example.
