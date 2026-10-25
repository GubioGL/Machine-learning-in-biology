# Notes for Claude

Research-paper repository made from GubioGL/paper-template. The author writes notes in Portuguese;
code comments and the paper are in English. Update the first line of README.md and this file with
the paper's subject once it is known.

## Layout rules
- `src/` = importable library code. `run/` = entry points (experiments, analysis, figures).
- Every script in `run/` starts with the repository header block (copy it from `run/example_figure.py`):
  it puts `run/`, `src/` and the script's folder on `sys.path` and `chdir`s to the repository root.
- Figures: always write to `paper/figures/` as PDF via `paths.save_figure(fig, name)`. Never commit PNGs.
- Data the figures need lives in `data/`; huge raw data goes to an archive (see `data/README.md`).
- Before stating a number in the paper, check it against the data in `data/`.

## Paper / Overleaf
- `paper/` is synced two-way with the Overleaf project in the repository variable `OVERLEAF_PROJECT`
  by `.github/workflows/overleaf-sync.yml` (script `.github/scripts/overleaf_sync.sh`; bookkeeping in
  `.github/overleaf-sync-state`, do not edit by hand).
- To change the paper: edit `paper/` and push to `main`; the workflow pushes to Overleaf.
- Branch `overleaf-conflict` = unresolved same-line edits; merge it into `main` and push.
- `paper/.placeholder` (until the first sync) lists the starter files an existing Overleaf draft replaces.
- Workflow status without a GitHub login: `https://api.github.com/repos/<owner>/<repo>/actions/runs`
  (public repositories), and error messages in `.../check-runs/<job id>/annotations`.
