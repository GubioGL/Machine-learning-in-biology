# Notes for Claude

Research-paper repository made from GubioGL/paper-template: "Lost signatures", validated machine
learning (classifiers, pairwise AUC) to measure biotic homogenisation in social-wasp assemblages
(Vespidae: Polistinae) of a preserved floodplain, an anthropised Pantanal wetland and the Cerrado
(Mato Grosso do Sul, Brazil). The author writes notes in Portuguese; code comments and the paper
are in English (LaTeX comments that guide the writing are in Portuguese).

## Project specifics
- Field data: `data/raw/dados_vespas.csv` (git-ignored; public repo, data unpublished). Never commit
  `data/raw/` unless the author says the data are published.
- Analysis = `run/01`..`run/05` (+ `run/run_all.py`); library code in `src/wasps.py`,
  `src/community.py`, `src/ml.py`, `src/report.py`. `WASPS_QUICK=1` for smoke tests.
- Numbers in the paper come from `paper/generated/res_*.tex` via `\res{key}` (keys: lowercase and
  '-'); tables via `\inputtable{tab_*}`. Never type a result number by hand in main.tex.
- `paper/guia_ponte_ml_biologia.tex` is a Portuguese helper document, not part of the paper.

## Layout rules
- `src/` = importable library code. `run/` = entry points (experiments, analysis, figures).
- Every script in `run/` starts with the repository header block (copy it from any script in `run/`):
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
- `paper/.placeholder` was removed on purpose: `paper/` holds the real manuscript, so on the first sync
  GitHub wins over any Overleaf draft (the Overleaf version stays in Overleaf History).
- Workflow status without a GitHub login: `https://api.github.com/repos/<owner>/<repo>/actions/runs`
  (public repositories), and error messages in `.../check-runs/<job id>/annotations`.
