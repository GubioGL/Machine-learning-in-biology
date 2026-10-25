# data/

Inputs and results that the scripts in `run/` read or write, and that the paper's numbers come from.

## `raw/` — field data (not in git)

The scripts read `data/raw/dados_vespas.csv`: one row per captured wasp, separator `;`, decimal `,`.

| column | meaning |
|---|---|
| `Tribo`, `Genero`, `Especie` | taxonomy (Polistinae: Epiponini, Polistini, Mischocyttarini) |
| `Ponto_Coleta` | protected area (`PEVRI`, `Baia Negra`, `BODOQUENA`) |
| `Data` | collection date, `dd/mm/yyyy` |
| `Armadilha` | `suspension` (attractive suspended trap), `malaise`, `puca` (net) |
| `Isca` | bait of suspended traps: `guava`, `orange`, `passion fruit` |
| `Paisagem` | landscape: `floodplain`, `pantanal`, `cerrado` |
| `Latitude`, `Longitude` | point coordinates (decimal degrees); a point is identified by them |

**Getting it:** copy `dados_.csv.csv` from the project Drive (folder *ML for biology*) to
`data/raw/dados_vespas.csv`. Expected after `run/01`: 46 points (18 floodplain, 17 Cerrado,
11 Pantanal), 17 species, 99 independent records; the script warns if the counts differ.

The folder is git-ignored because the repository is public and the data are not published yet.
When the dataset gets a DOI (e.g. Zenodo), either commit the file here or add a small download
script, and cite the DOI in the paper's data-availability statement. `run/01` also writes
`raw/site_by_species.csv`, the point x species matrix with readable point IDs (useful for R users).

## `results/` — outputs of the analysis (in git)

Machine-readable copies of everything the paper reports. Point-level files use anonymous IDs
(`FPL01`, `PAN03`, ...) and no coordinates.

| file | written by | content |
|---|---|---|
| `01.json` | `run/01` | PERMANOVA, PERMDISP, dissimilarities, distances, sampling table |
| `02.json`, `02_cv_metrics.csv` | `run/02` | metrics of every model and CV repetition, permutation tests |
| `02_oof_probabilities.csv` | `run/02` | mean out-of-fold class probabilities per point (main model) |
| `03.json`, `03_point_scores.csv` | `run/03` | pairwise AUCs, bootstrap, per-point scores |
| `04.json`, `04_species.csv` | `run/04` | Table S1: occurrence, importance, partial dependence, IndVal per species |
| `05.json` | `run/05` | sensitivity scenarios and sampling-campaign checks |

Each `*.json` has a `latex` block (the strings that go into `paper/generated/res_*.tex`) and a
`raw` block with full-precision values.
