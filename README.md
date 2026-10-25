# Lost signatures: machine learning and social-wasp assemblages of wetlands and savannas

Artigo *Lost signatures*: aprendizado de máquina validado para medir a perda de identidade
ecológica (homogeneização biótica) em assembleias de vespas sociais (Vespidae: Polistinae) de três
áreas protegidas de Mato Grosso do Sul — várzea preservada (PEVRI), Pantanal antropizado (APA Baía
Negra) e Cerrado (PN Serra da Bodoquena).

**Pergunta:** a lista de vespas de um ponto identifica a paisagem de origem? O Pantanal antropizado
ficou mais parecido com o Cerrado seco do que com a várzea preservada?

## O que há aqui

| onde | o quê |
|---|---|
| `paper/main.tex` | artigo v0.1 (Overleaf): introdução, métodos, resultados, discussão, apêndices com a matemática de cada algoritmo; comentários `%` em português dizem o que cada seção deve conter |
| `paper/guia_ponte_ml_biologia.tex` | guia-ponte (não é o artigo): biologia para quem vem do ML, ML para quem vem da biologia, o que cada algoritmo faz e por que ajuda |
| `paper/generated/` | números (`res_*.tex`) e tabelas (`tab_*.tex`) escritos pelos scripts e lidos pelo artigo |
| `docs/relatorio_pesquisa.md` | revisão da literatura de alto impacto, diagnóstico dos dados e lista de algoritmos |
| `src/` | `wasps.py` (dados), `community.py` (Jaccard, PCoA, PERMANOVA, PERMDISP, IndVal), `ml.py` (modelos, validação, testes), `report.py` (números e tabelas para o LaTeX) |
| `run/01`–`05` | uma etapa da análise por script (ver abaixo); `run/run_all.py` roda tudo |
| `data/` | `raw/` (dados de campo, fora do git) e `results/` (saídas em JSON/CSV) |

## Como rodar a análise

```bash
pip install -r run/requirements.txt
# copie o CSV de campo para data/raw/dados_vespas.csv (ver data/README.md)
WASPS_QUICK=1 python run/run_all.py   # teste rápido, ~2 min
python run/run_all.py                 # análise completa, ~40 min em 2 núcleos
git add paper data/results && git commit -m "Atualiza resultados" && git push
```

| script | objetivo | figura / tabela |
|---|---|---|
| `01_community_description.py` | descrever: Jaccard, PCoA, PERMANOVA, PERMDISP, distâncias | Fig. 1, Tab. 1, Tab. 3 |
| `02_classifier_benchmark.py` | O1: 7 classificadores, CV 5×50, testes de permutação | Fig. 2, Tab. 4 |
| `03_homogenization_test.py` | O2: AUC por pares, bootstrap pareado, probabilidades | Fig. 3, Tab. 5 |
| `04_species_interpretation.py` | O3: importância fora da amostra + Altmann, dependência parcial, SHAP, IndVal | Figs. 4–5, S1, Tab. S1 |
| `05_sensitivity.py` | O4: blocos espaciais, n igual, sem espécies, hiperparâmetros, campanhas | Fig. S2, Tab. S2 |

Depois do push, os números do texto (`\res{...}`) e as figuras se atualizam no Overleaf.

---

# Paper template: code, data and LaTeX, synced with Overleaf

Template for a research paper whose figures come from code. The repository holds the code, the data
and the manuscript. The manuscript folder, `paper/`, stays in sync with an Overleaf project in both
directions, so you can write in Overleaf and regenerate figures here.

## Novo artigo em 5 passos

1. **Criar o repositório:** no GitHub, abra este modelo e clique em **Use this template → Create a new
   repository**. Dê o nome do artigo.
2. **Criar o projeto no Overleaf:** *New Project → Blank Project*. Se você já tem um rascunho no Overleaf,
   use esse projeto. Copie o endereço da página do projeto (`https://www.overleaf.com/project/...`).
3. **Conectar:** no repositório novo, abra *Settings → Secrets and variables → Actions*:
   - aba **Variables** → *New repository variable*: nome `OVERLEAF_PROJECT`, valor = o endereço
     copiado;
   - aba **Secrets** → *New repository secret*: nome `OVERLEAF_TOKEN`, valor = um token Git do Overleaf
     (*Account Settings → Git integration*; o mesmo token serve para todos os projetos e vale um ano).
4. **Primeira sincronização:** *Actions → Overleaf sync → Run workflow*.
   - **Projeto em branco:** o artigo inicial deste modelo vai para o Overleaf.
   - **Projeto com rascunho:** o rascunho vem para `paper/`, substituindo o artigo inicial.
5. **Personalizar:** troque o `\documentclass` da revista em `paper/main.tex`. Depois apague os exemplos
   (`src/analysis.py`, `run/example_figure.py`, `data/example/`) quando tiver o seu código
   (neste repositório já foram apagados).

Pronto. Daqui em diante cada push que mexe em `paper/` vai para o Overleaf. O que você escrever no
Overleaf volta para cá a cada 3 horas, ou na hora com *Run workflow*.

## Layout

```text
├── src/          reusable code (models, solvers, loaders); importable from every script
│   └── paths.py  repository paths + save_figure()
├── run/          scripts you execute (experiments, analysis, figures) + requirements.txt
├── data/         inputs and results the scripts and the paper's numbers come from
├── paper/        LaTeX manuscript, synced with Overleaf
│   ├── main.tex, references.bib
│   └── figures/  written by the scripts in run/
└── .github/      Overleaf sync (workflow + script)
```

## Figures go straight into the paper

Save every figure through `save_figure` and it lands in `paper/figures/`, the folder LaTeX reads:

```python
from paths import save_figure

fig, ax = plt.subplots()
...
save_figure(fig, "fig2_results")      # -> paper/figures/fig2_results.pdf (+ local .png preview)
```

Then `\includegraphics{fig2_results.pdf}` in `paper/main.tex`, commit and push, and the figure appears in
Overleaf. Any script in `run/` (e.g. `run/01_community_description.py`) is a complete example.

Scripts start with a short header block that puts `src/` and `run/` on the import path and makes
`data/...` paths relative to the repository root. Keep it in new scripts (copy it from any script in `run/`), and
`python run/<script>.py` works from any folder.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r run/requirements.txt
python run/run_all.py
```

## How the Overleaf sync works

`.github/workflows/overleaf-sync.yml` runs `.github/scripts/overleaf_sync.sh`. The script uses
Overleaf's Git integration, which needs an Overleaf premium plan.

| Direction | When |
|---|---|
| GitHub → Overleaf | every push to `main` that changes `paper/` |
| Overleaf → GitHub | every 3 hours (`cron` in the workflow), or right away with *Run workflow* |

- **Edits on both sides** to different parts of the text are merged automatically.
- **The same lines edited on both sides:** the run stops and puts the Overleaf version on the branch
  `overleaf-conflict`. Merge it into `main` and push, and the next run finishes the sync.
- **Nothing is lost:** Overleaf keeps every earlier version in its *History* (top right of the editor).
- **Not connected yet:** while `OVERLEAF_PROJECT` is not set, the job is skipped. It costs no Actions
  minutes, so this template itself never syncs.
- **Errors:** a failed run shows the reason, including Overleaf's own message, in the run summary.
- The sync's bookkeeping is `.github/overleaf-sync-state`; do not edit it.

Overleaf limits: up to 2,000 files per project and 50 MB per file. Keep only the manuscript and its
figures in `paper/`.
