"""Social-wasp data: loading the field records and building the site x species matrix.

The raw file (one row per captured individual) is data/raw/dados_vespas.csv:

    Tribo;Genero;Especie;Ponto_Coleta;Data;Armadilha;Isca;Paisagem;Latitude;Longitude

with ';' as separator and ',' as decimal mark. A sampling point is identified by its coordinates
(the file has no point ID). For the main analysis only the attractive suspended traps are used and
captures are converted into independent records: one record = species x point x date, so that many
workers of the same colony count once.

    from wasps import load_records, site_matrix
    rec = load_records()
    X, meta = site_matrix(rec)          # X: 46 points x 17 species; meta: landscape, month, ...
"""
from pathlib import Path

import numpy as np
import pandas as pd

from paths import DATA

RAW_CSV = DATA / "raw" / "dados_vespas.csv"

# Class order used everywhere (tables, figures, probability columns).
LANDSCAPES = ["floodplain", "pantanal", "cerrado"]
LABEL = {"floodplain": "Floodplain", "pantanal": "Pantanal", "cerrado": "Cerrado"}
SHORT = {"floodplain": "Fpl", "pantanal": "Pan", "cerrado": "Cer"}
COLOR = {"floodplain": "#2A78D6", "pantanal": "#1B9E77", "cerrado": "#D95F02"}
PROTECTED_AREA = {
    "floodplain": "Várzeas do Rio Ivinhema State Park (PEVRI)",
    "pantanal": "Baía Negra Environmental Protection Area (APA)",
    "cerrado": "Serra da Bodoquena National Park",
}

# Taxonomic authorities (from Table S1 of the first draft), used only to print tables.
AUTHORITY = {
    "Agelaia bequaerti": "(Richards, 1951)",
    "Agelaia pallipes": "(Olivier, 1791)",
    "Apoica flavissima": "Van der Vecht, 1972",
    "Apoica pallens": "(Fabricius, 1804)",
    "Brachygastra lecheguana": "(Latreille, 1824)",
    "Mischocyttarus consimilis": "Zikán, 1949",
    "Parachartergus smithii": "(de Saussure, 1854)",
    "Polistes lanio": "(Fabricius, 1775)",
    "Polistes simillimus": "Zikán, 1951",
    "Polistes versicolor": "(Olivier, 1792)",
    "Polybia chrysothorax": "(Lichtenstein, 1796)",
    "Polybia ignobilis": "(Haliday, 1836)",
    "Polybia jurinei": "de Saussure, 1854",
    "Polybia occidentalis": "(Olivier, 1791)",
    "Polybia paulista": "H. von Ihering, 1896",
    "Polybia ruficeps": "Schrottky, 1902",
    "Polybia sericea": "(Olivier, 1792)",
}

# Dry season in Mato Grosso do Sul (approximately May-September/October).
DRY_MONTHS = {5, 6, 7, 8, 9, 10}


def load_records(path=RAW_CSV):
    """Read the raw CSV (one row per individual) and return a clean DataFrame.

    Adds the columns `point` (coordinate key), `month`, `year` and `campaign` (YYYY-MM).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Copy the field data (dados_.csv from the project Drive) to "
            "data/raw/dados_vespas.csv; see data/README.md.")
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, sep=";", decimal=",", encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    df.columns = [c.strip() for c in df.columns]
    for c in ("Tribo", "Genero", "Especie", "Ponto_Coleta", "Armadilha", "Isca", "Paisagem"):
        df[c] = df[c].astype("string").str.strip()
    df["Paisagem"] = df["Paisagem"].str.lower()
    df["Armadilha"] = df["Armadilha"].str.lower()
    df["Data"] = pd.to_datetime(df["Data"], dayfirst=True)
    df["point"] = df["Latitude"].round(6).astype(str) + "_" + df["Longitude"].round(6).astype(str)
    df["month"] = df["Data"].dt.month
    df["year"] = df["Data"].dt.year
    df["campaign"] = df["Data"].dt.strftime("%Y-%m")
    return df


def site_matrix(rec, traps=("suspension",), binary=False, min_points=1, dry_only=False):
    """Site x species matrix of independent records (species-point-date = 1 record).

    Parameters
    ----------
    traps      : trap types to keep ("suspension", "malaise", "puca").
    binary     : True -> presence/absence instead of record frequency.
    min_points : drop species present in fewer than `min_points` points (1 keeps all).
    dry_only   : keep only records from dry-season months (sensitivity analysis).

    Returns
    -------
    X    : DataFrame (points x species), integer frequencies (or 0/1).
    meta : DataFrame indexed like X with landscape, coordinates, campaign, season and a readable
           point ID (e.g. FPL07) used in every saved result instead of the coordinates.
    """
    d = rec[rec["Armadilha"].isin(traps)]
    if dry_only:
        d = d[d["month"].isin(DRY_MONTHS)]
    indep = d.drop_duplicates(["Especie", "point", "Data"])
    X = indep.pivot_table(index="point", columns="Especie", values="Data",
                          aggfunc="count", fill_value=0).astype(int)
    X.columns.name = None

    g = d.groupby("point")
    meta = pd.DataFrame({
        "landscape": g["Paisagem"].first(),
        "lat": g["Latitude"].first(),
        "lon": g["Longitude"].first(),
        "n_dates": g["Data"].nunique(),
        "campaign": g["campaign"].agg(lambda s: s.mode().iloc[0]),
        "month": g["month"].agg(lambda s: s.mode().iloc[0]),
    }).loc[X.index]
    meta["season"] = np.where(meta["month"].isin(DRY_MONTHS), "dry", "wet")

    # stable readable IDs: landscape prefix + order along latitude, longitude
    meta = meta.sort_values(["landscape", "lat", "lon"])
    meta["point_id"] = (meta.groupby("landscape").cumcount() + 1).astype(str).str.zfill(2)
    meta["point_id"] = meta["landscape"].map(lambda k: SHORT[k].upper()) + meta["point_id"]
    X = X.loc[meta.index]

    if min_points > 1:
        X = X.loc[:, (X > 0).sum(axis=0) >= min_points]
    if binary:
        X = (X > 0).astype(int)
    X = X[sorted(X.columns)]
    return X, meta


def species_tribe(rec):
    """Species -> tribe (Epiponini, Polistini, Mischocyttarini)."""
    return rec.drop_duplicates("Especie").set_index("Especie")["Tribo"].astype(str).to_dict()


def summary(X, meta):
    """Counts used in the Methods (points per landscape, species, independent records)."""
    n_by = meta["landscape"].value_counts().reindex(LANDSCAPES).to_dict()
    return {"n_points": len(X), "n_species": X.shape[1], "n_records": int(X.values.sum()),
            **{f"n_{k}": int(v) for k, v in n_by.items()}}
