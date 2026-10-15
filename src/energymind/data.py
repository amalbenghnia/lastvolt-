"""Couche Data & Provenance : chargement, validation, nettoyage TRACE (rien n'est corrige en silence)."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from .config import ROOT

def _num(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace("\u202f", "").replace(" ", "")
    if re.fullmatch(r"\d+,\d+", s):
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan

def load_invoices(path=None):
    """Retourne (factures_propres, journal_qualite). Chaque correction est journalisee."""
    raw = pd.read_csv(path or ROOT / "data/raw/campus_invoices_2022_2024.csv", dtype=str)
    log = []
    df = raw.copy()
    for c in ["Consumption_kWh", "Cost_TND", "Tax_TND", "Total_TND"]:
        df[c] = df[c].map(_num)
    # mois reconstruits par SEQUENCE : lignes chronologiques; un libelle qui rompt la monotonie est reaffecte
    months, cur = [], None
    for i, lab in enumerate(raw["month"]):
        if pd.isna(lab):
            months.append(pd.NaT)
            log.append((i, "orphan_row", "ligne sans mois (total seul) : exclue"))
            continue
        p = pd.Period(f"{lab[3:]}-{lab[:2]}", freq="M")
        if cur is not None and p <= cur:
            cand = cur + 1
            while cand.month != p.month:
                cand += 1
            log.append((i, "month_relabeled", f"{lab} -> {cand.strftime('%m/%Y')} (rupture de sequence, A CONFIRMER avec la direction technique)"))
            p = cand
        months.append(p)
        cur = p
    df["period"] = months
    df = df.dropna(subset=["period"]).copy()
    for i, r in df[df.Consumption_kWh.isna()].iterrows():
        log.append((i, "missing_consumption", f"{r.period} : facture absente"))
    df["price"] = df.Cost_TND / df.Consumption_kWh
    df["tot_ratio"] = df.Total_TND / (df.Cost_TND + df.Tax_TND)
    # bornes de plausibilite explicites (tarif moyen 0.20-0.40 TND/kWh; Total/(Cost+Tax) entre 1.0 et 1.35; kWh entier)
    nonint = (df.Consumption_kWh % 1 != 0) & df.Consumption_kWh.notna()
    bad_price = ~df.price.between(0.20, 0.40) & df.price.notna()
    bad_ratio = ~df.tot_ratio.between(1.0, 1.35) & df.tot_ratio.notna()
    df["suspect"] = (bad_price | bad_ratio | nonint).fillna(False)
    for i, r in df[df.suspect].iterrows():
        log.append((i, "suspect_values", f"{r.period} : incoherence prix/total/format (kWh={r.Consumption_kWh}, cost={r.Cost_TND}, total={r.Total_TND})"))
    df["valid"] = df.Consumption_kWh.notna() & ~df.suspect
    return df.reset_index(drop=True), pd.DataFrame(log, columns=["row", "issue", "detail"])

def monthly_energy_for_calibration(inv, months):
    """Energie mensuelle pour le simulateur. Mois invalides -> mediane saisonniere des mois valides, avec indicateur 'imputed'."""
    ok = inv[inv.valid].set_index("period").Consumption_kWh
    out = []
    for m in months:
        if m in ok.index:
            out.append((m, float(ok[m]), False))
        else:
            same = ok[[p.month == m.month for p in ok.index]]
            out.append((m, float(same.median() if len(same) else ok.median()), True))
    return pd.DataFrame(out, columns=["period", "kwh", "imputed"]).set_index("period")

def load_pv(path=None):
    df = pd.read_csv(path or ROOT / "data/raw/pv_dataset.csv", index_col=0, parse_dates=True)
    df.index.name = "timestamp"
    df.columns = [c.strip() for c in df.columns]
    return df

def pv_quality_report(df):
    idx = df.index
    full = pd.date_range(idx.min(), idx.max(), freq="5min")
    night = df.Irradiation < 1
    return {
        "rows": len(df), "start": str(idx.min()), "end": str(idx.max()),
        "missing_timestamps": int(len(full.difference(idx))), "duplicates": int(idx.duplicated().sum()),
        "nan_cells": int(df.isna().sum().sum()),
        "power_at_night_nonzero_pct": round(float((df.Power[night] > 1).mean() * 100), 3),
        "power_max": float(df.Power.max()), "irradiation_max": float(df.Irradiation.max()),
        "unit_of_power": "INCONNUE - a confirmer (W suppose : max 2.35 kW pour ~1.2 kW/m2)",
    }
