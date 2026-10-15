"""Physics Simulation Engine : microgrid de campus (PV + charge + BESS + reseau STEG). Verite physique = etat reel, avant mesure."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import load_config, val
from .data import load_invoices, load_pv, monthly_energy_for_calibration

def _load_profile(idx, rng):
    """Profil intra-mensuel HYPOTHETIQUE (hypothese documentee) : occupation campus jour ouvre / week-end + bruit AR(1)."""
    h = idx.hour + idx.minute / 60
    weekday = (idx.dayofweek < 5).astype(float)
    occ = 0.35 + weekday * 0.65 * np.exp(-((h - 12.5) / 4.5) ** 2) + (1 - weekday) * 0.15 * np.exp(-((h - 12) / 5) ** 2)
    eps = rng.normal(0, 0.04, len(idx))
    ar = np.zeros(len(idx))
    for i in range(1, len(idx)):
        ar[i] = 0.9 * ar[i - 1] + eps[i]
    return np.clip(occ * (1 + ar), 0.05, None)

def build_baseline(cfg=None, seed=7) -> pd.DataFrame:
    cfg = cfg or load_config()
    rng = np.random.default_rng(seed)
    pv5 = load_pv()
    pv = pv5.resample("15min").mean()
    dt = 0.25
    pnom = val(cfg["pv"]["pnom_kwp"])
    pv_kw = pv.Power / pv.Power.quantile(0.999) * pnom        # PV normalise : on ne suppose PAS que max = Pnom
    inv, _ = load_invoices()
    months = pv.index.to_period("M").unique()
    me = monthly_energy_for_calibration(inv, months)
    w = pd.Series(_load_profile(pv.index, rng), index=pv.index)
    load = pd.Series(0.0, index=pv.index)
    for m in months:
        mask = pv.index.to_period("M") == m
        full_steps = m.days_in_month * 96
        e_target = me.loc[m, "kwh"] * mask.sum() / full_steps   # mois partiel (24 fev 2022, fin 05/2024)
        load[mask] = w[mask] * e_target / (w[mask].sum() * dt)
    b = cfg["bess"]
    cap, pmax = val(b["capacity_kwh"]), val(b["power_kw"])
    ec, ed = val(b["eta_charge"]), val(b["eta_discharge"])
    lo, hi = b["soc_min"], b["soc_max"]
    n = len(pv)
    soc = np.empty(n); bess = np.zeros(n)
    s = 0.5
    net = (load - pv_kw).values        # >0 : manque, <0 : surplus PV
    for i in range(n):
        if net[i] < 0:                 # surplus -> charge
            p = min(-net[i], pmax, (hi - s) * cap / (dt * ec))
            bess[i] = -p; s += p * dt * ec / cap
        else:                          # manque -> decharge (autoconsommation), reserve 10 %
            p = min(net[i], pmax, (s - lo) * cap * ed / dt)
            bess[i] = p; s -= p * dt / (ed * cap)
        soc[i] = s
    nw = cfg["network"]
    loss = val(nw["loss_load_frac"]) * load + val(nw["loss_pv_frac"]) * pv_kw
    grid = load + loss - pv_kw - bess                          # import STEG (>0 = import)
    vnom = val(nw["v_nominal_v"])
    v = vnom * (1 - 0.012 * (load / load.max())) + rng.normal(0, 0.8, n)
    out = pd.DataFrame({
        "pv_kw": pv_kw, "load_kw": load, "bess_kw": bess, "soc": soc, "grid_kw": grid, "loss_kw": loss,
        "v_pcc": v, "v_inv": v + rng.normal(0, 0.4, n),
        "irradiance": pv.Irradiation, "t_cell": pv.Cell_temperature, "t_air": pv.Temperature,
        "wind": pv.Wind_speed, "humidity": pv.Humidity,
    }, index=pv.index)
    out.attrs["imputed_months"] = [str(p) for p in me.index[me.imputed]]
    return out

def energy_balance_error(df) -> float:
    """Controle de conservation : doit etre ~0 (qualite du simulateur)."""
    r = df.grid_kw + df.pv_kw + df.bess_kw - df.load_kw - df.loss_kw
    return float(r.abs().max())
