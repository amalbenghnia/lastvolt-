"""Injection de defauts REPRODUCTIBLE : verite physique separee de la telemetrie observee. Chaque fenetre injectee est documentee dans un registre."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import load_config, val

CLASSES = ["normal", "pv_fault", "grid_event", "pv_sensor_fault", "voltage_sensor_fault", "coordinated_falsification"]
# scenarios -> classe de diagnostic. M1 (biais) et M2 (derive) fusionnent en pv_sensor_fault (sous-type journalise)
SCENARIO_CLASS = {"P2": "pv_fault", "G1": "grid_event", "M1": "pv_sensor_fault", "M2": "pv_sensor_fault",
                  "V1": "voltage_sensor_fault", "A1": "coordinated_falsification"}

def observe(truth: pd.DataFrame, cfg: dict, rng) -> pd.DataFrame:
    """Telemetrie = verite + bruit de mesure (jamais la verite directement)."""
    nz = cfg["noise"]; rel, ab, vrel = nz["power_rel"], nz["power_abs_kw"], nz["voltage_rel"]
    def p(x): return x * (1 + rng.normal(0, rel, len(x))) + rng.normal(0, ab, len(x))
    obs = pd.DataFrame(index=truth.index)
    obs["pv_meas"] = p(truth.pv_kw); obs["load_meas"] = p(truth.load_kw)
    obs["bess_meas"] = p(truth.bess_kw); obs["grid_meas"] = p(truth.grid_kw)
    obs["inv_ac"] = p(truth.pv_kw)                     # telemetrie onduleur : mesure INDEPENDANTE du compteur PV
    obs["v_pcc"] = truth.v_pcc * (1 + rng.normal(0, vrel / 3, len(truth)))
    obs["v_inv"] = truth.v_inv * (1 + rng.normal(0, vrel / 3, len(truth)))
    for c in ["irradiance", "t_cell", "t_air", "wind"]:
        obs[c] = truth[c]
    return obs

def inject(truth: pd.DataFrame, n_events: int, seed: int, cfg=None):
    """Retourne (telemetrie observee, labels par pas de temps, registre des evenements)."""
    cfg = cfg or load_config()
    rng = np.random.default_rng(seed)
    t = truth.copy()
    obs_rng = np.random.default_rng(seed + 1)
    # etape 1 : defauts PHYSIQUES (modifient l'etat reel)
    n = len(t)
    day = ((t.irradiance > 300).values)
    labels = pd.Series("normal", index=t.index)
    sub = pd.Series("", index=t.index)
    reg = []
    scen = ["P2", "G1", "M1", "M2", "V1", "A1"]
    # evenements non chevauchants, positions tirees en jour pour les defauts PV
    cand = np.where(day)[0]
    cand = cand[(cand > 200) & (cand < n - 200)]
    taken = np.zeros(n, bool)
    plan = []
    for k in range(n_events):
        s = scen[k % len(scen)]
        for _ in range(200):
            i = int(rng.choice(cand)) if s != "G1" and s != "V1" else int(rng.integers(200, n - 200))
            dur = int(rng.integers(8, 28)) if s != "M2" else int(rng.integers(24, 48))
            if not taken[max(0, i - 8):i + dur + 8].any() and day[i:i + dur].all() if s not in ("G1", "V1") else not taken[max(0, i - 8):i + dur + 8].any():
                taken[i:i + dur] = True; plan.append((s, i, dur)); break
    # PV/charge/reseau physiques recalcules pour P2 et G1
    truth_mod = t.copy()
    for s, i, dur in plan:
        sl = slice(i, i + dur)
        if s == "P2":
            f = rng.uniform(0.25, 0.65)
            truth_mod.loc[truth_mod.index[sl], "pv_kw"] = t.pv_kw.iloc[sl] * f
            d = t.pv_kw.iloc[sl] * (1 - f)
            truth_mod.loc[truth_mod.index[sl], "grid_kw"] = t.grid_kw.iloc[sl] + d      # le reseau compense (BESS inchange)
        if s == "G1":
            sag = rng.uniform(0.08, 0.18)
            for c in ["v_pcc", "v_inv"]:
                truth_mod.loc[truth_mod.index[sl], c] = t[c].iloc[sl] * (1 - sag)
    obs = observe(truth_mod, cfg, obs_rng)
    # etape 2 : corruptions de MESURE (etat physique inchange)
    for s, i, dur in plan:
        idx = obs.index[i:i + dur]
        cls = SCENARIO_CLASS[s]
        labels.loc[idx] = cls
        sub.loc[idx] = s
        if s == "M1":
            obs.loc[idx, "pv_meas"] = obs.loc[idx, "pv_meas"] * rng.uniform(0.35, 0.7)
        elif s == "M2":
            ramp = np.linspace(0, rng.uniform(0.4, 0.7), dur)
            obs.loc[idx, "pv_meas"] = obs.loc[idx, "pv_meas"] * (1 - ramp)
        elif s == "V1":
            obs.loc[idx, "v_pcc"] = obs.loc[idx, "v_pcc"] * (1 - rng.uniform(0.08, 0.18))
        elif s == "A1":
            d = obs.loc[idx, "pv_meas"] * rng.uniform(0.35, 0.65)
            obs.loc[idx, "pv_meas"] -= d
            obs.loc[idx, "grid_meas"] += d            # bilan conserve : imite un vrai defaut PV
        reg.append({"scenario": s, "class": cls, "start": str(idx[0]), "steps": dur})
    obs["label"] = labels
    obs["scenario"] = sub
    return obs, pd.DataFrame(reg)
