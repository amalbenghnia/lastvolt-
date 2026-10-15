"""
Fault Catalog V2 — 8 diagnostic scenario classes.

Classes:
  - normal                    : Normal baseline operation
  - pv_fault (P2)             : Physical PV fault (shading, inverter trip, degradation)
  - grid_event (G1)           : Upstream utility grid voltage sag
  - load_peak (L1)            : Genuine demand surge (HVAC / EV charging spike)
  - pv_sensor_fault (M1/M2)   : Biased or drifting PV generation sensor
  - voltage_sensor_fault (V1) : Biased PCC voltage sensor
  - bess_sensor_fault (B1)    : Biased battery state-of-charge (SOC) sensor
  - coordinated_falsification : Coordinated cyber spoofing (PV & grid altered symmetrically)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import load_config

CLASSES_V2 = [
    "normal",
    "pv_fault",
    "grid_event",
    "load_peak",
    "pv_sensor_fault",
    "voltage_sensor_fault",
    "bess_sensor_fault",
    "coordinated_falsification",
]

SCENARIO_CLASS_V2 = {
    "P2": "pv_fault",
    "G1": "grid_event",
    "L1": "load_peak",
    "M1": "pv_sensor_fault",
    "M2": "pv_sensor_fault",
    "V1": "voltage_sensor_fault",
    "B1": "bess_sensor_fault",
    "A1": "coordinated_falsification",
}


def observe_v2(truth: pd.DataFrame, cfg: dict, rng) -> pd.DataFrame:
    """Generate observed telemetry with additive and multiplicative measurement noise."""
    nz = cfg["noise"]
    rel = nz["power_rel"]
    ab = nz["power_abs_kw"]
    vrel = nz["voltage_rel"]

    def p(x):
        return x * (1 + rng.normal(0, rel, len(x))) + rng.normal(0, ab, len(x))

    obs = pd.DataFrame(index=truth.index)
    obs["pv_meas"] = p(truth.pv_kw)
    obs["load_meas"] = p(truth.load_kw)
    obs["bess_meas"] = p(truth.bess_kw)
    obs["grid_meas"] = p(truth.grid_kw)
    obs["inv_ac"] = p(truth.pv_kw)
    obs["soc_meas"] = truth.soc * (1 + rng.normal(0, 0.01, len(truth)))
    obs["v_pcc"] = truth.v_pcc * (1 + rng.normal(0, vrel / 3, len(truth)))
    obs["v_inv"] = truth.v_inv * (1 + rng.normal(0, vrel / 3, len(truth)))
    for c in ["irradiance", "t_cell", "t_air", "wind", "humidity"]:
        if c in truth.columns:
            obs[c] = truth[c]
    return obs


def inject_v2(truth: pd.DataFrame, n_events: int, seed: int, cfg=None):
    """
    Inject n_events fault episodes across 8 scenario classes.

    Physical faults (P2, G1, L1) modify the true power balance and voltages.
    Measurement faults (M1, M2, V1, B1, A1) corrupt telemetry while physical
    states remain governed by electrical laws.
    """
    cfg = cfg or load_config()
    rng = np.random.default_rng(seed)
    obs_rng = np.random.default_rng(seed + 1)

    t = truth.copy()
    n = len(t)
    day_mask = (t.irradiance > 300).values if "irradiance" in t.columns else np.ones(n, bool)
    labels = pd.Series("normal", index=t.index)
    sub = pd.Series("", index=t.index)
    reg = []

    scen_all = ["P2", "G1", "L1", "M1", "M2", "V1", "B1", "A1"]
    cand_day = np.where(day_mask)[0]
    cand_day = cand_day[(cand_day > 200) & (cand_day < n - 200)]
    cand_all = np.arange(200, n - 200)
    taken = np.zeros(n, bool)
    plan = []

    for k in range(n_events):
        s = scen_all[k % len(scen_all)]
        needs_day = s in ("P2", "M1", "M2", "B1")
        cand = cand_day if needs_day and len(cand_day) > 0 else cand_all

        for _ in range(300):
            i = int(rng.choice(cand))
            dur = int(rng.integers(8, 28)) if s != "M2" else int(rng.integers(24, 48))
            buf = 8
            if not taken[max(0, i - buf): i + dur + buf].any():
                taken[i: i + dur] = True
                plan.append((s, i, dur))
                break

    truth_mod = t.copy()
    for s, i, dur in plan:
        sl = slice(i, i + dur)
        idx_sl = truth_mod.index[sl]

        if s == "P2":
            f = rng.uniform(0.25, 0.65)
            truth_mod.loc[idx_sl, "pv_kw"] = t.pv_kw.iloc[sl] * f
            truth_mod.loc[idx_sl, "grid_kw"] = t.grid_kw.iloc[sl] + t.pv_kw.iloc[sl] * (1 - f)

        elif s == "G1":
            sag = rng.uniform(0.08, 0.18)
            for c in ["v_pcc", "v_inv"]:
                truth_mod.loc[idx_sl, c] = t[c].iloc[sl] * (1 - sag)

        elif s == "L1":
            factor = rng.uniform(1.40, 1.80)
            truth_mod.loc[idx_sl, "load_kw"] = t.load_kw.iloc[sl] * factor
            truth_mod.loc[idx_sl, "grid_kw"] = (
                truth_mod.loc[idx_sl, "load_kw"]
                + t.loss_kw.iloc[sl] * factor
                - t.pv_kw.iloc[sl]
                - t.bess_kw.iloc[sl]
            )

    obs = observe_v2(truth_mod, cfg, obs_rng)

    for s, i, dur in plan:
        idx = obs.index[i: i + dur]
        cls = SCENARIO_CLASS_V2[s]
        labels.loc[idx] = cls
        sub.loc[idx] = s

        if s == "M1":
            bias = rng.uniform(0.35, 0.70)
            obs.loc[idx, "pv_meas"] = obs.loc[idx, "pv_meas"] * bias

        elif s == "M2":
            ramp = np.linspace(0, rng.uniform(0.40, 0.70), dur)
            obs.loc[idx, "pv_meas"] = obs.loc[idx, "pv_meas"] * (1 - ramp)

        elif s == "V1":
            obs.loc[idx, "v_pcc"] = obs.loc[idx, "v_pcc"] * (1 - rng.uniform(0.08, 0.18))

        elif s == "B1":
            obs.loc[idx, "soc_meas"] = obs.loc[idx, "soc_meas"] * rng.uniform(0.50, 0.80)
            obs.loc[idx, "bess_meas"] = obs.loc[idx, "bess_meas"] * rng.uniform(1.20, 1.60)

        elif s == "A1":
            delta = obs.loc[idx, "pv_meas"] * rng.uniform(0.35, 0.65)
            obs.loc[idx, "pv_meas"] -= delta
            obs.loc[idx, "grid_meas"] += delta

        reg.append({
            "scenario": s,
            "class": cls,
            "start": str(idx[0]),
            "steps": dur,
            "physical_injection": s in ("P2", "G1", "L1"),
            "measurement_corruption": s in ("M1", "M2", "V1", "B1", "A1"),
        })

    obs["label"] = labels
    obs["scenario"] = sub
    return obs, pd.DataFrame(reg)
