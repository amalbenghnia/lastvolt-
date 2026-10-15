"""
Experimental Evaluation Protocol V2 — EnergyMind Phase 1.

Key capabilities:
  - 8-class diagnostic scenarios including load peaks and BESS telemetry faults.
  - One-vs-rest isotonic probability calibration (ECE <= 0.015 target).
  - Expected-loss Bayesian decision policy routing.
  - 5-seed repeated trials for 95% confidence intervals.
  - pandapower 7-bus distribution feeder AC power-flow verification.
  - Cross-node Noosphere federation transfer validation.

Data-leakage prevention protocol:
  - Training period    : 2022-02 to 2023-03 (healthy operation only).
  - Calibration period : 2023-04 to 2023-10 (48 injected episodes).
  - Testing period     : 2023-11 to 2024-05 (5 random seeds x 72 episodes).
  - Isotonic split     : held-out 30% partition of calibration interval.
"""
from __future__ import annotations
import json
import copy
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

from .config import load_config, ROOT
from .simulate import build_baseline
from .faults import inject, observe, CLASSES
from .faults_v2 import inject_v2, observe_v2, CLASSES_V2, SCENARIO_CLASS_V2
from .engine import FeatureBuilder, Diagnoser, persistent, FEATURES
from .drift import cusum_scan, calibrate_cusum
from .calibration import IsotonicCalibrator, ece_score, brier_score, reliability_data
from .risk_policy import route_v2, risk_coverage_curve, LOSS_MATRIX, ACTIONS
from .noosphere_v2 import export_v2, import_v2, domain_shift_report, FactoryNode, FEDERATION_PEERS
from .governance import AuditLog

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="energymind")
pd.set_option("future.no_silent_downcasting", True)


def prf(y: np.ndarray, yhat: np.ndarray) -> dict:
    tp = int((y & yhat).sum())
    fp = int((~y & yhat).sum())
    fn = int((y & ~yhat).sum())
    tn = int((~y & ~yhat).sum())
    return {
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "fpr": fp / max(fp + tn, 1),
        "tp": tp, "fp": fp, "fn": fn,
    }


def event_stats(obs: pd.DataFrame, flag: np.ndarray, reg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in reg.iterrows():
        s = obs.index.get_loc(pd.Timestamp(r.start))
        sl = slice(s, s + r.steps)
        f = flag[sl]
        rows.append({
            "scenario": r.scenario,
            "class": r["class"],
            "detected": bool(f.any()),
            "latency_steps": int(np.argmax(f)) if f.any() else None,
        })
    return pd.DataFrame(rows)


def ci_mean(values: list, confidence: float = 0.95) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return float(arr.mean()), float(arr.mean()), float(arr.mean())
    se = stats.sem(arr)
    h = se * stats.t.ppf((1 + confidence) / 2, df=len(arr) - 1)
    return float(arr.mean()), float(arr.mean() - h), float(arr.mean() + h)


def run_v2(
    n_cal: int = 48,
    n_test: int = 72,
    hide: tuple = ("inv",),
    seeds_test: tuple = (22, 33, 44, 55, 66),
    out: str | None = "outputs",
    use_v2_faults: bool = True,
) -> dict:
    """Run full benchmark experiment across all 5 test seeds."""
    cfg = load_config()
    base = build_baseline(cfg)
    P = cfg["periods"]
    sl = lambda k: base.loc[P[k][0]: P[k][1]]

    tr = sl("train_pv_model")
    o_tr = observe(tr, cfg, np.random.default_rng(1))
    fb = FeatureBuilder().fit_pv_model(o_tr).fit_scales(o_tr)

    inject_fn = inject_v2 if use_v2_faults else inject
    o_cal, reg_cal = inject_fn(sl("calibrate_diagnosis"), n_cal, cfg["seed_calibration"], cfg)
    classes = CLASSES_V2 if use_v2_faults else CLASSES

    n_split = int(len(o_cal) * 0.70)
    o_diag = o_cal.iloc[:n_split]
    o_iso = o_cal.iloc[n_split:]

    Zc = fb.z(o_diag, hidden=())
    dg = Diagnoser(prior_normal=0.9)
    if use_v2_faults:
        dg.fit_v2(Zc, o_diag.label, classes)
    else:
        dg.fit(Zc, o_diag.label)

    Zi = fb.z(o_iso, hidden=())
    Pi = dg.posterior(Zi)
    cal = IsotonicCalibrator(classes)
    cal.fit(Pi, o_iso.label.values)

    from .network_v2 import PandapowerSurrogate
    surrogate = PandapowerSurrogate()
    surrogate.fit(rng=np.random.default_rng(99))
    surrogate_report = surrogate.validation_report

    results_per_seed = []

    for seed_idx, seed in enumerate(seeds_test):
        o_te, reg_te = inject_fn(sl("test"), n_test, seed, cfg)
        Z = fb.z(o_te, hidden=hide)
        Zfull = fb.z(o_te, hidden=())
        y_anom = (o_te.label != "normal").values

        post = dg.posterior(Z)
        post_cal = cal.transform(post)
        flagE = persistent(post_cal[:, 0] < 0.2)

        cusum_p = calibrate_cusum(fb.z(o_tr, hidden=hide), ["pvexp", "bal"], target_fpr=0.01)
        cusum_a = cusum_scan(Z, ["pvexp", "bal"], k=cusum_p["k"], h=cusum_p["h"])
        flagC = persistent(cusum_a.any(axis=1).values)

        raw_cols = ["pv_meas", "load_meas", "grid_meas", "bess_meas", "v_pcc", "v_inv", "irradiance"]
        iso_raw = IsolationForest(n_estimators=200, contamination=0.005, random_state=0).fit(o_tr[raw_cols])
        flagA = persistent(iso_raw.predict(o_te[raw_cols]) == -1)
        flagB = persistent((Z.bal.abs() > 4).values | ((Z.vpcc - Z.vinv).abs() > 4).values)
        Z_tr = fb.z(o_tr, hidden=hide).fillna(0)
        isoR = IsolationForest(n_estimators=200, contamination=0.005, random_state=0).fit(Z_tr)
        flagIF = persistent(isoR.predict(Z.fillna(0)) == -1)

        det_methods = {
            "A_isoforest_brut": prf(y_anom, flagA),
            "B_physique_seule": prf(y_anom, flagB),
            "C_isoforest_resid": prf(y_anom, flagIF),
            "EnergyMind": prf(y_anom, flagE),
            "CUSUM": prf(y_anom, flagC),
        }
        for k, v in det_methods.items():
            es = event_stats(o_te, {
                "A_isoforest_brut": flagA,
                "B_physique_seule": flagB,
                "C_isoforest_resid": flagIF,
                "EnergyMind": flagE,
                "CUSUM": flagC,
            }[k], reg_te)
            v["event_recall"] = float(es.detected.mean())
            v["event_recall_by_class"] = es.groupby("class").detected.mean().round(3).to_dict()
            lat = es.latency_steps.dropna()
            v["median_latency_steps"] = float(lat.median()) if len(lat) > 0 else None

        fa = flagE & ~y_anom
        ep = int(((fa.astype(int) - np.r_[0, fa[:-1].astype(int)]) == 1).sum())
        fa_per_day = ep / (len(o_te) / 96)

        rows_ev, ece_list = [], []
        for _, r in reg_te.iterrows():
            s = o_te.index.get_loc(pd.Timestamp(r.start))
            idx_ev = np.arange(s, s + r.steps)
            fl = idx_ev[flagE[idx_ev]]
            if len(fl) == 0:
                rows_ev.append({"truth": r["class"], "scenario": r.scenario, "detected": False})
                continue
            p_ev = post_cal[fl].mean(axis=0)
            top = int(p_ev.argmax())
            conf = float(p_ev[top])
            pred = classes[top]
            route = route_v2(p_ev)
            decided = conf >= 0.90
            rows_ev.append({
                "truth": r["class"], "scenario": r.scenario, "detected": True,
                "pred": pred, "conf": conf, "decided": bool(decided),
                "action": route["action"], "human_required": route["human_required"],
            })
            if decided:
                ece_list.append((conf, pred == r["class"]))

        ev_df = pd.DataFrame(rows_ev)
        det_ev = ev_df[ev_df.detected.fillna(False).infer_objects(copy=False).astype(bool)].copy()
        dec_ev = det_ev[det_ev.decided.fillna(False).infer_objects(copy=False).astype(bool)]

        ece_val = ece_score(post_cal[flagE], o_te.label[flagE].values, classes) if flagE.any() else None
        brier_val = brier_score(post_cal[flagE], o_te.label[flagE].values, classes) if flagE.any() else None

        results_per_seed.append({
            "seed": seed,
            "detection": det_methods,
            "fa_per_day": fa_per_day,
            "diag": {
                "n_detected": int(len(det_ev)),
                "abstention_rate": float(1 - len(dec_ev) / max(len(det_ev), 1)),
                "acc_forced": float((det_ev.pred == det_ev.truth).mean()) if len(det_ev) else None,
                "acc_decided": float((dec_ev.pred == dec_ev.truth).mean()) if len(dec_ev) else None,
                "ece_calibrated": ece_val,
                "brier_score": brier_val,
                "human_review_rate": float(det_ev.human_required.fillna(True).infer_objects(copy=False).astype(bool).mean()) if len(det_ev) else None,
            },
        })

    def _agg(values, ci=True):
        m, lo, hi = ci_mean(values)
        return {"mean": round(m, 4), "ci95_low": round(lo, 4), "ci95_high": round(hi, 4)} if ci else round(m, 4)

    summary = {"n_seeds": len(seeds_test), "n_test_events_per_seed": n_test}
    for method in ["A_isoforest_brut", "B_physique_seule", "C_isoforest_resid", "EnergyMind", "CUSUM"]:
        summary.setdefault("detection", {})[method] = {
            "event_recall": _agg([r["detection"][method]["event_recall"] for r in results_per_seed]),
            "precision": _agg([r["detection"][method]["precision"] for r in results_per_seed]),
            "recall": _agg([r["detection"][method]["recall"] for r in results_per_seed]),
            "fpr": _agg([r["detection"][method]["fpr"] for r in results_per_seed]),
        }

    summary["false_alarms_per_day"] = _agg([r["fa_per_day"] for r in results_per_seed])
    summary["diagnosis"] = {
        "acc_decided": _agg([r["diag"]["acc_decided"] for r in results_per_seed if r["diag"]["acc_decided"] is not None]),
        "acc_forced": _agg([r["diag"]["acc_forced"] for r in results_per_seed if r["diag"]["acc_forced"] is not None]),
        "abstention_rate": _agg([r["diag"]["abstention_rate"] for r in results_per_seed]),
        "ece_calibrated": _agg([r["diag"]["ece_calibrated"] for r in results_per_seed if r["diag"]["ece_calibrated"] is not None]),
        "brier_score": _agg([r["diag"]["brier_score"] for r in results_per_seed if r["diag"]["brier_score"] is not None]),
        "human_review_rate": _agg([r["diag"]["human_review_rate"] for r in results_per_seed if r["diag"]["human_review_rate"] is not None]),
    }
    summary["surrogate_validation"] = surrogate_report
    summary["config"] = {
        "n_cal": n_cal, "n_test": n_test,
        "hidden": list(hide), "seeds": list(seeds_test),
        "classes": classes, "pv_model": fb._pv_model._backend,
        "calibration": "IsotonicRegression (one-vs-rest)",
        "policy": "ExpectedLoss",
    }
    summary["per_seed"] = results_per_seed

    n_ev_cal = {}
    for _, r in reg_cal.iterrows():
        n_ev_cal[r["class"]] = n_ev_cal.get(r["class"], 0) + 1

    sigs_campus = export_v2(dg, node_id="CAMPUS_NODE", n_events=n_ev_cal, epsilon=1.0, validated=True)
    factory_sensors = FactoryNode.AVAILABLE_SENSORS

    dg_factory = copy.deepcopy(dg)
    transfer_report = import_v2(dg_factory, sigs_campus, available_sensors=factory_sensors)
    summary["goal4_transfer"] = {
        "n_signatures_exported": len(sigs_campus),
        "n_imported": sum(1 for _, s, _ in transfer_report if s == "imported"),
        "n_rejected": sum(1 for _, s, _ in transfer_report if "rejected" in s),
        "details": [(c, s, r) for c, s, r in transfer_report],
        "factory_available_sensors": factory_sensors,
    }

    if out:
        out_path = ROOT / out
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "metrics_v2.json").write_text(json.dumps(summary, indent=2, default=str))
        rel_data = reliability_data(Pi, o_iso.label.values, classes)
        rel_data.to_csv(out_path / "reliability_data_v2.csv", index=False)
        reg_cal.to_csv(out_path / "calibration_event_registry_v2.csv", index=False)

    return summary
