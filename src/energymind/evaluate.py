"""
Protocole expérimental complet (chronologique, graines fixées) :
  Baselines A / B / C vs EnergyMind.

Nouveautés Phase 1 v2 :
  - Modèle PV : LightGBM (avec fallback HistGBM documenté)
  - CUSUM inclus dans les métriques de détection de dérive
  - Résidus pandapower calculés sur un échantillon (performance)
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from .config import load_config, ROOT
from .simulate import build_baseline, energy_balance_error
from .faults import inject, observe, CLASSES
from .engine import FeatureBuilder, Diagnoser, persistent, FEATURES
from .drift import cusum_scan, calibrate_cusum


# ---------------------------------------------------------------------------
# Métriques utilitaires
# ---------------------------------------------------------------------------

def prf(y: np.ndarray, yhat: np.ndarray) -> dict:
    tp = int((y & yhat).sum())
    fp = int((~y & yhat).sum())
    fn = int((y & ~yhat).sum())
    tn = int((~y & ~yhat).sum())
    return {
        "precision": tp / max(tp + fp, 1),
        "recall":    tp / max(tp + fn, 1),
        "fpr":       fp / max(fp + tn, 1),
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


def ece(conf: list, correct: list, bins: int = 5) -> float:
    conf, correct = np.asarray(conf), np.asarray(correct)
    e, edges = 0.0, np.linspace(0.5, 1.0, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf >= a) & (conf < b + (1e-9 if b == 1.0 else 0))
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


# ---------------------------------------------------------------------------
# Pipeline expérimental principal
# ---------------------------------------------------------------------------

def run(n_cal: int = 48, n_test: int = 72, hide: tuple = ("inv",),
        thr_decide: float = 0.90, out=None):
    """
    Lance l'expérience complète et retourne (res, ev, (base, o_te, Z, post, flagE, dg, fb)).

    Paramètres
    ----------
    n_cal      : nombre d'événements injectés dans la période de calibration
    n_test     : nombre d'événements injectés dans la période de test
    hide       : canaux de télémétrie masqués (simulation mesure indisponible)
    thr_decide : seuil de confiance pour décider (sinon abstention)
    out        : répertoire de sortie (ou None pour ne pas écrire)
    """
    cfg = load_config()
    base = build_baseline(cfg)
    P = cfg["periods"]
    sl = lambda k: base.loc[P[k][0]:P[k][1]]

    # 1) Entraînement du modèle PV (LightGBM) sur période saine
    tr = sl("train_pv_model")
    o_tr = observe(tr, cfg, np.random.default_rng(1))
    fb = FeatureBuilder().fit_pv_model(o_tr).fit_scales(o_tr)

    # 2) Calibration du diagnostic sur défauts injectés (graine séparée)
    o_cal, reg_cal = inject(sl("calibrate_diagnosis"), n_cal, cfg["seed_calibration"], cfg)
    Zc = fb.z(o_cal, hidden=())
    dg = Diagnoser().fit(Zc, o_cal.label)

    # 3) Test sur période séparée avec autre graine
    o_te, reg_te = inject(sl("test"), n_test, cfg["seed_test"], cfg)
    Z = fb.z(o_te, hidden=hide)
    Zfull = fb.z(o_te, hidden=())
    y_anom = (o_te.label != "normal").values

    res = {
        "config": {
            "n_cal_events": n_cal,
            "n_test_events": n_test,
            "hidden": list(hide),
            "test_steps": len(o_te),
            "pv_model": "LightGBM" if fb._pv_model._backend == "lgb" else "HistGBM (fallback)",
        },
        "simulator_energy_balance_max_error_kw": energy_balance_error(base),
    }

    # --- Baselines ---
    raw_cols = ["pv_meas", "load_meas", "grid_meas", "bess_meas", "v_pcc", "v_inv", "irradiance"]
    # A — IsolationForest sur télémétrie brute
    iso = IsolationForest(n_estimators=200, contamination=0.005, random_state=0).fit(o_tr[raw_cols])
    flagA = persistent(iso.predict(o_te[raw_cols]) == -1)
    # B — Contraintes physiques seules
    flagB = persistent((Z.bal.abs() > 4).values | ((Z.vpcc - Z.vinv).abs() > 4).values)
    # C — Fusion ML (IsolationForest sur résidus + RandomForest supervisé)
    Z_tr = fb.z(o_tr, hidden=hide).fillna(0)
    isoC = IsolationForest(n_estimators=200, contamination=0.005, random_state=0).fit(Z_tr)
    flagC = persistent(isoC.predict(Z.fillna(0)) == -1)
    rf = RandomForestClassifier(n_estimators=200, random_state=0, class_weight="balanced").fit(
        fb.z(o_cal, hidden=hide).fillna(0), o_cal.label
    )
    # EnergyMind
    post = dg.posterior(Z)
    flagE = persistent(post[:, 0] < 0.2)

    # --- Détection CUSUM (dérive progressive — scénario M2) ---
    Z_normal_tr = fb.z(o_tr, hidden=hide)
    cusum_params = calibrate_cusum(Z_normal_tr, ["pvexp", "bal"], target_fpr=0.01)
    cusum_alarms = cusum_scan(Z, ["pvexp", "bal"],
                              k=cusum_params["k"], h=cusum_params["h"])
    flagCUSUM = persistent(cusum_alarms.any(axis=1).values)

    methods = {
        "A_isolation_forest_brut": flagA,
        "B_physique_seule": flagB,
        "C_fusion_ML_isoforest": flagC,
        "EnergyMind": flagE,
        "CUSUM_drift": flagCUSUM,
    }

    res["detection"] = {}
    for k_m, f in methods.items():
        d = prf(y_anom, f)
        es = event_stats(o_te, f, reg_te)
        d["event_recall"] = float(es.detected.mean())
        d["median_latency_steps"] = (
            float(es.latency_steps.dropna().median()) if es.detected.any() else None
        )
        d["event_recall_by_class"] = (
            es.groupby("class").detected.mean().round(3).to_dict()
        )
        res["detection"][k_m] = d

    res["cusum_calibration"] = cusum_params

    # --- Diagnostic au niveau événement ---
    rows, ece_c, ece_ok = [], [], []
    names = np.array(CLASSES)
    postF = dg.posterior(Zfull)
    rfp = rf.predict(Z.fillna(0))

    for _, r in reg_te.iterrows():
        s = o_te.index.get_loc(pd.Timestamp(r.start))
        idx = np.arange(s, s + r.steps)
        fl = idx[flagE[idx]]
        if len(fl) == 0:
            rows.append({"truth": r["class"], "scenario": r.scenario, "detected": False})
            continue
        p = post[fl].mean(axis=0)
        top = int(p.argmax())
        decided = p[top] >= thr_decide
        row = {
            "truth": r["class"],
            "scenario": r.scenario,
            "detected": True,
            "pred": names[top],
            "conf": float(p[top]),
            "decided": bool(decided),
            "rf_pred": pd.Series(rfp[fl]).mode()[0],
        }
        amb = [names[i] for i in np.argsort(-p)[:2] if p[i] > 0.1]
        row["ambiguity_set"] = "|".join(amb)
        if not decided:
            z_row = Z.iloc[fl[-1]]
            nb = dg.next_best_check(z_row, p, hidden=list(hide))
            row["next_check"] = nb[0][0] if nb else None
            p2 = postF[fl].mean(axis=0)
            t2 = int(p2.argmax())
            row["pred_after_check"] = names[t2]
            row["conf_after_check"] = float(p2[t2])
        rows.append(row)

    ev = pd.DataFrame(rows)
    det = ev[ev.detected.fillna(False).astype(bool)].copy()
    det["decided"] = det["decided"].infer_objects(copy=False).fillna(False).astype(bool)
    dec = det[det.decided]

    diag = {
        "events_detected": int(len(det)),
        "abstention_rate": float(1 - len(dec) / max(len(det), 1)),
        "accuracy_all_forced": float((det.pred == det.truth).mean()) if len(det) else None,
        "accuracy_when_decided": float((dec.pred == dec.truth).mean()) if len(dec) else None,
        "rf_blackbox_accuracy_forced": float((det.rf_pred == det.truth).mean()) if len(det) else None,
        "ece_decided": ece(dec.conf.tolist(), (dec.pred == dec.truth).tolist()) if len(dec) else None,
    }
    ab = det[~det.decided]
    if len(ab):
        diag["abstained_truth_distribution"] = ab.truth.value_counts().to_dict()
        diag["next_check_chosen"] = ab.next_check.value_counts().to_dict()
        diag["accuracy_after_active_check"] = float((ab.pred_after_check == ab.truth).mean())
        diag["accuracy_abstained_before_check_forced"] = float((ab.pred == ab.truth).mean())

    res["diagnosis"] = diag

    if len(det) > 0:
        res["confusion"] = pd.crosstab(det.truth, det.pred).to_dict()

    # Taux de fausses alarmes (épisodes/jour)
    fa = flagE & ~y_anom
    ep = int(((fa.astype(int) - np.r_[0, fa[:-1].astype(int)]) == 1).sum())
    res["false_alarm_episodes_per_day"] = ep / (len(o_te) / 96)

    if out:
        out_path = ROOT / out if not str(out).startswith("/") else __import__("pathlib").Path(out)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "metrics.json").write_text(json.dumps(res, indent=2, default=str))
        ev.to_csv(out_path / "event_level_results.csv", index=False)
        reg_te.to_csv(out_path / "test_event_registry.csv", index=False)
        reg_cal.to_csv(out_path / "calibration_event_registry.csv", index=False)

    return res, ev, (base, o_te, Z, post, flagE, dg, fb)
