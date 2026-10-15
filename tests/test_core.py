import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd, pytest
from energymind.data import load_invoices
from energymind.simulate import build_baseline, energy_balance_error
from energymind.governance import route, AuditLog
from energymind.faults import CLASSES, inject, observe
from energymind.engine import FeatureBuilder, Diagnoser, persistent
from energymind.drift import CUSUMDetector, cusum_scan, calibrate_cusum
from energymind.network import run_powerflow
from energymind.noosphere import export_signatures, import_signatures
from energymind.config import load_config


@pytest.fixture(scope="module")
def base(): return build_baseline()

@pytest.fixture(scope="module")
def fitted_engine(base):
    """Entraîne FeatureBuilder + Diagnoser une seule fois (coûteux)."""
    cfg = load_config()
    P = cfg["periods"]
    tr = base.loc[P["train_pv_model"][0]:P["train_pv_model"][1]]
    o_tr = observe(tr, cfg, np.random.default_rng(1))
    fb = FeatureBuilder().fit_pv_model(o_tr).fit_scales(o_tr)
    o_cal, _ = inject(base.loc[P["calibrate_diagnosis"][0]:P["calibrate_diagnosis"][1]],
                      12, cfg["seed_calibration"], cfg)
    Zc = fb.z(o_cal, hidden=())
    dg = Diagnoser().fit(Zc, o_cal.label)
    return fb, dg


# --- Tests existants ---

def test_energy_conservation(base):
    assert energy_balance_error(base) < 1e-9

def test_soc_limits(base):
    assert base.soc.between(0.10 - 1e-9, 0.90 + 1e-9).all()

def test_monthly_energy_respected(base):
    inv, _ = load_invoices()
    m = pd.Period("2024-01", "M")
    kwh = base.load_kw[base.index.to_period("M") == m].sum() * 0.25
    assert abs(kwh - inv.set_index("period").Consumption_kWh[m]) < 1

def test_invoice_sequence_fix():
    inv, log = load_invoices()
    assert inv.period.is_monotonic_increasing and "month_relabeled" in set(log.issue)

def test_policy_never_auto_for_physical_or_security():
    for c in ["pv_fault", "grid_event", "coordinated_falsification"]:
        p = np.full(len(CLASSES), 0.0); p[CLASSES.index(c)] = 0.99
        assert route(p)["human_required"] and route(p)["state"] != "AUTO_QUARANTINE"

def test_abstain_when_ambiguous():
    p = np.zeros(len(CLASSES))
    p[CLASSES.index("pv_fault")] = 0.5
    p[CLASSES.index("coordinated_falsification")] = 0.5
    assert route(p)["action"] == "abstain"

def test_audit_chain_tamper_detection(tmp_path):
    a = AuditLog(str(tmp_path / "a.db"))
    for i in range(3): a.append(f"e{i}", {}, {}, {}, "approved", "op1", "ok")
    assert a.verify()
    a.db.execute("UPDATE audit SET decision='rejected' WHERE id=2"); a.db.commit()
    assert not a.verify()


# --- Tests : LightGBM + FeatureBuilder ---

def test_lightgbm_pv_model_fitted(fitted_engine):
    """LightGBM est bien entraîné."""
    fb, _ = fitted_engine
    assert fb._pv_model._model is not None, "Modèle PV non initialisé"

def test_feature_builder_z_shape(base, fitted_engine):
    """Les résidus normalisés ont la bonne forme et les colonnes attendues."""
    fb, _ = fitted_engine
    cfg = load_config()
    P = cfg["periods"]
    tr = base.loc[P["train_pv_model"][0]:P["train_pv_model"][1]]
    o_tr = observe(tr, cfg, np.random.default_rng(1))
    Z = fb.z(o_tr)
    assert set(["bal", "pvexp", "inv", "vpcc", "vinv"]).issubset(Z.columns)
    assert len(Z) == len(o_tr)

def test_hidden_channel_is_nan(base, fitted_engine):
    """Un canal masqué doit être NaN dans les résidus."""
    fb, _ = fitted_engine
    cfg = load_config()
    P = cfg["periods"]
    tr = base.loc[P["train_pv_model"][0]:P["train_pv_model"][1]]
    o_tr = observe(tr, cfg, np.random.default_rng(1))
    Z = fb.z(o_tr, hidden=["inv"])
    assert Z["inv"].isna().all(), "Canal masqué devrait être NaN"

def test_posterior_sums_to_one(fitted_engine, base):
    """Le posterieur doit sommer à 1 pour chaque pas de temps."""
    fb, dg = fitted_engine
    cfg = load_config()
    P = cfg["periods"]
    tr = base.loc[P["train_pv_model"][0]:P["train_pv_model"][1]]
    o_tr = observe(tr, cfg, np.random.default_rng(1))
    Z = fb.z(o_tr)
    post = dg.posterior(Z)
    np.testing.assert_allclose(post.sum(axis=1), 1.0, atol=1e-6)


# --- Tests : CUSUM ---

def test_cusum_detects_step_change():
    """CUSUM doit détecter un saut de moyenne après suffisamment de pas."""
    rng = np.random.default_rng(0)
    detector = CUSUMDetector(k=0.5, h=5.0)
    for _ in range(20):
        detector.update(float(rng.normal(0, 0.5)))
    assert not detector.alarm, "Fausse alarme en phase normale"
    for _ in range(30):
        detector.update(2.0)
    assert detector.alarm, "CUSUM n'a pas détecté le saut"

def test_cusum_reset():
    """Après reset(), les statistiques et l'alarme sont remis à zéro."""
    d = CUSUMDetector(k=0.5, h=5.0)
    for _ in range(30):
        d.update(3.0)
    assert d.alarm
    d.reset()
    assert not d.alarm
    assert d.S_pos == 0.0 and d.S_neg == 0.0

def test_cusum_scan_dataframe():
    """cusum_scan retourne un DataFrame booléen avec les mêmes colonnes."""
    idx = pd.date_range("2024-01-01", periods=100, freq="15min")
    Z = pd.DataFrame({"pvexp": np.zeros(100), "bal": np.zeros(100)}, index=idx)
    result = cusum_scan(Z, ["pvexp", "bal"], k=0.5, h=5.0)
    assert result.shape == (100, 2)
    assert list(result.columns) == ["pvexp", "bal"]

def test_cusum_calibrate_returns_params():
    """calibrate_cusum retourne k, h et achieved_fpr."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=500, freq="15min")
    Z = pd.DataFrame({
        "pvexp": rng.normal(0, 1, 500),
        "bal": rng.normal(0, 1, 500),
    }, index=idx)
    params = calibrate_cusum(Z, ["pvexp", "bal"], target_fpr=0.05)
    assert "k" in params and "h" in params and "achieved_fpr" in params
    assert params["h"] > 0


# --- Tests : pandapower ---

def test_powerflow_returns_dict():
    """run_powerflow retourne un dict avec les clés attendues."""
    res = run_powerflow(pv_kw=20.0, load_kw=40.0, bess_kw=5.0)
    assert "v_pcc_V" in res
    assert "grid_kw" in res
    assert "r_balance_kw" in res

def test_powerflow_balance_residual_small():
    """Le résidu du bilan de puissance doit être dans les tolérances."""
    res = run_powerflow(pv_kw=30.0, load_kw=45.0, bess_kw=0.0)
    assert abs(res["r_balance_kw"]) < 5.0, f"Résidu bilan : {res['r_balance_kw']}"

def test_powerflow_voltage_reasonable():
    """La tension au PCC doit rester dans ±15% de 400 V."""
    res = run_powerflow(pv_kw=50.0, load_kw=60.0, bess_kw=10.0)
    assert 340 <= res["v_pcc_V"] <= 460, f"Tension PCC : {res['v_pcc_V']} V"


# --- Tests : Noosphère ---

def test_export_signatures_format(fitted_engine):
    """Les signatures exportées ont les champs requis."""
    _, dg = fitted_engine
    n_ev = {c: 10 for c in CLASSES[1:]}
    sigs = export_signatures(dg, n_events=n_ev, validated=True)
    assert len(sigs) > 0
    for s in sigs:
        assert "mu" in s and "sd" in s and "incident_class" in s
        assert s["validated_by_human"] is True

def test_import_rejects_wrong_schema(fitted_engine):
    """Un paquet avec un schéma incorrect doit être rejeté."""
    import copy
    _, dg = fitted_engine
    dg2 = copy.deepcopy(dg)
    bad_sig = {"schema_version": "99.0", "incident_class": "pv_fault",
               "validated_by_human": True, "mu": {}, "sd": {}, "required_sensors": []}
    report = import_signatures(dg2, [bad_sig], available_sensors=["bal", "pvexp"])
    assert any(status == "rejete" for _, status, _ in report)

def test_import_rejects_missing_sensors(fitted_engine):
    """Un paquet nécessitant des capteurs absents doit être rejeté."""
    import copy
    _, dg = fitted_engine
    dg2 = copy.deepcopy(dg)
    n_ev = {c: 10 for c in CLASSES[1:]}
    sigs = export_signatures(dg, n_events=n_ev, validated=True)
    report = import_signatures(dg2, sigs, available_sensors=[])
    for _, status, _ in report:
        assert status == "rejete"
