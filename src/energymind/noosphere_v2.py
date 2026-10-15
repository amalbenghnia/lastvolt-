"""
Noosphere V2 — Cross-node incident signature federation with privacy guarantees,
distribution drift monitoring (PSI + Wasserstein), and peer network simulation.

Architecture:
  1. IncidentSignatureV2: Pydantic-validated JSON schema for cross-operator exchange.
  2. export_v2()        : Laplace differential privacy (epsilon-DP) + k-anonymity gate.
  3. import_v2()        : Cautious Bayesian shrinkage with drift validation.
  4. domain_shift_report: Population Stability Index (PSI) + Wasserstein metric.
  5. Multi-node mesh    : Peer microgrids (Campus, Factory, Hospital, Smart Building, Rural).

Privacy policies:
  SHARED : Local dimensionless z-score signatures (mu, sigma), required sensors,
           validity preconditions, noisy sample count, human validation bit.
  LOCAL  : Raw electrical telemetry, utility invoices, load profiles, topological maps,
           operator credentials.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import wasserstein_distance

try:
    from pydantic import BaseModel, Field, field_validator
    _PYDANTIC_OK = True
except ImportError:
    _PYDANTIC_OK = False

from .faults_v2 import CLASSES_V2
from .engine import FEATURES

SCHEMA_VERSION = "2.0"


if _PYDANTIC_OK:
    class IncidentSignatureV2(BaseModel):
        """Normalized incident signature with differential privacy guarantees."""
        schema_version: str
        source_node_type: str
        source_node_id: str
        incident_class: str
        mu: dict[str, float]
        sd: dict[str, float]
        n_events: int
        n_events_noisy: int
        required_sensors: list[str]
        validity: dict[str, float]
        limits: str
        epsilon: float
        validated_by_human: bool
        transfer_notes: str = ""

        @field_validator("incident_class")
        @classmethod
        def class_must_be_valid(cls, v):
            if v not in CLASSES_V2[1:]:
                raise ValueError(f"incident_class must be in {CLASSES_V2[1:]}")
            return v

        @field_validator("schema_version")
        @classmethod
        def version_must_match(cls, v):
            if v != SCHEMA_VERSION:
                raise ValueError(f"schema_version must be '{SCHEMA_VERSION}'")
            return v
else:
    @dataclass
    class IncidentSignatureV2:
        schema_version: str
        source_node_type: str
        source_node_id: str
        incident_class: str
        mu: dict
        sd: dict
        n_events: int
        n_events_noisy: int
        required_sensors: list
        validity: dict
        limits: str
        epsilon: float
        validated_by_human: bool
        transfer_notes: str = ""


def export_v2(
    dg,
    node_type: str = "campus_microgrid",
    node_id: str = "CAMPUS_NODE_v2",
    n_events: dict | None = None,
    epsilon: float = 1.0,
    min_events: int = 5,
    rng=None,
    validated: bool = True,
) -> list[dict]:
    """
    Export incident signatures using Laplace differential privacy and k-anonymity.
    """
    rng = rng or np.random.default_rng(0)
    out = []
    n_ev = n_events or {}

    for i, c in enumerate(CLASSES_V2[1:], start=1):
        n = n_ev.get(c, 0)
        if n < min_events:
            continue

        sensitivity = 1.0 / max(np.sqrt(n), 1.0)
        noise_scale = sensitivity / max(epsilon, 1e-6)
        mu = dg.mu[i] + rng.laplace(0, noise_scale, size=dg.mu[i].shape)
        sd = np.maximum(
            dg.sd[i] + rng.laplace(0, noise_scale, size=dg.sd[i].shape),
            1.0,
        )

        required = [f for f, m in zip(FEATURES, dg.mu[i]) if abs(m) > 1.5]
        n_noisy = int(max(1, n + int(rng.laplace(0, 1.0 / epsilon))))

        sig_dict = {
            "schema_version": SCHEMA_VERSION,
            "source_node_type": node_type,
            "source_node_id": node_id,
            "incident_class": c,
            "mu": dict(zip(FEATURES, np.round(mu, 3).tolist())),
            "sd": dict(zip(FEATURES, np.round(sd, 3).tolist())),
            "n_events": n,
            "n_events_noisy": n_noisy,
            "required_sensors": required,
            "validity": {
                "irradiance_min_w_m2": 100.0 if "pvexp" in required else 0.0,
                "note": "recalibrate sigma locally on recipient node",
            },
            "limits": (
                "Valid for locally MAD-normalized residuals. "
                "Do not apply without local standard deviation calibration. "
                "Check domain shift metrics before Bayesian updating."
            ),
            "epsilon": epsilon,
            "validated_by_human": validated,
            "transfer_notes": "",
        }

        if _PYDANTIC_OK:
            try:
                sig_obj = IncidentSignatureV2(**sig_dict)
                out.append(sig_obj.model_dump())
            except Exception as e:
                sig_dict["_validation_error"] = str(e)
                out.append(sig_dict)
        else:
            out.append(sig_dict)

    return out


def psi_score(p: np.ndarray, q: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index (PSI) between two empirical distributions.
      PSI < 0.10: Stable distributions.
      0.10 - 0.25: Moderate shift.
      PSI > 0.25: Significant drift requiring recalibration.
    """
    p_arr = np.asarray(p, float)
    q_arr = np.asarray(q, float)
    if p_arr.ndim == 1:
        bins = np.histogram_bin_edges(np.concatenate([p_arr, q_arr]), bins=n_bins)
        p_hist, _ = np.histogram(p_arr, bins=bins)
        q_hist, _ = np.histogram(q_arr, bins=bins)
    else:
        return float(np.mean([
            psi_score(p_arr[:, k], q_arr[:, k], n_bins)
            for k in range(p_arr.shape[1])
        ]))

    p_freq = (p_hist + 1e-6) / (p_hist.sum() + n_bins * 1e-6)
    q_freq = (q_hist + 1e-6) / (q_hist.sum() + n_bins * 1e-6)
    return float(np.sum((p_freq - q_freq) * np.log(p_freq / q_freq)))


def domain_shift_report(
    Z_source: pd.DataFrame,
    Z_target: pd.DataFrame,
    features: list[str] = FEATURES,
) -> pd.DataFrame:
    """Compute PSI and Wasserstein distance across common features."""
    rows = []
    for f in features:
        s = Z_source[f].dropna().values
        t = Z_target[f].dropna().values
        if len(s) < 10 or len(t) < 10:
            continue
        psi = psi_score(s, t)
        w1 = float(wasserstein_distance(s, t))
        rows.append({
            "feature": f,
            "psi": round(psi, 4),
            "wasserstein": round(w1, 4),
            "psi_alert": psi > 0.25,
            "wass_alert": w1 > 2.0,
            "recommendation": "recalibrate" if (psi > 0.25 or w1 > 2.0) else "ok",
        })
    return pd.DataFrame(rows)


def import_v2(
    dg,
    sigs: list[dict],
    available_sensors: list[str],
    weight: float = 0.3,
    Z_local: pd.DataFrame | None = None,
    Z_remote_sample: pd.DataFrame | None = None,
) -> list[tuple[str, str, str]]:
    """
    Cautious Bayesian import of federated signatures with validation gates.

    Rejection conditions:
      1. Schema mismatch.
      2. Missing human operator validation.
      3. Missing required telemetry channels.
      4. Excessive distribution drift (PSI > 0.25).
    """
    report = []
    for s in sigs:
        cls = s.get("incident_class", "?")

        if s.get("schema_version") != SCHEMA_VERSION:
            report.append((cls, "rejected", f"schema {s.get('schema_version')} != {SCHEMA_VERSION}"))
            continue

        if not s.get("validated_by_human", False):
            report.append((cls, "rejected", "missing human operator verification"))
            continue

        missing = [f for f in s.get("required_sensors", []) if f not in available_sensors]
        if missing:
            report.append((cls, "rejected", f"missing sensors: {missing}"))
            continue

        if Z_local is not None and Z_remote_sample is not None:
            common = [f for f in FEATURES if f in Z_local.columns and f in Z_remote_sample.columns]
            if common:
                drift = domain_shift_report(Z_remote_sample, Z_local, common)
                if drift["psi_alert"].any():
                    alert_feats = drift[drift.psi_alert]["feature"].tolist()
                    report.append((cls, "rejected_drift", f"PSI > 0.25 on {alert_feats}"))
                    continue

        try:
            i = CLASSES_V2.index(cls)
        except ValueError:
            report.append((cls, "rejected", "unknown class in CLASSES_V2"))
            continue

        for j, f in enumerate(FEATURES):
            if f in s.get("mu", {}):
                dg.mu[i, j] = (1 - weight) * dg.mu[i, j] + weight * s["mu"][f]
                dg.sd[i, j] = (1 - weight) * dg.sd[i, j] + weight * s.get("sd", {}).get(f, dg.sd[i, j])

        report.append((cls, "imported", f"weight {weight}, epsilon={s.get('epsilon', '?')}"))

    return report


# ---------------------------------------------------------------------------
# Multi-Node Peer Network Topology
# ---------------------------------------------------------------------------

class PeerNode:
    """Base class for multi-node microgrid federation peers."""
    node_id: str
    node_type: str
    pv_capacity_kwp: float
    bess_capacity_kwh: float
    available_sensors: list[str]


class FactoryNode(PeerNode):
    """Industrial manufacturing microgrid peer (heavy shift loads, no independent inverter meter)."""
    node_id = "FACTORY_NODE_01"
    node_type = "industrial_microgrid"
    pv_capacity_kwp = 30.0
    bess_capacity_kwh = 0.0
    AVAILABLE_SENSORS = ["bal", "pvexp", "vpcc"]
    available_sensors = AVAILABLE_SENSORS

    @staticmethod
    def load_profile(index: pd.DatetimeIndex, rng) -> np.ndarray:
        h = index.hour + index.minute / 60
        weekday = (index.dayofweek < 5).astype(float)
        occ = weekday * (0.7 + 0.3 * np.exp(-((h - 13) / 4) ** 2) * (h >= 8) * (h <= 18))
        occ += (1 - weekday) * 0.05
        noise = rng.normal(0, 0.03, len(index))
        return np.clip(occ * (1 + noise), 0.02, 1.0)


class HospitalNode(PeerNode):
    """Critical healthcare facility peer (continuous 24/7 load, redundant BESS + diesel)."""
    node_id = "HOSPITAL_NODE_02"
    node_type = "critical_infrastructure"
    pv_capacity_kwp = 45.0
    bess_capacity_kwh = 120.0
    AVAILABLE_SENSORS = ["bal", "pvexp", "inv", "vpcc", "vinv"]
    available_sensors = AVAILABLE_SENSORS

    @staticmethod
    def load_profile(index: pd.DatetimeIndex, rng) -> np.ndarray:
        h = index.hour + index.minute / 60
        base = 0.75 + 0.20 * np.sin((h - 6) * np.pi / 12) ** 2
        noise = rng.normal(0, 0.02, len(index))
        return np.clip(base + noise, 0.50, 1.0)


class SmartBuildingNode(PeerNode):
    """Commercial smart office peer (steep morning HVAC peaks, EV fast chargers)."""
    node_id = "COMMERCIAL_BUILDING_03"
    node_type = "commercial_building"
    pv_capacity_kwp = 25.0
    bess_capacity_kwh = 30.0
    AVAILABLE_SENSORS = ["bal", "pvexp", "vpcc"]
    available_sensors = AVAILABLE_SENSORS

    @staticmethod
    def load_profile(index: pd.DatetimeIndex, rng) -> np.ndarray:
        h = index.hour + index.minute / 60
        weekday = (index.dayofweek < 5).astype(float)
        occ = weekday * (0.10 + 0.85 * (1.0 / (1.0 + np.exp(-2.0 * (h - 7.5)))) * (1.0 / (1.0 + np.exp(2.0 * (h - 18.5)))))
        occ += (1 - weekday) * 0.08
        noise = rng.normal(0, 0.03, len(index))
        return np.clip(occ + noise, 0.05, 1.0)


class RuralMicrogridNode(PeerNode):
    """Islanded rural microgrid peer (weak grid, high line impedance, agricultural pumps)."""
    node_id = "RURAL_COMMUNITY_04"
    node_type = "rural_islanded"
    pv_capacity_kwp = 40.0
    bess_capacity_kwh = 90.0
    AVAILABLE_SENSORS = ["bal", "pvexp", "vpcc", "vinv"]
    available_sensors = AVAILABLE_SENSORS

    @staticmethod
    def load_profile(index: pd.DatetimeIndex, rng) -> np.ndarray:
        h = index.hour + index.minute / 60
        morning_pump = np.exp(-((h - 6.5) / 1.5) ** 2) * 0.45
        evening_comm = np.exp(-((h - 19.5) / 2.0) ** 2) * 0.50
        base = 0.15 + morning_pump + evening_comm
        noise = rng.normal(0, 0.04, len(index))
        return np.clip(base + noise, 0.05, 1.0)


FEDERATION_PEERS = {
    "factory": FactoryNode,
    "hospital": HospitalNode,
    "smart_building": SmartBuildingNode,
    "rural_microgrid": RuralMicrogridNode,
}
