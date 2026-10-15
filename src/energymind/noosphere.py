"""Noosphere : partage inter-noeuds de CONNAISSANCES VALIDEES (signatures normalisees), jamais de series brutes."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict
from .faults import CLASSES
from .engine import FEATURES

SCHEMA_VERSION = "1.0"
SHARED = ["signature normalisee (moyenne/ecart des residus en unites sigma)", "capteurs requis", "conditions de validite", "limites"]
LOCAL = ["series temporelles brutes", "profils de charge", "topologie exacte", "identite des validateurs", "factures"]

@dataclass
class IncidentSignature:
    schema_version: str
    source_node_type: str
    incident_class: str
    mu: dict
    sd: dict
    n_events: int
    required_sensors: list
    validity: dict
    limits: str
    validated_by_human: bool

def export_signatures(dg, node_type="campus_microgrid", n_events=None, epsilon=1.0, min_events=5, rng=None, validated=True):
    """Export avec bruit de Laplace (confidentialite) et seuil k-anonymat : une classe rare n'est pas partagee."""
    rng = rng or np.random.default_rng(0)
    out = []
    for i, c in enumerate(CLASSES[1:], start=1):
        n = (n_events or {}).get(c, 0)
        if n < min_events:
            continue
        noise = lambda a: a + rng.laplace(0, 1.0 / (epsilon * np.sqrt(n)), size=a.shape)
        mu, sd = noise(dg.mu[i]), np.maximum(noise(dg.sd[i]), 1.0)
        req = [f for f, m in zip(FEATURES, dg.mu[i]) if abs(m) > 1.5]
        out.append(IncidentSignature(SCHEMA_VERSION, node_type, c, dict(zip(FEATURES, map(float, np.round(mu, 2)))),
                                     dict(zip(FEATURES, map(float, np.round(sd, 2)))), n, req,
                                     {"irradiance_min_w_m2": 100 if "pvexp" in req else 0},
                                     "signature valide pour residus normalises localement; recalibrer sigma sur le noeud destinataire", validated).__dict__)
    return out

def import_signatures(dg, sigs, available_sensors, weight=0.3):
    """Import prudent : rejet si schema/capteurs incompatibles, sinon melange pondere avec la connaissance locale."""
    report = []
    for s in sigs:
        if s["schema_version"] != SCHEMA_VERSION or not s["validated_by_human"]:
            report.append((s["incident_class"], "rejete", "schema ou validation humaine manquante")); continue
        missing = [f for f in s["required_sensors"] if f not in available_sensors]
        if missing:
            report.append((s["incident_class"], "rejete", f"capteurs absents localement: {missing}")); continue
        i = CLASSES.index(s["incident_class"])
        for j, f in enumerate(FEATURES):
            dg.mu[i, j] = (1 - weight) * dg.mu[i, j] + weight * s["mu"][f]
            dg.sd[i, j] = (1 - weight) * dg.sd[i, j] + weight * s["sd"][f]
        report.append((s["incident_class"], "importe", f"poids {weight}"))
    return report
