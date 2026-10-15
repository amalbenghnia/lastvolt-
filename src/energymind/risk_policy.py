"""
Policy Engine V2 — Expected-loss decision routing.

Decision logic:
  The routing action depends not only on the maximum posterior probability
  but also on the severity and cost of each possible error. A low-probability
  but high-consequence hypothesis (e.g. coordinated falsification) can trigger
  escalation even when ambiguity is high.

  Example: P(falsification) = 0.45 < threshold 0.90 → V1 would abstain.
  With the loss matrix: L(action=MONITOR | falsification) = 100 >> threshold.
  V2 escalates despite ambiguity.

Loss matrix L[a, h]: cost when action a is taken and hypothesis h is true.

Available actions:
  MONITOR     — log only, no operational action
  QUARANTINE  — isolate the measurement channel (data action only)
  RECOMMEND   — send operator recommendation without automatic action
  ESCALATE    — page human operator + security team
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .faults_v2 import CLASSES_V2

ACTIONS = ["MONITOR", "QUARANTINE", "RECOMMEND", "ESCALATE"]

LOSS_MATRIX = np.array([
    #  normal  pv_f  grid  load  pv_sen  v_sen  bess_s  falsif
    [     0,    20,   15,    5,     10,      8,      8,    100],   # MONITOR
    [     3,    15,   20,   15,      2,      2,      2,     80],   # QUARANTINE
    [     2,     5,    5,    3,      5,      5,      5,     20],   # RECOMMEND
    [     5,     2,    2,    2,      8,      8,      8,      1],   # ESCALATE
], dtype=float)

POLICY_V2 = {
    "auto_threshold":    0.95,
    "auto_classes":      ["pv_sensor_fault", "voltage_sensor_fault", "bess_sensor_fault"],
    "human_always":      ["coordinated_falsification", "pv_fault", "grid_event"],
    "audit_sampling":    0.05,
    "abstain_threshold": 0.90,
}

MODEL_VERSION_V2 = "energymind-v2.0.0"


def expected_loss(p: np.ndarray) -> np.ndarray:
    """
    Compute expected loss for each action.

    EL[a] = Σ_h P(h) × L[a, h]

    Parameters
    ----------
    p : (K,) posterior probability vector over K classes

    Returns
    -------
    (4,) expected loss for each action [MONITOR, QUARANTINE, RECOMMEND, ESCALATE]
    """
    return LOSS_MATRIX @ p


def route_v2(p: np.ndarray, rng=None, policy: dict = POLICY_V2) -> dict:
    """
    V2 state machine: expected-loss decision routing.

    Algorithm:
    1. Compute expected loss of each action.
    2. Select the minimum-loss action.
    3. Apply governance constraints (human-always classes).
    4. Apply audit sampling (anti-overconfidence).

    Parameters
    ----------
    p : (K,) calibrated posterior probability vector

    Returns
    -------
    dict with state, action, human_required, reason, expected_loss, optimal_action
    """
    rng = rng or np.random.default_rng()
    top_idx = int(p.argmax())
    top_cls = CLASSES_V2[top_idx]
    conf = float(p[top_idx])

    el = expected_loss(p)
    opt_idx = int(el.argmin())
    opt_act = ACTIONS[opt_idx]
    el_dict = dict(zip(ACTIONS, el.round(2).tolist()))

    if top_cls == "normal" and conf >= 0.85:
        return {
            "state": "MONITOR",
            "action": "log",
            "human_required": False,
            "reason": "no anomaly detected",
            "expected_loss": el_dict,
            "optimal_action": opt_act,
        }

    if conf < policy["abstain_threshold"]:
        return {
            "state": "HUMAN_REVIEW",
            "action": "abstain",
            "human_required": True,
            "reason": (
                f"confidence {conf:.2f} < {policy['abstain_threshold']} — "
                f"ambiguous hypotheses; MONITOR expected loss={el[0]:.1f}, "
                f"optimal action={opt_act}"
            ),
            "expected_loss": el_dict,
            "optimal_action": opt_act,
        }

    if top_cls in policy["human_always"]:
        return {
            "state": "HUMAN_REVIEW",
            "action": "recommend_only",
            "human_required": True,
            "reason": f"{top_cls} — human decision required by policy",
            "expected_loss": el_dict,
            "optimal_action": opt_act,
        }

    if top_cls in policy["auto_classes"] and conf >= policy["auto_threshold"]:
        sampled = bool(rng.random() < policy["audit_sampling"])
        return {
            "state": "HUMAN_REVIEW" if sampled else "AUTO_QUARANTINE",
            "action": "quarantine_channel+alert",
            "human_required": sampled,
            "reason": "low-risk measurement fault" + (" (audit sample)" if sampled else ""),
            "expected_loss": el_dict,
            "optimal_action": opt_act,
        }

    return {
        "state": f"AUTO_{opt_act}",
        "action": opt_act.lower(),
        "human_required": opt_act in ("ESCALATE", "RECOMMEND"),
        "reason": (
            f"{top_cls} (conf. {conf:.0%}) — "
            f"optimal action {opt_act} (EL={el[opt_idx]:.1f})"
        ),
        "expected_loss": el_dict,
        "optimal_action": opt_act,
    }


def risk_coverage_curve(
    P_cal: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Risk–Coverage curve.

    For each abstention threshold τ ∈ [0, 1]:
      - Coverage = fraction of events on which the system decides
      - Risk      = error rate on decided events

    A good system has a curve that drops quickly (low risk at high coverage).

    Parameters
    ----------
    P_cal    : (N, K) calibrated posteriors
    labels   : ground-truth labels
    classes  : class name list

    Returns
    -------
    DataFrame [threshold, coverage, risk, n_decided]
    """
    if thresholds is None:
        thresholds = np.linspace(0.5, 1.0, 51)

    y = np.asarray(labels)
    confs = P_cal.max(axis=1)
    preds = np.array([classes[i] for i in P_cal.argmax(axis=1)])
    correct = (preds == y)

    rows = []
    for τ in thresholds:
        decided = confs >= τ
        n_dec = decided.sum()
        if n_dec > 0:
            risk = float((~correct[decided]).mean())
            coverage = float(n_dec / len(y))
        else:
            risk = 0.0
            coverage = 0.0
        rows.append({"threshold": float(τ), "coverage": coverage,
                     "risk": risk, "n_decided": int(n_dec)})

    return pd.DataFrame(rows)
