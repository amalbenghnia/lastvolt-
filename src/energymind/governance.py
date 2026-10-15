"""
Governance layer — Human-AI escalation policy with hash-chained audit log.

The audit log uses SHA-256 hash chaining: each record includes the hash of the
previous record, making retroactive modification detectable. This is a practical
tamper-evidence mechanism for operational review, not a cryptographic blockchain.
"""
from __future__ import annotations
import hashlib
import json
import sqlite3
import time
import numpy as np
from .faults import CLASSES
from .engine import EXPLAIN

MODEL_VERSION = "energymind-0.1.0"

POLICY = {
    "decide_threshold": 0.90,
    "auto_threshold": 0.95,
    "auto_classes": ["pv_sensor_fault", "voltage_sensor_fault"],
    "human_always": ["coordinated_falsification", "pv_fault", "grid_event"],
    "audit_sampling": 0.05,
}

LABEL_EN = {
    "pv_fault":                  "physical PV fault (inverter / curtailment)",
    "grid_event":                "real grid event (voltage sag)",
    "pv_sensor_fault":           "biased or drifting PV sensor",
    "voltage_sensor_fault":      "faulty voltage sensor",
    "coordinated_falsification": "coordinated measurement falsification",
    "normal":                    "normal operation",
}


def route(p: np.ndarray, rng=None, policy=POLICY) -> dict:
    """State machine: return proposed action and whether human review is required."""
    rng = rng or np.random.default_rng()
    top = int(p.argmax())
    cls, conf = CLASSES[top], float(p[top])

    if cls == "normal":
        return {"state": "MONITOR", "action": "log", "human_required": False, "reason": "no anomaly detected"}

    if conf < policy["decide_threshold"]:
        return {
            "state": "HUMAN_REVIEW", "action": "abstain", "human_required": True,
            "reason": f"confidence {conf:.2f} < {policy['decide_threshold']}: ambiguous hypotheses",
        }

    if cls in policy["auto_classes"] and conf >= policy["auto_threshold"]:
        sampled = bool(rng.random() < policy["audit_sampling"])
        return {
            "state": "HUMAN_REVIEW" if sampled else "AUTO_QUARANTINE",
            "action": "quarantine_channel+alert",
            "human_required": sampled,
            "reason": "low-risk measurement fault" + (" (audit sample)" if sampled else ""),
        }

    return {
        "state": "HUMAN_REVIEW", "action": "recommend_only", "human_required": True,
        "reason": f"{LABEL_EN.get(cls, cls)}: human decision required by policy",
    }


def explain(dg, z_row, p: np.ndarray, next_check=None) -> str:
    order = np.argsort(-p)
    top = CLASSES[order[0]]
    lines = [f"Verdict: {LABEL_EN.get(top, top)} (confidence {p[order[0]]:.0%})."]
    if p[order[1]] > 0.1:
        lines.append(f"Competing hypothesis: {LABEL_EN.get(CLASSES[order[1]], CLASSES[order[1]])} ({p[order[1]]:.0%}).")
    if top != "normal":
        c = sorted(dg.contributions(z_row, top).items(), key=lambda t: -abs(t[1]))[:3]
        for f, v in c:
            lines.append(f"- {EXPLAIN[f]}: {'supports' if v > 0 else 'contradicts'} hypothesis (weight {v:+.1f}).")
    if next_check:
        lines.append(f"Recommended check: read sensor '{next_check}' (maximizes uncertainty reduction).")
    return "\n".join(lines)


class AuditLog:
    def __init__(self, path="outputs/audit.db"):
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS audit("
            "id INTEGER PRIMARY KEY, ts REAL, model TEXT, event TEXT, "
            "posterior TEXT, evidence TEXT, proposed TEXT, decision TEXT, "
            "actor TEXT, justification TEXT, prev_hash TEXT, hash TEXT)"
        )

    def _last(self):
        r = self.db.execute("SELECT hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
        return r[0] if r else "GENESIS"

    def append(self, event, posterior, evidence, proposed, decision, actor, justification):
        prev, ts = self._last(), time.time()
        body = json.dumps(
            [ts, MODEL_VERSION, event, posterior, evidence, proposed, decision, actor, justification, prev],
            sort_keys=True, default=str,
        )
        h = hashlib.sha256(body.encode()).hexdigest()
        self.db.execute(
            "INSERT INTO audit(ts,model,event,posterior,evidence,proposed,decision,actor,"
            "justification,prev_hash,hash) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (ts, MODEL_VERSION, event,
             json.dumps(posterior, default=str), json.dumps(evidence, default=str),
             json.dumps(proposed, default=str), decision, actor, justification, prev, h),
        )
        self.db.commit()
        return h

    def verify(self) -> bool:
        prev = "GENESIS"
        for r in self.db.execute(
            "SELECT ts,model,event,posterior,evidence,proposed,decision,actor,"
            "justification,prev_hash,hash FROM audit ORDER BY id"
        ):
            ts, model, event, post, ev, prop, dec, actor, just, ph, h = r
            body = json.dumps(
                [ts, model, event, json.loads(post), json.loads(ev), json.loads(prop),
                 dec, actor, just, prev],
                sort_keys=True, default=str,
            )
            if ph != prev or hashlib.sha256(body.encode()).hexdigest() != h:
                return False
            prev = h
        return True
