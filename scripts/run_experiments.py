"""Rejoue TOUTES les experiences de Phase 1 : python scripts/run_experiments.py"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energymind.evaluate import run
from energymind.data import load_invoices, load_pv, pv_quality_report
from energymind.config import ROOT

res, ev, (base, o_te, Z, post, flagE, dg, fb) = run(out="outputs")
inv, log = load_invoices()
log.to_csv(ROOT / "outputs/invoice_quality_log.csv", index=False)
inv.drop(columns=[c for c in inv.columns if c.endswith("_z")], errors="ignore").to_csv(ROOT / "data/processed/invoices_clean.csv", index=False)
(ROOT / "outputs/pv_quality_report.json").write_text(json.dumps(pv_quality_report(load_pv()), indent=2))
base.to_csv(ROOT / "data/processed/microgrid_baseline_15min.csv.gz")
o_te.to_csv(ROOT / "data/processed/test_telemetry_with_labels.csv.gz")
print(json.dumps({k: res[k] for k in ["detection", "diagnosis", "false_alarm_episodes_per_day"]}, indent=1, default=str))
