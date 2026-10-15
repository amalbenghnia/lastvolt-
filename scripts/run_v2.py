"""
Pipeline V2 complet — exécute toutes les expériences et génère les 8 figures.
Commande : python scripts/run_v2.py

Reproductibilité : une seule commande régénère tous les chiffres et figures.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from energymind.evaluate_v2 import run_v2
from energymind.config import ROOT

print("=" * 60)
print("EnergyMind V2 — Pipeline expérimental complet")
print("=" * 60)

print("\n[1/3] Lancement des expériences (5 graines, 8 classes)...")
summary = run_v2(
    n_cal=80, n_test=96,
    hide=("inv",),
    seeds_test=(22, 33, 44, 55, 66),
    out="outputs",
    use_v2_faults=True,
)

print("\n[2/3] Métriques principales :")
det = summary["detection"]
print(f"\n{'Méthode':<30} {'Event Recall (mean)':>20} {'Precision':>12} {'FA/jour':>10}")
print("-" * 76)
for method, vals in det.items():
    er   = vals["event_recall"]["mean"]
    prec = vals["precision"]["mean"]
    print(f"  {method:<28} {er:>18.1%}   {prec:>10.1%}")

fa = summary["false_alarms_per_day"]["mean"]
diag = summary["diagnosis"]
print(f"\n  Fausses alarmes/jour (EnergyMind) : {fa:.3f}")
print(f"  Précision diagnostic (décidé)     : {diag['acc_decided']['mean']:.1%}"
      f"  [IC95 {diag['acc_decided']['ci95_low']:.1%}–{diag['acc_decided']['ci95_high']:.1%}]")
print(f"  ECE calibré                       : {diag['ece_calibrated']['mean']:.4f}")
print(f"  Taux abstention                   : {diag['abstention_rate']['mean']:.1%}")
print(f"  Taux revue humaine                : {diag['human_review_rate']['mean']:.1%}")

surr = summary["surrogate_validation"]
print(f"\n  Substitut pandapower :")
print(f"    MAE import        : {surr.get('grid_kw_mae', 'N/A')} kW")
print(f"    Max import        : {surr.get('grid_kw_max', 'N/A')} kW")
print(f"    Valide (< 10 kW)  : {surr.get('valid', 'N/A')}")

g4 = summary["goal4_transfer"]
print(f"\n  Goal 4 — Transfert vers nœud usine :")
print(f"    Signatures exportées : {g4['n_signatures_exported']}")
print(f"    Importées            : {g4['n_imported']}")
print(f"    Rejetées             : {g4['n_rejected']}")

print("\n[3/3] Génération des 8 figures...")
# Importation différée pour éviter les dépendances circulaires
exec(open(Path(__file__).parent / "make_figures_v2.py", encoding="utf-8").read())

print(f"\n{'='*60}")
print(f"✅ V2 terminé. Métriques : outputs/metrics_v2.json")
print(f"   Figures : outputs/fig_*.png")
print(f"{'='*60}")
