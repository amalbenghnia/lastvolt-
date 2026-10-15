import sys, json; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from energymind.evaluate import run
from energymind.faults import CLASSES
res, ev, (base, o, Z, post, flag, dg, fb) = run(out=None)
reg = pd.read_csv("outputs/test_event_registry.csv")
# Fig 1 : meme alerte, deux causes (P2 vs A1) - cas du jury
fig, ax = plt.subplots(2, 2, figsize=(11, 6), sharex="col")
for col, sc in enumerate(["P2", "A1"]):
    r = reg[reg.scenario == sc].iloc[0]; s = o.index.get_loc(pd.Timestamp(r.start))
    w = o.iloc[s - 12: s + r.steps + 12]
    ax[0, col].plot(w.index, w.pv_meas, label="PV (compteur)"); ax[0, col].plot(w.index, w.inv_ac, "--", label="Onduleur AC (independant, masque)")
    ax[0, col].plot(w.index, w.grid_meas, label="Import STEG"); ax[0, col].set_title(f"{sc}: {'defaut PV reel' if sc=='P2' else 'falsification coordonnee'}"); ax[0, col].legend(fontsize=7)
    ax[1, col].plot(w.index, Z.bal.loc[w.index], label="z bilan"); ax[1, col].plot(w.index, Z.pvexp.loc[w.index], label="z PV attendu"); ax[1, col].plot(w.index, Z.inv.loc[w.index] if False else fb.z(o.loc[w.index], hidden=()).inv, label="z onduleur (si lu)"); ax[1, col].legend(fontsize=7)
plt.suptitle("Meme signature observable (bilan ~0, PV bas) : seul l'onduleur separe les deux causes"); plt.tight_layout(); plt.savefig("outputs/fig_same_alert_two_causes.png", dpi=130)
# Fig 2 : comparaison de detection par classe
d = res["detection"]; cls = ["pv_fault", "grid_event", "pv_sensor_fault", "voltage_sensor_fault", "coordinated_falsification"]
df = pd.DataFrame({m: [d[m]["event_recall_by_class"].get(c, 0) for c in cls] for m in d}, index=cls)
df.plot.bar(figsize=(10, 4.5)); plt.ylabel("Rappel evenementiel"); plt.title("Detection par classe : baselines vs EnergyMind"); plt.xticks(rotation=20); plt.tight_layout(); plt.savefig("outputs/fig_detection_by_class.png", dpi=130)
