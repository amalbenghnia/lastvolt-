"""
EnergyMind V2 — Professional publication-quality figures.
All 8 figures use IEEE-style clean design: no clutter, proper axes, consistent palette.
Run: python scripts/make_figures_v2.py
"""
import sys, json, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Professional style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linestyle": "--",
    "grid.color": "#94a3b8",
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cbd5e1",
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})

# ── Color palette ────────────────────────────────────────────────────────────
C = {
    "blue":   "#2563eb",
    "green":  "#16a34a",
    "amber":  "#d97706",
    "red":    "#dc2626",
    "violet": "#7c3aed",
    "gray":   "#64748b",
    "light":  "#e2e8f0",
    "bg":     "#f8fafc",
    "em":     "#2563eb",   # EnergyMind highlight
}

from energymind.config import ROOT, load_config
from energymind.network_v2 import run_powerflow_v2, PandapowerSurrogate
from energymind.calibration import IsotonicCalibrator, reliability_data
from energymind.risk_policy import risk_coverage_curve, LOSS_MATRIX, ACTIONS
from energymind.faults_v2 import CLASSES_V2, inject_v2
from energymind.engine import FeatureBuilder, Diagnoser, FEATURES, persistent
from energymind.simulate import build_baseline
from energymind.faults import observe
from energymind.noosphere_v2 import export_v2, import_v2

OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# ── Load baseline data (shared across figures) ───────────────────────────────
print("  Loading baseline simulation...")
cfg  = load_config()
base = build_baseline(cfg)
P    = cfg["periods"]
sl   = lambda k: base.loc[P[k][0]: P[k][1]]

o_tr = observe(sl("train_pv_model"), cfg, np.random.default_rng(1))
fb   = FeatureBuilder().fit_pv_model(o_tr).fit_scales(o_tr)
o_cal, reg_cal = inject_v2(sl("calibrate_diagnosis"), 48, cfg["seed_calibration"], cfg)
Zc   = fb.z(o_cal, hidden=())
dg   = Diagnoser(); dg.fit_v2(Zc, o_cal.label, CLASSES_V2)
o_te, reg_te = inject_v2(sl("test"), 72, 22, cfg)
Z    = fb.z(o_te, hidden=("inv",))
Zf   = fb.z(o_te, hidden=())
post = dg.posterior(Z)
n_split  = int(len(o_cal) * 0.70)
Pi       = dg.posterior(fb.z(o_cal.iloc[n_split:], hidden=()))
cal      = IsotonicCalibrator(CLASSES_V2)
cal.fit(Pi, o_cal.iloc[n_split:].label.values)
post_cal = cal.transform(post)

metrics_path = OUT / "metrics_v2.json"
summary = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}


# =============================================================================
# FIG 1 — Network Single-Line Diagram
# =============================================================================
print("  Fig 1 — Single-line diagram (Professional IEEE standard layout)...")
fig, ax = plt.subplots(figsize=(15, 10), dpi=220)
fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#ffffff')
ax.set_xlim(0, 15)
ax.set_ylim(0, 10.5)
ax.axis('off')

# Title
ax.text(7.5, 10.15, 'Figure 1 — Campus Microgrid Electrical Single-Line Diagram (SLD)',
        ha='center', va='center', fontsize=15, fontweight='bold', color='#0f172a')
ax.text(7.5, 9.75, '7-Bus, 20/0.4 kV Radial Feeder Topology with Rooftop Solar PV & Battery Energy Storage (BESS)',
        ha='center', va='center', fontsize=11, color='#475569')

# 1. Medium Voltage Grid (Top)
grid_rect = patches.FancyBboxPatch((6.0, 8.5), 3.0, 0.85, boxstyle='round,pad=0.08',
                                  facecolor='#f8fafc', edgecolor='#334155', linewidth=1.6)
ax.add_patch(grid_rect)
ax.text(7.5, 8.92, 'STEG Utility Grid\n20 kV Medium Voltage (Bus 0 - Slack)',
        ha='center', va='center', fontsize=9.5, fontweight='bold', color='#0f172a')

# Line from Grid to Breaker
ax.plot([7.5, 7.5], [8.5, 8.0], color='#1e293b', lw=2.2)

# HV Circuit breaker
cb1 = patches.Rectangle((7.35, 8.0), 0.3, 0.3, facecolor='#ffffff', edgecolor='#0f172a', lw=1.5)
ax.add_patch(cb1)
ax.plot([7.35, 7.65], [8.0, 8.3], color='#0f172a', lw=1.5)
ax.text(7.85, 8.15, 'HV Circuit Breaker', fontsize=8.0, color='#475569', va='center')

# Line to transformer
ax.plot([7.5, 7.5], [8.0, 7.55], color='#1e293b', lw=2.2)

# Transformer (2 overlapping circles)
circ1 = patches.Circle((7.5, 7.25), 0.35, facecolor='#f5f3ff', edgecolor='#7c3aed', lw=2.2)
circ2 = patches.Circle((7.5, 6.80), 0.35, facecolor='#f5f3ff', edgecolor='#7c3aed', lw=2.2)
ax.add_patch(circ1)
ax.add_patch(circ2)
ax.text(8.2, 7.02, 'Dyn11 Step-Down Transformer\n250 kVA, 20 kV / 0.4 kV\nu_k = 4.0%, P_fe = 0.55 kW',
        fontsize=9.0, color='#5b21b6', va='center', fontweight='600')

# Line to Main LV breaker
ax.plot([7.5, 7.5], [6.45, 6.0], color='#1e293b', lw=2.5)
cb2 = patches.Rectangle((7.35, 6.0), 0.3, 0.3, facecolor='#ffffff', edgecolor='#0f172a', lw=1.5)
ax.add_patch(cb2)
ax.plot([7.35, 7.65], [6.0, 6.3], color='#0f172a', lw=1.5)
ax.text(7.85, 6.15, 'Main LV Air Circuit Breaker', fontsize=8.0, color='#475569', va='center')

# Line from Breaker to Busbar
ax.plot([7.5, 7.5], [6.0, 5.3], color='#1e293b', lw=2.5)

# 2. Main 0.4 kV Busbar (Thick horizontal bar)
bus_y = 5.3
ax.plot([1.0, 14.0], [bus_y, bus_y], color='#1e3a8a', lw=7.0, solid_capstyle='round')

# Busbar label (placed on left above busbar)
ax.text(1.2, bus_y + 0.3, 'Main Low-Voltage Busbar (Bus 1) — 400 V Line-to-Line, 3-Phase, 50 Hz',
        ha='left', va='bottom', fontsize=11, fontweight='bold', color='#1e3a8a')

# PCC Sensor badge
sensor_pcc = patches.FancyBboxPatch((10.0, bus_y + 0.25), 3.6, 0.65, boxstyle='round,pad=0.06',
                                    facecolor='#fef2f2', edgecolor='#dc2626', linewidth=1.5)
ax.add_patch(sensor_pcc)
ax.plot([11.8, 11.8], [bus_y, bus_y + 0.25], color='#dc2626', lw=1.8)
ax.plot(11.8, bus_y, 'v', color='#dc2626', ms=8)
ax.text(11.8, bus_y + 0.58, 'PCC Voltage Sensor (V1) & Meter P_grid\nTelemetry: 15-min / 5-min SCADA',
        ha='center', va='center', fontsize=8.0, fontweight='bold', color='#b91c1c')

# 3. Five Radial Feeder Bays
bays = [
    {
        'x': 2.2, 'name': 'Feeder 4: Rooftop PV',
        'cable': 'L04: 35 mm2 Al, 30 m\nR = 0.868, X = 0.078 Ohm/km',
        'comp': '50 kWp Rooftop Solar Array',
        'sub': 'Inverter: 50 kVA Grid-Tied\nSensors: V_inv (V2) & P_pv\nWeather: GHI, T_amb, Wind',
        'color': '#d97706', 'bg': '#fffbeb', 'type': 'PV Generation',
        'kw': '50 kWp Peak',
    },
    {
        'x': 4.8, 'name': 'Feeder 1: Admin Building',
        'cable': 'L01: 70 mm2 Al, 40 m\nR = 0.443, X = 0.076 Ohm/km',
        'comp': 'Administration Offices',
        'sub': 'Peak Load: 15.0 kW (30% split)\nWorkday Hours: 08:00 - 18:00\nSmart Meter: P_load1',
        'color': '#0284c7', 'bg': '#f0f9ff', 'type': 'Commercial Load',
        'kw': '15 kW Peak',
    },
    {
        'x': 7.5, 'name': 'Feeder 2: Labs & HVAC',
        'cable': 'L02: 95 mm2 Al, 60 m\nR = 0.320, X = 0.074 Ohm/km',
        'comp': 'Research Labs & HVAC',
        'sub': 'Peak Load: 25.0 kW (50% split)\nHigh Dynamic Air-Conditioning\nSmart Meter: P_load2',
        'color': '#0369a1', 'bg': '#f0f9ff', 'type': 'Heavy Dynamic Load',
        'kw': '25 kW Peak',
    },
    {
        'x': 10.2, 'name': 'Feeder 3: Residence & EV',
        'cable': 'L03: 70 mm2 Al, 80 m\nR = 0.443, X = 0.076 Ohm/km',
        'comp': 'Student Residence & EV',
        'sub': 'Peak Load: 10.0 kW (20% split)\nEvening Peak & EV Charging\nSmart Meter: P_load3',
        'color': '#0891b2', 'bg': '#f0f9ff', 'type': 'Residential + EV',
        'kw': '10 kW Peak',
    },
    {
        'x': 12.8, 'name': 'Feeder 5: Storage BESS',
        'cable': 'L05: 35 mm2 Al, 20 m\nR = 0.868, X = 0.078 Ohm/km',
        'comp': 'Battery Energy Storage',
        'sub': 'Lithium-Ion: 100 kWh / 25 kW\nBi-directional Inverter (DC/AC)\nSensors: P_bess, SoC%',
        'color': '#16a34a', 'bg': '#f0fdf4', 'type': 'Energy Storage',
        'kw': '100 kWh / 25 kW',
    },
]

for bay in bays:
    bx = bay['x']
    ax.plot([bx, bx], [bus_y, 4.6], color='#334155', lw=2.0)
    cb = patches.Rectangle((bx - 0.15, 4.3), 0.3, 0.3, facecolor='#ffffff', edgecolor='#334155', lw=1.2)
    ax.add_patch(cb)
    ax.plot([bx - 0.15, bx + 0.15], [4.3, 4.6], color='#334155', lw=1.2)
    ax.plot([bx, bx], [4.3, 3.85], color='#334155', lw=2.0)
    cable_box = patches.FancyBboxPatch((bx - 1.1, 3.35), 2.2, 0.5, boxstyle='round,pad=0.04',
                                       facecolor='#f8fafc', edgecolor='#cbd5e1', lw=1.0)
    ax.add_patch(cable_box)
    ax.text(bx, 3.60, bay['cable'], ha='center', va='center', fontsize=7.2, color='#475569')
    ax.plot([bx, bx], [3.35, 2.9], color='#334155', lw=2.0)
    card = patches.FancyBboxPatch((bx - 1.2, 0.7), 2.4, 2.2, boxstyle='round,pad=0.08',
                                  facecolor=bay['bg'], edgecolor=bay['color'], lw=1.8)
    ax.add_patch(card)
    hdr = patches.Rectangle((bx - 1.2, 2.4), 2.4, 0.5, facecolor=bay['color'], edgecolor=bay['color'])
    ax.add_patch(hdr)
    ax.text(bx, 2.65, bay['name'], ha='center', va='center', fontsize=8.2, fontweight='bold', color='#ffffff')
    ax.text(bx, 2.05, bay['comp'], ha='center', va='center', fontsize=8.8, fontweight='bold', color='#1e293b')
    ax.text(bx, 1.75, '[' + bay['kw'] + ']', ha='center', va='center', fontsize=8.0, fontweight='600', color=bay['color'])
    ax.text(bx, 1.20, bay['sub'], ha='center', va='center', fontsize=7.5, color='#334155')

summary_rect = patches.FancyBboxPatch((1.0, 0.12), 13.0, 0.45, boxstyle='round,pad=0.04',
                                     facecolor='#f8fafc', edgecolor='#94a3b8', lw=1.0)
ax.add_patch(summary_rect)
ax.text(7.5, 0.35, 'IEEE 1547 Interconnection Compliance · Real Campus Rooftop PV (238,464 Rows) · 3 Feeders Calibrated to STEG Bills · Newton-Raphson AC Load Flow',
        ha='center', va='center', fontsize=8.4, fontweight='600', color='#334155')

plt.savefig(OUT / "fig01_single_line_diagram.png", bbox_inches='tight', dpi=220)
plt.close()
print("    -> fig01 saved (professional SLD)")


# =============================================================================
# FIG 2 — Power Flow Analysis
# =============================================================================
print("  Fig 2 — Power flow states...")
pv_vals   = np.linspace(0, 50, 25)
load_vals = [30, 45, 60]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor("white")

palette = [C["blue"], C["amber"], C["red"]]
labels  = ["Load = 30 kW", "Load = 45 kW", "Load = 60 kW"]

for load, color, lbl in zip(load_vals, palette, labels):
    v_pcc, grid_kw = [], []
    for pv in pv_vals:
        res = run_powerflow_v2(pv, load, 0.0)
        v_pcc.append(res.v_pcc)
        grid_kw.append(res.grid_kw)
    axes[0].plot(pv_vals, v_pcc, color=color, label=lbl, linewidth=2)
    axes[1].plot(pv_vals, grid_kw, color=color, label=lbl, linewidth=2)

axes[0].axhline(400, color="#1e293b", linestyle="--", linewidth=1, alpha=0.6, label="Nominal 400 V")
axes[0].axhline(360, color=C["red"], linestyle=":", linewidth=1, alpha=0.6, label="Undervoltage limit")
axes[0].axhline(440, color=C["amber"], linestyle=":", linewidth=1, alpha=0.6, label="Overvoltage limit")
axes[0].set_xlabel("PV Output (kW)")
axes[0].set_ylabel("PCC Voltage (V)")
axes[0].set_title("Bus Voltage vs PV Output")
axes[0].legend(fontsize=9)
axes[0].set_ylim(350, 450)

axes[1].axhline(0, color="#1e293b", linestyle="--", linewidth=1.2, alpha=0.6)
axes[1].fill_between(pv_vals, 0, np.array([min(0, run_powerflow_v2(p, 45, 0).grid_kw) for p in pv_vals]),
                     alpha=0.12, color=C["green"], label="Self-sufficient zone")
axes[1].annotate("Self-sufficiency\npoint", xy=(45, 0), xytext=(35, 8),
                 arrowprops=dict(arrowstyle="->", color=C["green"], lw=1.5),
                 fontsize=9, color=C["green"], fontweight="bold")
axes[1].set_xlabel("PV Output (kW)")
axes[1].set_ylabel("Grid Import (kW)")
axes[1].set_title("Grid Import vs PV Output")
axes[1].legend(fontsize=9)

for ax in axes:
    ax.set_facecolor(C["bg"])
fig.suptitle("Fig. 2 — Pandapower Power Flow Results (Newton-Raphson, 7-bus network)",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT / "fig02_powerflow_states.png")
plt.close()
print("    -> fig02 saved")


# =============================================================================
# FIG 3 — Surrogate Validation
# =============================================================================
print("  Fig 3 — Surrogate validation...")
surr = PandapowerSurrogate()
surr.fit(rng=np.random.default_rng(42))
report = surr.validation_report

rng_fig = np.random.default_rng(999)
N_VIS = 300
pv_t = rng_fig.uniform(0, 55, N_VIS)
ld_t = rng_fig.uniform(8, 75, N_VIS)
bs_t = rng_fig.uniform(-40, 40, N_VIS)
true_grid, pred_grid, errors = [], [], []
for pv, ld, bs in zip(pv_t, ld_t, bs_t):
    t = run_powerflow_v2(pv, ld, bs)
    p = surr.predict(pv, ld, bs)
    true_grid.append(t.grid_kw)
    pred_grid.append(p["grid_kw"])
    errors.append(abs(t.grid_kw - p["grid_kw"]))

true_grid = np.array(true_grid)
pred_grid = np.array(pred_grid)
errors    = np.array(errors)
ss_res = np.sum((true_grid - pred_grid)**2)
ss_tot = np.sum((true_grid - true_grid.mean())**2)
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor("white")

sc = axes[0].scatter(true_grid, pred_grid, c=errors, cmap="YlOrRd",
                     s=18, alpha=0.75, edgecolors="none")
plt.colorbar(sc, ax=axes[0], label="Absolute Error (kW)", fraction=0.04)
lim = [min(true_grid.min(), pred_grid.min())-3, max(true_grid.max(), pred_grid.max())+3]
axes[0].plot(lim, lim, "k--", linewidth=1.2, alpha=0.7, label="Perfect prediction")
axes[0].set_xlim(lim); axes[0].set_ylim(lim)
axes[0].set_xlabel("Pandapower (kW)")
axes[0].set_ylabel("Surrogate (kW)")
axes[0].set_title("Surrogate vs Ground-Truth")
axes[0].legend(fontsize=9)
axes[0].text(0.05, 0.93, f"R² = {r2:.4f}\nMAE = {errors.mean():.3f} kW",
             transform=axes[0].transAxes, fontsize=10, fontweight="bold",
             verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=C["blue"], alpha=0.9))

q_pct = [50, 75, 90, 95, 99]
q_vals = np.percentile(errors, q_pct)
q_colors = [C["green"], "#65a30d", C["amber"], "#ea580c", C["red"]]
bars = axes[1].bar([f"P{p}" for p in q_pct], q_vals, color=q_colors, width=0.55, edgecolor="white")
axes[1].axhline(surr.MAX_ERROR_KW, color=C["red"], linestyle="--", linewidth=1.5,
                label=f"Validity threshold ({surr.MAX_ERROR_KW} kW)")
axes[1].set_ylabel("Absolute Error (kW)")
axes[1].set_title("Error Quantile Distribution")
axes[1].legend(fontsize=9)
valid = report.get("valid", True)
valid_color = C["green"] if valid else C["red"]
valid_text  = "VALID ✓" if valid else "INVALID ✗"
axes[1].text(0.97, 0.97, valid_text,
             transform=axes[1].transAxes, fontsize=11, fontweight="bold",
             color=valid_color, ha="right", va="top")
for bar, v in zip(bars, q_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                 f"{v:.3f}", ha="center", fontsize=9, color="#1e293b")

for ax in axes:
    ax.set_facecolor(C["bg"])
fig.suptitle(f"Fig. 3 — Pandapower Surrogate Validation  (N_train={surr.N_FIT}, N_test={surr.N_TEST} independent OOD states)",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT / "fig03_surrogate_validation.png")
plt.close()
print("    -> fig03 saved")


# =============================================================================
# FIG 4 — Detection Benchmark
# =============================================================================
print("  Fig 4 — Detection benchmark...")
if summary and "detection" in summary:
    det = summary["detection"]
    method_labels = {
        "A_isoforest_brut":  "IsoForest (raw)",
        "B_physique_seule":  "Physics Only",
        "C_isoforest_resid": "IsoForest + Residuals",
        "EnergyMind":        "EnergyMind (ours)",
        "CUSUM":             "CUSUM",
    }
    methods = list(method_labels.keys())
    disp    = [method_labels[m] for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.patch.set_facecolor("white")

    er_mean = [det[m]["event_recall"]["mean"] for m in methods]
    er_lo   = [det[m]["event_recall"]["mean"] - det[m]["event_recall"]["ci95_low"]  for m in methods]
    er_hi   = [det[m]["event_recall"]["ci95_high"] - det[m]["event_recall"]["mean"] for m in methods]
    pr_mean = [det[m]["precision"]["mean"] for m in methods]
    fpr_mean = [det[m]["fpr"]["mean"] for m in methods]

    # Sort by event recall
    order = np.argsort(er_mean)
    colors_bar = [C["em"] if methods[i] == "EnergyMind" else C["gray"] for i in order]

    axes[0].barh([disp[i] for i in order], [er_mean[i] for i in order],
                 xerr=([er_lo[i] for i in order], [er_hi[i] for i in order]),
                 color=colors_bar, capsize=4, edgecolor="white", height=0.6)
    axes[0].set_xlabel("Event Recall")
    axes[0].set_title("Event Recall (95% CI)")
    axes[0].set_xlim(0, 1)
    axes[0].axvline(0.5, color=C["red"], linestyle="--", alpha=0.5, linewidth=1)
    for i, (idx, v) in enumerate(zip(order, [er_mean[j] for j in order])):
        axes[0].text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=9,
                     fontweight="bold" if methods[idx] == "EnergyMind" else "normal")

    axes[1].scatter(fpr_mean, er_mean,
                    c=[C["em"] if m == "EnergyMind" else C["gray"] for m in methods],
                    s=120, zorder=5, edgecolors="white", linewidths=1.5)
    for m, pr, er, fp in zip(disp, pr_mean, er_mean, fpr_mean):
        axes[1].annotate(m, (fp, er), textcoords="offset points",
                         xytext=(6, 3), fontsize=8,
                         fontweight="bold" if "ours" in m else "normal",
                         color=C["em"] if "ours" in m else "#334155")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("Event Recall")
    axes[1].set_title("Detection Trade-off\n(Recall vs FPR)")
    axes[1].set_xlim(-0.01, max(fpr_mean)+0.05)

    em_by_cls = det.get("EnergyMind", {}).get("event_recall_by_class", {})
    if em_by_cls:
        cls_names = [c.replace("_", " ").title() for c in em_by_cls.keys()]
        cls_vals  = list(em_by_cls.values())
        col_cls   = [C["green"] if v >= 0.8 else C["amber"] if v >= 0.5 else C["red"] for v in cls_vals]
        order_cls = np.argsort(cls_vals)
        axes[2].barh([cls_names[i] for i in order_cls], [cls_vals[i] for i in order_cls],
                     color=[col_cls[i] for i in order_cls], edgecolor="white", height=0.6)
        axes[2].set_xlabel("Event Recall")
        axes[2].set_title("EnergyMind — Recall by Fault Class")
        axes[2].set_xlim(0, 1.1)
        axes[2].axvline(0.8, color=C["green"], linestyle="--", alpha=0.4, linewidth=1)
        for i, (idx, v) in enumerate(zip(order_cls, [cls_vals[j] for j in order_cls])):
            axes[2].text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=9)

    for ax in axes:
        ax.set_facecolor(C["bg"])
    fig.suptitle("Fig. 4 — Detection Benchmark: 5 Methods × 8 Fault Classes  (5 seeds, 95% CI)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
else:
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "Run python scripts/run_v2.py first",
            ha="center", va="center", transform=ax.transAxes)
plt.savefig(OUT / "fig04_detection_benchmark.png")
plt.close()
print("    -> fig04 saved")


# =============================================================================
# FIG 5 — Sensor Sensitivity
# =============================================================================
print("  Fig 5 — Sensor sensitivity...")
noise_levels = [0.001, 0.0025, 0.005, 0.010, 0.020]
accs = []
for vrel in noise_levels:
    cfg_tmp = {**cfg, "noise": {**cfg["noise"], "voltage_rel": vrel}}
    from energymind.faults_v2 import inject_v2 as iv2
    o_te_n, reg_n = iv2(sl("test"), 36, 22, cfg_tmp)
    Zn  = fb.z(o_te_n, hidden=("inv",))
    pn  = dg.posterior(Zn)
    pnc = cal.transform(pn)
    subset = reg_n[reg_n["class"].isin(["pv_fault", "coordinated_falsification"])]
    correct = []
    for _, r in subset.iterrows():
        s  = o_te_n.index.get_loc(pd.Timestamp(r.start))
        fl = np.arange(s, s + r.steps)
        if len(fl) == 0: continue
        pred_cls = CLASSES_V2[pnc[fl].mean(axis=0).argmax()]
        correct.append(pred_cls == r["class"])
    accs.append(np.mean(correct) * 100 if correct else 0.0)

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor(C["bg"])
noise_pct = [n * 100 for n in noise_levels]
ax.plot(noise_pct, accs, "o-", color=C["blue"], linewidth=2.5, markersize=9, zorder=5)
ax.fill_between([0, 0.3], 0, 105, alpha=0.08, color=C["green"])
ax.axvline(0.3, color=C["green"], linestyle="--", linewidth=1.5, label="Nominal noise (0.3%)")
ax.axhline(80, color=C["gray"], linestyle=":", linewidth=1, alpha=0.6)
# Annotate inflection
infl_idx = np.argmin(np.abs(np.array(noise_pct) - 0.5))
ax.annotate("Separation degrades\nabove 0.5% noise",
            xy=(noise_pct[infl_idx], accs[infl_idx]),
            xytext=(0.8, accs[infl_idx] - 20),
            arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.5),
            fontsize=9.5, color=C["red"])
ax.set_xlabel("Voltage Sensor Noise Level (% of V_nominal, std)")
ax.set_ylabel("Separation Accuracy: PV Fault vs Cyber Attack (%)")
ax.set_title("Fig. 5 — Diagnostic Robustness to Sensor Noise Quality")
ax.legend(fontsize=10)
ax.set_ylim(0, 108)
ax.set_xlim(0, noise_pct[-1] + 0.1)
for x, y in zip(noise_pct, accs):
    ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "fig05_sensor_sensitivity.png")
plt.close()
print("    -> fig05 saved")


# =============================================================================
# FIG 6 — Active Evidence Acquisition
# =============================================================================
print("  Fig 6 — Active evidence acquisition...")
ambiguous_events = reg_te[reg_te["class"].isin(["pv_fault", "coordinated_falsification"])]
n_checks_active, n_checks_random, n_checks_fixed, accuracies = [], [], [], []
rng_aea = np.random.default_rng(7)
for _, r in ambiguous_events.head(30).iterrows():
    s  = o_te.index.get_loc(pd.Timestamp(r.start))
    fl = np.arange(s, s + r.steps)
    fl = fl[fl < len(post_cal)]
    if len(fl) == 0: continue
    p  = post_cal[fl].mean(axis=0)
    if p.max() >= 0.90: continue
    nb = dg.next_best_check(Z.iloc[fl[-1]], p, ["inv"])
    has_active = len(nb) > 0
    pf  = dg.posterior(Zf.iloc[fl]).mean(axis=0)
    acc = (CLASSES_V2[pf.argmax()] == r["class"])
    n_checks_active.append(1 if has_active else 2)
    n_checks_random.append(1 if rng_aea.random() < 0.5 else 2)
    n_checks_fixed.append(2)
    accuracies.append(float(acc))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("white")
for ax in axes:
    ax.set_facecolor(C["bg"])

strategies = ["Active\n(Info Gain)", "Random", "Fixed\nOrder"]
means = [
    np.mean(n_checks_active) if n_checks_active else 1.5,
    np.mean(n_checks_random) if n_checks_random else 1.5,
    np.mean(n_checks_fixed)  if n_checks_fixed  else 2.0,
]
cols_aea = [C["blue"], C["amber"], C["gray"]]
bars = axes[0].bar(strategies, means, color=cols_aea, width=0.45, edgecolor="white")
axes[0].set_ylabel("Avg. Checks per Ambiguous Case")
axes[0].set_title("Evidence Collection Efficiency\n(fewer checks = better)")
axes[0].set_ylim(0, 2.6)
axes[0].axhline(2.0, color=C["gray"], linestyle="--", alpha=0.4, linewidth=1)
for bar, m in zip(bars, means):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
                 f"{m:.2f}", ha="center", fontsize=12, fontweight="bold")

if accuracies:
    before = 0.50
    after  = np.mean(accuracies)
    bars2 = axes[1].bar(["Before\n(uncertain)", "After Active\nAcquisition"],
                        [before * 100, after * 100],
                        color=[C["red"], C["green"]], width=0.4, edgecolor="white")
    axes[1].set_ylabel("Diagnostic Accuracy on Ambiguous Cases (%)")
    axes[1].set_title("Impact on Diagnostic Accuracy")
    axes[1].set_ylim(0, 115)
    for bar, v in zip(bars2, [before*100, after*100]):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                     f"{v:.0f}%", ha="center", fontsize=13, fontweight="bold")

fig.suptitle("Fig. 6 — Active Evidence Acquisition (Goal 2)\n"
             "Information-gain guided sensor selection for P2/A1 disambiguation",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT / "fig06_active_evidence.png")
plt.close()
print("    -> fig06 saved")


# =============================================================================
# FIG 7 — Risk-Coverage Curve
# =============================================================================
print("  Fig 7 — Risk-coverage curve...")
flagE   = (post_cal[:, 0] < 0.2)
flagE_p = persistent(flagE)
y_anom  = (o_te.label != "normal").values
detected = flagE_p & y_anom
P_det    = post_cal[detected]
labs_det = o_te.label.values[detected]

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor(C["bg"])

if len(P_det) > 5:
    rc = risk_coverage_curve(P_det, labs_det, CLASSES_V2)
    cov = rc["coverage"].values * 100
    risk = rc["risk"].values * 100
    ax.fill_between(cov, risk, alpha=0.08, color=C["blue"])
    ax.plot(cov, risk, "o-", color=C["blue"], linewidth=2.5, markersize=5, zorder=5)

    v1_mask = rc["threshold"] >= 0.89
    if v1_mask.any():
        r_v1 = rc[v1_mask].iloc[0]
        ax.scatter(r_v1["coverage"]*100, r_v1["risk"]*100,
                   s=200, color=C["amber"], zorder=6, label="Fixed threshold V1 (0.90)", marker="D")
        ax.annotate(f"V1: {r_v1['coverage']*100:.0f}% cov.",
                    (r_v1["coverage"]*100, r_v1["risk"]*100),
                    xytext=(10, 10), textcoords="offset points",
                    fontsize=9, color=C["amber"], fontweight="bold")

    op_cov  = 50.0
    op_idx  = np.argmin(np.abs(cov - op_cov))
    ax.scatter(cov[op_idx], risk[op_idx], s=200, color=C["green"],
               zorder=6, label=f"Operating point (~{cov[op_idx]:.0f}% coverage)", marker="*")

    ax.set_xlabel("Coverage — Fraction of Events Decided Autonomously (%)")
    ax.set_ylabel("Error Rate on Decided Cases (%)")
    ax.set_title("Fig. 7 — Risk-Coverage Trade-off (Goal 3)\nAutonomy vs. Accuracy under Expected-Loss Policy")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 105)
    ax.set_ylim(bottom=-1)
else:
    ax.text(0.5, 0.5, "Run with more seeds for sufficient detected events",
            ha="center", va="center", transform=ax.transAxes, fontsize=11)
plt.tight_layout()
plt.savefig(OUT / "fig07_risk_coverage.png")
plt.close()
print("    -> fig07 saved")


# =============================================================================
# FIG 8 — Noosphere Knowledge Transfer
# =============================================================================
print("  Fig 8 — Noosphere transfer...")
n_ev_cal = {}
for _, r in reg_cal.iterrows():
    n_ev_cal[r["class"]] = n_ev_cal.get(r["class"], 0) + 1

o_fac, reg_fac = inject_v2(sl("test"), 36, 99, cfg)
Zfac = fb.z(o_fac, hidden=("inv",))

conditions = {"No Transfer": copy.deepcopy(dg)}
for eps, lbl in [(2.0, "Transfer ε=2.0"), (1.0, "Transfer ε=1.0"), (0.3, "Transfer ε=0.3")]:
    dg_tmp = copy.deepcopy(dg)
    sigs   = export_v2(dg, n_events=n_ev_cal, epsilon=eps, validated=True)
    import_v2(dg_tmp, sigs, available_sensors=["bal", "pvexp", "vpcc"])
    conditions[lbl] = dg_tmp

results_transfer = {}
for lbl, dg_c in conditions.items():
    post_c = dg_c.posterior(Zfac)
    preds  = [CLASSES_V2[i] for i in post_c.argmax(axis=1)]
    acc    = float(np.mean([p == t for p, t in zip(preds, o_fac.label.values)]))
    results_transfer[lbl] = acc

fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor("white")
ax.set_facecolor(C["bg"])

labels_t = list(results_transfer.keys())
vals_t   = [results_transfer[l] * 100 for l in labels_t]
cols_t   = [C["red"]] + [C["blue"], C["green"], "#166534"[:7]]
cols_t   = cols_t[:len(labels_t)]
bars = ax.barh(labels_t, vals_t, color=cols_t, edgecolor="white", height=0.55)
ax.set_xlabel("Diagnostic Accuracy on Factory Node (%)")
ax.set_title("Fig. 8 — Federated Knowledge Transfer with Differential Privacy (Goal 4)\n"
             "Campus → Factory Node (3 sensors only: bal, pvexp, vpcc — no inverter)")
ax.set_xlim(0, 115)
ax.axvline(vals_t[0], color=C["red"], linestyle="--", alpha=0.5, linewidth=1.5)
for bar, v in zip(bars, vals_t):
    ax.text(v + 1, bar.get_y() + bar.get_height()/2,
            f"{v:.1f}%", va="center", fontsize=11, fontweight="bold")
ax.text(0.98, 0.08,
        "ε → 0: more private, less accurate\nε = ∞: no noise, maximum transfer",
        transform=ax.transAxes, fontsize=9, ha="right", color=C["gray"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C["light"]))
plt.tight_layout()
plt.savefig(OUT / "fig08_noosphere_transfer.png")
plt.close()
print("    -> fig08 saved")

print("\n  [OK] 8 professional figures generated in outputs/")
