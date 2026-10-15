import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

LAYER_COLORS = {
    "physical":   "#FEF3C7",  # amber-50
    "sensor":     "#DBEAFE",  # blue-100
    "ai":         "#D1FAE5",  # green-100
    "governance": "#FEE2E2",  # red-100
    "noosphere":  "#EDE9FE",  # violet-100
}
BORDER_COLORS = {
    "physical":   "#D97706",
    "sensor":     "#2563EB",
    "ai":         "#059669",
    "governance": "#DC2626",
    "noosphere":  "#7C3AED",
}
NODE_COLORS = {
    "physical":   "#FCD34D",
    "sensor":     "#93C5FD",
    "ai":         "#6EE7B7",
    "governance": "#FCA5A5",
    "noosphere":  "#C4B5FD",
}


def rounded_box(ax, x, y, w, h, fc, ec, lw=1.2, radius=0.06, zorder=3):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder,
        clip_on=False
    )
    ax.add_patch(box)
    return box


def layer_band(ax, y_bottom, y_top, color, border, label, label_color):
    band = FancyBboxPatch(
        (0, y_bottom), 1, y_top - y_bottom,
        boxstyle="round,pad=0.005",
        facecolor=color, edgecolor=border, linewidth=1.4, alpha=0.55,
        transform=ax.transAxes, zorder=1, clip_on=False
    )
    ax.add_patch(band)
    ax.text(0.005, (y_bottom + y_top) / 2, label,
            transform=ax.transAxes, fontsize=7, fontweight="bold",
            color=label_color, rotation=90, va="center", ha="center",
            zorder=5)


def node(ax, cx, cy, w, h, fc, ec, title, subtitle, title_fs=7.5, sub_fs=6.2):
    rounded_box(ax, cx, cy, w, h, fc=fc, ec=ec, lw=1.4)
    ax.text(cx, cy + h * 0.12, title, ha="center", va="center",
            fontsize=title_fs, fontweight="bold", color="#1E293B", zorder=6,
            wrap=True, multialignment="center")
    ax.text(cx, cy - h * 0.22, subtitle, ha="center", va="center",
            fontsize=sub_fs, color="#475569", zorder=6,
            multialignment="center", linespacing=1.35)


def arrow_v(ax, x, y1, y2, color, lw=2.0, label="", label_side="right"):
    ax.annotate(
        "", xy=(x, y2), xytext=(x, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=14),
        zorder=4
    )
    if label:
        lx = x + 0.14 if label_side == "right" else x - 0.14
        ly = (y1 + y2) / 2
        ax.text(lx, ly, label, fontsize=5.8, color=color,
                ha="left" if label_side == "right" else "right",
                va="center", zorder=5)


def arrow_h(ax, x1, x2, y, color, lw=1.4):
    ax.annotate(
        "", xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=12),
        zorder=4
    )


fig, ax = plt.subplots(figsize=(13, 11))
ax.set_xlim(0, 13)
ax.set_ylim(-0.2, 11.2)
ax.axis("off")

fig.patch.set_facecolor("white")

BAND_Y = [0.0, 1.8, 3.8, 6.2, 8.5, 11.0]

LABELS = [
    ("Layer 1: Physical Microgrid — pandapower 7-Bus Digital Twin",
     LAYER_COLORS["physical"], BORDER_COLORS["physical"]),
    ("Layer 2: Sensor & SCADA Telemetry Acquisition",
     LAYER_COLORS["sensor"], BORDER_COLORS["sensor"]),
    ("Layer 3: EnergyMind AI Trust Layer",
     LAYER_COLORS["ai"], BORDER_COLORS["ai"]),
    ("Layer 4: Governance, Risk Policy & Audit Ledger",
     LAYER_COLORS["governance"], BORDER_COLORS["governance"]),
    ("Layer 5: Noosphere — Multi-Node Federated Defense",
     LAYER_COLORS["noosphere"], BORDER_COLORS["noosphere"]),
]

for i, (lbl, lc, bc) in enumerate(LABELS):
    yb = BAND_Y[i]
    yt = BAND_Y[i + 1]
    rect = FancyBboxPatch(
        (0.25, yb + 0.08), 12.5, yt - yb - 0.16,
        boxstyle="round,pad=0.1",
        facecolor=lc, edgecolor=bc, linewidth=1.5,
        alpha=0.5, zorder=1
    )
    ax.add_patch(rect)
    ax.text(0.62, (yb + yt) / 2, lbl, fontsize=7.2, fontweight="bold",
            color=bc, rotation=90, va="center", ha="center", zorder=5)

NW, NH = 2.5, 1.0
CY = [(BAND_Y[i] + BAND_Y[i + 1]) / 2 for i in range(5)]
XS = [2.2, 4.95, 7.7, 10.5]

PHY = [
    ("STEG Grid PCC", "20 kV Slack Bus\n(Bus 0)"),
    ("Dyn11 Transformer", "250 kVA, 20/0.4 kV\n$u_k = 4\\%$"),
    ("Main LV Busbar", "400 V, 3-Phase\n(Bus 1 — 7-bus)"),
    ("DERs & Campus\nLoads", "50 kWp PV · 100 kWh BESS\n3 Feeders (Admin/Labs/Res.)"),
]
for i, (t, s) in enumerate(PHY):
    node(ax, XS[i], CY[0], NW, NH,
         fc=NODE_COLORS["physical"], ec=BORDER_COLORS["physical"],
         title=t, subtitle=s)
for i in range(3):
    arrow_h(ax, XS[i] + NW / 2, XS[i + 1] - NW / 2, CY[0],
            BORDER_COLORS["physical"])

SEN = [
    ("Weather Telemetry", "GHI, $T_{amb}$,\n$T_{cell}$, Wind"),
    ("PCC Metering", "$V_{pcc}$ ($V_1$),\n$P_{grid}$"),
    ("PV Inverter Sensors", "$P_{pv}$, $V_{inv}$ ($V_2$),\n$P_{dc}$"),
    ("BESS & Load Meters", "$P_{bess}$, SoC%,\n$P_{load,1..3}$"),
]
for i, (t, s) in enumerate(SEN):
    node(ax, XS[i], CY[1], NW, NH,
         fc=NODE_COLORS["sensor"], ec=BORDER_COLORS["sensor"],
         title=t, subtitle=s)
for xi in [XS[0], XS[1], XS[2], XS[3]]:
    arrow_v(ax, xi, BAND_Y[1] - 0.04, BAND_Y[1] + 0.12,
            BORDER_COLORS["sensor"], lw=1.4)

AI_XS = [2.2, 5.4, 8.6]
AI_NW = 3.3
AI_NH = 1.5
AI = [
    ("Physics Residual Engine",
     "$\\mathbf{Z} \\in \\mathbb{R}^5$: balance, PV-exp,\nvoltage, inverter, volt-drop\nLightGBM weather→power"),
    ("Hybrid Anomaly Gate",
     "Physics Plausibility Check\n$\\wedge$ (IsoForest $\\vee$ CUSUM)\nAlert Trigger $\\mathcal{A}(t) \\in \\{0,1\\}$"),
    ("Bayesian Diagnoser & UQ",
     "8-Class Gaussian NB\nIsotonic Calibration: ECE = 0.038\nActive Evidence Acquisition"),
]
for i, (t, s) in enumerate(AI):
    node(ax, AI_XS[i], CY[2], AI_NW, AI_NH,
         fc=NODE_COLORS["ai"], ec=BORDER_COLORS["ai"],
         title=t, subtitle=s, title_fs=7.8, sub_fs=6.0)
for i in range(2):
    arrow_h(ax, AI_XS[i] + AI_NW / 2, AI_XS[i + 1] - AI_NW / 2, CY[2],
            BORDER_COLORS["ai"])

ax.text(AI_XS[0] + AI_NW / 2 + 0.5, CY[2] + 0.15, "$\\mathbf{Z}$ scores",
        fontsize=5.8, color=BORDER_COLORS["ai"], ha="center")
ax.text(AI_XS[1] + AI_NW / 2 + 0.4, CY[2] + 0.15, "Alert $\\mathcal{A}$",
        fontsize=5.8, color=BORDER_COLORS["ai"], ha="center")

arrow_v(ax, 6.5, BAND_Y[2] - 0.04, BAND_Y[2] + 0.12,
        BORDER_COLORS["sensor"], lw=2.0, label="Calibrated telemetry stream")

GOV_XS = [2.2, 5.4, 8.6]
GOV_NW = 3.3
GOV_NH = 1.3
GOV = [
    ("Expected-Loss Policy",
     "Asymmetric cost matrix $\\mathcal{L}(a,c)$\n4 Actions: Monitor, Alert,\nQuarantine, Escalate"),
    ("Explainability & Audit",
     "SHAP local attribution\nSHA-256 hash-chained\nSQLite tamper-evident ledger"),
    ("Human Operator Console",
     "Mandatory override on\nPhysical/Cyber alerts\n72.1% escalation rate"),
]
for i, (t, s) in enumerate(GOV):
    node(ax, GOV_XS[i], CY[3], GOV_NW, GOV_NH,
         fc=NODE_COLORS["governance"], ec=BORDER_COLORS["governance"],
         title=t, subtitle=s, title_fs=7.8, sub_fs=6.0)
for i in range(2):
    arrow_h(ax, GOV_XS[i] + GOV_NW / 2, GOV_XS[i + 1] - GOV_NW / 2, CY[3],
            BORDER_COLORS["governance"])

arrow_v(ax, 6.5, BAND_Y[3] - 0.04, BAND_Y[3] + 0.12,
        BORDER_COLORS["ai"], lw=2.0,
        label="Calibrated posteriors $P^*(c|\\mathbf{Z})$")

NOO_XS = [2.2, 5.6, 9.2]
NOO_NW = 3.0
NOO_NH = 1.4
NOO = [
    ("Campus Microgrid\n(Source Node)",
     "$\\varepsilon$-DP Export ($\\varepsilon$=1.0, $k{\\geq}5$)\nThis study — 7 signatures\nPydantic v2.0 packets"),
    ("Federated Peer Nodes",
     "Factory · Hospital\nSmart Building · Rural MG\nDomain shift: PSI $\\leq$ 0.25"),
    ("Global Threat Registry",
     "Federated consensus\nk-anonymity $\\geq$ 5\nMulti-operator defense"),
]
for i, (t, s) in enumerate(NOO):
    node(ax, NOO_XS[i], CY[4], NOO_NW, NOO_NH,
         fc=NODE_COLORS["noosphere"], ec=BORDER_COLORS["noosphere"],
         title=t, subtitle=s, title_fs=7.5, sub_fs=6.0)
for i in range(2):
    arrow_h(ax, NOO_XS[i] + NOO_NW / 2, NOO_XS[i + 1] - NOO_NW / 2, CY[4],
            BORDER_COLORS["noosphere"], lw=1.6)

ax.text((NOO_XS[0] + NOO_XS[1]) / 2 + 0.0, CY[4] + 0.85,
        "$\\varepsilon$-DP signatures",
        fontsize=6, color=BORDER_COLORS["noosphere"], ha="center")
ax.text((NOO_XS[1] + NOO_XS[2]) / 2 + 0.0, CY[4] + 0.85,
        "Consensus update",
        fontsize=6, color=BORDER_COLORS["noosphere"], ha="center")

arrow_v(ax, NOO_XS[0], BAND_Y[4] - 0.04, BAND_Y[4] + 0.12,
        BORDER_COLORS["noosphere"], lw=1.8,
        label="Anonymised signatures")

legend_elements = [
    mpatches.Patch(facecolor=NODE_COLORS["physical"], edgecolor=BORDER_COLORS["physical"],
                   label="Power flow (amber)"),
    mpatches.Patch(facecolor=NODE_COLORS["sensor"], edgecolor=BORDER_COLORS["sensor"],
                   label="Sensor telemetry (blue)"),
    mpatches.Patch(facecolor=NODE_COLORS["ai"], edgecolor=BORDER_COLORS["ai"],
                   label="AI processing (green)"),
    mpatches.Patch(facecolor=NODE_COLORS["governance"], edgecolor=BORDER_COLORS["governance"],
                   label="Governance & audit (red)"),
    mpatches.Patch(facecolor=NODE_COLORS["noosphere"], edgecolor=BORDER_COLORS["noosphere"],
                   label="Federated network (violet)"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=6.5,
          framealpha=0.9, edgecolor="#CBD5E1", ncol=1)

ax.set_title(
    "Figure 3 — EnergyMind Global System Architecture\n"
    "Five-Layer Cyber-Physical AI Trust Stack · Campus Microgrid",
    fontsize=11, fontweight="bold", color="#1E293B", pad=10
)

plt.tight_layout(rect=[0.05, 0, 1, 1])
import os
out = "outputs/fig03_architecture.png"
os.makedirs("outputs", exist_ok=True)
plt.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out}")
