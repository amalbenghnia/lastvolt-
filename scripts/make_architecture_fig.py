import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

LAYER_THEMES = [
    {
        "name": "LAYER 1: PHYSICAL MICROGRID & DIGITAL TWIN (pandapower 7-Bus Model)",
        "band_bg": "#FFFBEB", "band_edge": "#F59E0B",
        "node_bg": "#FEF3C7", "node_edge": "#D97706",
        "accent": "#B45309",
    },
    {
        "name": "LAYER 2: SENSORS & SCADA TELEMETRY (15-min / 5-min Multi-Stream Acquisition)",
        "band_bg": "#EFF6FF", "band_edge": "#3B82F6",
        "node_bg": "#DBEAFE", "node_edge": "#2563EB",
        "accent": "#1D4ED8",
    },
    {
        "name": "LAYER 3: ENERGYMIND AI TRUST LAYER (Physics Residuals, Hybrid Gate & Isotonic Calibration)",
        "band_bg": "#ECFDF5", "band_edge": "#10B981",
        "node_bg": "#D1FAE5", "node_edge": "#059669",
        "accent": "#047857",
    },
    {
        "name": "LAYER 4: GOVERNANCE, RISK POLICY & AUDIT (Expected Loss, SHAP & Tamper-Evident Ledger)",
        "band_bg": "#FEF2F2", "band_edge": "#EF4444",
        "node_bg": "#FEE2E2", "node_edge": "#DC2626",
        "accent": "#B91C1C",
    },
    {
        "name": "LAYER 5: NOOSPHERE FEDERATED COLLECTIVE DEFENSE (Multi-Node ε-DP Privacy-Preserving Sharing)",
        "band_bg": "#F5F3FF", "band_edge": "#8B5CF6",
        "node_bg": "#EDE9FE", "node_edge": "#7C3AED",
        "accent": "#6D28D9",
    },
]

fig, ax = plt.subplots(figsize=(15, 13))
ax.set_xlim(0, 15)
ax.set_ylim(-0.3, 13.0)
ax.axis("off")
fig.patch.set_facecolor("white")

# Define vertical bands (y-bottom, y-top)
BANDS = [
    (0.2, 2.3),    # Layer 1
    (2.6, 4.7),    # Layer 2
    (5.0, 7.3),    # Layer 3
    (7.6, 9.8),    # Layer 4
    (10.1, 12.3),  # Layer 5
]

def draw_node(ax, cx, cy, w, h, bg, edge, title, subtitle, title_fs=8.2, sub_fs=6.8):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.06",
        facecolor=bg, edgecolor=edge, linewidth=1.3, zorder=3
    )
    ax.add_patch(box)
    ax.text(cx, cy + h * 0.16, title, ha="center", va="center",
            fontsize=title_fs, fontweight="bold", color="#0F172A", zorder=4)
    ax.text(cx, cy - h * 0.20, subtitle, ha="center", va="center",
            fontsize=sub_fs, color="#334155", zorder=4, linespacing=1.3)

def arrow_h(ax, x1, x2, y, color, lw=1.5, label=""):
    ax.annotate(
        "", xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=13),
        zorder=5
    )
    if label:
        ax.text((x1 + x2) / 2, y + 0.18, label, fontsize=6.5, color=color,
                ha="center", va="bottom", fontweight="bold", zorder=6)

def arrow_v(ax, x, y1, y2, color, lw=1.8, label="", label_side="right"):
    ax.annotate(
        "", xy=(x, y2), xytext=(x, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=14),
        zorder=5
    )
    if label:
        lx = x + 0.15 if label_side == "right" else x - 0.15
        ha = "left" if label_side == "right" else "right"
        ax.text(lx, (y1 + y2) / 2, label, fontsize=6.8, color=color,
                ha=ha, va="center", fontweight="bold", zorder=6)

# Draw bands & headers
for i, (yb, yt) in enumerate(BANDS):
    theme = LAYER_THEMES[i]
    band = FancyBboxPatch(
        (0.3, yb), 14.4, yt - yb,
        boxstyle="round,pad=0.08",
        facecolor=theme["band_bg"], edgecolor=theme["band_edge"],
        linewidth=1.4, zorder=1
    )
    ax.add_patch(band)
    ax.text(0.6, yt - 0.23, theme["name"], fontsize=8.8, fontweight="bold",
            color=theme["accent"], ha="left", va="top", zorder=2)

# --- LAYER 1: PHYSICAL MICROGRID ---
L1_Y = 1.05
L1_W, L1_H = 2.7, 1.05
L1_X = [2.0, 5.6, 9.2, 12.8]
L1_NODES = [
    ("STEG Grid PCC", "20 kV MV Slack Bus (Bus 0)\nGrid Connection & PCC"),
    ("Dyn11 Transformer", "250 kVA, 20/0.4 kV\nDyn11 Step-Down, uk=4%"),
    ("Main LV Busbar", "400 V Line-to-Line (Bus 1)\n3-Phase Radial Backbone"),
    ("DERs & Campus Load", "50 kWp PV · 100 kWh BESS\n3 Feeders (Admin/Labs/Res)"),
]
for j, (t, s) in enumerate(L1_NODES):
    draw_node(ax, L1_X[j], L1_Y, L1_W, L1_H,
              LAYER_THEMES[0]["node_bg"], LAYER_THEMES[0]["node_edge"], t, s)
for j in range(3):
    arrow_h(ax, L1_X[j] + L1_W / 2 + 0.05, L1_X[j+1] - L1_W / 2 - 0.05, L1_Y,
            LAYER_THEMES[0]["accent"], lw=1.6)

# --- LAYER 2: SENSORS & SCADA ---
L2_Y = 3.45
L2_W, L2_H = 2.7, 1.05
L2_X = [2.0, 5.6, 9.2, 12.8]
L2_NODES = [
    ("Weather Station", "GHI, Tamb, Tcell, Wind\n15-min Ambient Telemetry"),
    ("PCC Metering (V1)", "Vpcc voltage & Pgrid meter\n15-min / 5-min SCADA"),
    ("PV Inverter Sensors", "Ppv, Vinv (V2), Pdc\nDedicated Inverter Channel"),
    ("BESS & Feeders", "Pbess, SoC%, Smart Meters\nPload across 3 branches"),
]
for j, (t, s) in enumerate(L2_NODES):
    draw_node(ax, L2_X[j], L2_Y, L2_W, L2_H,
              LAYER_THEMES[1]["node_bg"], LAYER_THEMES[1]["node_edge"], t, s)
    arrow_v(ax, L1_X[j], L1_Y + L1_H / 2 + 0.05, L2_Y - L2_H / 2 - 0.05,
            LAYER_THEMES[1]["accent"], lw=1.2)

# --- LAYER 3: ENERGYMIND AI TRUST LAYER ---
L3_Y = 5.95
L3_W, L3_H = 3.6, 1.25
L3_X = [2.6, 7.5, 12.4]
L3_NODES = [
    ("Physics Residual Engine", "Z in R^5: [bal, pvexp, vpcc, inv, vinv]\nLightGBM Surrogate (MAE=0.096 kW)"),
    ("Hybrid Anomaly Gate", "Physics Plausibility Residuals\nAND (Isolation Forest OR Dual CUSUM)"),
    ("Bayesian Diagnoser & UQ", "8-Class Gaussian Naive Bayes\nIsotonic Calibration (ECE=0.038)\nActive Evidence Acquisition"),
]
for j, (t, s) in enumerate(L3_NODES):
    draw_node(ax, L3_X[j], L3_Y, L3_W, L3_H,
              LAYER_THEMES[2]["node_bg"], LAYER_THEMES[2]["node_edge"], t, s)

arrow_h(ax, L3_X[0] + L3_W / 2 + 0.05, L3_X[1] - L3_W / 2 - 0.05, L3_Y,
        LAYER_THEMES[2]["accent"], lw=1.8, label="Z Scores")
arrow_h(ax, L3_X[1] + L3_W / 2 + 0.05, L3_X[2] - L3_W / 2 - 0.05, L3_Y,
        LAYER_THEMES[2]["accent"], lw=1.8, label="Alert Trigger")

# Upward flow from Layer 2 to Layer 3
arrow_v(ax, 7.5, 4.7, 5.0, LAYER_THEMES[1]["accent"], lw=2.2,
        label="Synchronized Multi-Sensor Telemetry Stream", label_side="right")

# --- LAYER 4: GOVERNANCE, RISK POLICY & AUDIT ---
L4_Y = 8.50
L4_W, L4_H = 3.6, 1.20
L4_X = [2.6, 7.5, 12.4]
L4_NODES = [
    ("Expected-Loss Risk Policy", "Asymmetric Cost Matrix L(a, c)\n4 Actions: Alert, Monitor, Quarantine, Escalate"),
    ("Explainability & Audit Ledger", "SHAP Local Feature Attribution\nSHA-256 Hash-Chained SQLite Ledger"),
    ("Human Operator Sign-Off", "Mandatory Review on Uncertain Events\n100% Accuracy on Decided (50.6% Abstention)"),
]
for j, (t, s) in enumerate(L4_NODES):
    draw_node(ax, L4_X[j], L4_Y, L4_W, L4_H,
              LAYER_THEMES[3]["node_bg"], LAYER_THEMES[3]["node_edge"], t, s)

arrow_h(ax, L4_X[0] + L4_W / 2 + 0.05, L4_X[1] - L4_W / 2 - 0.05, L4_Y,
        LAYER_THEMES[3]["accent"], lw=1.8, label="Audit Trail")
arrow_h(ax, L4_X[1] + L4_W / 2 + 0.05, L4_X[2] - L4_W / 2 - 0.05, L4_Y,
        LAYER_THEMES[3]["accent"], lw=1.8, label="Escalation")

# Upward flow from Layer 3 to Layer 4
arrow_v(ax, 7.5, 7.3, 7.6, LAYER_THEMES[2]["accent"], lw=2.2,
        label="Calibrated Posteriors P*(c|Z) & Decision Actions", label_side="right")

# --- LAYER 5: NOOSPHERE FEDERATED COLLECTIVE DEFENSE ---
L5_Y = 11.00
L5_W, L5_H = 3.6, 1.20
L5_X = [2.6, 7.5, 12.4]
L5_NODES = [
    ("Campus Microgrid (Source Node)", "epsilon-DP Export (epsilon=1.0, k >= 5)\n7 Signatures Exported · Pydantic v2.0"),
    ("Federated Peer Nodes", "Multi-Node: Factory · Hospital · Smart Bldg · Rural\nDomain Shift Validation (PSI <= 0.25)"),
    ("Global Threat Registry", "Federated Consensus Defense\nk-Anonymity >= 5 Privacy Guarantee"),
]
for j, (t, s) in enumerate(L5_NODES):
    draw_node(ax, L5_X[j], L5_Y, L5_W, L5_H,
              LAYER_THEMES[4]["node_bg"], LAYER_THEMES[4]["node_edge"], t, s)

arrow_h(ax, L5_X[0] + L5_W / 2 + 0.05, L5_X[1] - L5_W / 2 - 0.05, L5_Y,
        LAYER_THEMES[4]["accent"], lw=1.8, label="epsilon-DP Export")
arrow_h(ax, L5_X[1] + L5_W / 2 + 0.05, L5_X[2] - L5_W / 2 - 0.05, L5_Y,
        LAYER_THEMES[4]["accent"], lw=1.8, label="Federated Update")

# Upward flow from Layer 4 to Layer 5
arrow_v(ax, 2.6, 9.8, 10.1, LAYER_THEMES[4]["accent"], lw=2.2,
        label="Anonymized Anomaly Signatures", label_side="right")

# Overall title
ax.text(7.5, 12.70, "Figure 3 — EnergyMind Global System Architecture",
        fontsize=13, fontweight="bold", color="#0F172A", ha="center", va="center")
ax.text(7.5, 12.46, "Five-Layer Cyber-Physical AI Trust Stack · University Campus Microgrid",
        fontsize=9.5, color="#475569", ha="center", va="center")

# Bottom Legend
leg_items = [
    mpatches.Patch(facecolor="#FEF3C7", edgecolor="#D97706", label="Physical Microgrid & Twin"),
    mpatches.Patch(facecolor="#DBEAFE", edgecolor="#2563EB", label="Telemetry & SCADA Acquisition"),
    mpatches.Patch(facecolor="#D1FAE5", edgecolor="#059669", label="AI Trust Layer & UQ"),
    mpatches.Patch(facecolor="#FEE2E2", edgecolor="#DC2626", label="Governance, Risk & Audit"),
    mpatches.Patch(facecolor="#EDE9FE", edgecolor="#7C3AED", label="Noosphere Federated Defense"),
]
ax.legend(handles=leg_items, loc="lower center", bbox_to_anchor=(0.5, -0.02),
          ncol=5, fontsize=7.2, frameon=True, facecolor="white", edgecolor="#CBD5E1")

plt.tight_layout()
out_path = "outputs/fig03_architecture.png"
plt.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Generated clean architecture diagram: {out_path}")
