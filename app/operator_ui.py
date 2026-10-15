"""
Interface opérateur EnergyMind — Phase 1.
Démontre les 4 Goals du challenge.

Lancer : streamlit run app/operator_ui.py

Goal 1 — Détection & Diagnostic      : onglet "Diagnostic Temps Réel"
Goal 2 — Confiance & Explainabilité  : onglet "Confiance & SHAP"
Goal 3 — Gouvernance Humain-IA       : onglet "Gouvernance & Audit"
Goal 4 — Noosphère                   : onglet "Transfert de Connaissances"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from energymind.evaluate import run
from energymind.faults import CLASSES
from energymind.governance import route, explain, AuditLog, LABEL_FR, POLICY
from energymind.noosphere import export_signatures, import_signatures
from energymind.drift import cusum_scan, calibrate_cusum

# ---------------------------------------------------------------------------
# Configuration page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EnergyMind — Campus Microgrid",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS léger pour les badges de confiance
st.markdown("""
<style>
.badge-high   { background:#22c55e; color:white; padding:2px 10px; border-radius:12px; font-weight:bold; }
.badge-medium { background:#f59e0b; color:white; padding:2px 10px; border-radius:12px; font-weight:bold; }
.badge-low    { background:#ef4444; color:white; padding:2px 10px; border-radius:12px; font-weight:bold; }
.metric-box   { background:#f8fafc; border-left:4px solid #3b82f6; padding:10px 16px; border-radius:4px; margin:6px 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Chargement (mis en cache — un seul run par session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="⏳ Calibration du moteur EnergyMind (environ 30 s)…")
def _load():
    res, ev, (base, o, Z, post, flag, dg, fb) = run(out=None)
    reg = pd.read_csv(
        Path(__file__).resolve().parents[1] / "outputs" / "test_event_registry.csv"
    )
    return res, ev, o, Z, post, flag, dg, fb, base, reg


res, ev, o, Z, post, flag, dg, fb, base, reg = _load()
audit = AuditLog(
    str(Path(__file__).resolve().parents[1] / "outputs" / "audit.db")
)

# ---------------------------------------------------------------------------
# Sidebar — Sélection de l'incident
# ---------------------------------------------------------------------------
st.sidebar.title("⚡ EnergyMind")
st.sidebar.caption("Microgrid Cyber-Physical Defense — Phase 1 Prototype")

incident_idx = st.sidebar.selectbox(
    "Sélectionner un incident injecté",
    range(len(reg)),
    format_func=lambda k: f"#{k}  {reg.scenario.iloc[k]} — {reg['class'].iloc[k][:20]}",
)

r = reg.iloc[incident_idx]
s = o.index.get_loc(pd.Timestamp(r.start))
idx_event = np.arange(s, s + r.steps)
fl = idx_event[flag[idx_event]]
w = o.iloc[max(0, s - 24): s + r.steps + 24]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Scénario :** `{r.scenario}`")
st.sidebar.markdown(f"**Classe réelle :** `{r['class']}`")
st.sidebar.markdown(f"**Durée :** {r.steps} pas × 15 min = {r.steps * 15} min")

telemetrie_masquee = st.sidebar.checkbox("🔌 Télémétrie onduleur masquée (`inv`)", value=True)
hidden = ["inv"] if telemetrie_masquee else []

# ---------------------------------------------------------------------------
# Recalcul du posterieur avec/sans mesure masquée
# ---------------------------------------------------------------------------
Z_view = fb.z(o, hidden=hidden)
post_view = dg.posterior(Z_view)

if len(fl) > 0:
    p = post_view[fl].mean(axis=0)
else:
    p = np.ones(len(CLASSES)) / len(CLASSES)

rt = route(p)

# ---------------------------------------------------------------------------
# Onglets principaux
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Diagnostic Temps Réel",
    "📊 Confiance & SHAP",
    "⚖️ Gouvernance & Audit",
    "🌐 Transfert de Connaissances",
])

# ===========================================================================
# TAB 1 — Goal 1 : Détection & Diagnostic
# ===========================================================================
with tab1:
    st.header("Goal 1 — Détection & Diagnostic d'anomalies")
    st.caption(
        "Fusion de 5 canaux de preuves physiques → diagnostic probabiliste multi-hypothèses."
    )

    col_a, col_b, col_c = st.columns([1, 1, 1])
    col_a.metric("Événements détectés", f"{int(flag[idx_event].any())}/1",
                 delta="✅ Détecté" if flag[idx_event].any() else "❌ Manqué")
    col_b.metric("Pas alertés", f"{len(fl)} / {len(idx_event)}")
    col_c.metric("Confiance diagnostic", f"{p.max():.0%}")

    st.subheader("📈 Télémétrie autour de l'incident")
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axes[0].plot(w.index, w.pv_meas, label="PV mesuré (kW)", color="#f59e0b")
    axes[0].plot(w.index, w.load_meas, label="Charge (kW)", color="#3b82f6")
    axes[0].plot(w.index, w.grid_meas, label="Réseau STEG (kW)", color="#6b7280")
    # Zone de l'incident
    if s < len(o):
        ts_start = o.index[s]
        ts_end = o.index[min(s + r.steps - 1, len(o) - 1)]
        for ax in axes:
            ax.axvspan(ts_start, ts_end, alpha=0.15, color="red", label="Fenêtre incident")
    axes[0].set_ylabel("Puissance (kW)")
    axes[0].legend(fontsize=8, ncol=4)
    axes[0].grid(alpha=0.3)
    axes[1].plot(w.index, w.v_pcc, label="V_PCC (V)", color="#8b5cf6")
    axes[1].plot(w.index, w.v_inv, label="V_INV (V)", color="#ec4899")
    axes[1].set_ylabel("Tension (V)")
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("🎯 Posterieur bayésien par hypothèse")
    if len(fl) == 0:
        st.warning("⚠️ Incident non détecté (faux négatif). "
                   "Désactivez le filtre de persistance pour analyser en post-mortem.")
    else:
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        colors = ["#22c55e" if i == p.argmax() else "#94a3b8" for i in range(len(CLASSES))]
        labels_fr = [LABEL_FR[c] for c in CLASSES]
        bars = ax2.barh(labels_fr, p, color=colors, edgecolor="white", linewidth=0.5)
        ax2.axvline(0.9, color="#ef4444", linestyle="--", linewidth=1, label="Seuil décision 90%")
        ax2.set_xlim(0, 1)
        ax2.set_xlabel("Probabilité postérieure")
        ax2.legend(fontsize=8)
        for bar, val in zip(bars, p):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1%}", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        st.subheader("📋 Explication en langage naturel")
        nb = (dg.next_best_check(Z_view.iloc[fl[-1]], p, hidden)
              if p.max() < POLICY["decide_threshold"] and hidden else [])
        explanation = explain(dg, Z_view.iloc[fl[-1]], p, nb[0][0] if nb else None)
        st.code(explanation, language="markdown")

    st.subheader("📊 Résidus physiques normalisés (z-scores)")
    z_w = Z_view.iloc[max(0, s - 24): s + r.steps + 24]
    st.line_chart(z_w[FEATURES].fillna(0))
    st.caption(
        "Seuil ≈ ±2σ (ligne non tracée). Résidus hors norme → preuve d'anomalie. "
        "NaN = mesure masquée (marginalisée dans le diagnostic)."
    )

    # CUSUM drift detection
    st.subheader("📉 Détection de dérive CUSUM (scénario M2 — biais progressif)")
    calib = calibrate_cusum(Z_view.iloc[:s], ["pvexp", "bal"], target_fpr=0.01)
    cusum_alarms = cusum_scan(Z_view, ["pvexp", "bal"],
                              k=calib["k"], h=calib["h"])
    fig3, axes3 = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
    for ax, col, color in zip(axes3, ["pvexp", "bal"], ["#f59e0b", "#3b82f6"]):
        ax.plot(z_w.index, z_w[col].fillna(0), color=color, linewidth=0.8, label=col)
        alarm_mask = cusum_alarms[col].reindex(z_w.index, fill_value=False)
        ax.fill_between(z_w.index, -5, 5, where=alarm_mask,
                        alpha=0.25, color="#ef4444", label="Alarme CUSUM")
        ax.set_ylabel(f"z({col})")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
    axes3[-1].set_xlabel("Temps")
    plt.suptitle(f"CUSUM : k={calib['k']}, h={calib['h']:.1f} "
                 f"(FPR cible=1%, obtenu={calib['achieved_fpr']:.2%})", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

# ===========================================================================
# TAB 2 — Goal 2 : Confiance & Explainabilité
# ===========================================================================
with tab2:
    st.header("Goal 2 — Confiance, Calibration & Explainabilité")

    col1, col2, col3 = st.columns(3)
    import json
    m = res.get("diagnosis", {})
    col1.metric("Précision (quand décidé)", f"{m.get('accuracy_when_decided', 0):.0%}")
    col2.metric("Taux d'abstention", f"{m.get('abstention_rate', 0):.0%}")
    col3.metric("ECE (calibration)", f"{m.get('ece_decided', 0):.4f}")

    st.markdown("""
    > **Interprétation ECE :** Expected Calibration Error. Une valeur proche de 0 signifie
    > que quand le système annonce 90% de confiance, environ 90% des diagnostics sont corrects.
    """)

    st.subheader("📊 Diagramme de fiabilité (Reliability Diagram)")
    # Calcul sur les événements détectés
    if len(ev) > 0:
        det_ev = ev[ev.detected.fillna(False).astype(bool)].copy()
        if "conf" in det_ev.columns and "pred" in det_ev.columns and "truth" in det_ev.columns:
            det_ev["correct"] = (det_ev["pred"] == det_ev["truth"])
            confs = det_ev["conf"].dropna().values
            correct = det_ev.loc[det_ev["conf"].notna(), "correct"].values
            bins = np.linspace(0.5, 1.0, 6)
            fig_cal, ax_cal = plt.subplots(figsize=(6, 5))
            bin_centers, bin_acc, bin_counts = [], [], []
            for a, b in zip(bins[:-1], bins[1:]):
                m_bin = (confs >= a) & (confs < b + 1e-9)
                if m_bin.sum() > 0:
                    bin_centers.append((a + b) / 2)
                    bin_acc.append(correct[m_bin].mean())
                    bin_counts.append(m_bin.sum())
            ax_cal.plot([0.5, 1.0], [0.5, 1.0], "k--", linewidth=1, label="Calibration parfaite")
            ax_cal.bar(bin_centers, bin_acc, width=0.09, alpha=0.6,
                       color="#3b82f6", label="Précision observée")
            ax_cal.set_xlabel("Confiance prédite")
            ax_cal.set_ylabel("Précision réelle")
            ax_cal.set_xlim(0.45, 1.05); ax_cal.set_ylim(0, 1.05)
            ax_cal.legend(); ax_cal.grid(alpha=0.3)
            for cx, acc, cnt in zip(bin_centers, bin_acc, bin_counts):
                ax_cal.text(cx, acc + 0.03, f"n={cnt}", ha="center", fontsize=8)
            plt.title("Reliability Diagram — Diagnostic EnergyMind")
            plt.tight_layout()
            st.pyplot(fig_cal)
            plt.close()
        else:
            st.info("Les événements détectés ne contiennent pas suffisamment de données de confiance pour tracer le diagramme.")
    else:
        st.info("Aucun événement dans les résultats de l'évaluation.")

    st.subheader("🔍 Explainabilité SHAP — Modèle LightGBM PV")
    if fb.shap_explainer is not None and fb.shap_explainer.available:
        # SHAP sur le pas central de l'incident
        if len(fl) > 0:
            mid_step = fl[len(fl) // 2]
            obs_row = o.iloc[mid_step]
            shap_vals = fb.shap_pv(obs_row)
            if shap_vals:
                fig_shap, ax_shap = plt.subplots(figsize=(7, 3))
                feats = [t[0] for t in shap_vals]
                vals = [t[1] for t in shap_vals]
                colors_s = ["#ef4444" if v < 0 else "#22c55e" for v in vals]
                ax_shap.barh(feats, vals, color=colors_s, edgecolor="white")
                ax_shap.axvline(0, color="black", linewidth=0.5)
                ax_shap.set_xlabel("Contribution SHAP (kW)")
                ax_shap.set_title(f"SHAP — Prédiction PV au pas {mid_step}")
                plt.tight_layout()
                st.pyplot(fig_shap)
                plt.close()
                st.caption(
                    "Vert = contribution positive à la prédiction PV. "
                    "Rouge = contribution négative. "
                    "Un résidu négatif (PV mesuré << PV prédit) peut indiquer un défaut ou un biais capteur."
                )
    else:
        st.info("ℹ️ SHAP non disponible (`pip install shap`). "
                "Les contributions bayésiennes sont utilisées à la place (voir onglet Diagnostic).")

    st.subheader("🤐 Politique d'abstention (Selective Prediction)")
    st.markdown(f"""
    Le système s'abstient si la confiance maximale est < **{POLICY['decide_threshold']:.0%}**.

    | Situation | Action |
    |-----------|--------|
    | Confiance ≥ {POLICY['auto_threshold']:.0%}, classe à faible risque | Action automatique (quarantaine capteur) |
    | Confiance ≥ {POLICY['decide_threshold']:.0%}, classe à risque élevé | Recommandation + revue humaine obligatoire |
    | Confiance < {POLICY['decide_threshold']:.0%} | **Abstention → revue humaine + prochaine vérification** |

    Seuil `{POLICY['decide_threshold']:.0%}` = hypothèse de Phase 1, à ajuster selon le coût des erreurs (Phase 2).
    """)

    # Active Evidence Acquisition demo
    if len(fl) > 0 and p.max() < POLICY["decide_threshold"] and hidden:
        nb = dg.next_best_check(Z_view.iloc[fl[-1]], p, hidden)
        if nb:
            st.success(
                f"🔎 **Active Evidence Acquisition :** "
                f"Lire la mesure `{nb[0][0]}` réduirait l'incertitude de "
                f"{nb[0][1]:.2f} nats (gain d'information espéré − coût)."
            )

# ===========================================================================
# TAB 3 — Goal 3 : Gouvernance & Audit
# ===========================================================================
with tab3:
    st.header("Goal 3 — Gouvernance Humain-IA & Audit")

    col1, col2 = st.columns(2)
    with col1:
        badge_color = ("high" if rt["state"] == "MONITOR"
                       else "medium" if rt["state"] == "AUTO_QUARANTINE"
                       else "low")
        st.markdown(f"**État :** <span class='badge-{badge_color}'>{rt['state']}</span>",
                    unsafe_allow_html=True)
        st.markdown(f"**Action proposée :** `{rt['action']}`")
        st.markdown(f"**Raison :** {rt['reason']}")
        st.markdown(f"**Revue humaine requise :** {'✅ Oui' if rt['human_required'] else '❌ Non'}")

    with col2:
        st.markdown("""
        **Politique de gouvernance (POLICY)**
        ```
        Seuil décision  : 90%
        Seuil auto      : 95%
        Classes auto    : capteur PV, capteur tension
        Classes humain  : falsification, défaut PV, événement réseau
        Échantillon audit : 5% des décisions automatiques
        ```
        """)

    st.subheader("🔒 Revue et décision opérateur")
    if rt["human_required"] and len(fl) > 0:
        st.info("Une décision humaine est requise pour cet incident.")
        actor = st.text_input("Identifiant du validateur", "operateur_1",
                              help="Identifiant fonctionnel (non nominatif en production)")
        why = st.text_area("Justification de la décision",
                           placeholder="Ex. : Vérification terrain confirmée, capteur remplacé…")
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        for label, col in zip(
            ["✅ Approuver", "❌ Rejeter", "🔄 Override (autre cause)"],
            [col_btn1, col_btn2, col_btn3]
        ):
            if col.button(label) and why.strip():
                h = audit.append(
                    event=f"incident_{incident_idx}",
                    posterior=dict(zip(CLASSES, p.round(3).tolist())),
                    evidence={"flagged_steps": int(len(fl)), "hidden": hidden},
                    proposed=rt,
                    decision=label,
                    actor=actor,
                    justification=why,
                )
                st.success(f"✅ Décision enregistrée dans le journal SHA-256 (hash : `{h[:16]}…`)")
    elif not rt["human_required"]:
        st.success("✅ Décision automatique autorisée par politique (capteur à faible risque).")

    st.subheader("📜 Journal d'audit SHA-256 (infalsifiable)")
    integrity = audit.verify()
    if integrity:
        st.success("🔐 Journal intègre — chaîne de hachage vérifiée.")
    else:
        st.error("⚠️ Journal altéré — anomalie détectée dans la chaîne de hachage !")

    try:
        import sqlite3
        conn = sqlite3.connect(
            str(Path(__file__).resolve().parents[1] / "outputs" / "audit.db")
        )
        log_df = pd.read_sql(
            "SELECT ts, event, decision, actor, hash FROM audit ORDER BY id DESC LIMIT 10",
            conn,
        )
        log_df["ts"] = pd.to_datetime(log_df["ts"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
        log_df["hash"] = log_df["hash"].str[:12] + "…"
        conn.close()
        st.dataframe(log_df, use_container_width=True)
    except Exception:
        st.caption("Aucune entrée dans le journal pour l'instant.")

    st.subheader("📈 Métriques de performance globales (Test Set)")
    det_m = res.get("detection", {})
    rows_m = []
    for method, vals in det_m.items():
        rows_m.append({
            "Méthode": method,
            "Précision": f"{vals['precision']:.1%}",
            "Rappel": f"{vals['recall']:.1%}",
            "Event Recall": f"{vals['event_recall']:.1%}",
            "FPR": f"{vals['fpr']:.3%}",
        })
    st.dataframe(pd.DataFrame(rows_m), use_container_width=True)

# ===========================================================================
# TAB 4 — Goal 4 : Noosphère — Transfert de Connaissances
# ===========================================================================
with tab4:
    st.header("Goal 4 — Noosphère : Transfert de Connaissances Inter-nœuds")

    st.markdown("""
    **Principe :** Partager des *signatures d'incidents normalisées* (pas les séries brutes)
    entre nœuds de microgrid différents. La confidentialité est assurée par :
    - Bruit de Laplace (confidentialité différentielle, ε-DP)
    - Seuil k-anonymat : une classe rare (< 5 événements) n'est pas partagée
    - Contrôle des capteurs requis : un nœud sans capteur compatible rejette la signature
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📤 Export — Nœud Campus")
        epsilon = st.slider("Budget ε (confidentialité, plus petit = plus privé)", 0.1, 2.0, 1.0, 0.1)
        n_events_by_class = (
            ev[ev.detected.fillna(False).astype(bool)]
            .groupby("truth").size().to_dict()
            if len(ev) > 0 else {}
        )
        sigs = export_signatures(dg, node_type="campus_microgrid_university",
                                 n_events=n_events_by_class, epsilon=epsilon,
                                 validated=True)
        st.metric("Signatures exportées", len(sigs))
        if sigs:
            st.json(sigs[0], expanded=False)
            st.caption("Exemple de signature (première classe). "
                       "Valeurs = résidus normalisés avec bruit DP.")

    with col2:
        st.subheader("📥 Import — Nœud Destinataire (simulation)")
        import copy
        dg_node2 = copy.deepcopy(dg)
        available_sensors = st.multiselect(
            "Capteurs disponibles sur le nœud destinataire",
            FEATURES,
            default=["bal", "pvexp", "vpcc"],
        )
        weight = st.slider("Poids du transfert (0 = ignorer, 1 = remplacer)", 0.0, 1.0, 0.3, 0.05)
        if st.button("🚀 Simuler le transfert"):
            report = import_signatures(dg_node2, sigs, available_sensors, weight=weight)
            for cls, status, reason in report:
                if status == "importe":
                    st.success(f"✅ `{cls}` — {status} ({reason})")
                else:
                    st.warning(f"⚠️ `{cls}` — {status} : {reason}")

    st.subheader("📋 Ce qui est partagé vs. ce qui reste local")
    st.markdown("""
    | Partagé ✅ | Reste local 🔒 |
    |-----------|---------------|
    | Signatures normalisées (µ/σ des résidus en unités sigma) | Séries temporelles brutes |
    | Capteurs requis pour utiliser la signature | Profils de charge campus |
    | Conditions de validité (irradiance min, etc.) | Topologie réseau exacte |
    | Nombre d'événements (bruité par DP) | Identité des validateurs |
    | Validé par humain : oui/non | Factures STEG |
    """)

    st.subheader("🔬 Validation du transfert (Leave-One-Node-Out)")
    st.markdown("""
    **Protocole documenté (à exécuter en Phase 2 avec 2 nœuds réels) :**

    1. Entraîner le Diagnoser sur le Nœud A (Campus Universitaire)
    2. Exporter les signatures → Nœud B (autre campus ou installation PV)
    3. Évaluer le Diagnoser sur Nœud B *sans* transfert → précision baseline
    4. Importer les signatures → évaluer *avec* transfert → amélioration mesurée
    5. Comparer avec et sans bruit DP pour quantifier le trade-off confidentialité/performance
    """)
    st.info(
        "ℹ️ En Phase 1, le second nœud est simulé avec des paramètres différents "
        "(Pnom différent, profil de charge différent, graine aléatoire différente). "
        "Le protocole est identique à la démonstration réelle."
    )
