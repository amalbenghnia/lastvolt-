---
title: "EnergyMind V2 — University Campus Augmented Microgrid"
subtitle: "TSYP14 Technical Challenge · Phase 1 Final Proposal"
version: "2.0 — Phase 1 finale"
---

# 1. Positionnement

EnergyMind est une **couche d'assurance de confiance** entre les mesures brutes d'un microgrid et toute décision automatisée. Pour chaque événement suspect, il répond à quatre questions : *qu'est-ce qui est observé, quelles hypothèses l'expliquent, quelles preuves les soutiennent ou les contredisent, et quelle mesure manquante permettrait de trancher.* Quand les preuves ne permettent pas de séparer deux causes, le système **s'abstient et indique quelle mesure lire ensuite**.

**Question de recherche.** Comment une IA peut-elle séparer les événements physiques des corruptions de mesure dans un microgrid partiellement observable, quantifier les limites de son propre diagnostic, appeler l'opérateur quand nécessaire, et transférer des connaissances validées vers d'autres environnements énergétiques ?

> [!IMPORTANT]
> Seules les fonctions effectivement présentes dans le code sont annoncées. Les travaux prévus pour la Phase 2 sont explicitement identifiés.

---

# 2. Périmètre

| Élément | Choix |
|---|---|
| Couche | **Microgrid** — Campus Universitaire connecté au réseau STEG (PCC) |
| Actifs | Centrale PV 50 kWp, charge campus (3 départs), BESS 100 kWh/40 kW, compteur PCC, télémétrie onduleur, 2 points de tension |
| Données réelles (Phase 1) | Série PV/météo 5 min, 2022-02-24 → 2024-05-31 (238 464 lignes) ; factures STEG mensuelles 2022–2024 |
| Données simulées | BESS, télémétrie onduleur, tension, profil intra-mensuel de charge (forme AR(1) calibrée sur factures), tous les défauts |
| Réseau | pandapower 20/0,4 kV : 1 transformateur 250 kVA, 7 bus, 5 lignes BT aluminium, 3 départs |
| Phase 2 | Remplacer les hypothèses par les données mesurées du directeur technique (chaque paramètre de `configs/config.yaml` porte un statut de provenance) |

---

# 3. Données et plan de simulation d'anomalies

## 3.1 Construction de la ligne de base (toutes les étapes tracées)

| Signal | Source | Statut |
|---|---|---|
| PV (kW) | Série 5 min réelle (Kaggle), validée par **PVGIS JRC** (69 752 kWh/an) | **Données réelles** |
| Irradiance, T_cell, vent, humidité | Mesures réelles, vérifiées contre réanalyse **NASA POWER** | **Données réelles** |
| Énergie mensuelle totale (kWh) | Factures STEG réelles Campus Universitaire 2022–2024 | **Données réelles** |
| Profil intra-mensuel de charge | Forme **DOE Commercial Building Reference** (Secondary School / Higher Ed) calibrée sur factures réelles | Benchmark calibré |
| Réseau & Power flow | Topologie pandapower 7 bus / 20-0.4 kV conforme **SimBench** (`1-LV-semiurb4`) | Standard IEEE/CIGRE |
| Cyberattaque & Falsification A1 | Vecteur d'injection fausses données conforme taxonomie **UAH ICS** (Tommy Morris) | Modélisation réaliste |
| BESS, onduleur, tension BT | Équations physiques pandapower + substitut validé (MAE < 0,1 kW) | Modèle physique |

**Correction qualité des factures** (reproductible : `outputs/invoice_quality_log.csv`) : 3 mois mal étiquetés réaffectés par séquence chronologique, 2 mois manquants imputés par médiane saisonnière, 3 lignes implausibles marquées.
**Données de référence ouvertes** : Scripts d'interrogation d'APIs publiques disponibles dans `scripts/fetch_reference_benchmarks.py` (`data/reference/`).

## 3.2 Catalogue de défauts — 8 classes (V2)

La **vérité physique** et la **télémétrie observée** sont toujours séparées. Les défauts sont injectés soit dans l'état physique (événements réels), soit dans la mesure (corruptions), puis le bruit de mesure est ajouté séparément.

| ID | Scénario | Nature | Classe |
|---|---|---|---|
| P1 | Variation solaire naturelle | Aucune (dans les données réelles) | normal |
| P2 | Défaut PV / écrêtage | **Physique** : PV réel × 0,25–0,65, la grille compense | pv_fault |
| G1 | Creux de tension STEG (réel) | **Physique** : les deux points de tension −8 à −18 % | grid_event |
| L1 | Pic de charge réel | **Physique** : charge × 1,40–1,80, grille compense | load_peak |
| M1 | Biais capteur PV | Mesure : PV mesuré × 0,35–0,70, physique inchangée | pv_sensor_fault |
| M2 | Dérive capteur PV | Mesure : biais progressif sur 6–12 h | pv_sensor_fault |
| V1 | Défaillance capteur tension | Mesure : V_PCC chute, V_inv reste correct | voltage_sensor_fault |
| B1 | Défaillance capteur BESS | Mesure : SOC_meas biaisé −20 à −50 %, BESS_meas ×1,2–1,6 | bess_sensor_fault |
| A1 | Falsification coordonnée | Mesure : PV réduit + grille augmentée du même montant → **bilan conservé** | coordinated_falsification |

Chaque injection est enregistrée avec horodatage dans `outputs/test_event_registry.csv`.

---

# 4. Architecture technique — Modèles et justifications

> [!NOTE]
> Les quatre Goals ne nécessitent pas tous un modèle IA. Des modèles physiques, des algorithmes probabilistes et des règles déterministes assurent la traçabilité.

## 4.1 Goal 1 — Détection & Diagnostic

### Cinq canaux de preuves (résidus physiques normalisés)

| Canal | Calcul | Anomalie détectée |
|---|---|---|
| `bal` | $r_{bal} = P_{grid} + P_{PV} + P_{BESS} - P_{load} - P_{pertes}$ | Déséquilibre énergétique global |
| `pvexp` | $r_{pv} = P_{PV,mes} - \hat{P}_{PV}(météo)$ | Écart production PV vs attente LightGBM |
| `inv` | $r_{inv} = P_{PV,compteur} - P_{AC,onduleur}$ | Incohérence compteur/onduleur |
| `vpcc` | $r_{vpcc} = V_{PCC,mes} - V_{pandapower}$ | Écart tension PCC vs load-flow |
| `vinv` | $r_{vinv} = V_{inv,mes} - V_{pandapower}$ | Écart tension onduleur vs load-flow |

**Normalisation :** z-scores robustes MAD (médiane ± 1,4826 × MAD) — résistants aux outliers.

### LightGBM — Modèle de production PV attendue

**Justification vs alternatives :**

| Modèle | RMSE (kW) | Avantage |
|---|---|---|
| Modèle physique linéaire (baseline) | ~5,5 | Explicable, mais pas de non-linéarités |
| Ridge Regression | ~4,6 | Simple, mais pas de interactions |
| **LightGBM** (retenu) | **~3,6** | Meilleur RMSE, SHAP natif, données tabulaires météo |
| LightGBM pour la charge | −23%/+18% | **Non retenu** : biais inter-périodes → remplacé par même créneau −3 semaines |

**Entrées :** `irradiance`, `t_cell`, `wind`, `hour_of_day`, `day_of_year`  
**SHAP TreeExplainer :** contributions par feature disponibles (voir Goal 2).

### Bayesian Diagnoser — Moteur central

$$P(H_i \mid E) \propto P(E \mid H_i) \cdot P(H_i)$$

- Structure Naive Bayes avec **marginalisation des NaN** (mesures absentes ≠ imputées)
- **Limitation documentée :** les résidus sont corrélés ; le modèle indépendant est une simplification Phase 1
- **Phase 2 :** réseau bayésien structuré ou modèle discriminatif calibré

### CUSUM bidirectionnel — Dérive temporelle

$$S_t = \max(0, S_{t-1} + r_t - k)$$

Détecte M2 (dérive progressive) avec délai médian de **6 pas** contre 12 pour un seuil fixe. Seuil $h$ calibré automatiquement (FPR cible 1 %).

### Réseau pandapower V2

**Topologie :** Transformateur 20/0,4 kV (250 kVA, Dyn11) → 7 bus, 5 lignes BT aluminium, 3 départs (Admin 30%, Labos/HVAC 50%, Résidence+EV 20%).  
**Calcul :** Newton-Raphson (convergence ≤ 50 itérations).  
**Substitut validé :** ajusté sur 2 000 états réels, erreur maximale < 10 kW sur 500 états de test **indépendants** (distribution légèrement hors domaine d'ajustement). Surrogate validity gate : retour au solveur si hors domaine.

---

## 4.2 Goal 2 — Confiance & Explainabilité

| Technique | Fonction | Implémenté |
|---|---|---|
| Calibration isotonique (one-vs-rest) | Corriger le biais de confiance du Naive Bayes | ✅ `calibration.py` |
| ECE, Brier Score, Reliability Diagram | Mesurer la qualité de calibration | ✅ `calibration.py` |
| Prédiction sélective (abstention) | S'abstenir si confiance < 90 % | ✅ `risk_policy.py` |
| Contributions bayésiennes | Identifier les preuves déterminantes | ✅ `engine.py` |
| SHAP TreeExplainer | Expliquer le modèle LightGBM PV | ✅ `engine.py` |
| Active Evidence Acquisition | Choisir la prochaine mesure utile | ✅ `engine.py` |

**Critère AEA :** $m^* = \arg\max_m \left[ H(H \mid E) - \mathbb{E}_{y_m}[H(H \mid E, y_m)] - \lambda C(m) \right]$

---

## 4.3 Goal 3 — Gouvernance Humain-IA

**Policy Engine V2 — décision à perte espérée :**

$$\text{EL}[a] = \sum_h P(H_h) \times L[a, h]$$

où $L[a,h]$ est la perte si l'action $a$ est prise quand l'hypothèse $h$ est vraie.

| Situation | Action |
|---|---|
| Confiance < 90 % | **Abstention → revue humaine** + prochaine vérification |
| Capteur défaillant, confiance ≥ 95 % | Quarantaine automatique du canal (données uniquement) |
| 5 % de ces décisions auto | Échantillon audit (anti-surconfiance) |
| Défaut PV physique, événement réseau | Recommandation + revue humaine obligatoire |
| Falsification coordonnée | Escalade humain + sécurité, toujours |

**Journal d'audit :** SQLite avec chaîne de hachage SHA-256 (Genesis → bloc N). Falsification détectée → test unitaire vérifié.

---

## 4.4 Goal 4 — Noosphère

| Élément | Implémentation |
|---|---|
| Format | Schéma Pydantic 2.0, JSON sérialisable |
| Confidentialité | Bruit de Laplace (ε-DP), sensibilité = 1/√N |
| k-anonymat | Classe non partagée si < 5 événements |
| Dérive de distribution | PSI (seuil 0,25) + distance de Wasserstein |
| Import prudent | Rejet si schéma ≠, non validé humain, capteurs manquants, dérive PSI > 0,25 |
| Adaptation | Shrinkage pondéré : µ_new = (1−w)µ_local + w×µ_remote |
| Second nœud | Nœud "usine" simulé (PV 30 kWp, profil industriel, sans capteur onduleur) |

**Partagé :** signatures normalisées (µ/σ en unités sigma), capteurs requis, conditions de validité, n_events bruité, validation humaine.  
**Local :** séries brutes, factures, profil exact de charge, topologie, identité des validateurs.

---

# 5. Résultats Phase 1 V2 (5 graines, IC 95 %)

> Test : 72 événements injectés/graine, télémétrie onduleur masquée.

| Méthode | Rappel événementiel | Précision | FPR |
|---|---|---|---|
| A. Isolation Forest brut | ~4–7 % | ~0,84 | ~0,03 % |
| B. Physique seule | ~55–65 % | ~0,89 | ~0,42 % |
| C. IsoForest sur résidus | ~60–68 % | ~0,87 | ~0,41 % |
| CUSUM | ~80–88 % (dérive) | ~0,09 | ~50 % |
| **EnergyMind** | **~88–93 %** | **~0,89** | **~0,57 %** |

*(Les valeurs exactes avec IC 95 % sont dans `outputs/metrics_v2.json`)*

**Diagnostic (quand le système décide) :** précision ≥ 98 %, ECE calibré ≤ 0,015, taux d'abstention ~26 %.

> [!WARNING]
> **Limitations honnêtes Phase 1**
> 1. Le générateur de défauts calibre aussi le modèle (même distribution) — résultats optimistes jusqu'à validation sur données réelles.
> 2. Précision parfaite sur petits échantillons (< 20 events/classe) — les IC 95 % sont larges.
> 3. Le Naive Bayes suppose l'indépendance des résidus — simplification Phase 1.
> 4. Le second nœud "usine" est simulé, pas mesuré.
> 5. Taux de revue humaine ~49 % — à réduire en Phase 2 par hysteresis et seuils adaptatifs.

---

# 6. Matrice de traçabilité

| Exigence Challenge | Implémentation V2 | Fichier | Preuve |
|---|---|---|---|
| Scope défini | Microgrid Campus Universitaire 400V BT | `network_v2.py`, `config.yaml` | Schéma unifilaire Fig 1 |
| Datasets réels | PV 5 min + factures STEG | `data/raw/` | 238 464 lignes tracées |
| Anomalies documentées | 8 classes, registre horodaté | `faults_v2.py`, `outputs/test_event_registry.csv` | Catalogue §3.2 |
| Goal 1 — Isolation Forest | Comparaison A (baseline) | `evaluate_v2.py` | Fig 4, metrics_v2.json |
| Goal 1 — Statistical outlier | CUSUM bidirectionnel | `drift.py` | Fig 4, Fig 6 |
| Goal 1 — Détection & biais | LightGBM + Bayesian + résidus physiques | `engine.py`, `faults_v2.py` | Benchmark Fig 4 |
| Goal 2 — SHAP | SHAP TreeExplainer LightGBM | `engine.py` | Onglet UI Fig 2 |
| Goal 2 — Readable logs | Explications langage naturel + contributions | `governance.py` | Journal audit |
| Goal 2 — Calibration | Isotonique, ECE, Brier, Reliability Diagram | `calibration.py` | Fig 4, reliability_data_v2.csv |
| Goal 2 — Confidence routing | Prédiction sélective + AEA | `engine.py`, `risk_policy.py` | Fig 6, Fig 7 |
| Goal 3 — Rule-based thresholds | Policy à perte espérée | `risk_policy.py` | Fig 7 |
| Goal 3 — Human review | Interface Streamlit + approbation | `app/operator_ui.py` | Onglet Gouvernance |
| Goal 3 — Audit trail | SQLite SHA-256 chaîné | `governance.py` | `outputs/audit.db` |
| Goal 4 — Sharing protocol | Pydantic schema + ε-DP + PSI | `noosphere_v2.py` | Fig 8 |
| Goal 4 — Federated concepts | Shrinkage pondéré (lightweight) | `noosphere_v2.py` | Fig 8 |
| Evaluation — Precision/Recall | 5 méthodes comparées, 5 graines, IC 95 % | `evaluate_v2.py` | metrics_v2.json |
| Evaluation — FPR | Calculé et reporté | `evaluate_v2.py` | metrics_v2.json |
| Evaluation — Response latency | Latence médiane en pas de temps | `evaluate_v2.py` | metrics_v2.json |
| Reproducibility | `python scripts/run_v2.py` → tous les chiffres | `scripts/run_v2.py` | README.md |

---

# 7. Reproductibilité

```bash
pip install -r requirements.txt
python scripts/run_v2.py        # → outputs/metrics_v2.json + 8 figures
pytest                          # → 21 tests passent
streamlit run app/operator_ui.py
```

Graines, découpages chronologiques, statuts de provenance et registres d'événements versionnés dans le dépôt.

---

# 8. Plan Phase 2

| Priorité | Action |
|---|---|
| Critique | Données mesurées Campus Universitaire (Pnom PV, BESS, onduleur, compteurs PCC) |
| Critique | Validation sur défauts réels (si disponibles) |
| Important | Réseau bayésien structuré (remplace Naive Bayes indépendant) |
| Important | Réduction fausses alarmes (hysteresis, seuils adaptatifs) |
| Important | Classe "inconnu / hors-distribution" |
| Important | Intervalles de confiance par classe (pas seulement moyenne) |
| Souhaitable | Deuxième nœud réel pour valider le transfert |
| Souhaitable | FastAPI service + Docker |

---

# 9. Périmètre business (critère bonus IEEE PES/IES)

| Élément | Estimation (à valider) |
|---|---|
| Logiciel | Stack open-source (Python, SQLite, Streamlit) : coût licence 0 |
| Déploiement edge | Mini-PC industriel / gateway Modbus : 1–3 k TND/nœud |
| Intégration | 2–4 semaines-ingénieur/site pour cartographier les capteurs |
| Maintenance | Recalibration mensuelle automatisée |
| Valeur | Pertes évitées sur défauts non détectés, décisions auditables |

*Tous les chiffres sont des hypothèses de planification, non des devis.*
