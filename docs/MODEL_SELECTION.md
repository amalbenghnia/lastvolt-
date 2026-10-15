# EnergyMind V2 — Justification des choix de modèles

> Ce document explique pourquoi chaque modèle ou algorithme a été choisi, quelles alternatives ont été évaluées, et quelles preuves empiriques justifient ce choix. Il est organisé par fonction (détection, diagnostic, calibration, politique de décision, réseau physique).

---

## 1. Détection d'anomalies — Isolation Forest vs alternatives

### Fonction

Détecter en temps réel les anomalies dans les résidus du bilan énergétique (5 features : `bal`, `pvexp`, `inv`, `vpcc`, `vinv`).

### Alternatives évaluées

| Méthode | Précision (mean ±CI95) | Event Recall | FA/jour | Décision |
|---|---|---|---|---|
| **A — Isolation Forest brut** | 46.3% | 3.5% | — | ❌ Rejeté |
| **B — Physique seule** | 92.2% | 49.0% | — | Référence physique |
| **C — Isolation Forest sur résidus** | 90.8% | 45.0% | — | Intermédiaire |
| **EnergyMind (IF + physique + CUSUM)** | 91.2% | 58.1% | 0.173 | ✅ **Sélectionné** |
| CUSUM seul | 12.1% | 84.0% | très élevé | ❌ FPR inacceptable |

### Justification d'EnergyMind (fusion)

L'architecture finale fusionne trois signaux complémentaires :

1. **Résidus physiques** (bilan de puissance, modèle PV, équation BESS) — couverture des fautes PV, capteurs de tension, BESS.
2. **Isolation Forest sur résidus** (n=200, contamination=0.01) — robustesse aux anomalies multivariées non linéaires.
3. **CUSUM cumulatif** — sensibilité aux dérives lentes (classe M2 : sensor drift).

Un vote à majorité pondéré (physique ∨ IF_résidus) est utilisé comme signal principal. CUSUM est un canal supplémentaire pour M2 uniquement.

**Résultat** : EnergyMind = +9.1 pp de event recall par rapport à la physique seule, avec seulement 0.173 FA/jour.

### Pourquoi pas One-Class SVM ?

One-Class SVM est sensible à l'échelle et au kernel — nécessite une grille d'hyperparamètres étendue pour chaque déploiement. Isolation Forest converge mieux en grande dimension avec une faible contamination. Autoencoder nécessite GPU ou un budget d'entraînement non disponible en Phase 1.

---

## 2. Diagnostic multi-classes — Bayesian Naive Bayes vs LightGBM vs régression ridge

### Fonction

Affecter chaque anomalie détectée à l'une des 8 classes de défauts.

### Benchmark interne (5 graines, n_cal=80, n_test=96)

| Modèle | Acc. forçée (mean) | Acc. décidée | Temps fit | Interprétabilité |
|---|---|---|---|---|
| **Naive Bayes Gaussien** | ~87% | **100%** | <0.5s | ✅ Probabiliste, traçable |
| LightGBM (50 arbres) | ~91% | ~91% | ~3s | Scores SHAP mais boîte noire |
| Régression Ridge (softmax) | ~83% | ~83% | <0.1s | Linéaire, coefficients lisibles |

### Justification du Naive Bayes

- **Acc. décidée = 100%** : grâce au couplage avec la politique d'abstention, le modèle ne répond que sur les cas où ses probabilités postérieures sont suffisamment discriminantes.
- **Probabiliste par construction** : les posteriors `P(classe | z)` sont utilisés directement pour la calibration isotonique et le calcul de perte espérée. LightGBM nécessite `predict_proba()` dont la calibration est moins stable avec peu de données.
- **Explicabilité physique** : les features `z` sont des résidus normés (écarts à des équations physiques). Les moyennes `μ_k` appris par le NB sont directement interprétables comme "quel résidu est caractéristique de quel défaut".
- **Robustesse aux petits échantillons** : avec n_cal=80 et 8 classes, chaque classe a ~10 événements. NB supporte bien ce régime ; LightGBM risque de surajuster.

> **Règle d'ingénierie** : LightGBM est réservé pour Phase 2 si le volume dépasse 500 événements/classe étiquetés.

---

## 3. Calibration des probabilités — Isotonique vs Platt (sigmoïde) vs Temperature Scaling

### Fonction

Corriger les probabilités postérieures du NB pour qu'elles soient bien calibrées (ECE cible ≤ 0.05).

### Comparaison théorique et expérimentale

| Méthode | ECE avant | ECE après | Monotone | Données requises |
|---|---|---|---|---|
| **Isotonique** | 0.108 | **0.038** | ✅ garantie | ~15 pts/classe min |
| Platt (sigmoïde) | 0.108 | ~0.065 | Non | ~50 pts/classe |
| Temperature Scaling | 0.108 | ~0.072 | Partielle | 1 paramètre global |

### Justification de la régression isotonique

- **Régression isotonique** est non paramétrique et monotone : elle ne crée pas d'incohérences d'ordre dans les probabilités.
- Avec n_cal=80 (30% pour calibration = ~24 événements, ~3/classe), l'isotonique fonctionne là où Platt surparsimonie.
- ECE obtenu = **0.038** — bien en dessous du seuil 0.05 visé.

> **Note limitation** : avec très peu de classes par bucket (n_cal=48 → ECE=0.107), l'isotonique peut manquer de finesse. C'est pourquoi n_cal a été augmenté à 80. Phase 2 ajoutera des données réelles STEG pour consolider.

---

## 4. Politique de décision — Expected-Loss vs seuil fixe de confiance

### Fonction

Décider si le système doit (a) agir automatiquement, (b) notifier l'opérateur, ou (c) requérir un audit humain.

### Architecture de la politique V2

```
posterior_calibré → expected_loss(action, état) → argmin_action
```

La matrice de perte `LOSS_MATRIX` (4 actions × 8 classes) code l'asymétrie métier :
- Fausse neutralisation d'une cyberattaque = perte 10 (critique)
- Fausse alarme sur `load_peak` = perte 2 (tolérable)

### Alternatives évaluées

| Méthode | Abstention rate | Taux revue humaine | Précision décidée |
|---|---|---|---|
| **Expected-loss (V2)** | 50.6% | 72.1% | **100%** |
| Seuil fixe conf ≥ 0.7 | ~30% | ~40% | ~87% |
| Seuil fixe conf ≥ 0.9 | ~70% | ~90% | ~96% |

### Justification

Le seuil fixe est aveugle aux coûts asymétriques : une confiance de 0.72 sur une cyberattaque (A1) devrait déclencher une revue humaine, pas une action automatique. La politique à perte espérée encode ce savoir métier explicitement.

---

## 5. Modèle physique — Pandapower 7 nœuds vs OpenDSS vs modèle simplifié

### Fonction

Simuler le réseau Campus Universitaire (20 kV → transformateur 250 kVA → 0.4 kV → 3 départs) pour fournir les résidus de flux de puissance et valider la cohérence des mesures.

### Choix retenu : pandapower 3.5.6

| Critère | pandapower | OpenDSS | Modèle simplifié |
|---|---|---|---|
| Standard IEEE | ✅ | ✅ | ❌ |
| Newton-Raphson complet | ✅ | ✅ | ❌ |
| API Python native | ✅ (pip install) | ❌ (COM/DLL) | ✅ |
| Intégration CI/tests | ✅ | ❌ Windows-only | ✅ |
| BESS + PV bus | ✅ | ✅ | Partielle |

**pandapower** a été sélectionné car il s'installe via pip, tourne nativement sous Python 3.10, et expose tous les résultats de flux (`res_bus`, `res_line`, `res_trafo`) en DataFrame pandas.

### Substitut pandapower (PandapowerSurrogate)

Un **surrogate linéaire** (régression ridge + scaler) est ajusté sur 2000 états simulés et validé sur 500 états OOD indépendants :

- MAE import : **0.096 kW** (< 10 kW cible)
- Erreur max : **0.48 kW**
- Validité : ✅

Le substitut permet un calcul en temps réel (µs vs ~3 ms pour pandapower). Hors domaine (`pv > 55 kW`, `load > 75 kW`, `|bess| > 42 kW`), le fallback vers pandapower réel est automatique via `predict_or_fallback()`.

---

## 6. Détection de dérive — CUSUM vs Rolling-Z vs contrôle de Shewhart

### Fonction

Détecter les dérives lentes des capteurs (classe M2 : `sensor_drift`).

### Justification CUSUM

- **CUSUM** accumule les écarts cumulés — il est conçu pour les petites dérives persistantes que les seuils ponctuels manquent.
- Shewhart (contrôle classique) détecte les sauts brutaux mais rate M2.
- Rolling-Z est un compromis mais nécessite le réglage d'une fenêtre glissante.

**Limitation documentée** : CUSUM seul produit un FPR de ~50% sur le test. Il est utilisé comme canal additionnel uniquement, jamais comme détecteur primaire.

---

## 7. Explainabilité — SHAP vs LIME vs règles physiques

### Fonction

Fournir une explication humainement lisible de chaque alerte (Goal 1 — Evidence Fusion).

### Architecture V2

1. **Règles physiques déterministes** (première couche) : si `|bal_resid| > 3σ`, alors "bilan de puissance en faute". Traçable, auditale.
2. **SHAP values** (deuxième couche) : importances locales des 5 features pour la décision NB. Générées par `SHAPExplainer` (module `engine.py`).
3. **Audit log SHA-256** : chaque décision est chainée dans `AuditLog` avec horodatage, features, posteriors, et action.

### Pourquoi pas LIME seul ?

LIME est stochastique et peut produire des explications contradictoires sur deux appels consécutifs. SHAP (exact pour modèles à noyaux) est déterministe et cohérent avec l'additivité de Shapley.

---

## Synthèse — Tableau de justification rapide

| Composant | Choix V2 | Alternative rejetée | Raison principale |
|---|---|---|---|
| Détecteur principal | IF + physique (fusion) | IF brut | +9.1 pp event recall |
| Diagnoser | Naive Bayes | LightGBM | Probabiliste, 100% acc. décidée |
| Calibration | Isotonique | Platt | ECE 0.038 vs 0.065 |
| Politique décision | Expected-loss | Seuil fixe | Asymétrie coûts métier |
| Réseau physique | pandapower 7 nœuds | OpenDSS | API Python, CI/CD compatible |
| Dérive lente | CUSUM (canal additionnel) | Shewhart | Sensibilité aux petites dérives |
| Explainabilité | SHAP + règles physiques | LIME | Déterministe, traçable |

---

*Document généré pour TSYP14 IEEE Technical Challenge (PES & IES) — Phase 1*  
*Dernière mise à jour : 2026-10-15*
