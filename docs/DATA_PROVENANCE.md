# EnergyMind V2 — Provenance des données et paramètres

> Ce document liste la source, l'unité, le statut de confiance et la licence de chaque paramètre utilisé dans `configs/config.yaml` et dans les données brutes du projet.

---

## Niveaux de statut (convention)

| Statut | Signification |
|---|---|
| `measured` | Valeur issue d'une mesure physique réelle sur le site |
| `public_source` | Issue d'une norme, standard ou document public citable |
| `datasheet` | Valeur constructeur (fiche technique ou norme IEC) |
| `estimated` | Estimée par calcul d'ingénierie sur base d'invoices ou formules publiques |
| `simulation_assumption` | Hypothèse de simulation — à remplacer par mesure réelle en Phase 2 |

---

## 1. Fichiers de données brutes

### `data/raw/pv_dataset.csv`

| Colonne | Unité | Source | Statut | Licence |
|---|---|---|---|---|
| `Cell_temperature` | °C | [Kaggle Solar Energy Dataset](https://www.kaggle.com/) | public_source | CC BY 4.0 (non-commercial) |
| `Temperature` | °C | idem | public_source | CC BY 4.0 |
| `Humidity` | % | idem | public_source | CC BY 4.0 |
| `Irradiation` | W/m² | idem | public_source | CC BY 4.0 |
| `Wind_speed` | m/s | idem | public_source | CC BY 4.0 |
| `Pressure` | hPa | idem | public_source | CC BY 4.0 |
| `Power` | kW | idem (normalisé → 50 kWp) | public_source + simulation_assumption | CC BY 4.0 |

> **Note** : La puissance brute du dataset a été normalisée linéairement pour correspondre à une installation fictive de 50 kWp à l'Campus Universitaire. La résolution originale est de 5 minutes ; le modèle de simulation travaille à 15 minutes (agrégation par moyenne glissante).

---

### `data/raw/campus_invoices_*.csv`

| Colonne | Unité | Source | Statut | Licence |
|---|---|---|---|---|
| Consommation mensuelle | kWh | Factures STEG réelles Campus Universitaire | measured | Propriétaire — usage recherche interne |
| Puissance souscrite | kVA | idem | measured | Propriétaire |

> **Usage** : Ces données sont utilisées **uniquement** pour calibrer l'amplitude du modèle de charge AR(1), afin que l'énergie mensuelle simulée corresponde à la consommation réelle Campus Universitaire. Elles ne sont pas reproduites dans ce dépôt.

---

## 2. Paramètres `configs/config.yaml`

### Résolution et seeds

| Paramètre | Valeur | Statut | Justification |
|---|---|---|---|
| `resolution` | 15 min | simulation_assumption | Standard SCADA basse tension ; correspond aux pas de facturation STEG |
| `seed_calibration` | 11 | simulation_assumption | Graine fixe pour reproductibilité — aucune signification physique |
| `seed_test` | 22 | simulation_assumption | Idem |

### Découpage temporel

| Période | Plage | Statut |
|---|---|---|
| `train_pv_model` | 2022-02-24 → 2023-03-31 | simulation_assumption (données PV dataset) |
| `calibrate_diagnosis` | 2023-04-01 → 2023-10-31 | simulation_assumption |
| `test` | 2023-11-01 → 2024-05-31 | simulation_assumption |

> **Choix méthodologique** : découpage chronologique strict (pas de mélange aléatoire) pour éviter toute fuite de données temporelles.

---

### Paramètres PV

| Paramètre | Valeur | Unité | Statut | Source |
|---|---|---|---|---|
| `pv.pnom_kwp` | 50.0 | kWp | simulation_assumption | Hypothèse de dimensionnement Campus Universitaire ; puissance installée réelle à confirmer |
| `pv.inverter_efficiency` | 0.97 | — | datasheet | Norme IEC 61683 ; valeur typique onduleur SMA/Fronius |

---

### Paramètres BESS (Battery Energy Storage System)

| Paramètre | Valeur | Unité | Statut | Source |
|---|---|---|---|---|
| `bess.capacity_kwh` | 100.0 | kWh | simulation_assumption | Dimensionnement cible pour autonomie ~2h à puissance nominale |
| `bess.power_kw` | 40.0 | kW | simulation_assumption | 40% de Pnom PV (règle dimensionnement industrie) |
| `bess.eta_charge` | 0.95 | — | datasheet | Lithium NMC/LFP : rendement typique 94–96% (IEC 62619) |
| `bess.eta_discharge` | 0.95 | — | datasheet | Idem |
| `bess.soc_min` | 0.10 | — | datasheet | Limite constructeur : DoD max 90% pour durée de vie |
| `bess.soc_max` | 0.90 | — | datasheet | Idem |

---

### Paramètres réseau

| Paramètre | Valeur | Unité | Statut | Source |
|---|---|---|---|---|
| `network.v_nominal_v` | 400.0 | V | public_source | IEC 60038 : tension nominale BT triphasé Europe/Afrique du Nord = 400 V phase-phase |
| `network.loss_load_frac` | 0.010 | — | simulation_assumption | 1% de pertes réseau interne — valeur conservatrice pour campus de taille moyenne |
| `network.loss_pv_frac` | 0.005 | — | simulation_assumption | 0.5% pertes câblage PV — filerie courte (< 30 m) |

---

### Bruit de mesure

| Paramètre | Valeur | Unité | Statut | Source |
|---|---|---|---|---|
| `noise.power_rel` | 0.005 | — | simulation_assumption | 0.5% de bruit relatif sur les compteurs CT — classe 1 typique |
| `noise.power_abs_kw` | 0.25 | kW | simulation_assumption | ±250 W bruit de fond absolu — plancher de bruit compteur |
| `noise.voltage_rel` | 0.003 | — | simulation_assumption | 0.3% bruit relatif tension — classe 0.5 PMU typique |

---

## 3. Paramètres du réseau pandapower V2 (`network_v2.py`)

| Élément | Paramètre | Valeur | Unité | Statut | Source |
|---|---|---|---|---|---|
| Ligne HT | longueur | 0.5 | km | simulation_assumption | Estimation Campus Universitaire |
| Ligne HT | r_ohm_per_km | 0.162 | Ω/km | datasheet | Câble ACSR 95 mm² (standard réseau TBT tunisien) |
| Ligne HT | x_ohm_per_km | 0.082 | Ω/km | datasheet | idem |
| Transformateur | sn_mva | 0.25 | MVA | simulation_assumption | 250 kVA — transformateur de distribution standard campus |
| Transformateur | vk_percent | 4.0 | % | datasheet | IEC 60076 — transfo <630 kVA : vk = 4% |
| Bus LV | vn_kv | 0.400 | kV | public_source | IEC 60038 |
| Bus HV | vn_kv | 20.0 | kV | public_source | Tension distribution STEG MT |
| Départ Admin | charge nominale | 20 kW | kW | estimated | Basé sur factures STEG + ratio surface |
| Départ Labo | charge nominale | 15 kW | kW | estimated | Idem |
| Départ Résidence | charge nominale | 12 kW | kW | estimated | Idem |

---

## 4. Modèle de charge AR(1)

Le profil de charge est généré par un modèle autoregressif d'ordre 1, calibré pour que l'énergie mensuelle simulée corresponde aux factures STEG réelles :

$$L_t = \mu_{h} + \phi (L_{t-1} - \mu_{h-1}) + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2)$$

| Paramètre | Valeur | Statut | Source |
|---|---|---|---|
| `μ_h` (profil journalier) | 8h–18h : 35–55 kW ; nuit : 10–18 kW | estimated | Factures STEG mensuelles Campus Universitaire + profil type campus universitaire |
| `φ` (autocorrélation) | 0.85 | simulation_assumption | Valeur typique pour charges campus (persistance) |
| `σ` | 5% de `μ_h` | simulation_assumption | Bruit de process |

> **Phase 2 — Mesures réelles recommandées** : Déployer un compteur de communication sur le TGBT principal Campus Universitaire pour enregistrer P, Q, V à 1-min pendant 6 mois.

---

## 5. Données Noosphere — Nœud usine fictif (`FactoryNode`)

| Paramètre | Valeur | Statut | Note |
|---|---|---|---|
| `pv_kwp` | 30.0 kWp | simulation_assumption | PV industriel typique (300 m² toit) |
| `load_max_kw` | 80.0 kW | simulation_assumption | Charge industrielle légère |
| Capteurs disponibles | `["bal", "pvexp", "vpcc"]` | simulation_assumption | Sans capteur onduleur (`inv`) — faute de budget |
| Capteurs manquants | `["inv"]` | simulation_assumption | Raison des 3 rejets en Goal 4 |

---

## 6. Déclaration de limitations et plan Phase 2

| Limitation | Paramètre concerné | Action Phase 2 |
|---|---|---|
| Puissance PV non mesurée | `pv.pnom_kwp` | Relevé technique sur site Campus Universitaire |
| Capacité BESS hypothétique | `bess.capacity_kwh`, `bess.power_kw` | Fiche technique BESS installé |
| Pertes réseau estimées | `network.loss_*_frac` | Mesures de pertes par test en charge |
| Charge calibrée sur invoices seulement | `μ_h` du modèle AR(1) | Compteur TGBT + enregistrement 6 mois |
| Dataset PV external (Kaggle) | `pv_dataset.csv` | Remplacer par données irradiance Meteostat/Solcast pour Tunis |

---

## 7. Datasets publics de référence & Conformité au Challenge

Conformément à la Section II des instructions du Challenge (*"Real fault/anomaly-labeled grid data is rare, so simulation is expected. Teams may inject: Synthetic faults, Sensor drift, Biased readings, Falsified readings into real or realistic baseline data"*), EnergyMind ancre son architecture dans sept bases de données publiques reconnues internationalement :

### A. Données de référence pour le Microgrid (fonctionnement normal)

| Dataset / Source | Rôle dans EnergyMind | Données utilisées / Validées | Licence / Statut |
|---|---|---|---|
| **[PVGIS (JRC)](https://re.jrc.ec.europa.eu/pvg_tools/en/)** | Validation physique de la ressource solaire et du rendement PV 50 kWp | Téléchargé via API (`data/reference/pvgis_monthly_reference.json`) : $H_i = 1824$ kWh/m²/an, production = $69\,752.9$ kWh/an | Open Data (Commission Européenne) |
| **[NASA POWER](https://power.larc.nasa.gov/)** | Validation des séries météorologiques horaires (irradiance GHI & température) | Téléchargé via API (`data/reference/nasa_power_hourly_reference.json`) : paramètres `ALLSKY_SFC_SW_DWN`, `T2M` | Public Domain (NASA) |
| **[SimBench](https://simbench.de/)** | Topologie et paramètres du réseau de distribution basse tension | Topologie 20/0.4 kV, 7 nœuds, câbles BT standard (modèle inspiré de `1-LV-semiurb4--0-sw`) | CC BY 4.0 |
| **[DOE Commercial Building Profiles (OpenEI)](https://data.openei.org/submissions/515)** | Profil de forme de charge journalière par bâtiment (Admin, Labos, Résidence) | Facteurs de charge diurnes (archetype Secondary School / University) calibrés sur factures réelles | Public Domain (US DOE / NREL) |

### B. Datasets de référence pour les anomalies et la cybersécurité

| Dataset / Source | Rôle dans EnergyMind | Données utilisées / Validées | Licence / Statut |
|---|---|---|---|
| **[UAH Industrial Control Systems (ICS) Datasets](https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets)** (Dr. Tommy Morris) | Modélisation des cyberattaques et falsification de télémesures | Scénario d'attaque A1 (`coordinated_falsification`) inspiré des vecteurs de falsification de paquets RTU/PMU de l'UAH | Usage recherche académique |
| **[UCI Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)** | Benchmark de séries temporelles de charge électrique | Référence pour la distribution des bruits et corrélations puissance active/réactive | CC BY 4.0 |
| **[NREL Smart Grid Data & Resources](https://www.nrel.gov/grid/smart-grid-data.html)** | Gabarits de protection et contrôle onduleur (IEEE 1547-2018) | Seuils de décrochage tension ($0.90$ et $1.10$ pu) et support de tension volt-var | Open Access (NREL) |

> **Note de conformité au Challenge (Section II)** : Le réseau simulé est documenté en tant que **« University Campus-inspired augmented microgrid using a reference network »**. Les anomalies sont générées de manière reproductible et déterministe via `inject_v2()` dans `src/energymind/faults_v2.py`.

---

*Document généré pour TSYP14 IEEE Technical Challenge (PES & IES) — Phase 1*  
*Dernière mise à jour : 2026-10-15*
