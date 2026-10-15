# EnergyMind — AI Trust Layer for Campus Microgrid Anomaly Detection

> **TSYP14 Technical Challenge — IEEE PES & IES — Phase 1 Project Proposal & Prototype**  
> *Track: Energy System Forensic Analysis, Anomaly Detection & Collective Defense*

[![Python](https://img.shields.io/badge/python-3.10.11-blue?logo=python)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-21%2F21%20PASS-brightgreen)](#reproducibility--testing)
[![FastAPI](https://img.shields.io/badge/FastAPI-SCADA%20Live%20Dashboard-009688?logo=fastapi)](http://localhost:8000)
[![Overleaf](https://img.shields.io/badge/Overleaf-Report%20Package%20Ready-47A141?logo=overleaf)](docs/energymind_overleaf_package.zip)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

EnergyMind is an **evidence-driven AI trust layer** designed to detect, diagnose, calibrate, govern, and collectively federate cyber-physical anomalies across distributed microgrids. Rather than treating microgrids as black boxes or relying exclusively on deep learning, EnergyMind combines **Kirchhoff AC power flow physics**, **machine learning residuals**, **Bayesian hypothesis testing**, **isotonic probability calibration**, and **differential privacy knowledge transfer** across all four TSYP14 challenge goals.

---

## 🏛️ End-to-End System Architecture

The complete system spans five interconnected layers, bridging physical electrical distribution with multi-node peer microgrid defense:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: PHYSICAL MICROGRID & DIGITAL TWIN (pandapower 7-Bus Radial Feeder)                 │
│  Bus 0: 20 kV Utility Slack ──[250 kVA Dyn11 Transfo]──► Bus 1: 0.4 kV Main Switchboard     │
│  ├── Feeder 1: Admin Building (15 kW)      ├── Bus 5: Inverter-Interfaced PV Array (50 kWp) │
│  ├── Feeder 2: Labs & HVAC (25 kW)         └── Bus 6: Battery Energy Storage BESS (40 kWh)  │
│  └── Feeder 3: Student Residence & EV (10 kW)                                               │
└──────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                           │ Real-time Telemetry (15-min / 5-min SCADA)
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 2: TELEMETRY & PHYSICAL SENSOR LAYER (Observed vs. True State)                        │
│  P_grid, P_pv, P_bess, P_load, V_pcc, V_inv, SoC_batt, GHI, T_amb, T_cell                  │
└──────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                           │ Raw Measurements + Environmental Telemetry
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 3: ENERGYMIND AI TRUST LAYER (Detection, Diagnostics & Calibration)                   │
│                                                                                             │
│  1. Physics & Residual Extraction (Z ∈ ℝ⁵ robust MAD-scaled z-scores):                      │
│     • bal: Energy conservation residual (P_grid + P_pv + P_bess - P_load - P_loss)          │
│     • pvexp: LightGBM weather-to-power surrogate error (GHI, T_cell, Wind, Hour → P_pv)     │
│     • inv: Inverter AC power meter vs. DC generation consistency                            │
│     • vpcc: PCC voltage deviation from pandapower AC power flow expected profile            │
│     • vinv: Terminal voltage residual across PV inverter feeder                             │
│                                                                                             │
│  2. Multi-Method Anomaly Detection Gate:                                                    │
│     • Fast Physical Thresholds    • Isolation Forest on Residuals    • Dual-Channel CUSUM   │
│                                                                                             │
│  3. Bayesian Multi-Hypothesis Diagnosis:                                                    │
│     Gaussian Naive Bayes over 8 scenario classes with missing-sensor marginalization:       │
│     P(H | Z) ∝ P(H) ∏ P(Z_j | H)                                                           │
│                                                                                             │
│  4. Uncertainty Quantification & Isotonic Calibration:                                      │
│     One-vs-Rest Isotonic Regression post-hoc mapper: uncalibrated conf → calibrated prob.   │
│     Achieves Expected Calibration Error (ECE) = 0.038 (Target ≤ 0.05)                       │
└──────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                           │ Calibrated Probabilities P*(H | Z)
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 4: GOVERNANCE & ACTIVE RISK POLICY ENGINE                                             │
│                                                                                             │
│  • Expected Loss Routing: a* = argmin_a ∑_h P*(h|Z) · L(a, h)                               │
│    Actions: MONITOR (log only) | QUARANTINE (isolate data stream) | RECOMMEND | ESCALATE    │
│  • Human-in-the-Loop Safeguards: mandatory operator sign-off on physical / cyber alerts    │
│  • Active Evidence Acquisition: next-best sensor query via Mutual Information Gain          │
│  • SHA-256 Hash-Chained Audit Ledger: tamper-evident cryptographic log with genesis hash   │
└──────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                           │ Anonymized Signatures (μ, σ, validity, ε-budget)
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 5: NOOSPHERE MULTI-NODE PEER FEDERATION (Goal 4 Collective Defense)                  │
│                                                                                             │
│              ┌────────────────────────────────────────────────────────┐                     │
│              │             Global Threat Registry Node                │                     │
│              │        (Federated Incident Catalog & Policies)         │                     │
│              └───────────────────────────┬────────────────────────────┘                     │
│                         ▲                │               ▲                                  │
│                         │                │               │                                  │
│          ┌──────────────┴───────┐        │        ┌──────┴───────────────┐                  │
│          │  Campus Microgrid    │◄───────┼───────►│  Industrial Facility │                  │
│          │  (50 kWp PV, BESS)   │        │        │  (30 kWp PV, Heavy)  │                  │
│          └──────────────┬───────┘        │        └──────┬───────────────┘                  │
│                         │                │               │                                  │
│                         ▼                ▼               ▼                                  │
│          ┌──────────────────────┐                 ┌──────────────────────┐                  │
│          │  Hospital Node       │                 │  Commercial Smart    │                  │
│          │  (45 kWp, Critical)  │                 │  Building (25 kWp)   │                  │
│          └──────────────┬───────┘                 └──────┬───────────────┘                  │
│                         │                                │                                  │
│                         └───────────────┬────────────────┘                                  │
│                                         ▼                                                   │
│                          ┌──────────────────────────────┐                                   │
│                          │  Rural Islanded Community    │                                   │
│                          │  (40 kWp PV, 90 kWh Storage) │                                   │
│                          └──────────────────────────────┘                                   │
│                                                                                             │
│  Guarantees:                                                                                │
│  • Differential Privacy: Laplace mechanism on signature centroids (ε = 1.0)                │
│  • k-Anonymity: signature suppression when incident count n < 5                             │
│  • Domain Shift Gating: Population Stability Index (PSI) & Wasserstein metric test          │
│  • Zero Raw Telemetry Leakage: only dimensionless z-scores and validity metadata shared     │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

> **Standalone Vector Architecture Diagram**: A full TikZ color-coded vector source is provided at [`docs/overleaf/architecture.tex`](docs/overleaf/architecture.tex).

---

## 📊 Benchmark Results (5 Random Seeds, 95% Confidence Intervals)

All figures and tables reflect multi-seed evaluation across 5 independent test partitions with zero data leakage:

| Challenge Goal & Metric | EnergyMind V2 Result | Baseline / Target | Status |
|:-----------------------|:--------------------:|:-----------------:|:------:|
| **Goal 1: Event Detection Recall** | **58.1%** [54.5%, 61.7%] | IsoForest: 43.1%, Pure Physics: 34.7% | **Highest (+15.0%)** |
| **Goal 1: Detection Precision** | **91.2%** [90.8%, 91.7%] | Baseline: ~78.4% | **+12.8%** |
| **Goal 1: False Alarms / Day** | **0.173** (1 per 5.8 days) | Target: < 0.50 / day | **Exceeded** |
| **Goal 2: Expected Calibration Error (ECE)** | **0.038** | Uncalibrated: 0.142 (Target ≤ 0.05) | **Calibrated** |
| **Goal 2: Decided Case Accuracy** | **100.0%** (5/5 seeds) | Forced Choice: 88.9% | **Zero Error** |
| **Goal 2: Pandapower Surrogate MAE** | **0.096 kW** (Max: 0.41 kW) | Target: < 0.50 kW | **Exceeded** |
| **Goal 3: Human Review Escalation Rate** | **72.1%** of alerts | Strict safety mandate for physical/cyber | **Verified** |
| **Goal 3: SHA-256 Audit Chain Verification** | **100% Cryptographic Match** | Genesis tamper-detection test | **Passed** |
| **Goal 4: Cross-Node Signature Export** | **7 signatures (ε = 1.0)** | Privacy budget enforced | **Verified** |
| **Goal 4: Cross-Node Signature Import** | **4 imported, 3 rejected** | Sensor/schema/drift gates | **Secure** |

---

## ⚡ Quick Start

### 1. Environment Setup
```powershell
# Clone and enter directory
cd energymind

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies (pandapower, lightgbm, fastapi, uvicorn, shap, etc.)
pip install -r requirements.txt
```

### 2. Run Complete Phase 1 Pipeline (Multi-Seed Experiments)
```powershell
$env:PYTHONUTF8=1
python scripts/run_v2.py
```
*Outputs `outputs/metrics_v2.json`, `reliability_data_v2.csv`, and event registries.*

### 3. Generate All Publication-Quality Figures (150 DPI)
```powershell
python scripts/make_figures_v2.py
```
*Produces Figures 1 through 8 in `outputs/` and copies them to `docs/overleaf/figures/`.*

### 4. Run Complete Test Suite
```powershell
pytest tests/test_core.py -v
```
*Runs all 21 unit tests covering AC power flow, energy balance, isotonic calibration, audit hashing, and federated transfer.*

### 5. Launch Live SCADA FastAPI Dashboard
```powershell
uvicorn app.dashboard:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser at **[http://localhost:8000](http://localhost:8000)** to experience the real-time interactive platform:
- **Tab 1: Live SCADA & Physics Residuals**: Real-time pandapower simulation, single-line diagram, dynamic residual meters ($Z \in \mathbb{R}^5$), and live anomaly injection controls (Grid sag, PV drift, Degradation, Coordinated falsification).
- **Tab 2: Trust, Governance & SHA-256 Ledger**: Interactive calibrated posteriors, expected-loss action routing, and tamper-evident audit table updating live with cryptographic hashes.
- **Tab 3: Noosphere Multi-Node Federation**: Live status of 5 federated peer nodes (Campus, Factory, Hospital, Smart Building, Rural Microgrid), differential privacy sliders ($\varepsilon$), and domain shift acceptance matrix.

---

## 📄 Overleaf Single-Column Report Package

A complete, self-contained Overleaf project package is prepared in:
- **`docs/energymind_overleaf_package.zip`** (Direct upload to Overleaf)
- Source folder: **`docs/overleaf/`**
  - `main.tex`: Formatted single-column technical proposal and report (~7 pages, article class, standard 2.5 cm margins, no two-column IEEE format).
  - `architecture.tex`: Standalone TikZ color architecture diagram.
  - `figures/`: All 8 high-resolution PNG figures.

To compile on Overleaf:
1. Log in to [Overleaf](https://www.overleaf.com/).
2. Click **New Project** $\rightarrow$ **Upload Project**.
3. Select `energymind_overleaf_package.zip`.
4. Click **Recompile** with pdfLaTeX.

---

## 🌐 Fault Taxonomy (8 Diagnostic Classes)

| Code | Diagnostic Class | Physical vs Telemetry | Mechanism & Signature |
|:---:|:---|:---:|:---|
| **P2** | `pv_fault` | Physical | Inverter clipping or module degradation: $P_{\text{pv}}$ drops, grid compensates, $r_{\text{pvexp}} < 0$. |
| **G1** | `grid_event` | Physical | Upstream utility voltage sag: $V_{\text{pcc}}$ and $V_{\text{inv}}$ drop simultaneously by 8–18%. |
| **L1** | `load_peak` | Physical | Genuine demand surge (+40% to +80%): balance holds, grid compensates, no sensor discrepancy. |
| **M1** | `pv_sensor_fault` | Telemetry | Fixed scaling bias on PV sensor: $r_{\text{bal}} \ne 0$, $r_{\text{inv}} \ne 0$, weather prediction mismatches. |
| **M2** | `pv_sensor_fault` | Telemetry | Progressive sensor drift: detected via dual CUSUM change-point detector before threshold triggers. |
| **V1** | `voltage_sensor_fault` | Telemetry | Local PCC voltage transducer offset: $V_{\text{pcc}}$ diverges while inverter terminal $V_{\text{inv}}$ remains normal. |
| **B1** | `bess_sensor_fault` | Telemetry | State-of-charge (SoC) sensor bias: causes false dispatch commands and apparent storage imbalance. |
| **A1** | `coordinated_falsification`| Telemetry | Multi-sensor cyber spoofing: $P_{\text{pv}}$ decreased while $P_{\text{grid}}$ is increased symmetrically to preserve basic power balance. Unmasked by LightGBM weather surrogate and inverter cross-check. |

---

## 📁 Repository Structure

```
energymind/
├── app/
│   └── dashboard.py                  # Live SCADA FastAPI platform (3 interactive tabs, no CDN)
├── configs/
│   └── config.yaml                   # Experiment parameters, network cables, seeds, noise models
├── data/
│   ├── raw/                          # Real campus PV data (238,464 rows) & invoices (2022-2024)
│   ├── processed/                    # Calibrated 15-min baseline and test telemetry datasets
│   └── reference/                    # NASA POWER hourly and PVGIS monthly benchmarks
├── docs/
│   ├── energymind_overleaf_package.zip # Ready-to-upload complete Overleaf zip bundle
│   ├── REPORT_PHASE1.md              # Complete Markdown technical report
│   ├── REPORT_PHASE1.tex              # Single-column LaTeX technical proposal
│   ├── DATA_PROVENANCE.md            # Dataset sources, units, and validation report
│   ├── MODEL_SELECTION.md            # Engineering justifications for all models
│   └── overleaf/                     # Overleaf project source tree (main.tex, architecture.tex, figures/)
├── outputs/                          # Generated figures (fig01–fig08), metrics_v2.json, audit database
├── scripts/
│   ├── run_v2.py                     # Multi-seed evaluation runner
│   ├── make_figures_v2.py            # High-resolution figure generation script
│   └── fetch_reference_benchmarks.py # Reference data downloader (PVGIS / NASA POWER)
├── src/energymind/                   # Core Python library
│   ├── calibration.py                # Isotonic probability calibration & ECE metrics
│   ├── engine.py                     # LightGBM surrogate, residual extraction & Bayesian diagnoser
│   ├── faults_v2.py                  # 8-class anomaly simulation & ground-truth injection
│   ├── governance.py                 # SHA-256 audit ledger & explainability
│   ├── network_v2.py                 # pandapower 7-bus AC power flow & polynomial surrogate
│   ├── noosphere_v2.py               # Multi-node peer federation, DP export & domain shift gating
│   ├── risk_policy.py                # Expected loss decision engine & risk-coverage curve
│   └── simulate.py                   # Baseline microgrid load synthesizer
└── tests/
    └── test_core.py                  # 21 unit tests (all passing)
```

---

## 🛡️ Reproducibility & Testing

EnergyMind enforces strict determinism and reproducible evaluation. No test data is used in surrogate training or probability calibration.

```bash
# Verify unit tests
pytest tests/test_core.py -v
```

All 21 test routines validate:
- Energy conservation ($P_{\text{gen}} = P_{\text{load}} + P_{\text{loss}}$)
- Battery state-of-charge constraints ($20\% \le \text{SoC} \le 90\%$)
- Isotonic posterior normalization ($\sum_k P_k = 1.0$)
- SHA-256 audit chain tamper detection
- Pandapower AC power flow convergence and balance residuals
- Differential privacy signature format and k-anonymity gate
- Cross-node schema and sensor validation

---

## 📜 Citation & Credits

Developed for the **TSYP14 Technical Challenge — IEEE PES & IES** (Phase 1).  
Designed under anonymous team submission guidelines (*Campus Universitaire Microgrid*).
