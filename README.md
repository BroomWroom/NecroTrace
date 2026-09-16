# NecroTrace

<div align="center">
  <img src="assets/logo_transparent.png" alt="NecroTrace Logo" width="380" />
  <br/>
  <h3>Forensic Metagenomics & Machine Learning Platform for Postmortem Interval Estimation</h3>
  
  [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-22c55e.svg?style=flat-square&logo=python&logoColor=white)](https://python.org)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-ff4b4b.svg?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
  [![XGBoost](https://img.shields.io/badge/XGBoost-Quantile%20ML-16a34a.svg?style=flat-square)](https://xgboost.readthedocs.io)
  [![SHAP](https://img.shields.io/badge/SHAP-TreeExplainer-0284c7.svg?style=flat-square)](https://shap.readthedocs.io)
  [![ReportLab](https://img.shields.io/badge/ReportLab-Form%20PM--5372-0284c7.svg?style=flat-square)](https://www.reportlab.com)
  [![License: MIT](https://img.shields.io/badge/License-MIT-cef79e.svg?style=flat-square&color=222f30&labelColor=cef79e)](LICENSE)
</div>

> ⚠️ **ACADEMIC RESEARCH PROTOTYPE** — NecroTrace is an educational demonstration project. It is NOT certified, validated, or approved for use in actual forensic investigations, criminal cases, or legal proceedings. All generated reports are for research and demonstration purposes only. The platform has NOT been accredited under ISO 17025 or validated under Daubert/Frye standards.

---

## Executive Abstract

Forensic determination of the time elapsed since death (Postmortem Interval, **PMI**) represents one of the most consequential yet challenging determinations in criminal investigations and medico-legal proceedings. Classical physical indicators—*algor mortis* (body cooling), *rigor mortis* (muscle rigidity), *livor mortis* (blood settling), and forensic entomology—suffer from rapid signal degradation beyond 24–48 hours and high sensitivity to ambient temperature and microclimates.

**NecroTrace** translates the continuous, reproducible ecological succession of the decomposing human necrobiome into quantifiable, court-defensible PMI estimations with rigorous probabilistic uncertainty bounds:

1. **Metagenomic Succession Clock:** Leverages 16S rRNA taxonomic abundance profiles to track predictable community blooms and crashes across postmortem decay stages.
2. **Quantile Machine Learning:** Employs tuned Gradient Boosted Quantile Regressors (XGBoost) outputting calibrated 10th, 50th, and 90th percentile bounds rather than subjective single-point guesses.
3. **Algorithmic Explainability (SHAP):** Decomposes non-linear predictions using TreeSHAP to compute exact Shapley bioindicator feature attributions ($\phi_i$) relative to baseline expectation ($E[f(X)]$), providing transparent, court-admissible justification.
4. **Medical Examiner Triage Workflow:** Bridges physical macroscopic observations (corneal opacity, skin marbling, bloating, purged fluids) with molecular bioindicators.
5. **Forensic Dossier Export (Research Demonstration):** Programmatic synthesis of **Form PM-5372**-style postmortem autopsy reports with cryptographic SHA-256 verification seals and embedded SHAP attribution audits.

```
       [ Metagenomic Data ] (16S rRNA OTU / ASV Abundance Tables)
                 │
                 ▼
   ┌────────────────────────────┐
   │ Bioinformatic Pipeline     │  Read Depth Filtering & Rare Taxa Pruning
   │ & Normalization            │  Compositional Transforms (TSS, Centered Log-Ratio)
   └─────────────┬──────────────┘
                 │
                 ▼
   ┌────────────────────────────┐
   │ Machine Learning Engine    │  Non-linear Quantile Regression (XGBoost)
   │ & Uncertainty Calibration  │  Probabilistic Interval Bounds (p10, p50, p90)
   └─────────────┬──────────────┘
                 │
                 ▼
   ┌────────────────────────────┐
   │ Algorithmic Explainability │  TreeSHAP Shapley Feature Attribution Engine
   │ & Bioindicator Auditing    │  Decomposition into Baseline E[f(X)] + Sum(phi_i)
   └─────────────┬──────────────┘
                 │
                 ▼
   ┌────────────────────────────┐
   │ Diagnostic Triage &        │  Interactive Darkroom Medical Examiner Suite
   │ Medico-Legal Dossier       │  Form PM-5372 PDF Research Export + SHAP Audit
   └────────────────────────────┘
```

---

## Comparison: Traditional PMI vs. NecroTrace

| Dimension | Classical Forensic Indicators | NecroTrace Metagenomic Platform |
| :--- | :--- | :--- |
| **Primary Biomarkers** | Body core temperature, muscle stiffness, insect colonization | Postmortem microbial succession dynamics (16S rRNA / shotgun metagenomics) |
| **Effective Temporal Window** | Accurate mostly within initial 24–48 hours | Robust across extended decay stages (1 to 25+ days) |
| **Environmental Robustness** | Highly volatile to ambient drafts, humidity, clothing | Normalized across temperature and microenvironment covariates |
| **Analytical Objectivity** | Subjective assessment by investigator | Algorithmic, reproducible mathematical feature extraction |
| **Uncertainty Quantification** | Heuristic, uncalibrated estimation windows | Formal statistical confidence intervals via Quantile Regression (p10, p50, p90) |
| **Algorithmic Explainability** | Subjective clinician heuristic / black box | Exact Shapley feature attributions (TreeSHAP) with top bioindicator drivers |
| **Evidentiary Standard** | Frequently disputed under Daubert / Frye challenges | Auditable bioinformatic provenance with SHA-256 cryptographic verification (research demonstration) |

---

## Platform Architecture & Core Workflows

### View 1: Forensic Overview & Succession Matrix
- **Kinetic WebGL Hero:** High-performance canvas visualizing dynamic microbial particle flow and postmortem dispersion.
- **Microbial Succession Waves:**
  - *Wave 1: Fresh / Early Stage (0–3 Days)* &bull; Dominated by aerotolerant mucosal and dermal colonizers (*Staphylococcus*, *Streptococcus*, *Cutibacterium*).
  - *Wave 2: Active Putrefaction (3–8 Days)* &bull; Shift toward hypoxic enteric bloomers and liquefaction catalysts (*Clostridium perfringens*, *Proteus mirabilis*, *Bacteroides fragilis*).
  - *Wave 3: Advanced Skeletonization & Soil Leaching (8–25+ Days)* &bull; Proliferation of environmental saprophytes and soil actinomycetes (*Pseudomonas fluorescens*, *Bacillus subtilis*, *Streptomyces albus*).
- **Minimalist Dispatch Footer:** Streamlined platform navigation, legal disclosures, and academic research advisories.

### View 2: Medical Examiner Diagnostic Triage Suite 
- **Step 01 &bull; Autopsy Particulars:** Standard case registry inputs (PM Report Number, Police Station, Inquest Reference, Deceased Demographics, Ambient Temperatures, Specimen Swab Anatomical Sites).
- **Step 02 &bull; Morphological Observation Matrix:** Interactive forensic triage correlating macroscopic postmortem findings (corneal clouding, algor status, venous marbling, abdominal bloating, purge fluid) with estimated physiological decay windows.
- **Step 03 &bull; Bioindicator Image Viewports:** 19 dedicated clinical photographic viewports featuring real microscopy JPEG assets, optical viewfinder reticles, Gram-stain classifications, and biochemical mechanism breakdowns.
- **Step 04 &bull; Quantile Estimation & SHAP Explainability:** Interactive Plotly timeline gauge (p10–p50–p90 bounds) paired with an automated **TreeSHAP feature attribution card**, plotting top microbial drivers shifting the interval upward or downward with biological narratives.
- **Step 05 &bull; Medico-Legal Dossier Synthesis (Form PM-5372):** Real-time on-screen preview of **Form PM-5372** and one-click export of forensic PDF documents with cryptographic SHA-256 verification seals, QR mobile verification, and embedded SHAP feature attribution audit tables.

### View 3: Mobile Evidence Verification Portal (`?view=verify`)
- **Digital Chain-of-Custody Verification:** Mobile-scannable QR code verification allowing authorized officers to validate tamper-evident SHA-256 checksums, view case particulars, inspect microbial bioindicator evidence, and generate synchronized 6-digit mortuary workstation release codes (`NC-XXXXXX`).

### View 4: Legal & Policy Compliance Suite
- **Dedicated Compliance Pages:** Includes accessible, high-contrast, centered policy documentation for Privacy Policy (`?view=privacy`), Terms of Service (`?view=terms`), Cookie & Third-Party Disclosure Policy (`?view=cookies`), and a custom Medico-Legal 404 Route Diagnostics terminal (`?view=404`).

---

## Dependencies & Core Libraries

NecroTrace is built on a specialized bioinformatic, machine learning, and forensic document generation stack:

| Category | Library | Minimum Version | Architectural Role in NecroTrace |
|:---|:---|:---|:---|
| **Machine Learning & Modeling** | `xgboost` | `>=2.0.0` | Multi-quantile gradient boosting (`reg:quantileerror`) for $p_{10}$, $p_{50}$, and $p_{90}$ non-parametric interval estimation |
| **Model Explainability (XAI)** | `shap` | `>=0.44.0` | `TreeExplainer` computing additive Shapley values ($\phi_i$) for non-black-box microbial bioindicator attribution |
| **Scientific & Bioinformatics** | `numpy` | `>=1.24.0` | High-performance vector operations, Centered Log-Ratio (CLR) offsets, and monotonic bound enforcement |
| **Scientific & Bioinformatics** | `pandas` | `>=2.0.0` | Abundance matrices, case particulars, specimen metadata indexing, and feature schema alignment |
| **Scientific & Bioinformatics** | `scipy` | `>=1.10.0` | Statistical distributions, microbial variance modeling, and bioinformatic dispersion calculations |
| **Scientific & Bioinformatics** | `scikit-learn` | `>=1.3.0` | Reproducible train/test splitting, baseline regressor benchmarking, and model evaluation metrics (MAE, RMSE, coverage) |
| **Pipeline Serialization** | `joblib` | `>=1.3.0` | Fast serialization and disk loading of pre-trained quantile regressors and feature schemas |
| **Interactive Web Application** | `streamlit` | `>=1.32.0` | Reactive application framework, session state management, form inputs, and view routing |
| **Dynamic Visualizations** | `plotly` | `>=5.18.0` | Interactive bioindicator abundance gauges and dynamic postmortem interval timeline fan charts |
| **Static & Forensic Plotting** | `matplotlib` | `>=3.7.0` | Headless rendering (`Agg` backend) of dark-themed in-app SHAP plots and publication-grade PDF graphics |
| **Forensic PDF Synthesis** | `reportlab` | `>=4.1.0` | Programmatic compilation of official Form PM-5372 autopsy dossiers, Platypus flowables, and native QR code generation |
| **Vector Graphic Rendering** | `kaleido` | `>=0.2.1` | Headless static image engine converting Plotly vector figures for report embedding |
| **Image Processing** | `Pillow` (PIL) | `>=10.0.0` | Photographic asset manipulation, optical reticle compositing, and Base64 badge encoding |
| **Cryptographic & Network Utilities** | `hashlib`, `secrets`, `urllib` | Standard Library | SHA-256 evidence hashing, tamper-proof seed token generation, and zero-dependency Firebase REST gateway |

---

## Repository Structure

```text
NecroTrace/
├── .env.example                       # Environment configuration template
├── .gitignore                          # Git ignore rules
├── .streamlit/
│   └── config.toml                    # Production server settings & darkroom theme
├── LICENSE                             # MIT license terms
├── README.md                           # Comprehensive platform documentation
├── requirements.txt                    # Python package dependencies
│
├── app.py                              # Primary Streamlit application entrypoint
├── app/
│   ├── app.py                          # Synchronized secondary application entrypoint
│   └── assets/                         # Mirrored production assets
│
├── artifacts/                          # Serialized pipeline assets
│   ├── feature_schema.joblib           # Pre-trained 16S rRNA taxonomic feature schema
│   ├── xgb_p10.joblib                  # Quantile XGBoost regressor (10th percentile bound)
│   ├── xgb_p50.joblib                  # Quantile XGBoost regressor (median estimate)
│   ├── xgb_p90.joblib                  # Quantile XGBoost regressor (90th percentile bound)
│   ├── sample_forensic_report.pdf      # Sample Form PM-5372 legal dossier
│   └── triage_test_report.pdf          # Triage validation test output
│
├── assets/                             # Platform design & biological assets
│   ├── favicon.png                     # 64x64 browser tab icon
│   ├── logo.png                        # High-resolution master logo (733x510)
│   ├── logo_transparent.png            # Transparent alpha logo
│   ├── logo_icon.png                   # Bio-geometric rosette emblem (360x360)
│   ├── logo_icon_128.png               # Web-optimized 128px inline base64 icon
│   ├── logo_horizontal.png             # Horizontal logo lockup
│   ├── logo_text.png                   # Isolated geometric typography wordmark
│   └── microbes/                       # 19 authentic scientific microscopy JPEGs
│       ├── Acinetobacter baumannii.jpg
│       ├── Bacillus subtilis.jpg
│       ├── Bacteroides fragilis.jpg
│       ├── Clostridium perfringens.jpg
│       ├── Corynebacterium striatum.jpg
│       ├── Cutibacterium acnes.jpg
│       ├── Enterococcus faecalis.jpg
│       ├── Ignatzschineria larvae.jpg
│       ├── Micrococcus luteus.jpg
│       ├── Morganella morganii.jpg
│       ├── Planococcus halocryophilus.jpg
│       ├── Proteus mirabilis.jpg
│       ├── Pseudomonas fluorescens.jpg
│       ├── Rothia dentocariosa.jpg
│       ├── Sphingobacterium multivorum.jpg
│       ├── Staphylococcus epidermidis.jpg
│       ├── Streptococcus oralis.jpg
│       ├── Streptomyces albus.jpg
│       └── Wohlfahrtiimonas chitiniclastica.jpg
│
├── components/                         # React / Tailwind / shadcn design system
│   ├── demo.tsx                        # React showcase wrapper
│   └── ui/
│       ├── button.tsx                  # shadcn CVA Button primitive
│       ├── footer-04.tsx               # Minimalist Next.js footer component
│       ├── input.tsx                   # shadcn Input primitive
│       ├── kinetic-grid.tsx            # Canvas kinetic grid hero component
│       └── separator.tsx               # Radix UI Separator primitive
│
├── data/                               # Data management
│   ├── metadata.csv                    # Specimen records and environmental context
│   └── synthetic_counts.csv            # Synthetic taxonomic abundance matrices
│
├── lib/                                # Frontend utility libraries
│   └── utils.ts                        # Tailwind class merging utility (`clsx` + `twMerge`)
│
├── scripts/                            # Synthesis and training utilities
│   └── make_data.py                    # Pipeline dataset synthesizer
│
└── src/                                # Core computational backend
    ├── __init__.py
    ├── auth/                           # Firebase Authentication & access control
    │   ├── __init__.py
    │   └── firebase_auth.py            # Firebase Identity Toolkit REST API client
    ├── explainability.py               # TreeSHAP algorithmic attribution & narrative engine
    ├── pipeline.py                     # Quantile XGBoost inference & CLR transformation
    ├── report.py                       # Case report structuring utilities
    ├── triage.py                       # Bioindicator catalog & morphological rule engine
    └── reporting/
        ├── __init__.py
        └── pdf_generator.py            # ReportLab Form PM-5372 court export engine
```

---

## Methodological Summary

### 1. Compositional Normalization (TSS & CLR)
Metagenomic sequencing count matrices $X$ reside in a simplex where total read depth is an arbitrary technical artifact. To address compositionality:

$$TSS(x_{ij}) = \frac{x_{ij}}{\sum_{k=1}^{p} x_{ik}}$$

To map simplex compositions into unconstrained Euclidean space for gradient boosting:

$$CLR(x_i) = \left[ \ln\frac{x_{i1}}{g(x_i)}, \ln\frac{x_{i2}}{g(x_i)}, \dots, \ln\frac{x_{ip}}{g(x_i)} \right]$$

where $g(x_i) = \left(\prod_{j=1}^{p} x_{ij}\right)^{1/p}$ is the geometric mean of observed taxa in sample $i$.

### 2. Pinball Quantile Loss Formulation
Rather than minimizing squared errors (which yields conditional means vulnerable to skewed biological variance), the engine optimizes asymmetric pinball loss for quantile $\alpha \in \{0.10, 0.50, 0.90\}$:

$$\mathcal{L}_\alpha(y, \hat{y}) = \max(\alpha (y - \hat{y}), (1 - \alpha)(\hat{y} - y))$$

This guarantees that:
- $\hat{y}_{0.10}$: 10% lower bound (earliest probable time of death)
- $\hat{y}_{0.50}$: Median point estimate
- $\hat{y}_{0.90}$: 90% upper bound (latest probable time of death)
- Interval $[\hat{y}_{0.10}, \hat{y}_{0.90}]$ provides an empirical 80% confidence window satisfying courtroom standards.

### 3. TreeSHAP Additive Feature Attribution
To eliminate "black-box" objections in judicial proceedings, individual quantile predictions are decomposed using TreeSHAP into exact, additive feature attribution scores:

$$f(x) = E[f(X)] + \sum_{i=1}^{M} \phi_i(x)$$

where $E[f(X)]$ represents the baseline expected postmortem interval across the reference metagenomic training cohort ($\approx 6.78$ days), and $\phi_i$ denotes the exact Shapley contribution (in elapsed days) exerted by bioindicator or environmental feature $i$:
- **$\phi_i > 0$ (Positive attribution / Crimson vectors):** Elevated taxa or environmental covariates (e.g., late-stage enteric putrefiers such as *Clostridium perfringens* or elevated ambient temperature) that push the estimated interval upward toward later postmortem decay.
- **$\phi_i < 0$ (Negative attribution / Sky blue vectors):** Fresh mucosal taxa (e.g., *Streptococcus*, *Cutibacterium*) or suppressing conditions that restrain the estimated interval toward biological cessation.

Each generated autopsy dossier directly embeds the top 8 driving bioindicators alongside automated, plain-English forensic findings to satisfy transparency and legal scrutiny.

---

## Firebase Authentication & QR-Gated Chain of Custody

Inspired by ISO 17025 chain-of-custody principles and Federal Rules of Evidence, report access is gated behind authentication:

1. **Step 5 Report Locking**: When an autopsy dossier is synthesized, the PDF download is encrypted/locked by default.
2. **High-Contrast QR Code**: The system encodes a case-specific payload into a high-contrast SVG QR seal pointing to `?view=verify` with a cryptographic session seed.
3. **Mandatory Mobile QR Verification**:
   - Direct browser links to the verification portal are blocked to preserve physical chain of custody.
   - An authorized officer **compulsorily scans the physical QR code seal** displayed on the report preview using a mobile device.
   - The mobile portal verifies that the officer's credentials are authenticated in Cloud Firestore.
   - Once authenticated, the officer can:
     - **Directly download** the Form PM-5372 PDF onto their mobile device.
     - Obtain a **Dynamic 6-Digit Workstation Release Passcode** (e.g. `NC-148895`) dynamically seeded from the scanned QR code.
4. **Physical Mortuary Terminal Release**: Entering the 6-digit release passcode back into the mortuary workstation terminal confirms chain of custody and unlocks desktop PDF downloading and physical printing.

---


## Evidentiary Standards & Academic Scope

NecroTrace was designed with awareness of judicial evidentiary standards for scientific expert testimony (e.g., *Daubert v. Merrell Dow Pharmaceuticals, Inc.* and *Frye v. United States*). However, **the platform has NOT been formally validated or certified under these standards**:
- **Known Error Rates:** Quantile uncertainty bounds explicitly report empirical coverage and confidence intervals.
- **Standardized Protocols:** Form PM-5372 integrates clinical specimen collection sites, DNA adequacy clearance, and institutional registration numbers.
- **Cryptographic Auditability:** Every exported PDF includes a SHA-256 checksum generated over the specimen particulars and estimated intervals.

> ⚠️ **Disclaimer**: NecroTrace is an academic research prototype developed by Tanish Walture (Team BroomWroom). Generated reports are for educational and research demonstration purposes only and must NOT be used as evidence in any legal proceeding.

---

## Deployment Link :
<div align="center">
  <img src="assets/logo_horizontal.png" alt="NecroTrace Logo" width="380" />
  <br/> 
  https://necrotrace.streamlit.app/</a>
</div>

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
