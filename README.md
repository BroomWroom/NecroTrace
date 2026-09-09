# NecroTrace
### Forensic Metagenomics & Machine Learning Platform for Postmortem Interval Estimation

NecroTrace translates microbial community succession—the "necrobiome clock"—into quantifiable, court-defensible Postmortem Interval (PMI) estimates with probabilistic uncertainty bounds.

---

## Overview

Forensic determination of the time of death is critical in death investigations and criminal proceedings. Traditional methods (such as *algor mortis*, *rigor mortis*, *livor mortis*, and forensic entomology) often yield wide, subjective time windows that degrade in accuracy after the initial 24–48 hours or under fluctuating environmental conditions.

**NecroTrace** provides an objective, data-driven alternative. By analyzing metagenomic 16S rRNA or shotgun sequencing data from decomposing remains, the platform identifies reproducible waves of microbial succession and applies machine learning regression to calculate high-confidence PMI windows.

```
       [ Metagenomic Data ] (OTU / ASV Abundance Tables)
                 │
                 ▼
   ┌────────────────────────────┐
   │ Bioinformatic Preprocessing│  Read Depth Filtering & Rare Taxa Pruning
   │ & Normalization            │  Compositional Transforms (TSS, CSS, CLR)
   └─────────────┬──────────────┘
                 │
                 ▼
   ┌────────────────────────────┐
   │ Machine Learning Pipeline  │  Non-linear Regression (Random Forest & XGBoost)
   │ & Uncertainty Estimation   │  Quantile Bounds (10th, 50th, 90th percentiles)
   └─────────────┬──────────────┘
                 │
                 ▼
   ┌────────────────────────────┐
   │ Decision Support Dashboard │  Interactive Streamlit Application
   │ & Forensic Reporting       │  Courtroom-Ready PDF Summary Export
   └────────────────────────────┘
```

---

## Comparison: Traditional PMI vs. NecroTrace

| Dimension | Traditional Forensic Indicators | NecroTrace Platform |
| :--- | :--- | :--- |
| **Primary Biomarkers** | Body temperature, muscle stiffness, insect larvae | Postmortem microbial succession dynamics (bacteria & fungi) |
| **Temporal Window** | Accurate mostly within 24–48 hours; irregular thereafter | Robust across extended decomposition stages (days to weeks) |
| **Objectivity** | Subject to investigator interpretation | Algorithmic analysis with repeatable feature extraction |
| **Uncertainty Quantification** | Coarse estimation ranges without formal probabilities | Statistical confidence intervals via Quantile Regression |
| **Courtroom Defensibility** | Challenged under rigorous evidentiary standards | Auditable bioinformatics pipeline with transparent metrics |

---

## Core Capabilities

### 1. Bioinformatic Preprocessing
- **Quality Control & Read-Depth Filtering:** Excludes low-coverage samples below sequencing thresholds to eliminate spurious noise.
- **Prevalence & Abundance Thresholding:** Prunes rare or transient taxa that do not contribute generalizable succession signal.
- **Compositional Normalization:** Supports Total Sum Scaling (TSS), Cumulative Sum Scaling (CSS), and Centered Log-Ratio (CLR) transformations to account for varying library sizes and the simplex nature of microbiome data.
- **Ecological Feature Engineering:** Computes alpha diversity (Shannon entropy), richness indices, and taxa log-ratios between bloomers and senescing organisms.

### 2. Machine Learning & Uncertainty Estimation
- **Non-linear Succession Modeling:** Employs tuned Random Forest and XGBoost regressors optimized to capture multi-phase microbial shifts.
- **Quantile Confidence Intervals:** Generates lower (10th percentile), median (50th percentile), and upper (90th percentile) intervals rather than a single point estimate, explicitly capturing real-world uncertainty.
- **Model Evaluation:** Benchmarks error metrics (MAE, RMSE) alongside empirical coverage probabilities.

### 3. Interactive Dashboard & Forensic Reporting
- **Streamlit Interface:** Step-by-step workflow spanning data ingestion, QC inspection, regression inference, and succession visualization.
- **Succession Analytics:** Dynamic Plotly decay and proliferation curves highlighting dominant bioindicator taxa.
- **Case Reporting:** Generates downloadable, structured forensic reports detailing sample metadata, environmental context, predicted interval bounds, and quality audit trails.

---

## Project Structure

```text
necrotrace/
├── .env.example                       # Environment configuration template
├── .gitignore                          # Standard git ignore rules
├── README.md                           # Project documentation
├── requirements.txt                    # Python package dependencies
├── setup.py                            # Package installation script
│
├── data/                               # Data management
│   ├── raw/                            # Raw count tables, OTU/ASV matrices
│   ├── processed/                      # Quality-filtered and normalized matrices
│   ├── metadata/                       # Sample records (temperature, soil, true PMI)
│   └── synthetic/                      # Simulated succession datasets for validation
│
├── notebooks/                          # Research & exploration
│   ├── 01_eda_microbial_decay.ipynb    # Succession patterns and diversity trends
│   └── 02_model_benchmarking.ipynb     # Model comparison and error benchmarking
│
├── src/                                # Core library
│   ├── preprocessing/                  # QC, filtering, and normalization
│   │   ├── filters.py                  # Read-depth cutoffs and taxa pruning
│   │   ├── normalizer.py               # TSS, CSS, and CLR transformations
│   │   └── feature_engineering.py      # Diversity indices and succession ratios
│   │
│   ├── models/                         # Regression and uncertainty estimation
│   │   ├── train.py                    # Cross-validated model training loop
│   │   ├── evaluate.py                 # MAE, RMSE, and coverage metrics
│   │   ├── baseline.py                 # Random Forest baseline regressor
│   │   ├── xgboost_regressor.py        # Tuned XGBoost regressor
│   │   └── uncertainty.py              # Quantile interval regressors
│   │
│   ├── visualization/                  # Chart generation
│   │   ├── decay_curves.py             # Microbial succession curves over time
│   │   └── pmi_intervals.py            # Probabilistic interval plots
│   │
│   └── reporting/                      # Forensic reporting
│       ├── pdf_generator.py            # ReportLab summary export engine
│       └── templates/                  # Document templates
│
├── artifacts/                          # Serialized pipeline assets (git-ignored)
│   ├── models/                         # Trained model binaries (.joblib)
│   └── preprocessors/                  # Saved feature scalers and taxon lists
│
├── app/                                # Streamlit web application
│   ├── app.py                          # Application entry point
│   ├── config.py                       # Theme, layout, and path constants
│   └── pages/                          # Multi-page dashboard
│       ├── 1_Upload_&_Process.py       # File ingestion and bioinformatic QC
│       ├── 2_PMI_Prediction.py         # Regression estimates and uncertainty
│       ├── 3_Taxa_Succession.py        # Interactive decay plots (Plotly)
│       └── 4_Case_Report.py            # Case summary and PDF export
│
└── tests/                              # Automated test suite
    ├── test_preprocessing.py           # Unit tests for filtering and normalization
    └── test_inference.py               # Unit tests for model prediction pipeline
```

---

## Methodological Summary

### 1. Relative Abundance & Total Sum Scaling (TSS)
For a count matrix $X$ where $x_{ij}$ represents the observed read count of taxon $j$ in sample $i$:

$$TSS(x_{ij}) = \frac{x_{ij}}{\sum_{k=1}^{p} x_{ik}}$$

### 2. Centered Log-Ratio (CLR) Transformation
To remove the unit-sum constraint inherent in compositional data:

$$CLR(x_i) = \left[ \ln\frac{x_{i1}}{g(x_i)}, \ln\frac{x_{i2}}{g(x_i)}, \dots, \ln\frac{x_{ip}}{g(x_i)} \right]$$

where $g(x_i) = \left(\prod_{j=1}^{p} x_{ij}\right)^{1/p}$ is the geometric mean of abundances for sample $i$.

### 3. Quantile Loss Function
For quantile $\alpha \in (0, 1)$, the pinball loss minimizes asymmetric residuals to output bounds:

$$\mathcal{L}_\alpha(y, \hat{y}) = \max(\alpha (y - \hat{y}), (1 - \alpha)(\hat{y} - y))$$

This provides empirical 10th and 90th percentile bounds, bounding the estimated PMI with an 80% confidence interval.

---



## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
