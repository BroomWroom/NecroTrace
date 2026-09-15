#!/usr/bin/env python3
"""
NecroTrace - Core Processing & Quantile Model Pipeline.
Consolidates bioinformatic quality control, Centered Log-Ratio (CLR) normalization,
Shannon diversity calculation, and multi-quantile XGBoost regression for
postmortem interval (PMI) estimation with defensible uncertainty bounds.
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


# ---------------------------------------------------------------------------
# 1. Bioinformatic Preprocessing & Normalization
# ---------------------------------------------------------------------------

def quality_control_filter(
    counts_df: pd.DataFrame,
    metadata_df: Optional[pd.DataFrame] = None,
    min_depth: int = 1000,
    min_prevalence: float = 0.05
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Apply standard bioinformatic quality control:
    1. Exclude samples with total sequencing reads below min_depth.
    2. Exclude taxa present in fewer than min_prevalence fraction of samples.

    Args:
        counts_df: Abundance matrix (samples x taxa).
        metadata_df: Optional sample metadata aligned by sample_id.
        min_depth: Minimum sequencing read count per sample.
        min_prevalence: Minimum prevalence fraction (e.g. 0.05 = 5%).

    Returns:
        Tuple of (filtered_counts, filtered_metadata, qc_summary_dict)
    """
    initial_samples, initial_taxa = counts_df.shape

    # 1. Sample read depth filtering
    sample_depths = counts_df.sum(axis=1)
    passing_samples = sample_depths >= min_depth
    counts_clean = counts_df[passing_samples].copy()

    # 2. Taxa prevalence filtering
    prevalence = (counts_clean > 0).mean(axis=0)
    passing_taxa = prevalence >= min_prevalence
    counts_clean = counts_clean.loc[:, passing_taxa].copy()

    # Align metadata if provided
    filtered_metadata = None
    if metadata_df is not None:
        filtered_metadata = metadata_df.copy()
        if "sample_id" in filtered_metadata.columns:
            filtered_metadata = filtered_metadata.set_index("sample_id")
        filtered_metadata = filtered_metadata.loc[counts_clean.index].reset_index()

    qc_summary = {
        "initial_samples": initial_samples,
        "retained_samples": counts_clean.shape[0],
        "dropped_samples": initial_samples - counts_clean.shape[0],
        "initial_taxa": initial_taxa,
        "retained_taxa": counts_clean.shape[1],
        "dropped_taxa": initial_taxa - counts_clean.shape[1],
        "min_depth_threshold": min_depth,
        "min_prevalence_threshold": min_prevalence,
    }

    return counts_clean, filtered_metadata, qc_summary


def compute_clr(counts_df: pd.DataFrame, pseudocount: float = 1.0) -> pd.DataFrame:
    """
    Compute Centered Log-Ratio (CLR) transformation for compositional data:
    CLR(x_ij) = ln(x_ij) - (1/p) * sum_{k=1}^p ln(x_ik)

    Args:
        counts_df: Abundance matrix (samples x taxa).
        pseudocount: Small offset added to avoid ln(0).

    Returns:
        pd.DataFrame containing CLR-transformed features.
    """
    counts_offset = counts_df.astype(float) + pseudocount
    log_counts = np.log(counts_offset)
    geometric_mean_log = log_counts.mean(axis=1)
    clr_matrix = log_counts.sub(geometric_mean_log, axis=0)
    return clr_matrix


def compute_shannon_entropy(counts_df: pd.DataFrame) -> pd.Series:
    """
    Compute Shannon Diversity Index (entropy) per sample using relative abundances:
    H = - sum p_i ln(p_i)

    Args:
        counts_df: Raw or relative abundance matrix.

    Returns:
        pd.Series of Shannon entropy values aligned by sample index.
    """
    row_sums = counts_df.sum(axis=1).replace(0, 1.0)
    proportions = counts_df.div(row_sums, axis=0)

    # Calculate entropy only for non-zero entries
    entropy_vals = []
    for _, row in proportions.iterrows():
        p = row[row > 0].values
        h = -np.sum(p * np.log(p))
        entropy_vals.append(h)

    return pd.Series(entropy_vals, index=counts_df.index, name="alpha_shannon_entropy")


def extract_features(
    counts_df: pd.DataFrame,
    metadata_df: Optional[pd.DataFrame] = None,
    feature_schema: Optional[list] = None
) -> pd.DataFrame:
    """
    Combine CLR-transformed taxa abundances, Shannon entropy, and environmental covariates.

    Args:
        counts_df: Filtered count matrix (samples x taxa).
        metadata_df: Metadata containing 'ambient_temp_c' and 'humidity_pct'.
        feature_schema: Optional fixed list of column names for inference alignment.

    Returns:
        pd.DataFrame of model-ready features.
    """
    # 1. Compositional CLR
    features = compute_clr(counts_df)

    # 2. Alpha diversity
    features["alpha_shannon_entropy"] = compute_shannon_entropy(counts_df)

    # 3. Environmental covariates
    if metadata_df is not None:
        meta_indexed = metadata_df.copy()
        if "sample_id" in meta_indexed.columns:
            meta_indexed = meta_indexed.set_index("sample_id")

        if "ambient_temp_c" in meta_indexed.columns:
            features["ambient_temp_c"] = meta_indexed.loc[features.index, "ambient_temp_c"]
        else:
            features["ambient_temp_c"] = 22.0

        if "humidity_pct" in meta_indexed.columns:
            features["humidity_pct"] = meta_indexed.loc[features.index, "humidity_pct"]
        else:
            features["humidity_pct"] = 65.0
    else:
        features["ambient_temp_c"] = 22.0
        features["humidity_pct"] = 65.0

    # Align to trained feature schema if provided
    if feature_schema is not None:
        features = features.reindex(columns=feature_schema, fill_value=0.0)

    return features


# ---------------------------------------------------------------------------
# 2. Quantile Regression Engine (XGBoost)
# ---------------------------------------------------------------------------

class NecroQuantileRegressor:
    """
    Quantile regression wrapper utilizing 3 tuned XGBoost models
    to output median PMI estimates (p50) along with an 80% credible interval (p10 to p90).
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_names_: list[str] = []

        # Pinball loss objective at 10th, 50th, and 90th percentiles
        self.model_p10 = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=0.10,
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model_p50 = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=0.50,
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model_p90 = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=0.90,
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "NecroQuantileRegressor":
        """Fit all three quantile models on training features."""
        self.feature_names_ = list(X.columns)

        print("[*] Training 10th percentile XGBoost regressor (lower bound)...")
        self.model_p10.fit(X, y)

        print("[*] Training 50th percentile XGBoost regressor (median estimate)...")
        self.model_p50.fit(X, y)

        print("[*] Training 90th percentile XGBoost regressor (upper bound)...")
        self.model_p90.fit(X, y)

        self.is_fitted = True
        return self

    def predict_intervals(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict median PMI and 80% credible interval bounds.

        Returns:
            pd.DataFrame with ['lower_bound', 'predicted_pmi', 'upper_bound', 'interval_width']
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() or load() first.")

        p10_preds = self.model_p10.predict(X)
        p50_preds = self.model_p50.predict(X)
        p90_preds = self.model_p90.predict(X)

        # Enforce non-negativity and monotonic bounds (p10 <= p50 <= p90)
        p10_clean = np.maximum(0.0, p10_preds)
        p50_clean = np.maximum(p10_clean, p50_preds)
        p90_clean = np.maximum(p50_clean, p90_preds)

        results = pd.DataFrame({
            "lower_bound": np.round(p10_clean, 2),
            "predicted_pmi": np.round(p50_clean, 2),
            "upper_bound": np.round(p90_clean, 2),
            "interval_width": np.round(p90_clean - p10_clean, 2)
        }, index=X.index)

        return results

    def save(self, artifacts_dir: str = "artifacts") -> None:
        """Serialize models and feature schema to disk."""
        path = Path(artifacts_dir)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model_p10, path / "xgb_p10.joblib")
        joblib.dump(self.model_p50, path / "xgb_p50.joblib")
        joblib.dump(self.model_p90, path / "xgb_p90.joblib")
        joblib.dump(self.feature_names_, path / "feature_schema.joblib")
        print(f"[+] Models and feature schema saved to '{artifacts_dir}/'.")

    @classmethod
    def load(cls, artifacts_dir: str = "artifacts") -> "NecroQuantileRegressor":
        """Load fitted models and schema from disk."""
        path = Path(artifacts_dir)
        instance = cls()
        instance.model_p10 = joblib.load(path / "xgb_p10.joblib")
        instance.model_p50 = joblib.load(path / "xgb_p50.joblib")
        instance.model_p90 = joblib.load(path / "xgb_p90.joblib")
        instance.feature_names_ = joblib.load(path / "feature_schema.joblib")
        instance.is_fitted = True
        return instance


# ---------------------------------------------------------------------------
# 3. Training & Validation Execution Loop
# ---------------------------------------------------------------------------

def train_pipeline(
    counts_path: str = "data/synthetic_counts.csv",
    metadata_path: str = "data/metadata.csv",
    artifacts_dir: str = "artifacts",
    test_size: float = 0.20,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Full end-to-end training and validation loop.

    1. Loads count matrix and metadata.
    2. Applies QC pruning (min_depth=1000, min_prevalence=0.05).
    3. Transforms features (CLR + Shannon + Temp/Humidity).
    4. Splits data into train and test sets.
    5. Trains 3-quantile XGBoost models.
    6. Calculates MAE, RMSE, and 80% empirical interval coverage.
    7. Serializes artifacts.
    """
    print("=" * 60)
    print("NecroTrace: Pipeline Training & Model Benchmarking")
    print("=" * 60)

    # 1. Load Data
    print(f"[*] Ingesting raw counts from '{counts_path}'...")
    counts_raw = pd.read_csv(counts_path, index_col=0)
    print(f"[*] Ingesting sample metadata from '{metadata_path}'...")
    metadata_raw = pd.read_csv(metadata_path)

    # 2. Quality Control
    print("[*] Running bioinformatic Quality Control...")
    counts_qc, meta_qc, qc_stats = quality_control_filter(
        counts_raw, metadata_raw, min_depth=1000, min_prevalence=0.05
    )
    print(f"    - Retained Samples: {qc_stats['retained_samples']} / {qc_stats['initial_samples']} "
          f"({qc_stats['dropped_samples']} dropped < {qc_stats['min_depth_threshold']} reads)")
    print(f"    - Retained Taxa: {qc_stats['retained_taxa']} / {qc_stats['initial_taxa']} "
          f"({qc_stats['dropped_taxa']} pruned < {qc_stats['min_prevalence_threshold']*100}% prevalence)")

    # 3. Feature Extraction
    print("[*] Computing CLR transformations, Shannon entropy, and environmental features...")
    X = extract_features(counts_qc, meta_qc)
    y = meta_qc.set_index("sample_id").loc[X.index, "true_pmi_days"]

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"[*] Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # 5. Model Training
    model = NecroQuantileRegressor(random_state=random_state)
    model.fit(X_train, y_train)

    # 6. Evaluation on Test Set
    print("[*] Evaluating on held-out test set...")
    intervals = model.predict_intervals(X_test)

    median_pred = intervals["predicted_pmi"]
    lower_bound = intervals["lower_bound"]
    upper_bound = intervals["upper_bound"]

    mae = mean_absolute_error(y_test, median_pred)
    rmse = root_mean_squared_error(y_test, median_pred)

    # Empirical coverage: proportion of true values falling between lower and upper bounds
    in_interval = (y_test >= lower_bound) & (y_test <= upper_bound)
    coverage = in_interval.mean() * 100.0
    mean_width = intervals["interval_width"].mean()

    print("-" * 60)
    print("Forensic Performance Metrics:")
    print(f"    - Mean Absolute Error (MAE):     {mae:.2f} days")
    print(f"    - Root Mean Squared Error (RMSE): {rmse:.2f} days")
    print(f"    - Empirical 80% CI Coverage:      {coverage:.1f}% (target: ~80%)")
    print(f"    - Mean Credible Interval Width:   {mean_width:.2f} days")
    print("-" * 60)

    # 7. Save Artifacts
    model.save(artifacts_dir=artifacts_dir)

    metrics = {
        "mae_days": round(float(mae), 2),
        "rmse_days": round(float(rmse), 2),
        "ci_coverage_pct": round(float(coverage), 1),
        "mean_interval_width_days": round(float(mean_width), 2),
        "qc_summary": qc_stats,
    }

    return metrics


# ---------------------------------------------------------------------------
# 4. Streamlit Inference Helper
# ---------------------------------------------------------------------------

def predict_pmi_from_raw(
    counts_df: pd.DataFrame,
    metadata_df: Optional[pd.DataFrame] = None,
    artifacts_dir: str = "artifacts"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    High-level inference function for the Streamlit dashboard:
    Loads pre-trained artifacts, processes raw counts, and outputs
    probabilistic interval predictions and QC metrics.
    """
    # 1. Load trained model & schema
    model = NecroQuantileRegressor.load(artifacts_dir=artifacts_dir)
    schema = model.feature_names_

    # 2. QC Filter
    counts_qc, meta_qc, qc_stats = quality_control_filter(
        counts_df, metadata_df, min_depth=1000, min_prevalence=0.0
    )

    if counts_qc.empty:
        raise ValueError("No samples passed QC threshold (minimum 1,000 reads required).")

    # 3. Extract aligned features
    X = extract_features(counts_qc, meta_qc, feature_schema=schema)

    # 4. Predict intervals
    preds = model.predict_intervals(X)

    # 5. Enrich output with Shannon index
    preds["shannon_entropy"] = np.round(compute_shannon_entropy(counts_qc), 3)

    return preds, qc_stats


if __name__ == "__main__":
    train_pipeline()
