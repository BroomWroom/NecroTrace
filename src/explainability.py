"""
NecroTrace Algorithmic Explainability & Feature Attribution Engine (SHAP).
Provides court-admissible, non-black-box forensic attributions for postmortem interval (PMI)
quantile predictions derived from metagenomic microbial succession signatures.
"""

import io
import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional


def get_tree_explainer(model) -> shap.TreeExplainer:
    """
    Initialize and return a shap.TreeExplainer on the median (xgb_p50) regressor.
    Supports either a raw XGBRegressor or a NecroQuantileRegressor instance.
    """
    median_regressor = getattr(model, "model_p50", model)
    return shap.TreeExplainer(median_regressor)


def explain_sample_prediction(
    model,
    input_features_df: pd.DataFrame,
    top_n: int = 8
) -> Dict[str, Any]:
    """
    Compute exact Shapley values (phi_i) for a single sample input vector.

    Args:
        model: NecroQuantileRegressor or trained XGBRegressor (p50).
        input_features_df: DataFrame with shape (1, n_features).
        top_n: Number of top driving bioindicators to highlight.

    Returns:
        Dict containing:
            - base_value: E[f(X)], expected model baseline in days.
            - predicted_pmi: Model point prediction f(x) in days.
            - shap_values: Numpy array of Shapley values for all features.
            - feature_names: List of all feature column names.
            - feature_values: Numpy array of input values for all features.
            - drivers: List of dicts for top_n features sorted by |phi_i| descending.
            - narrative_bullet_points: Human-readable forensic interpretation points.
    """
    explainer = get_tree_explainer(model)
    shap_vals = explainer.shap_values(input_features_df)

    # Base value handling (float or array)
    expected_val = explainer.expected_value
    if hasattr(expected_val, "__len__"):
        base_value = float(expected_val[0])
    else:
        base_value = float(expected_val)

    if hasattr(shap_vals, "values"):
        # If Explanation object returned
        phi = np.array(shap_vals.values)[0]
    elif isinstance(shap_vals, list):
        phi = np.array(shap_vals[0])[0] if len(np.shape(shap_vals[0])) > 1 else np.array(shap_vals)[0]
    else:
        phi = np.array(shap_vals)[0]

    feature_names = list(input_features_df.columns)
    feature_vals = input_features_df.values[0]
    predicted_pmi = base_value + float(np.sum(phi))

    # Rank features by absolute Shapley magnitude
    abs_phi = np.abs(phi)
    top_indices = np.argsort(abs_phi)[::-1][:top_n]

    drivers = []
    narrative_points = []

    for idx in top_indices:
        feat_name = feature_names[idx]
        val = float(feature_vals[idx])
        contrib = float(phi[idx])
        direction = "upward" if contrib > 0 else "downward"
        
        # Clean up microbial Latin name
        clean_name = feat_name.replace("_", " ")

        driver_info = {
            "feature": feat_name,
            "display_name": clean_name,
            "value": val,
            "shap_value": contrib,
            "abs_shap_value": abs(contrib),
            "direction": direction,
        }
        drivers.append(driver_info)

        # Biological narrative phrasing
        if contrib > 0:
            narrative = (
                f"<b><i>{clean_name}</i></b> (abundance/value: {val:.2f}) increased the estimated PMI "
                f"by <b>+{contrib:.2f} days</b> relative to the baseline cohort average."
            )
        else:
            narrative = (
                f"<b><i>{clean_name}</i></b> (abundance/value: {val:.2f}) restrained the estimated PMI "
                f"by <b>{contrib:.2f} days</b> relative to the baseline cohort average."
            )
        narrative_points.append(narrative)

    return {
        "base_value": base_value,
        "predicted_pmi": predicted_pmi,
        "shap_values": phi,
        "feature_names": feature_names,
        "feature_values": feature_vals,
        "drivers": drivers,
        "narrative_bullet_points": narrative_points,
    }


def generate_attribution_plot(
    explanation_data: Dict[str, Any],
    top_n: int = 8,
    dark_theme: bool = True
) -> plt.Figure:
    """
    Generate an attribution bar chart displaying the top bioindicators driving the estimate.
    
    Args:
        explanation_data: Dictionary returned by explain_sample_prediction.
        top_n: Number of top features to display.
        dark_theme: If True, uses midnight laboratory theme (#1d2728). If False, clean light theme.
    """
    drivers = explanation_data["drivers"][:top_n]
    
    # Reverse so top feature is at the top of the horizontal bar chart
    features = [d["display_name"] for d in reversed(drivers)]
    values = [d["shap_value"] for d in reversed(drivers)]

    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=180)

    bg_color = "#1d2728" if dark_theme else "#ffffff"
    text_color = "#e2e8f0" if dark_theme else "#0f172a"
    sub_text_color = "#94a3b8" if dark_theme else "#475569"
    grid_color = "#2d3c3d" if dark_theme else "#e2e8f0"
    pos_color = "#ef4444" if dark_theme else "#dc2626"  # Red/amber increases PMI (older decomposition)
    neg_color = "#38bdf8" if dark_theme else "#0284c7"  # Cyan/blue decreases PMI (fresher)

    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    bar_colors = [pos_color if v > 0 else neg_color for v in values]
    bars = ax.barh(features, values, color=bar_colors, height=0.6, edgecolor="none", alpha=0.9)

    ax.axvline(0, color=sub_text_color, linestyle="--", linewidth=0.8, alpha=0.7)

    # Annotate bar values
    max_val = max(abs(min(values)), abs(max(values)), 0.1)
    offset = max_val * 0.03

    for bar, val in zip(bars, values):
        x_pos = val + (offset if val >= 0 else -offset)
        ha = "left" if val >= 0 else "right"
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.2f}d",
            va="center",
            ha=ha,
            color=text_color,
            fontsize=8,
            fontfamily="sans-serif",
            fontweight="bold"
        )

    ax.set_xlim(-max_val * 1.35, max_val * 1.35)
    ax.set_xlabel("SHAP Value Impact (Days on Median PMI Prediction)", color=text_color, fontsize=8.5, labelpad=6)
    
    base_val = explanation_data["base_value"]
    pred_val = explanation_data["predicted_pmi"]
    title_text = (
        f"Microbial Bioindicator Feature Attribution (Top {len(drivers)})\n"
        f"Baseline E[f(X)] = {base_val:.2f}d  ➔  Estimated PMI f(x) = {pred_val:.2f}d"
    )
    ax.set_title(title_text, color=text_color, fontsize=9.5, fontweight="bold", pad=8)

    ax.tick_params(axis="y", colors=text_color, labelsize=8.5)
    ax.tick_params(axis="x", colors=sub_text_color, labelsize=7.5)
    ax.grid(axis="x", color=grid_color, linestyle=":", linewidth=0.6, alpha=0.7)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(grid_color)

    plt.tight_layout()
    return fig


def generate_attribution_plot_bytes(
    explanation_data: Dict[str, Any],
    top_n: int = 8
) -> io.BytesIO:
    """
    Render a clean, publication-ready light-themed attribution plot to an in-memory BytesIO buffer.
    """
    fig = generate_attribution_plot(explanation_data, top_n=top_n, dark_theme=False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf
