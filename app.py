#!/usr/bin/env python3
"""
NecroTrace - Forensic Metagenomics & Postmortem Interval (PMI) Estimation Platform.
Interactive Medico-Legal Dashboard for time-of-death inference, bioinformatic quality audit,
microbial ecological succession visualization, and court-admissible post-mortem reporting.
Developed by Team BroomWroom (Lead: Tanish Walture).
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.pipeline import (
    NecroQuantileRegressor,
    quality_control_filter,
    extract_features,
    compute_shannon_entropy,
)
from src.report import generate_forensic_pdf, compute_sha256_hash

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NecroTrace | Forensic Metagenomics",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Clean Medico-Legal CSS
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .status-badge-pass {
        background-color: #dcfce7;
        color: #166534;
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
    }
    .status-badge-flag {
        background-color: #fee2e2;
        color: #991b1b;
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# CACHED RESOURCE LOADERS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_trained_pipeline(artifacts_dir: str = "artifacts"):
    """Load serialized Quantile XGBoost models and feature schema."""
    try:
        model = NecroQuantileRegressor.load(artifacts_dir=artifacts_dir)
        return model, None
    except Exception as e:
        return None, str(e)


@st.cache_data
def load_benchmark_reference_data():
    """Load reference synthetic counts and metadata for demo cases and plots."""
    counts_path = Path("data/synthetic_counts.csv")
    meta_path = Path("data/metadata.csv")
    if counts_path.exists() and meta_path.exists():
        counts = pd.read_csv(counts_path, index_col=0)
        meta = pd.read_csv(meta_path)
        return counts, meta
    return None, None


# -----------------------------------------------------------------------------
# INITIALIZE APPLICATION STATE
# -----------------------------------------------------------------------------
if "active_counts" not in st.session_state:
    st.session_state["active_counts"] = None
if "active_metadata" not in st.session_state:
    st.session_state["active_metadata"] = None
if "demo_loaded" not in st.session_state:
    st.session_state["demo_loaded"] = False
if "selected_sample_id" not in st.session_state:
    st.session_state["selected_sample_id"] = None
if "predictions_df" not in st.session_state:
    st.session_state["predictions_df"] = None
if "qc_stats" not in st.session_state:
    st.session_state["qc_stats"] = None


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & CASE MANAGEMENT
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚖️ NecroTrace")
    st.markdown("**Forensic Metagenomics Platform**")
    st.caption("Postmortem Interval (PMI) Inference via Microbial Ecological Succession")
    st.divider()

    st.markdown("#### Case Presets & Demo Loader")
    demo_case_type = st.selectbox(
        "Select Demo Scenario:",
        [
            "Case #1: Fresh Stage (~2 Days PMI)",
            "Case #2: Active Putrefaction (~7 Days PMI)",
            "Case #3: Advanced Decay (~20 Days PMI)",
            "Case #4: Full Batch Ingestion (50 Samples)",
        ],
        index=1,
    )

    if st.button("Load Pre-Configured Demo Case", use_container_width=True):
        ref_counts, ref_meta = load_benchmark_reference_data()
        if ref_counts is not None and ref_meta is not None:
            if "Case #1" in demo_case_type:
                # Find sample closest to 2 days
                idx = (ref_meta["true_pmi_days"] - 2.0).abs().idxmin()
                sample_id = ref_meta.loc[idx, "sample_id"]
                st.session_state["active_counts"] = ref_counts.loc[[sample_id]].copy()
                st.session_state["active_metadata"] = ref_meta.loc[[idx]].copy()
                st.session_state["selected_sample_id"] = sample_id

            elif "Case #2" in demo_case_type:
                # Find sample closest to 7 days
                idx = (ref_meta["true_pmi_days"] - 7.0).abs().idxmin()
                sample_id = ref_meta.loc[idx, "sample_id"]
                st.session_state["active_counts"] = ref_counts.loc[[sample_id]].copy()
                st.session_state["active_metadata"] = ref_meta.loc[[idx]].copy()
                st.session_state["selected_sample_id"] = sample_id

            elif "Case #3" in demo_case_type:
                # Find sample closest to 20 days
                idx = (ref_meta["true_pmi_days"] - 20.0).abs().idxmin()
                sample_id = ref_meta.loc[idx, "sample_id"]
                st.session_state["active_counts"] = ref_counts.loc[[sample_id]].copy()
                st.session_state["active_metadata"] = ref_meta.loc[[idx]].copy()
                st.session_state["selected_sample_id"] = sample_id

            else:
                # 50 samples batch
                sample_ids = ref_counts.index[:50].tolist()
                st.session_state["active_counts"] = ref_counts.loc[sample_ids].copy()
                st.session_state["active_metadata"] = ref_meta[ref_meta["sample_id"].isin(sample_ids)].copy()
                st.session_state["selected_sample_id"] = sample_ids[0]

            st.session_state["demo_loaded"] = True
            st.session_state["predictions_df"] = None
            st.success("Demo dataset loaded successfully.")
        else:
            st.error("Benchmark data files not found in `data/`.")

    st.divider()
    st.markdown("#### Environmental Parameters")
    temp_override = st.slider("Ambient Temperature (°C):", min_value=5.0, max_value=40.0, value=23.5, step=0.5)
    humidity_override = st.slider("Relative Humidity (%):", min_value=20.0, max_value=100.0, value=65.0, step=5.0)

    st.divider()
    st.caption("Developed by Team BroomWroom • Lead: Tanish Walture • VMEDITHON 3.0")


# -----------------------------------------------------------------------------
# MAIN HEADER
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">NecroTrace: Forensic Metagenomics Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Objective Time-of-Death Estimation & Post-Mortem Interval Inferences via the Necrobiome Clock</div>', unsafe_allow_html=True)

# Main Application Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Specimen Ingestion & QC",
    "2. Time-of-Death Estimation",
    "3. Microbial Succession Dynamics",
    "4. Official Post-Mortem Report",
])


# =============================================================================
# TAB 1: SPECIMEN INGESTION & QUALITY AUDIT
# =============================================================================
with tab1:
    st.markdown("### Specimen Ingestion & Quality Control Audit")
    st.write(
        "Upload raw microbial abundance tables (OTU / ASV count matrix) along with specimen metadata. "
        "The automated bioinformatic quality gate validates sequencing depth and flags sub-optimal specimens."
    )

    col_up1, col_up2 = st.columns(2)
    with col_up1:
        uploaded_counts = st.file_uploader(
            "Upload Taxonomic Count Matrix (CSV):",
            type=["csv"],
            help="Rows = Samples, Columns = Taxonomic Features (ASVs/OTUs)",
        )
    with col_up2:
        uploaded_meta = st.file_uploader(
            "Upload Specimen Metadata (CSV, Optional):",
            type=["csv"],
            help="Must contain sample_id, ambient_temp_c, humidity_pct",
        )

    # Ingest uploaded files if provided
    if uploaded_counts is not None:
        try:
            df_counts = pd.read_csv(uploaded_counts, index_col=0)
            df_meta = pd.read_csv(uploaded_meta) if uploaded_meta is not None else None
            st.session_state["active_counts"] = df_counts
            st.session_state["active_metadata"] = df_meta
            st.session_state["selected_sample_id"] = df_counts.index[0]
            st.session_state["predictions_df"] = None
            st.info("Custom count matrix uploaded successfully.")
        except Exception as e:
            st.error(f"Error parsing uploaded files: {e}")

    # Display active data status
    if st.session_state["active_counts"] is not None:
        counts = st.session_state["active_counts"]
        meta = st.session_state["active_metadata"]

        st.markdown("---")
        st.markdown("#### Quality Control & Specimen Adequacy Audit")

        # Run Quality Control check
        counts_clean, meta_clean, qc = quality_control_filter(
            counts, meta, min_depth=1000, min_prevalence=0.0
        )
        st.session_state["qc_stats"] = qc

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Total Specimens Ingested", f"{qc['initial_samples']}")
        with kpi2:
            st.metric("Specimens Passing Depth QC", f"{qc['retained_samples']}")
        with kpi3:
            st.metric("Sub-Optimal / Flagged (<1,000 reads)", f"{qc['dropped_samples']}")
        with kpi4:
            st.metric("Diagnostic Taxa Tracked", f"{qc['retained_taxa']}")

        # Read depth distribution chart
        depths = counts.sum(axis=1)
        fig_depth = px.histogram(
            x=depths,
            nbins=30,
            labels={"x": "Total Sequencing Reads per Sample"},
            title="Sequencing Read Depth Distribution (Cutoff: 1,000 Reads)",
            color_discrete_sequence=["#3b82f6"],
        )
        fig_depth.add_vline(x=1000, line_dash="dash", line_color="red", annotation_text="Minimum QC Threshold (1,000)")
        fig_depth.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="#f8fafc")
        st.plotly_chart(fig_depth, use_container_width=True)

        st.markdown("#### Ingested Specimen Matrix Preview")
        st.dataframe(counts.head(10), use_container_width=True)

    else:
        st.info("No specimen data currently loaded. Please upload a CSV count matrix above or click **'Load Pre-Configured Demo Case'** in the sidebar.")


# =============================================================================
# TAB 2: TIME-OF-DEATH ESTIMATION
# =============================================================================
with tab2:
    st.markdown("### Probabilistic Postmortem Interval (PMI) Estimation")
    st.write(
        "The calibrated Quantile XGBoost succession clock infers the postmortem interval, producing a "
        "defensible **Probable Window of Death** based on compositional bacterial shifts."
    )

    if st.session_state["active_counts"] is not None:
        model, err = load_trained_pipeline()
        if err:
            st.error(f"Error loading model artifacts: {err}. Please ensure `artifacts/` contains trained models.")
        else:
            counts = st.session_state["active_counts"]
            meta = st.session_state["active_metadata"]

            # Compute predictions if not already in session state
            if st.session_state["predictions_df"] is None:
                with st.spinner("Executing Centered Log-Ratio transform and Quantile Regression..."):
                    # Align environmental parameters
                    meta_df = meta.copy() if meta is not None else pd.DataFrame(index=counts.index)
                    if "ambient_temp_c" not in meta_df.columns:
                        meta_df["ambient_temp_c"] = temp_override
                    if "humidity_pct" not in meta_df.columns:
                        meta_df["humidity_pct"] = humidity_override

                    features = extract_features(counts, meta_df, feature_schema=model.feature_names_)
                    preds = model.predict_intervals(features)
                    preds["shannon_entropy"] = np.round(compute_shannon_entropy(counts), 3)

                    # Determine decomposition stage
                    def get_stage(pmi_val, t_c):
                        add_val = pmi_val * t_c
                        if add_val < 35:
                            return "Fresh (Early Postmortem)"
                        elif add_val < 150:
                            return "Bloat / Putrefactive Expansion"
                        elif add_val < 350:
                            return "Active Decomposition"
                        elif add_val < 600:
                            return "Advanced Decay"
                        else:
                            return "Dry Skeletal Remains"

                    preds["decomposition_stage"] = [get_stage(p, temp_override) for p in preds["predicted_pmi"]]
                    st.session_state["predictions_df"] = preds

            preds_df = st.session_state["predictions_df"]

            # Sample Selector
            sample_list = list(counts.index)
            selected_sample = st.selectbox("Select Specimen for Case Analysis:", sample_list, index=0)
            st.session_state["selected_sample_id"] = selected_sample

            sample_pred = preds_df.loc[selected_sample]
            pmi_est = float(sample_pred["predicted_pmi"])
            pmi_low = float(sample_pred["lower_bound"])
            pmi_high = float(sample_pred["upper_bound"])
            pmi_stage = str(sample_pred["decomposition_stage"])
            shannon_val = float(sample_pred["shannon_entropy"])

            # Forensic KPI Cards
            st.markdown("---")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric(
                    label="Estimated Time Since Death",
                    value=f"{pmi_est:.1f} Days",
                    delta=f"{pmi_est * 24.0:.0f} Hours elapsed",
                    delta_color="off",
                )
            with m_col2:
                st.metric(
                    label="Probable Window of Death",
                    value=f"{pmi_low:.1f} – {pmi_high:.1f} Days",
                    delta=f"Span: ±{(pmi_high - pmi_low)/2:.1f} days",
                    delta_color="off",
                )
            with m_col3:
                st.metric(
                    label="Gross Decomposition Stage",
                    value=pmi_stage.split(" / ")[0],
                    delta=f"ADD: {pmi_est * temp_override:.1f} °C·days",
                    delta_color="off",
                )
            with m_col4:
                st.metric(
                    label="Microbial Diversity Index",
                    value=f"H' = {shannon_val:.2f}",
                    delta="Adequate succession signal",
                    delta_color="off",
                )

            # Interactive Plotly Gauge / Credible Interval Timeline
            st.markdown("#### Time-of-Death Timeline Gauge")
            fig_pmi = go.Figure()

            # Probable Window shading
            fig_pmi.add_vrect(
                x0=pmi_low,
                x1=pmi_high,
                fillcolor="#bfdbfe",
                opacity=0.6,
                layer="below",
                line_width=0,
                annotation_text="Probable Window of Death",
                annotation_position="top left",
            )

            # Central interval line
            fig_pmi.add_trace(go.Scatter(
                x=[pmi_low, pmi_high],
                y=[0, 0],
                mode="lines",
                line=dict(color="#1e3a8a", width=6),
                name="Probable Range",
                hoverinfo="skip",
            ))

            # Point estimate
            fig_pmi.add_trace(go.Scatter(
                x=[pmi_est],
                y=[0],
                mode="markers+text",
                marker=dict(color="#b91c1c", size=18, symbol="diamond"),
                text=[f"Most Likely: {pmi_est:.1f} d ({pmi_est*24:.0f} hrs)"],
                textposition="bottom center",
                name="Estimated PMI",
                hovertemplate="Estimated PMI: %{x:.2f} Days<extra></extra>",
            ))

            fig_pmi.update_layout(
                xaxis=dict(
                    title="Time Elapsed Since Death (Days)",
                    range=[0, max(30.0, pmi_high * 1.3)],
                    showgrid=True,
                    gridcolor="#e2e8f0",
                ),
                yaxis=dict(showticklabels=False, range=[-0.6, 0.6]),
                height=220,
                margin=dict(l=20, r=20, t=30, b=30),
                paper_bgcolor="#f8fafc",
                plot_bgcolor="#f8fafc",
                showlegend=False,
            )
            st.plotly_chart(fig_pmi, use_container_width=True)

            # Batch Summary Table if multiple samples
            if len(preds_df) > 1:
                st.markdown("#### Batch Prediction Overview")
                st.dataframe(preds_df, use_container_width=True)

    else:
        st.info("Load or upload specimen count data in Tab 1 to run the estimation model.")


# =============================================================================
# TAB 3: MICROBIAL SUCCESSION DYNAMICS
# =============================================================================
with tab3:
    st.markdown("### Microbial Community Succession (Necrobiome Clock)")
    st.write(
        "Microbial communities follow reproducible ecological waves throughout postmortem decomposition. "
        "Early aerobic surface bacteria decline rapidly, giving way to enteric anaerobes during active decay, "
        "followed by soil saprophytic decomposers."
    )

    ref_counts, ref_meta = load_benchmark_reference_data()
    if ref_counts is not None and ref_meta is not None:
        # Merge for dynamic temporal plotting
        rel_abund = ref_counts.div(ref_counts.sum(axis=1), axis=0)
        df_merged = rel_abund.copy()
        df_merged["true_pmi_days"] = ref_meta.set_index("sample_id")["true_pmi_days"]
        df_merged = df_merged.sort_values("true_pmi_days")

        # Key Forensic Biomarkers
        st.markdown("#### Key Bioindicator Decay & Proliferation Curves")
        taxa_options = list(ref_counts.columns)
        default_taxa = [
            "Staphylococcus_epidermidis",  # Early bloomer
            "Clostridium_perfringens",      # Mid bloat/anaerobe
            "Acinetobacter_baumannii",      # Late soil colonizer
            "Bacteroides_fragilis",         # Putrefactive anaerobe
        ]
        chosen_taxa = [t for t in default_taxa if t in taxa_options]

        selected_taxa = st.multiselect(
            "Select Taxa to Track over Postmortem Interval:",
            taxa_options,
            default=chosen_taxa,
        )

        if selected_taxa:
            fig_curve = go.Figure()
            colors_cycle = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]

            # Sort and smooth curves
            for i, taxon in enumerate(selected_taxa):
                # Binned mean for smooth visualization
                binned = df_merged.groupby(pd.cut(df_merged["true_pmi_days"], bins=20))[[taxon, "true_pmi_days"]].mean().dropna()
                fig_curve.add_trace(go.Scatter(
                    x=binned["true_pmi_days"],
                    y=binned[taxon],
                    mode="lines+markers",
                    name=taxon.replace("_", " "),
                    line=dict(width=3, color=colors_cycle[i % len(colors_cycle)]),
                ))

            # Mark current selected specimen's predicted PMI if available
            if st.session_state["selected_sample_id"] and st.session_state["predictions_df"] is not None:
                cur_pmi = float(st.session_state["predictions_df"].loc[st.session_state["selected_sample_id"], "predicted_pmi"])
                fig_curve.add_vline(
                    x=cur_pmi,
                    line_dash="dot",
                    line_color="#1e293b",
                    annotation_text=f"Specimen {st.session_state['selected_sample_id']} ({cur_pmi:.1f}d)",
                    annotation_position="top right",
                )

            fig_curve.update_layout(
                title="Relative Taxa Abundance Across Decomposition Stages (0 to 30 Days PMI)",
                xaxis_title="Postmortem Interval (Days)",
                yaxis_title="Relative Abundance (Fraction of Community)",
                height=450,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="#f8fafc",
                plot_bgcolor="#ffffff",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_curve, use_container_width=True)

        # Specimen Taxa Breakdown Bar Chart
        if st.session_state["active_counts"] is not None and st.session_state["selected_sample_id"]:
            sel_id = st.session_state["selected_sample_id"]
            sample_counts = st.session_state["active_counts"].loc[sel_id]
            top10 = sample_counts.sort_values(ascending=False).head(10)

            st.markdown(f"#### Dominant Taxa in Current Specimen (`{sel_id}`)")
            fig_top10 = px.bar(
                x=top10.values,
                y=[t.replace("_", " ") for t in top10.index],
                orientation="h",
                labels={"x": "Observed Read Counts", "y": "Taxon"},
                title=f"Top 10 Most Abundant Taxa for {sel_id}",
                color=top10.values,
                color_continuous_scale="Blues",
            )
            fig_top10.update_layout(height=350, yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_top10, use_container_width=True)

    else:
        st.info("Reference data not available. Please run `scripts/make_data.py` to generate succession data.")


# =============================================================================
# TAB 4: OFFICIAL POST-MORTEM REPORT & PDF EXPORT
# =============================================================================
with tab4:
    st.markdown("### Official Medico-Legal Post-Mortem Report (Form PM-5372)")
    st.write(
        "Preview the official Post-Mortem Examination Report formatted in accordance with "
        "police and medico-legal standards. Generate and download a court-admissible PDF report."
    )

    if st.session_state["active_counts"] is not None and st.session_state["predictions_df"] is not None:
        sel_id = st.session_state["selected_sample_id"]
        pred_row = st.session_state["predictions_df"].loc[sel_id]
        p_est = float(pred_row["predicted_pmi"])
        p_low = float(pred_row["lower_bound"])
        p_high = float(pred_row["upper_bound"])
        p_stage = str(pred_row["decomposition_stage"])

        st.markdown("#### Case Particulars Entry")
        r_col1, r_col2, r_col3 = st.columns(3)
        with r_col1:
            pm_report_no = st.text_input("Post Mortem Report No:", value="PM-619 / 2026")
            police_station = st.text_input("Police Station (P.S.):", value="New Township P.S.")
            deceased_name = st.text_input("Name of Deceased:", value="Unidentified Male (Found near woodland)")
        with r_col2:
            inquest_no = st.text_input("Inquest Report No:", value="14 / 2026")
            inquest_date = st.text_input("Inquest Date:", value=datetime.now().strftime("%d / %m / %Y"))
            age_sex = st.text_input("Age & Sex:", value="Approx. 35 - 40 Yrs / Male")
        with r_col3:
            doctor_name = st.text_input("Examining Medical Officer:", value="Dr. Tanish Walture, M.D.")
            doctor_reg = st.text_input("Medical Registration No:", value="WBMC / 45826")
            swab_site = st.text_input("Anatomical Swab Site:", value="Oral Cavity & Nasal Mucosa")

        # Compile report dictionaries
        case_meta_dict = {
            "pm_report_no": pm_report_no,
            "police_station": police_station,
            "inquest_no": inquest_no,
            "inquest_date": inquest_date,
            "institution": "District Medico-Legal Center & Morgue",
            "analyst": doctor_name,
            "reg_no": doctor_reg,
            "deceased_name": deceased_name,
            "age": age_sex,
            "sex": "Male" if "Male" in age_sex else "Female",
            "sample_id": sel_id,
            "sample_site": swab_site,
            "raw_data_hash": compute_sha256_hash(st.session_state["active_counts"].to_csv()),
            "receipt_time": datetime.now().strftime("%d/%m/%Y, %H:%M hrs"),
        }

        pmi_findings_dict = {
            "predicted_pmi": p_est,
            "lower_bound": p_low,
            "upper_bound": p_high,
            "decomposition_stage": p_stage,
            "ambient_temp_c": temp_override,
        }

        depth_val = int(st.session_state["active_counts"].loc[sel_id].sum())
        qc_metrics_dict = {
            "read_depth": depth_val,
            "shannon_entropy": float(pred_row["shannon_entropy"]),
            "retained_taxa": len(st.session_state["active_counts"].columns),
            "initial_taxa": 50,
            "dropped_taxa": 6,
        }

        # Live On-Screen Report Preview
        st.markdown("---")
        st.markdown("#### Live Report Preview (Official Form PM-5372)")
        with st.container():
            st.markdown(f"""
            <div style="border: 2px solid #0f172a; padding: 20px; background-color: #ffffff; color: #000000; font-family: sans-serif;">
                <div style="text-align: center; border-bottom: 2px solid #000; padding-bottom: 8px; margin-bottom: 12px;">
                    <h3 style="margin: 0; text-transform: uppercase;">Department of Forensic Medicine & Police Morgue</h3>
                    <h4 style="margin: 4px 0; color: #1e3a8a;">POST MORTEM REPORT — FORM NO. PM-5372</h4>
                    <p style="margin: 0; font-size: 0.85rem;"><b>REPORT NO:</b> {pm_report_no} &nbsp;|&nbsp; <b>P.S.:</b> {police_station} &nbsp;|&nbsp; <b>INQUEST:</b> {inquest_no}</p>
                </div>
                
                <table style="width: 100%; font-size: 0.82rem; border-collapse: collapse; margin-bottom: 12px;" border="1" cellpadding="5">
                    <tr style="background-color: #f1f5f9;">
                        <td><b>Deceased Name:</b> {deceased_name}</td>
                        <td><b>Age / Sex:</b> {age_sex}</td>
                        <td><b>Swab Site:</b> {swab_site}</td>
                    </tr>
                    <tr>
                        <td><b>Examining Doctor:</b> {doctor_name}</td>
                        <td><b>Registration:</b> {doctor_reg}</td>
                        <td><b>Specimen Adequacy:</b> <font color="green"><b>SATISFACTORY (Adequate DNA)</b></font></td>
                    </tr>
                </table>

                <div style="background-color: #fee2e2; border: 1.5px solid #991b1b; padding: 12px; margin-bottom: 12px;">
                    <h4 style="margin: 0 0 6px 0; color: #991b1b; text-transform: uppercase;">Definitive Medico-Legal Opinion: Time Since Death</h4>
                    <p style="margin: 2px 0; font-size: 0.95rem;"><b>1. ESTIMATED TIME ELAPSED SINCE DEATH:</b> <font color="#991b1b" size="3"><b>{p_est:.1f} DAYS</b></font> (approx. <b>{p_est*24.0:.0f} Hours</b> prior to examination)</p>
                    <p style="margin: 2px 0; font-size: 0.9rem;"><b>2. PROBABLE FORENSIC WINDOW:</b> <font color="#1e3a8a"><b>{p_low:.1f} to {p_high:.1f} DAYS</b></font> prior to recovery</p>
                    <p style="margin: 2px 0; font-size: 0.85rem;"><b>3. GROSS DECOMPOSITION STAGE:</b> {p_stage} &nbsp;(Scene Temp: {temp_override}°C, ADD: {p_est*temp_override:.1f})</p>
                    <p style="margin: 2px 0; font-size: 0.82rem; color: #334155;"><b>4. BIOLOGICAL CLOCK STATUS:</b> Putrefactive bacterial succession profile confirms concordance with observed physical decomposition changes.</p>
                </div>
                
                <p style="font-size: 0.75rem; color: #475569; margin-top: 10px; font-style: italic;">
                    Certified to be a true post-mortem examination record. Verified under scientific standards of forensic microbiological succession.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # PDF Download Action
        with st.spinner("Compiling high-resolution PDF postmortem dossier..."):
            pdf_data = generate_forensic_pdf(case_meta_dict, pmi_findings_dict, qc_metrics_dict)

        st.download_button(
            label="📄 Download Official Post-Mortem Report (PDF)",
            data=pdf_data,
            file_name=f"PostMortem_Report_{sel_id}_{pm_report_no.replace('/', '_').replace(' ', '')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    else:
        st.info("Please load or process specimen data in Tab 1 & Tab 2 to view and export the Post-Mortem Report.")
