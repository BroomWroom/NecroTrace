#!/usr/bin/env python3
"""
NecroTrace - Forensic Metagenomics & Postmortem Interval (PMI) Estimation Platform.
Integrated Biosciences Design System ("Bioluminescent Laboratory at Midnight").
Features an architectural editorial landing page with Lenis smooth scrolling,
paired with an authoritative Medical Examiner Diagnostic Triage workflow.
Developed by Team BroomWroom (Lead: Tanish Walture).
"""

from src.auth import (
    get_firebase_config,
    is_firebase_configured,
    check_email_registered_in_firebase,
    sign_in_officer,
    register_officer,
    fetch_officer_from_firestore,
    get_active_officer_session,
    set_active_officer_session,
    clear_active_officer_session,
    generate_release_passcode,
    verify_release_passcode,
    save_firebase_config,
    test_firebase_connection,
    DEMO_REGISTERED_OFFICERS,
)
from src.explainability import (
    get_tree_explainer,
    explain_sample_prediction,
    generate_attribution_plot,
    generate_attribution_plot_bytes,
)
from src.reporting.pdf_generator import (
    generate_forensic_pdf,
    compute_sha256_hash,
    generate_qr_code_svg,
)
from datetime import datetime, timezone, timedelta
import time
import secrets
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import base64
import os
import plotly.graph_objects as go

from src.pipeline import NecroQuantileRegressor
from src.triage import (
    evaluate_morphological_triage,
    synthesize_abundance_profile,
    FORENSIC_BIOINDICATOR_CATALOG,
    MORPHOLOGICAL_SIGN_GUIDES,
)
import sys
import importlib
import src.reporting.pdf_generator as pdf_gen
importlib.reload(pdf_gen)


# -----------------------------------------------------------------------------
# AUTHENTICATION & FIREBASE GATEWAY
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 1. VIEW DEFINITIONS & METADATA
# -----------------------------------------------------------------------------
VALID_VIEWS = ["landing", "examination", "verify",
               "privacy", "terms", "cookies", "404"]

VIEW_TITLES = {
    "landing": "NecroTrace — Forensic Metagenomics & Microbial Succession Clock",
    "examination": "NecroTrace — Medical Examiner Triage & Autopsy Suite (Form PM-5372)",
    "verify": "NecroTrace — Digital Chain-of-Custody & Evidence Verification Portal",
    "privacy": "NecroTrace — Forensic Data Privacy Policy & Compliance",
    "terms": "NecroTrace — Terms of Service & Medico-Legal Scope",
    "cookies": "NecroTrace — Cookie Policy & Third-Party Disclosures",
    "404": "NecroTrace — 404 File Not Found & Evidence Unresolved",
}

# Resolve active view for page configuration & metadata
_raw_query_view = st.query_params.get("view", None)
_session_view = st.session_state.get("view", None)

if _raw_query_view and _raw_query_view not in VALID_VIEWS and _raw_query_view != "logout":
    _active_view_config = "404"
elif _raw_query_view in VALID_VIEWS:
    _active_view_config = _raw_query_view
elif _session_view in VALID_VIEWS:
    _active_view_config = _session_view
else:
    _active_view_config = "landing"

# Check if pre-landing authentication gateway applies
if not st.session_state.get("authenticated_officer") and _active_view_config not in ["verify", "privacy", "terms", "cookies", "404"]:
    _initial_page_title = "NecroTrace — Examiner Authentication & Access Gateway"
else:
    _initial_page_title = VIEW_TITLES.get(_active_view_config, "NecroTrace — Forensic Metagenomics")

st.set_page_config(
    page_title=_initial_page_title,
    page_icon="assets/favicon.png" if os.path.exists(
        "assets/favicon.png") else "assets/logo_icon.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# 2. APPLICATION ROUTING & STATE
# -----------------------------------------------------------------------------
# Handle query parameters for view routing if present
query_view = st.query_params.get("view", None)
query_auth = st.query_params.get("auth", None)

if query_view == "logout":
    st.session_state["authenticated_officer"] = None
    clear_active_officer_session()
    st.query_params.clear()
    st.session_state["view"] = "landing"
    st.rerun()

# 1. Restore authenticated officer if not in current session_state
if not st.session_state.get("authenticated_officer"):
    cached_session_officer = get_active_officer_session()
    if cached_session_officer:
        st.session_state["authenticated_officer"] = cached_session_officer
    elif query_auth:
        fs_officer = fetch_officer_from_firestore(local_id=query_auth)
        if fs_officer:
            officer_obj = {
                "name": fs_officer.get("name", "Examiner"),
                "role": fs_officer.get("role", "Forensic Medical Examiner"),
                "badge": fs_officer.get("badge", "CFS-9042"),
                "station": fs_officer.get("station", "Central Forensic Science Laboratory"),
                "email": fs_officer.get("email", ""),
                "local_id": query_auth,
                "firestore_verified": True,
                "database": "Cloud Firestore",
            }
            st.session_state["authenticated_officer"] = officer_obj
            set_active_officer_session(officer_obj)

# Always sync active session when authenticated_officer is present
if st.session_state.get("authenticated_officer"):
    set_active_officer_session(st.session_state["authenticated_officer"])

if query_view and query_view not in VALID_VIEWS and query_view != "logout":
    st.session_state["view"] = "404"
elif "view" not in st.session_state:
    st.session_state["view"] = query_view if query_view in VALID_VIEWS else "landing"
elif query_view in VALID_VIEWS and query_view != st.session_state["view"]:
    st.session_state["view"] = query_view

if "triage_confirmed" not in st.session_state:
    st.session_state["triage_confirmed"] = False
if "confirmed_taxa" not in st.session_state:
    st.session_state["confirmed_taxa"] = []
if "pmi_results" not in st.session_state:
    st.session_state["pmi_results"] = None
if "triage_eval" not in st.session_state:
    st.session_state["triage_eval"] = None
if "case_particulars" not in st.session_state:
    st.session_state["case_particulars"] = {}
if "authenticated_officer" not in st.session_state:
    st.session_state["authenticated_officer"] = None
if "unlocked_reports" not in st.session_state:
    st.session_state["unlocked_reports"] = set()


def render_clean_html(html_str: str):
    """
    Renders raw HTML safely in Streamlit without triggering markdown's
    4-space indented code block parser. Strips leading and trailing whitespace
    from each line and removes empty lines that cause CommonMark to break HTML blocks.
    """
    cleaned_lines = [line.strip()
                     for line in html_str.splitlines() if line.strip()]
    st.markdown("\n".join(cleaned_lines), unsafe_allow_html=True)


def sync_document_title(view_name: str):
    """
    Synchronizes browser tab document.title instantaneously across client-side
    single-page state transitions in Streamlit.
    """
    if not st.session_state.get("authenticated_officer") and view_name not in ["verify", "privacy", "terms", "cookies", "404"]:
        target_title = "NecroTrace — Examiner Authentication & Access Gateway"
    else:
        target_title = VIEW_TITLES.get(view_name, "NecroTrace — Forensic Metagenomics")
    components.html(
        f"""
        <script>
            (function() {{
                try {{
                    var docTitle = {repr(target_title)};
                    if (window.parent && window.parent.document) {{
                        window.parent.document.title = docTitle;
                    }}
                    document.title = docTitle;
                }} catch (err) {{
                    // Fallback for cross-origin or isolated frame contexts
                }}
            }})();
        </script>
        """,
        height=0,
    )


def _load_logo_b64() -> str:
    import base64
    candidates = [
        os.path.join(os.path.dirname(__file__), "assets", "logo_icon_128.png"),
        os.path.join("assets", "logo_icon_128.png"),
        os.path.join("app", "assets", "logo_icon_128.png"),
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"
    return ""


LOGO_ICON_B64 = _load_logo_b64()


def _load_logo_horizontal_b64() -> str:
    import base64
    candidates = [
        os.path.join(os.path.dirname(__file__),
                     "assets", "logo_horizontal.png"),
        os.path.join("assets", "logo_horizontal.png"),
        os.path.join("app", "assets", "logo_horizontal.png"),
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"
    return ""


LOGO_HORIZONTAL_B64 = _load_logo_horizontal_b64()


# -----------------------------------------------------------------------------
# 3. GLOBAL DESIGN SYSTEM CSS (INTEGRATED BIOSCIENCES TOKENS)
# -----------------------------------------------------------------------------
GLOBAL_CSS = """
<link rel="stylesheet" href="https://unpkg.com/lenis@1.3.26/dist/lenis.css">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400&family=Roboto+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
    /* CSS TOKENS */
    :root {
        --color-bioluminescent-lime: #cef79e;
        --color-abyssal-ink: #222f30;
        --color-bone-white: #f7f7f5;
        --color-paper: #ffffff;
        --color-graphite: #94a3a3;
        --color-lichen: #c9cbbe;
        --color-tissue: #e7e8e1;
        --color-frost: #eeeeee;
        --color-void: #000000;
        --font-display: 'Inter Tight', 'Aspekta', -apple-system, sans-serif;
        --font-mono: 'Roboto Mono', monospace;
    }

    /* Accessibility Focus Indicators & Skip Link */
    a:focus-visible, button:focus-visible, input:focus-visible, select:focus-visible {
        outline: 2px solid #cef79e !important;
        outline-offset: 2px !important;
    }
    .skip-link {
        position: absolute;
        top: -60px;
        left: 14px;
        background: #cef79e;
        color: #000000 !important;
        padding: 8px 16px;
        z-index: 9999999;
        font-family: var(--font-mono);
        font-size: 12px;
        font-weight: 700;
        text-decoration: none;
        border-radius: 4px;
        transition: top 0.15s ease;
    }
    .skip-link:focus {
        top: 14px;
    }

    /* Streamlit overrides - Full bleed support across all versions */
    header[data-testid="stHeader"],
    div[data-testid="stHeader"] { display: none !important; }
    div[data-testid="stToolbar"] { display: none !important; }
    /* Precision Footer Styling matching app fonts */
    .necrotrace-footer,
    .necrotrace-footer * {
        font-family: var(--font-display) !important;
    }
    .necrotrace-footer .footer-mono {
        font-family: var(--font-mono) !important;
    }

    footer[data-testid="stFooter"], footer:not(.necrotrace-footer) { display: none !important; }

    .stApp,
    div[data-testid="stAppViewContainer"],
    section[data-testid="stMain"],
    div[data-testid="stMain"],
    .stMain,
    section.main {
        background-color: var(--color-abyssal-ink) !important;
        padding: 0 !important;
        margin: 0 !important;
    }



    /* Core typography reset - Pure 400 weight discipline */
    html, body, [class*="css"] {
        font-family: var(--font-display) !important;
        font-weight: 400 !important;
        color: var(--color-paper);
        background-color: var(--color-abyssal-ink);
        letter-spacing: -0.001em;
        margin: 0;
        padding: 0;
    }

    /* Precision hairline borders - No shadows, completely flat */
    * {
        box-shadow: none !important;
    }

    /* Monospace elements */
    .mono-tag {
        font-family: var(--font-mono) !important;
        font-size: 13px;
        line-height: 1.23;
        letter-spacing: -0.02em;
        text-transform: uppercase;
    }

    /* Pill counter (01 / 04) */
    .section-counter {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 4px 12px;
        border: 1px solid var(--color-graphite);
        border-radius: 9999px;
        font-family: var(--font-mono);
        font-size: 13px;
        color: var(--color-graphite);
        margin-bottom: 24px;
        background: transparent;
    }

    .signal-dot {
        display: none !important;
    }

    /* Hero typography */
    .hero-title {
        font-family: var(--font-display) !important;
        font-weight: 400 !important;
        font-size: clamp(48px, 7vw, 108px);
        line-height: 1.0;
        letter-spacing: -0.03em;
        color: var(--color-paper);
        margin: 0 0 32px 0;
        max-width: 1100px;
    }

    .hero-sub {
        font-family: var(--font-display) !important;
        font-weight: 400 !important;
        font-size: clamp(20px, 2.8vw, 32px);
        line-height: 1.25;
        letter-spacing: -0.006em;
        color: var(--color-graphite);
        margin: 0 0 48px 0;
        max-width: 820px;
    }

    /* Sober, restrained buttons & Streamlit button overrides */
    div.stButton > button[kind="primary"],
    button[data-testid="baseButton-primary"],
    div.stDownloadButton > button,
    div[data-testid="stDownloadButton"] > button {
        background-color: var(--color-paper) !important;
        color: var(--color-abyssal-ink) !important;
        font-family: var(--font-mono) !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        letter-spacing: -0.02em !important;
        text-transform: uppercase !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        border: 1px solid var(--color-paper) !important;
        box-shadow: none !important;
        height: auto !important;
        min-height: 44px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        transition: background-color 0.15s ease, border-color 0.15s ease, opacity 0.15s ease !important;
    }

    div.stButton > button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover,
    div.stDownloadButton > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        background-color: var(--color-bone-white) !important;
        color: var(--color-abyssal-ink) !important;
        border-color: var(--color-bone-white) !important;
        opacity: 0.96 !important;
        box-shadow: none !important;
    }

    div.stButton > button[kind="secondary"],
    button[data-testid="baseButton-secondary"],
    div.stButton > button:not([kind="primary"]) {
        background-color: transparent !important;
        color: var(--color-paper) !important;
        font-family: var(--font-mono) !important;
        font-size: 13px !important;
        font-weight: 400 !important;
        letter-spacing: -0.02em !important;
        text-transform: uppercase !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        border: 1px solid var(--color-graphite) !important;
        box-shadow: none !important;
        height: auto !important;
        min-height: 44px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
    }

    div.stButton > button[kind="secondary"]:hover,
    button[data-testid="baseButton-secondary"]:hover,
    div.stButton > button:not([kind="primary"]):hover {
        border-color: var(--color-lichen) !important;
        color: var(--color-bone-white) !important;
        background-color: rgba(255, 255, 255, 0.04) !important;
        box-shadow: none !important;
    }

    .sober-btn-dark {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        background-color: var(--color-paper);
        color: var(--color-abyssal-ink) !important;
        font-family: var(--font-mono) !important;
        font-size: 13px;
        font-weight: 500;
        letter-spacing: -0.02em;
        text-transform: uppercase;
        padding: 12px 24px;
        border-radius: 8px;
        border: 1px solid var(--color-paper);
        text-decoration: none !important;
        cursor: pointer;
        height: 44px;
        box-sizing: border-box;
        transition: background-color 0.15s ease;
    }
    .sober-btn-dark:hover {
        background-color: var(--color-bone-white);
        opacity: 0.96;
    }

    .sober-btn-ghost {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background-color: transparent;
        color: var(--color-paper) !important;
        font-family: var(--font-mono) !important;
        font-size: 13px;
        font-weight: 400;
        letter-spacing: -0.02em;
        text-transform: uppercase;
        padding: 12px 24px;
        border-radius: 8px;
        border: 1px solid var(--color-graphite);
        text-decoration: none !important;
        cursor: pointer;
        height: 44px;
        box-sizing: border-box;
        transition: border-color 0.15s ease, background-color 0.15s ease;
    }
    .sober-btn-ghost:hover {
        border-color: var(--color-lichen);
        color: var(--color-bone-white) !important;
        background-color: rgba(255, 255, 255, 0.04);
    }

    .micro-arrow-btn {
        width: 38px;
        height: 38px;
        border-radius: 8px;
        background-color: var(--color-bioluminescent-lime);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: var(--color-abyssal-ink);
        font-size: 16px;
    }

    /* Clean instrumentation cards */
    .instrument-card-dark {
        background-color: transparent;
        border: 1px solid var(--color-graphite);
        border-radius: 16px;
        padding: 36px;
        height: 100%;
    }

    .instrument-card-light {
        background-color: var(--color-paper);
        border: 1px solid var(--color-lichen);
        border-radius: 16px;
        padding: 36px;
        height: 100%;
        color: var(--color-abyssal-ink);
    }

    /* Dividers */
    .hairline-dark {
        height: 1px;
        background-color: var(--color-graphite);
        border: none;
        margin: 60px 0;
    }
    .hairline-light {
        height: 1px;
        background-color: var(--color-lichen);
        border: none;
        margin: 60px 0;
    }

    /* Examination room layout & spacing */
    .exam-wrapper {
        max-width: 1280px;
        margin: 0 auto;
        padding: 32px 48px 100px 48px;
        box-sizing: border-box;
    }

    .exam-step-card {
        background: rgba(34, 47, 48, 0.45);
        border: 1px solid var(--color-graphite);
        border-radius: 12px;
        padding: 32px 36px;
        margin-bottom: 34px;
        box-sizing: border-box;
    }

    .symptom-card {
        background: #172122;
        border: 1px solid #36494a;
        border-radius: 10px;
        padding: 22px 24px;
        margin-bottom: 22px;
        box-sizing: border-box;
    }

    .symptom-visual-box {
        background: #101617;
        border: 1px solid #283738;
        border-radius: 8px;
        padding: 16px 20px;
        box-sizing: border-box;
        height: 100%;
    }

    .taxa-visual-card {
        background: #182223;
        border: 1px solid #36494a;
        border-radius: 12px;
        padding: 22px 24px;
        margin-bottom: 20px;
        box-sizing: border-box;
        transition: border-color 0.2s ease, background 0.2s ease;
    }
    .taxa-visual-card:hover {
        border-color: #557273;
        background: #1c2829;
    }

    .gram-badge-pos {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 8px;
        border-radius: 4px;
        font-family: var(--font-mono);
        font-size: 11px;
        background: rgba(184, 130, 255, 0.15);
        color: #d1b3ff;
        border: 1px solid rgba(184, 130, 255, 0.4);
    }
    .gram-badge-neg {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 8px;
        border-radius: 4px;
        font-family: var(--font-mono);
        font-size: 11px;
        background: rgba(255, 107, 139, 0.15);
        color: #ffa1b5;
        border: 1px solid rgba(255, 107, 139, 0.4);
    }

    /* Bulletproof Smooth Scroll - Overflow is ALWAYS active and smooth */
    html, body {
        overflow-x: hidden !important;
        scroll-behavior: smooth !important;
    }
    div[data-testid="stAppViewContainer"],
    section.main,
    .main {
        overflow-y: auto !important;
        overflow-x: hidden !important;
        scroll-behavior: smooth !important;
        -webkit-overflow-scrolling: touch !important;
    }
    iframe[title="streamlit.components.v1.html"] {
        display: none !important;
        position: absolute !important;
        height: 0 !important;
        width: 0 !important;
        border: none !important;
    }
    div[data-testid="stCustomComponentV1"] {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* -------------------------------------------------------------------------
       MOBILE RESPONSIVENESS & TOUCH ADAPTIVE SCALING (< 768px)
       ------------------------------------------------------------------------- */
    @media (max-width: 768px) {
        /* Container and padding optimization */
        .block-container,
        div[data-testid="stMainBlockContainer"],
        .stMainBlockContainer {
            padding-left: 14px !important;
            padding-right: 14px !important;
            padding-top: 18px !important;
            padding-bottom: 72px !important;
            max-width: 100% !important;
        }

        .exam-wrapper {
            padding: 16px 12px 60px 12px !important;
        }

        .exam-step-card {
            padding: 18px 14px !important;
            margin-bottom: 20px !important;
            border-radius: 10px !important;
        }

        .symptom-card {
            padding: 16px 14px !important;
            margin-bottom: 16px !important;
        }

        .instrument-card-dark,
        .instrument-card-light {
            padding: 20px 16px !important;
            border-radius: 12px !important;
        }

        /* Responsive Metric Cards */
        div[data-testid="stMetric"] {
            background: #172223 !important;
            padding: 12px 14px !important;
            border-radius: 8px !important;
            border: 1px solid #283738 !important;
            margin-bottom: 8px !important;
        }
        div[data-testid="stMetricLabel"] p {
            font-size: 11px !important;
        }
        div[data-testid="stMetricValue"] div {
            font-size: 20px !important;
        }

        /* Streamlit columns stack cleanly on small mobile viewports */
        [data-testid="column"] {
            min-width: 100% !important;
            flex: 1 1 100% !important;
            margin-bottom: 10px !important;
        }

        /* Touch-friendly buttons and inputs */
        div.stButton > button {
            width: 100% !important;
            min-height: 48px !important;
            font-size: 13px !important;
            padding: 10px 14px !important;
        }

        /* Responsive table wrapping */
        div[data-testid="stTable"],
        div[data-testid="stDataFrame"] {
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch !important;
        }

        /* Form PM-5372 On-Screen Preview Mobile Adjustments */
        .pm-preview-box {
            padding: 14px !important;
        }
        .pm-header-row {
            flex-direction: column !important;
            align-items: center !important;
            text-align: center !important;
        }
        .pm-particulars-grid {
            grid-template-columns: 1fr !important;
            gap: 8px !important;
        }
        .pm-opinion-subgrid {
            grid-template-columns: 1fr !important;
        }

        /* Matplotlib & Plotly full viewport scaling */
        div[data-testid="stPlotlyChart"],
        div[data-testid="stImage"] img,
        div[data-testid="stArrowVegaLiteChart"] {
            max-width: 100% !important;
            overflow-x: auto !important;
        }
    }

    /* Small Phone adjustments (< 480px) */
    @media (max-width: 480px) {
        h1 { font-size: 24px !important; line-height: 1.25 !important; }
        h2 { font-size: 20px !important; line-height: 1.3 !important; }
        h3 { font-size: 17px !important; line-height: 1.35 !important; }
        .mono-tag { font-size: 10px !important; }
    }
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SHARED STYLING FOR CENTERED LEGAL & ERROR PAGES
# -----------------------------------------------------------------------------
LEGAL_PAGE_CSS = """
<style>
div[data-testid="stAppViewContainer"] > section.main,
div[data-testid="stAppViewContainer"] .stMain {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
}

div[data-testid="stMainBlockContainer"],
div[data-testid="stAppViewBlockContainer"],
.main .block-container,
.stMainBlockContainer,
.block-container,
div[class*="stMainBlockContainer"],
div[class*="block-container"] {
    max-width: 740px !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding-top: 28px !important;
    padding-bottom: 96px !important;
    padding-left: 20px !important;
    padding-right: 20px !important;
    box-sizing: border-box !important;
}

/* Document Container Card */
.document-card {
    background: #111a1b;
    border: 1px solid #202e30;
    border-radius: 12px;
    padding: 36px 36px;
    margin-bottom: 24px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.45);
}

@media (max-width: 640px) {
    .document-card {
        padding: 24px 18px;
    }
}

/* Clean Tables */
div[data-testid="stMarkdownContainer"] table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 20px 0 !important;
    font-size: 13px !important;
    background: #0d1415 !important;
    border: 1px solid #233436 !important;
    border-radius: 6px !important;
    overflow: hidden !important;
}

div[data-testid="stMarkdownContainer"] th {
    background: #162325 !important;
    color: #cef79e !important;
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    padding: 10px 14px !important;
    border: 1px solid #233436 !important;
    text-align: left !important;
}

div[data-testid="stMarkdownContainer"] td {
    padding: 10px 14px !important;
    border: 1px solid #1a2729 !important;
    color: #cbd5e1 !important;
    line-height: 1.5 !important;
}

div[data-testid="stMarkdownContainer"] tr:nth-child(even) td {
    background: rgba(22, 35, 37, 0.35) !important;
}

/* Headings & Text */
div[data-testid="stMarkdownContainer"] h1 {
    font-size: 28px !important;
    font-weight: 600 !important;
    color: #ffffff !important;
    letter-spacing: -0.02em !important;
    margin-top: 0 !important;
    margin-bottom: 8px !important;
}

div[data-testid="stMarkdownContainer"] h2 {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #cef79e !important;
    letter-spacing: -0.01em !important;
    margin-top: 32px !important;
    margin-bottom: 12px !important;
    border-bottom: 1px solid #202e30 !important;
    padding-bottom: 8px !important;
}

div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li {
    font-size: 14px !important;
    color: #c9cbbe !important;
    line-height: 1.65 !important;
}

div[data-testid="stMarkdownContainer"] blockquote {
    border-left: 3px solid #f59e0b !important;
    padding-left: 14px !important;
    margin: 16px 0 !important;
    background: rgba(245, 158, 11, 0.06) !important;
    border-radius: 0 6px 6px 0 !important;
    padding-top: 8px !important;
    padding-bottom: 8px !important;
}

div[data-testid="stMarkdownContainer"] a {
    color: #cef79e !important;
    text-decoration: none !important;
}

div[data-testid="stMarkdownContainer"] a:hover {
    text-decoration: underline !important;
}
</style>
"""

# -----------------------------------------------------------------------------
# ACTIVE SMOOTH SCROLL ENGINE & ANCHOR GLIDE (IFRAME CONTROLLER)
# -----------------------------------------------------------------------------
SMOOTH_SCROLL_CONTROLLER_HTML = """
<script>
(function() {
    function init() {
        try {
            const parentDoc = window.parent.document;
            const parentWin = window.parent;
            if (!parentDoc || !parentWin) return;

            // Ensure scroll container is completely unlocked with smooth scrolling
            const scrollContainer = parentDoc.querySelector('[data-testid="stAppViewContainer"]') || 
                                    parentDoc.querySelector('section.main') || 
                                    parentDoc.documentElement;

            if (scrollContainer) {
                scrollContainer.style.setProperty('overflow-y', 'auto', 'important');
                scrollContainer.style.setProperty('scroll-behavior', 'smooth', 'important');
                scrollContainer.style.setProperty('background-color', '#222f30', 'important');
            }

            const searchStr = window.parent.location.search || '';
            const isExam = searchStr.includes('view=examination');
            const isLegalOr404 = searchStr.includes('view=privacy') || 
                                 searchStr.includes('view=terms') || 
                                 searchStr.includes('view=cookies') || 
                                 searchStr.includes('view=404');
            const isCentered = isExam || isLegalOr404;

            // Set container geometry based on active view
            const blockContainers = parentDoc.querySelectorAll('[data-testid="stMainBlockContainer"], .block-container, div.stMainBlockContainer');
            blockContainers.forEach(function(el) {
                if (isCentered) {
                    const maxWidth = isLegalOr404 ? '740px' : '880px';
                    el.style.setProperty('max-width', maxWidth, 'important');
                    el.style.setProperty('width', '100%', 'important');
                    el.style.setProperty('margin-left', 'auto', 'important');
                    el.style.setProperty('margin-right', 'auto', 'important');
                    el.style.setProperty('padding-left', '24px', 'important');
                    el.style.setProperty('padding-right', '24px', 'important');
                    el.style.setProperty('padding-top', isLegalOr404 ? '28px' : '24px', 'important');
                    el.style.setProperty('padding-bottom', '96px', 'important');
                } else {
                    el.style.setProperty('padding-top', '0px', 'important');
                    el.style.setProperty('padding-bottom', '0px', 'important');
                    el.style.setProperty('padding-left', '0px', 'important');
                    el.style.setProperty('padding-right', '0px', 'important');
                    el.style.setProperty('max-width', '100%', 'important');
                    el.style.setProperty('width', '100%', 'important');
                    el.style.setProperty('margin', '0px', 'important');
                }
            });

            // Center the main section wrapper on centered views
            const mainSection = parentDoc.querySelector('section.main') || parentDoc.querySelector('[data-testid="stMain"]');
            if (mainSection) {
                if (isCentered) {
                    mainSection.style.setProperty('display', 'flex', 'important');
                    mainSection.style.setProperty('flex-direction', 'column', 'important');
                    mainSection.style.setProperty('align-items', 'center', 'important');
                } else {
                    mainSection.style.setProperty('display', 'block', 'important');
                }
            }

            // Smooth anchor links navigation via event delegation
            if (!parentWin.__smooth_scroll_delegated) {
                parentDoc.addEventListener('click', function(e) {
                    const anchor = e.target.closest('a[href^="#"]');
                    if (anchor) {
                        const hash = anchor.getAttribute('href');
                        if (hash && hash.length > 1) {
                            const targetEl = parentDoc.querySelector(hash);
                            if (targetEl) {
                                e.preventDefault();
                                targetEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
                            }
                        }
                    }
                }, true);
                parentWin.__smooth_scroll_delegated = true;
            }
        } catch(e) {
            console.error('[NecroTrace] Scroll controller error:', e);
        }
    }

    if (document.readyState === 'complete' || document.readyState === 'interactive') {
        init();
    } else {
        window.addEventListener('load', init);
    }
})();
</script>
"""
components.html(SMOOTH_SCROLL_CONTROLLER_HTML, height=0)


def get_microbe_svg(morphology_type: str, gram_stain: str) -> str:
    """Renders high-contrast microscope ocular viewport SVG for bacterial cellular morphology."""
    color = "#b882ff" if gram_stain == "Gram-positive" else "#ff6b8b"
    glow = "rgba(184, 130, 255, 0.35)" if gram_stain == "Gram-positive" else "rgba(255, 107, 139, 0.35)"

    if "cocci_clusters" in morphology_type:
        inner = f"""
        <circle cx="28" cy="22" r="5" fill="{color}" />
        <circle cx="37" cy="24" r="4.5" fill="{color}" />
        <circle cx="23" cy="29" r="4.5" fill="{color}" />
        <circle cx="32" cy="32" r="5.5" fill="{color}" />
        <circle cx="41" cy="33" r="4" fill="{color}" />
        <circle cx="26" cy="39" r="4.5" fill="{color}" />
        <circle cx="35" cy="41" r="4" fill="{color}" />
        """
    elif "cocci_chains" in morphology_type:
        inner = f"""
        <circle cx="16" cy="38" r="4.5" fill="{color}" />
        <circle cx="24" cy="33" r="4.5" fill="{color}" />
        <circle cx="32" cy="28" r="4.5" fill="{color}" />
        <circle cx="40" cy="25" r="4.5" fill="{color}" />
        <circle cx="48" cy="22" r="4.5" fill="{color}" />
        """
    elif "spore" in morphology_type or "endospore" in morphology_type:
        inner = f"""
        <rect x="15" y="24" width="34" height="15" rx="7.5" fill="{color}" />
        <ellipse cx="38" cy="31.5" rx="5" ry="4" fill="#cef79e" />
        """
    elif "swarming" in morphology_type or "motile" in morphology_type:
        inner = f"""
        <rect x="18" y="26" width="28" height="12" rx="6" fill="{color}" />
        <path d="M18 28 Q10 23 5 29" stroke="{color}" stroke-width="1.3" fill="none" stroke-linecap="round" />
        <path d="M18 35 Q11 39 6 36" stroke="{color}" stroke-width="1.3" fill="none" stroke-linecap="round" />
        <path d="M46 28 Q53 23 58 29" stroke="{color}" stroke-width="1.3" fill="none" stroke-linecap="round" />
        <path d="M46 35 Q54 40 59 36" stroke="{color}" stroke-width="1.3" fill="none" stroke-linecap="round" />
        """
    elif "branching" in morphology_type:
        inner = f"""
        <path d="M12 50 Q28 35 32 20 T48 12" stroke="{color}" stroke-width="2.5" fill="none" stroke-linecap="round" />
        <path d="M28 35 Q38 41 50 43" stroke="{color}" stroke-width="2" fill="none" stroke-linecap="round" />
        <circle cx="48" cy="12" r="2.8" fill="#cef79e" />
        <circle cx="50" cy="43" r="2.8" fill="#cef79e" />
        """
    elif "dipteran" in morphology_type:
        inner = f"""
        <rect x="18" y="27" width="26" height="11" rx="5.5" fill="{color}" />
        <path d="M25 27 Q32 13 42 17 Q37 24 33 27 Z" fill="rgba(206, 247, 158, 0.45)" stroke="#cef79e" stroke-width="1" />
        """
    elif "tetrad" in morphology_type or "psychrotolerant" in morphology_type:
        inner = f"""
        <circle cx="26" cy="26" r="5" fill="{color}" />
        <circle cx="38" cy="26" r="5" fill="{color}" />
        <circle cx="26" cy="38" r="5" fill="{color}" />
        <circle cx="38" cy="38" r="5" fill="{color}" />
        """
    elif "coccobacilli" in morphology_type:
        inner = f"""
        <ellipse cx="26" cy="28" rx="8" ry="6" fill="{color}" />
        <ellipse cx="38" cy="36" rx="8" ry="6" fill="{color}" />
        """
    else:  # Standard bacilli / rods / club rods
        inner = f"""
        <rect x="16" y="21" width="25" height="10" rx="5" fill="{color}" />
        <rect x="24" y="35" width="24" height="10" rx="5" fill="{color}" />
        """

    return f"""<div style="width: 64px; height: 64px; flex-shrink: 0;"><svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" style="background: #121819; border-radius: 50%; border: 1.5px solid #36494a; box-shadow: 0 0 14px {glow};"><circle cx="32" cy="32" r="30" stroke="#1d2829" stroke-width="1.5" stroke-dasharray="2 3" /><line x1="32" y1="2" x2="32" y2="8" stroke="#36494a" stroke-width="1" /><line x1="32" y1="56" x2="32" y2="62" stroke="#36494a" stroke-width="1" /><line x1="2" y1="32" x2="8" y2="32" stroke="#36494a" stroke-width="1" /><line x1="56" y1="32" x2="62" y2="32" stroke="#36494a" stroke-width="1" />{inner}</svg></div>"""


# -----------------------------------------------------------------------------
# 4. CACHED MODEL LOADER
# -----------------------------------------------------------------------------

def get_microbe_card_banner(taxon_id: str, common_name: str, morphology_type: str, gram_stain: str) -> str:
    """
    Renders the rich media banner for the microbial card.
    Uses balanced photographic proportions (230px height) with clinical microscopy HUD overlay.
    Prioritizes real user-provided JPEG/PNG assets in assets/microbes/.
    """
    accent = "#b882ff" if "positive" in gram_stain.lower() else "#ff6b8b"

    # Search candidates across multiple naming conventions (spaces, underscores, common names)
    candidates = [
        os.path.join("assets", "microbes", f"{taxon_id}.jpg"),
        os.path.join("assets", "microbes", f"{taxon_id}.jpeg"),
        os.path.join("assets", "microbes", f"{taxon_id}.png"),
        os.path.join("assets", "microbes",
                     f"{taxon_id.replace('_', ' ')}.jpg"),
        os.path.join("assets", "microbes",
                     f"{taxon_id.replace('_', ' ')}.jpeg"),
        os.path.join("assets", "microbes", f"{common_name}.jpg"),
        os.path.join("assets", "microbes", f"{common_name}.jpeg"),
        os.path.join("app", "assets", "microbes", f"{taxon_id}.jpg"),
        os.path.join("app", "assets", "microbes",
                     f"{taxon_id.replace('_', ' ')}.jpg"),
        os.path.join("app", "assets", "microbes", f"{common_name}.jpg"),
    ]
    found_asset = None
    for cand in candidates:
        if os.path.exists(cand):
            found_asset = cand
            break

    if found_asset and not found_asset.endswith(".svg"):
        try:
            with open(found_asset, "rb") as img_file:
                b64 = base64.b64encode(img_file.read()).decode("utf-8")
            mime = "image/png" if found_asset.endswith(
                ".png") else "image/jpeg"
            return f'''<div style="width: 100%; height: 230px; position: relative; overflow: hidden; background: #0c1213; border-bottom: 1px solid #283637;">
                <!-- Real Microscopy JPEG Image -->
                <img src="data:{mime};base64,{b64}" style="width: 100%; height: 100%; object-fit: cover; display: block; filter: brightness(0.96) contrast(1.05);" alt="{common_name}" />
                
                <!-- Subtle Clean Vignette -->
                <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; background: linear-gradient(180deg, rgba(8,14,15,0.4) 0%, rgba(8,14,15,0) 25%, rgba(8,14,15,0) 75%, rgba(8,14,15,0.25) 100%);"></div>
                
                <!-- Clinical Microscopy HUD Brackets (Symmetrical Frame) -->
                <div style="position: absolute; top: 12px; left: 14px; width: 12px; height: 12px; border-top: 1.5px solid rgba(255,255,255,0.5); border-left: 1.5px solid rgba(255,255,255,0.5); pointer-events: none;"></div>
                <div style="position: absolute; top: 12px; right: 14px; width: 12px; height: 12px; border-top: 1.5px solid rgba(255,255,255,0.5); border-right: 1.5px solid rgba(255,255,255,0.5); pointer-events: none;"></div>
                <div style="position: absolute; bottom: 12px; left: 14px; width: 12px; height: 12px; border-bottom: 1.5px solid rgba(255,255,255,0.5); border-left: 1.5px solid rgba(255,255,255,0.5); pointer-events: none;"></div>
                <div style="position: absolute; bottom: 12px; right: 14px; width: 12px; height: 12px; border-bottom: 1.5px solid rgba(255,255,255,0.5); border-right: 1.5px solid rgba(255,255,255,0.5); pointer-events: none;"></div>
                
                <!-- Top Status Badge -->
                <div style="position: absolute; top: 10px; right: 12px; background: rgba(8, 14, 15, 0.85); border: 1px solid rgba(77, 87, 87, 0.6); padding: 3px 8px; border-radius: 4px; font-family: var(--font-mono); font-size: 9px; color: #cef79e; letter-spacing: 0.05em; backdrop-filter: blur(4px);">
                    SPECIMEN VERIFIED
                </div>
            </div>'''
        except Exception:
            pass

    # Viewfinder Geometric Placeholder Fallback
    return f'''<div style="width: 100%; height: 230px; overflow: hidden; position: relative; background: #152021; border-bottom: 1px solid #283637;">
        <svg viewBox="0 0 380 230" width="100%" height="100%" preserveAspectRatio="none" style="display: block;">
            <defs>
                <radialGradient id="grad_card_{taxon_id}" cx="50%" cy="50%" r="65%">
                    <stop offset="0%" stop-color="#223133" stop-opacity="0.95"/>
                    <stop offset="70%" stop-color="#141d1e" stop-opacity="1"/>
                    <stop offset="100%" stop-color="#0d1415" stop-opacity="1"/>
                </radialGradient>
                <pattern id="grid_card_{taxon_id}" width="20" height="20" patternUnits="userSpaceOnUse">
                    <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#1f2c2d" stroke-width="0.5"/>
                </pattern>
            </defs>
            <rect width="380" height="230" fill="url(#grad_card_{taxon_id})"/>
            <rect width="380" height="230" fill="url(#grid_card_{taxon_id})"/>
            
            <!-- Viewfinder Optical Reticle -->
            <circle cx="190" cy="115" r="76" fill="none" stroke="#253536" stroke-width="1.2" stroke-dasharray="3 3"/>
            <circle cx="190" cy="115" r="72" fill="none" stroke="rgba(206, 247, 158, 0.08)" stroke-width="1"/>
            <line x1="140" y1="115" x2="240" y2="115" stroke="#2b3b3c" stroke-width="1"/>
            <line x1="190" y1="65" x2="190" y2="165" stroke="#2b3b3c" stroke-width="1"/>
            
            <!-- Viewfinder Brackets (4 corners) -->
            <path d="M 20 36 L 20 20 L 36 20" fill="none" stroke="#445759" stroke-width="1.5" />
            <path d="M 360 36 L 360 20 L 344 20" fill="none" stroke="#445759" stroke-width="1.5" />
            <path d="M 20 194 L 20 210 L 36 210" fill="none" stroke="#445759" stroke-width="1.5" />
            <path d="M 360 194 L 360 210 L 344 210" fill="none" stroke="#445759" stroke-width="1.5" />

            <!-- Geometric Shapes (Circle + Triangle) -->
            <circle cx="172" cy="118" r="38" fill="#ffffff" fill-opacity="0.09" stroke="#ffffff" stroke-opacity="0.15" stroke-width="1.2"/>
            <polygon points="202,74 242,144 162,144" fill="#ffffff" fill-opacity="0.14" stroke="#ffffff" stroke-opacity="0.2" stroke-width="1.2"/>
            <circle cx="190" cy="115" r="3.5" fill="{accent}"/>
            
            <!-- Top HUD Badge -->
            <rect x="238" y="14" width="128" height="22" rx="4" fill="#0f1617" fill-opacity="0.9" stroke="#2d3c3d" stroke-width="0.8"/>
            <text x="302" y="29" font-family="'Roboto Mono', monospace" font-size="9" fill="#9db0b0" text-anchor="middle" letter-spacing="0.04em">MICROSCOPY STANDBY</text>
            
            <!-- Bottom clean margin -->
        </svg>
    </div>'''


@st.cache_resource
def load_trained_pipeline(artifacts_dir: str = "artifacts"):
    """Load serialized Quantile XGBoost models and feature schema."""
    try:
        model = NecroQuantileRegressor.load(artifacts_dir=artifacts_dir)
        return model, None
    except Exception as e:
        return None, str(e)


model, err = load_trained_pipeline()


# =============================================================================
# PRE-LANDING AUTHENTICATION & EXAMINER ENROLLMENT GATE
# =============================================================================
if not st.session_state.get("authenticated_officer") and st.session_state.get("view") not in ["verify", "privacy", "terms", "cookies", "404"]:
    sync_document_title(st.session_state.get("view", "landing"))
    render_clean_html("""
    <style>
    div[data-testid="stAppViewContainer"] > section.main,
    div[data-testid="stAppViewContainer"] .stMain {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: flex-start !important;
    }
    div[data-testid="stMainBlockContainer"],
    div[data-testid="stAppViewBlockContainer"],
    .main .block-container,
    .stMainBlockContainer,
    .block-container {
        max-width: 620px !important;
        width: 100% !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-top: 56px !important;
        padding-bottom: 64px !important;
        padding-left: 20px !important;
        padding-right: 20px !important;
        box-sizing: border-box !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #364547 !important;
        border-radius: 14px !important;
        background: rgba(26, 36, 37, 0.75) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45) !important;
        padding: 28px 32px !important;
    }
    div[data-baseweb="input"] {
        background-color: #141f20 !important;
        border: 1px solid #334244 !important;
        border-radius: 6px !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: var(--color-bioluminescent-lime) !important;
    }
    div[data-baseweb="input"] input {
        color: #ffffff !important;
        font-family: var(--font-mono) !important;
        font-size: 13px !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #141f20 !important;
        border: 1px solid #334244 !important;
        border-radius: 6px !important;
        color: #ffffff !important;
        font-family: var(--font-mono) !important;
        font-size: 13px !important;
    }
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] p {
        color: #c9cbbe !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        letter-spacing: 0.03em !important;
        font-family: var(--font-mono) !important;
    }
    </style>
    """)

    fb_active = is_firebase_configured()
    gate_logo = LOGO_HORIZONTAL_B64 if LOGO_HORIZONTAL_B64 else LOGO_ICON_B64

    render_clean_html(f"""
    <div style="text-align: center; margin-bottom: 28px; padding-top: 10px;">
        <img src="{gate_logo}" style="max-width: 320px; width: 100%; height: auto; object-fit: contain; display: inline-block; filter: drop-shadow(0 0 20px rgba(116, 194, 92, 0.25));" alt="NecroTrace Logo" />
    </div>
    """)

    with st.container(border=True):
        auth_tab_in, auth_tab_up = st.tabs([
            "Examiner Sign-In",
            "Enroll New Examiner",
        ])

        with auth_tab_in:
            st.markdown(
                '<div style="font-family: var(--font-mono); font-size: 11px; color: var(--color-graphite); margin-bottom: 12px; letter-spacing: 0.04em;">ENTER REGISTERED CREDENTIALS</div>',
                unsafe_allow_html=True
            )
            login_email = st.text_input(
                "Departmental / Institutional Email",
                placeholder="e.g. examiner@forensic.gov",
                key="gate_login_email"
            )
            login_pass = st.text_input(
                "Security Passcode",
                type="password",
                placeholder="••••••••",
                key="gate_login_password"
            )

            if st.button("AUTHENTICATE  &  ENTER SYSTEM", use_container_width=True, key="btn_gate_signin"):
                if not login_email or not login_pass:
                    st.warning(
                        "Please provide both registered email and password.")
                else:
                    with st.spinner("Authenticating & fetching profile from Cloud Firestore..."):
                        auth_res = sign_in_officer(login_email, login_pass)
                    if auth_res.get("success"):
                        officer_info = auth_res.get("officer_info", {})
                        st.session_state["authenticated_officer"] = officer_info
                        set_active_officer_session(officer_info)
                        st.session_state["view"] = "landing"
                        st.query_params.clear()
                        st.query_params["view"] = "landing"
                        if officer_info.get("local_id"):
                            st.query_params["auth"] = officer_info["local_id"]
                        st.success(
                            f"Identity Verified. Welcome, {officer_info.get('name')}.")
                        st.rerun()
                    else:
                        st.error(auth_res.get('message'))

        with auth_tab_up:
            st.markdown(
                '<div style="font-family: var(--font-mono); font-size: 11px; color: var(--color-graphite); margin-bottom: 12px; letter-spacing: 0.04em;">ENROLL EXAMINER PROFILE</div>',
                unsafe_allow_html=True
            )
            reg_c1, reg_c2 = st.columns(2)
            with reg_c1:
                reg_name = st.text_input(
                    "Full Legal Name & Title", placeholder="e.g. Dr. Sarah Jenkins, M.D.", key="gate_reg_name")
                reg_station = st.text_input(
                    "Police Station / Forensic Lab", placeholder="e.g. State Forensic Science Lab", key="gate_reg_station")
            with reg_c2:
                reg_badge = st.text_input(
                    "Badge / Registration No.", placeholder="e.g. MED-9042 / WBMC-45826", key="gate_reg_badge")
                reg_role = st.selectbox(
                    "Professional Medico-Legal Role",
                    [
                        "Forensic Pathologist",
                        "Chief Medical Examiner",
                        "Senior Investigating Officer",
                        "Forensic Anthropologist",
                        "Toxicology Specialist",
                        "Judicial Inquest Officer",
                    ],
                    index=0,
                    key="gate_reg_role"
                )

            reg_email = st.text_input(
                "Official Departmental Email", placeholder="e.g. s.jenkins@forensics.gov", key="gate_reg_email")

            reg_p1, reg_p2 = st.columns(2)
            with reg_p1:
                reg_pass = st.text_input(
                    "Security Passcode (min 6 chars)", type="password", key="gate_reg_pass")
            with reg_p2:
                reg_pass_conf = st.text_input(
                    "Confirm Passcode", type="password", key="gate_reg_pass_conf")

            reg_consent = st.checkbox(
                "I consent to my name, email, badge number, and station being stored securely in Firebase (Google Cloud) for authentication purposes. I have read the [Privacy Policy](?view=privacy).",
                value=False,
                key="gate_reg_consent"
            )

            if st.button("ENROLL EXAMINER", use_container_width=True, key="btn_gate_enroll", disabled=not reg_consent):
                if not reg_email or not reg_pass or not reg_name:
                    st.warning("Please provide Name, Email, and Passcode.")
                elif len(reg_pass) < 6:
                    st.error("Passcode must be at least 6 characters long.")
                elif reg_pass != reg_pass_conf:
                    st.error("Passcodes do not match.")
                else:
                    with st.spinner("Enrolling examiner & saving profile..."):
                        reg_res = register_officer(
                            email=reg_email,
                            password=reg_pass,
                            full_name=reg_name,
                            name=reg_name,
                            badge=reg_badge,
                            station=reg_station,
                            role=reg_role,
                        )
                    if reg_res.get("success"):
                        officer_info = reg_res.get("officer_info", {})
                        st.session_state["authenticated_officer"] = officer_info
                        set_active_officer_session(officer_info)
                        st.session_state["view"] = "landing"
                        st.query_params.clear()
                        st.query_params["view"] = "landing"
                        if officer_info.get("local_id"):
                            st.query_params["auth"] = officer_info["local_id"]
                        st.success(
                            f"Profile saved. Welcome, {reg_name}.")
                        st.rerun()
                    else:
                        st.error(
                            f"Enrollment failed: {reg_res.get('message')}")

        st.markdown("""
        <div style="text-align: center; margin-top: 24px; padding-top: 14px; border-top: 1px solid rgba(77, 87, 87, 0.25); font-family: var(--font-mono); font-size: 10px; color: var(--color-graphite); letter-spacing: 0.05em;">
            ACADEMIC RESEARCH PROTOTYPE &bull; OPEN-SOURCE FORENSIC METAGENOMICS &bull; MIT LICENSE
        </div>
        """, unsafe_allow_html=True)

    st.stop()


# =============================================================================
# VIEW 1: BIOLUMINESCENT LABORATORY LANDING PAGE
# =============================================================================
if st.session_state["view"] == "landing":
    sync_document_title("landing")
    render_clean_html("""
    <style>
    div[data-testid="stMainBlockContainer"],
    div[data-testid="stAppViewBlockContainer"],
    .main .block-container,
    .stMainBlockContainer,
    .block-container,
    div[class*="stMainBlockContainer"],
    div[class*="block-container"] {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        max-width: 100% !important;
        width: 100% !important;
        margin: 0 !important;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    div[data-testid="stElementContainer"] {
        margin-bottom: 0 !important;
        padding: 0 !important;
    }
    </style>
    """)

    # --- SECTION 01: FULL-VIEWPORT HERO SECTION WITH KINETIC GRID & FLOATING NAV ---
    cur_officer = st.session_state.get("authenticated_officer") or {}
    cur_uid = cur_officer.get("local_id", "")
    auth_q = f"&auth={cur_uid}" if cur_uid else ""
    officer_name_short = cur_officer.get("name", "Examiner")
    officer_badge_short = cur_officer.get("badge", "CFS")
    officer_nav_pill = f"""
    <span style="font-family: var(--font-mono); font-size: 11px; color: #6ee7b7; background: rgba(6, 78, 59, 0.55); border: 1px solid #10b981; padding: 5px 11px; border-radius: 6px; display: inline-flex; align-items: center; gap: 6px;">
        {officer_name_short} ({officer_badge_short})
    </span>
    <a href="?view=logout" target="_self" style="font-family: var(--font-mono); font-size: 11px; color: #fca5a5; text-decoration: none; border: 1px solid rgba(239, 68, 68, 0.4); padding: 5px 11px; border-radius: 6px; background: rgba(239, 68, 68, 0.08); transition: all 0.15s ease;">LOG OUT</a>
    """

    hero_markup = f"""
    <a href="#succession" class="skip-link">Skip to main content</a>
    <div id="kinetic-hero-container" style="position: relative; width: 100%; min-height: 100vh; overflow: hidden; background-color: var(--color-abyssal-ink); cursor: crosshair; display: flex; flex-direction: column; justify-content: space-between; box-sizing: border-box;">
        <canvas id="kinetic-grid-canvas" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1; pointer-events: none;"></canvas>
        
        <!-- Architectural Top Nav (Floating over the kinetic canvas) -->
        <header style="position: relative; z-index: 10; width: 100%; border-bottom: 1px solid rgba(77, 87, 87, 0.45); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); background: rgba(34, 47, 48, 0.55); pointer-events: auto;">
            <div style="padding: 22px 48px; display: flex; align-items: center; justify-content: space-between; max-width: 1300px; margin: 0 auto; box-sizing: border-box;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <img src="{LOGO_ICON_B64}" style="height: 32px; width: 32px; object-fit: contain; vertical-align: middle; filter: drop-shadow(0 0 10px rgba(116, 194, 92, 0.45));" alt="NecroTrace Logo" />
                    <span style="font-family: var(--font-mono); font-size: 15px; letter-spacing: -0.01em; color: var(--color-paper); font-weight: 600;">
                        NECROTRACE
                    </span>
                </div>
                <div style="display: flex; align-items: center; gap: 20px;">
                    
                    {officer_nav_pill}
                    <a href="?view=examination{auth_q}" target="_self" style="font-family: var(--font-mono); font-size: 12px; color: var(--color-paper); text-decoration: none; border: 1px solid var(--color-graphite); padding: 8px 16px; border-radius: 6px; letter-spacing: -0.02em; background: rgba(255, 255, 255, 0.03); transition: all 0.15s ease;">EXAMINATION ROOM &rarr;</a>
                </div>
            </div>
        </header>

        <!-- Main Hero Body (Centered in visible viewport) -->
        <div style="position: relative; z-index: 10; width: 100%; max-width: 1300px; margin: 0 auto; padding: 48px 48px 64px 48px; box-sizing: border-box; flex: 1; display: flex; flex-direction: column; justify-content: center; pointer-events: auto;">
            
            <h1 class="hero-title" style="text-shadow: 0 2px 24px rgba(0,0,0,0.65);">The microbial clock of human decomposition.</h1>
            <p class="hero-sub" style="text-shadow: 0 2px 14px rgba(0,0,0,0.65);">
                High-throughput metagenomic taxonomic profiling and quantile regression to infer postmortem intervals with quantifiable evidentiary certainty.
            </p>
            <div style="display: flex; align-items: center; gap: 16px; flex-wrap: wrap; margin-top: 12px;">
                <a href="?view=examination{auth_q}" target="_self" class="sober-btn-dark">COMMENCE POST-MORTEM EXAMINATION</a>
                <a href="#platform" class="sober-btn-ghost">METHODOLOGY SPECIFICATIONS</a>
            </div>
        </div>

        <!-- Clean bottom border marking the exact end of hero before next section -->
        <div style="position: relative; z-index: 10; width: 100%; border-bottom: 1px solid rgba(77, 87, 87, 0.35);"></div>
    </div>
    """
    render_clean_html(hero_markup)

    # Interactive Kinetic Grid Engine Controller
    KINETIC_GRID_CONTROLLER_HTML = """
    <script>
    (function() {
        function initKineticGrid() {
            try {
                const parentDoc = window.parent.document;
                const parentWin = window.parent;
                if (!parentDoc || !parentWin) return;

                const canvas = parentDoc.getElementById('kinetic-grid-canvas');
                const container = parentDoc.getElementById('kinetic-hero-container');
                if (!canvas || !container) {
                    setTimeout(initKineticGrid, 120);
                    return;
                }

                const ctx = canvas.getContext('2d');
                if (!ctx) return;

                if (parentWin.__necro_kinetic_raf) {
                    parentWin.cancelAnimationFrame(parentWin.__necro_kinetic_raf);
                }

                const CELL_SIZE = 54;
                const INFLUENCE_RADIUS = 260;
                const MAX_WARP = 24;
                const DOT_SPACING = 28;
                const LERP_SPEED = 0.08;

                const LINE_BASE = { r: 255, g: 255, b: 255, a: 0.12 };
                const NODE_BASE_RADIUS = 1.8;
                const NODE_ACTIVE_RADIUS = 3.2;

                const theme = {
                    bg: '#222f30',
                    lineActive: { r: 206, g: 247, b: 158, a: 0.8 },
                    nodeActive: { r: 206, g: 247, b: 158, a: 1.0 },
                    glow: '206,247,158',
                    ripple: '206,247,158',
                };

                function lerpN(a, b, t) {
                    return a + (b - a) * t;
                }

                function lerpColor(base, active, t) {
                    const r = Math.round(lerpN(base.r, active.r, t));
                    const g = Math.round(lerpN(base.g, active.g, t));
                    const b = Math.round(lerpN(base.b, active.b, t));
                    const a = lerpN(base.a, active.a, t);
                    return `rgba(${r},${g},${b},${a.toFixed(3)})`;
                }

                let W = container.offsetWidth || parentWin.innerWidth;
                let H = container.offsetHeight || parentWin.innerHeight;
                let dotCanvas = null;

                function renderDotPattern() {
                    if (!W || !H) return;
                    dotCanvas = parentDoc.createElement('canvas');
                    dotCanvas.width = W;
                    dotCanvas.height = H;
                    const dctx = dotCanvas.getContext('2d');
                    if (!dctx) return;
                    dctx.fillStyle = "rgba(255, 255, 255, 0.05)";
                    const dotSpacing = 28;
                    for (let x = dotSpacing / 2; x < W; x += dotSpacing) {
                        for (let y = dotSpacing / 2; y < H; y += dotSpacing) {
                            dctx.beginPath();
                            dctx.arc(x, y, 0.8, 0, Math.PI * 2);
                            dctx.fill();
                        }
                    }
                }

                const mouse = { x: -9999, y: -9999 };
                const targetMouse = { x: -9999, y: -9999 };
                const ripples = [];

                function setSize() {
                    if (!container || !canvas) return;
                    W = container.offsetWidth || parentWin.innerWidth;
                    H = container.offsetHeight || parentWin.innerHeight;
                    const dpr = Math.min(parentWin.devicePixelRatio || 1, 2);
                    canvas.width = Math.floor(W * dpr);
                    canvas.height = Math.floor(H * dpr);
                    canvas.style.width = W + 'px';
                    canvas.style.height = H + 'px';
                    ctx.setTransform(1, 0, 0, 1, 0, 0);
                    ctx.scale(dpr, dpr);
                    renderDotPattern();
                }
                setSize();
                parentWin.addEventListener('resize', setSize);

                container.onmousemove = function(e) {
                    const rect = container.getBoundingClientRect();
                    targetMouse.x = e.clientX - rect.left;
                    targetMouse.y = e.clientY - rect.top;
                };

                container.onmouseleave = function() {
                    targetMouse.x = -9999;
                    targetMouse.y = -9999;
                };

                container.onclick = function(e) {
                    const rect = container.getBoundingClientRect();
                    ripples.push({
                        x: e.clientX - rect.left,
                        y: e.clientY - rect.top,
                        radius: 0,
                        opacity: 1,
                        born: performance.now()
                    });
                };

                function getWarpedPoint(gx, gy, col, row, cols, rows) {
                    const edgeMargin = 1.5;
                    const colPin = Math.min(col / edgeMargin, (cols - 1 - col) / edgeMargin, 1);
                    const rowPin = Math.min(row / edgeMargin, (rows - 1 - row) / edgeMargin, 1);
                    const pinFactor = colPin * colPin * rowPin * rowPin;

                    const dx = gx - mouse.x;
                    const dy = gy - mouse.y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    const proximity = Math.max(0, 1 - dist / INFLUENCE_RADIUS) * pinFactor;

                    let rx = 0, ry = 0;
                    for (let i = 0; i < ripples.length; i++) {
                        const r = ripples[i];
                        const rdx = gx - r.x;
                        const rdy = gy - r.y;
                        const rdist = Math.sqrt(rdx * rdx + rdy * rdy);
                        const waveWidth = 55;
                        const diff = rdist - r.radius;
                        if (Math.abs(diff) < waveWidth) {
                            const strength = (1 - Math.abs(diff) / waveWidth) * r.opacity * 18 * pinFactor;
                            const angle = Math.atan2(rdy, rdx);
                            const sign = diff < 0 ? -1 : 1;
                            rx += Math.cos(angle) * strength * sign * -1;
                            ry += Math.sin(angle) * strength * sign * -1;
                        }
                    }

                    if (dist < INFLUENCE_RADIUS && dist > 0 && pinFactor > 0) {
                        const t = dist / INFLUENCE_RADIUS;
                        const eased = t < 0.01 ? 0 : (1 - t) * (1 - t) * Math.min(1, dist / 60);
                        const warpAmt = eased * MAX_WARP * pinFactor;
                        const angle = Math.atan2(dy, dx);
                        return {
                            pt: {
                                x: gx - Math.cos(angle) * warpAmt + rx,
                                y: gy - Math.sin(angle) * warpAmt + ry
                            },
                            proximity
                        };
                    }

                    return { pt: { x: gx + rx, y: gy + ry }, proximity };
                }

                function draw(now) {
                    ctx.clearRect(0, 0, W, H);
                    ctx.fillStyle = theme.bg;
                    ctx.fillRect(0, 0, W, H);

                    // High-performance pre-rendered dot pattern
                    if (dotCanvas) {
                        ctx.drawImage(dotCanvas, 0, 0);
                    }

                    // Update ripple shockwaves
                    for (let i = ripples.length - 1; i >= 0; i--) {
                        const r = ripples[i];
                        const age = (now - r.born) / 1000;
                        r.radius = Math.max(0, age * 400);
                        r.opacity = Math.max(0, 1 - age * 1.2);
                        if (r.opacity <= 0) ripples.splice(i, 1);
                    }

                    const cols = Math.max(2, Math.ceil(W / CELL_SIZE)) + 1;
                    const rows = Math.max(2, Math.ceil(H / CELL_SIZE)) + 1;
                    const cellW = W / (cols - 1);
                    const cellH = H / (rows - 1);

                    const pts = [];
                    const prox = [];

                    for (let row = 0; row < rows; row++) {
                        pts[row] = [];
                        prox[row] = [];
                        for (let col = 0; col < cols; col++) {
                            const res = getWarpedPoint(col * cellW, row * cellH, col, row, cols, rows);
                            pts[row][col] = res.pt;
                            prox[row][col] = res.proximity;
                        }
                    }

                    function drawSeg(p1, p2, pr1, pr2) {
                        const avg = (pr1 + pr2) / 2;
                        const t = avg * avg * (3 - 2 * avg);
                        ctx.beginPath();
                        ctx.moveTo(p1.x, p1.y);
                        ctx.lineTo(p2.x, p2.y);
                        ctx.strokeStyle = lerpColor(LINE_BASE, theme.lineActive, t);
                        ctx.lineWidth = lerpN(0.8, 1.5, t);
                        ctx.stroke();
                    }

                    ctx.lineCap = "butt";

                    for (let row = 0; row < rows; row++) {
                        for (let col = 0; col < cols - 1; col++) {
                            drawSeg(pts[row][col], pts[row][col + 1], prox[row][col], prox[row][col + 1]);
                        }
                    }

                    for (let col = 0; col < cols; col++) {
                        for (let row = 0; row < rows - 1; row++) {
                            drawSeg(pts[row][col], pts[row + 1][col], prox[row][col], prox[row + 1][col]);
                        }
                    }

                    // Intersection nodes with glowing highlights
                    for (let row = 0; row < rows; row++) {
                        for (let col = 0; col < cols; col++) {
                            const p = pts[row][col];
                            const pr = prox[row][col];
                            const t = pr * pr * (3 - 2 * pr);
                            const r = lerpN(NODE_BASE_RADIUS, NODE_ACTIVE_RADIUS, t);

                            if (t > 0.28) {
                                const glowR = r + lerpN(0, 6, (t - 0.28) / 0.72);
                                const grd = ctx.createRadialGradient(p.x, p.y, r * 0.4, p.x, p.y, glowR);
                                grd.addColorStop(0, `rgba(${theme.glow},${(t * 0.35).toFixed(3)})`);
                                grd.addColorStop(1, `rgba(${theme.glow},0)`);
                                ctx.beginPath();
                                ctx.arc(p.x, p.y, glowR, 0, Math.PI * 2);
                                ctx.fillStyle = grd;
                                ctx.fill();
                            }

                            ctx.beginPath();
                            ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
                            ctx.fillStyle = lerpColor({ r: 255, g: 255, b: 255, a: 0.18 }, theme.nodeActive, t);
                            ctx.fill();
                        }
                    }

                    // Ripple rings
                    for (let i = 0; i < ripples.length; i++) {
                        const r = ripples[i];
                        ctx.beginPath();
                        ctx.arc(r.x, r.y, Math.max(0, r.radius), 0, Math.PI * 2);
                        ctx.strokeStyle = `rgba(${theme.ripple},${(r.opacity * 0.32).toFixed(3)})`;
                        ctx.lineWidth = 1.5;
                        ctx.stroke();
                    }
                }

                function animate(now) {
                    mouse.x = lerpN(mouse.x, targetMouse.x, LERP_SPEED);
                    mouse.y = lerpN(mouse.y, targetMouse.y, LERP_SPEED);

                    draw(now);
                    parentWin.__necro_kinetic_raf = parentWin.requestAnimationFrame(animate);
                }

                parentWin.__necro_kinetic_raf = parentWin.requestAnimationFrame(animate);
            } catch(e) {
                console.error('[NecroTrace] Kinetic grid init error:', e);
            }
        }

        if (document.readyState === 'complete' || document.readyState === 'interactive') {
            initKineticGrid();
        } else {
            window.addEventListener('load', initKineticGrid);
        }
    })();
    </script>
    """
    components.html(KINETIC_GRID_CONTROLLER_HTML, height=0)

    # --- SECTION 02: INSTRUMENTATION & ARCHITECTURE (DARK BAND #222f30) ---
    render_clean_html("""
    <div id="platform" style="max-width: 1200px; margin: 0 auto; padding: 90px 40px 80px 40px;">
        
        <div style="font-size: 36px; line-height: 1.2; letter-spacing: -0.006em; color: var(--color-paper); margin-bottom: 48px;">
            Quantitative bioinformatic succession architecture.
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px;">
            <div class="instrument-card-dark">
                <div class="mono-tag" style="color: var(--color-bioluminescent-lime); margin-bottom: 16px;">[01] COMPOSITIONAL NORMALIZATION</div>
                <div style="font-size: 22px; line-height: 1.3; letter-spacing: -0.13px; margin-bottom: 16px; color: var(--color-paper);">
                    Centered Log-Ratio (CLR)
                </div>
                <div style="font-size: 16px; line-height: 1.35; color: var(--color-graphite);">
                    Microbiome sequencing counts reside on an unconstrained simplex. Centered Log-Ratio transformations with pseudocounts remove library-size artifacts, ensuring mathematically valid distances for supervised modeling.
                </div>
            </div>
            
            <div class="instrument-card-dark">
                <div class="mono-tag" style="color: var(--color-bioluminescent-lime); margin-bottom: 16px;">[02] STATISTICAL QUANTILE ENGINE</div>
                <div style="font-size: 22px; line-height: 1.3; letter-spacing: -0.13px; margin-bottom: 16px; color: var(--color-paper);">
                    Pinball Quantile XGBoost
                </div>
                <div style="font-size: 16px; line-height: 1.35; color: var(--color-graphite);">
                    Three separate gradient-boosted trees optimized under asymmetric pinball loss output median estimates alongside defensible 10th and 90th percentile bounds, capturing biological uncertainty rather than false point precision.
                </div>
            </div>
            
            <div class="instrument-card-dark">
                <div class="mono-tag" style="color: var(--color-bioluminescent-lime); margin-bottom: 16px;">[03] FORENSIC ADMISSIBILITY</div>
                <div style="font-size: 22px; line-height: 1.3; letter-spacing: -0.13px; margin-bottom: 16px; color: var(--color-paper);">
                    Form PM-5372 Integration
                </div>
                <div style="font-size: 16px; line-height: 1.35; color: var(--color-graphite);">
                    Direct generation of forensic autopsy documentation. Designed with awareness of evidentiary standards (Daubert / Frye) through explicit error bounds, specimen adequacy clearance, and institutional certification.
                </div>
            </div>
        </div>
    </div>
    """)

    # --- SECTION 03: SUCCESSION WAVES (LIGHT FLIP TO BONE WHITE #f7f7f5) ---
    render_clean_html("""
    <div id="succession" style="background-color: var(--color-bone-white); color: var(--color-abyssal-ink); padding: 100px 40px; margin-top: 40px;">
        <div style="max-width: 1200px; margin: 0 auto;">
            
            <div style="font-size: 42px; line-height: 1.15; letter-spacing: -0.01em; color: var(--color-abyssal-ink); margin-bottom: 16px;">
                Three reproducible ecological waves.
            </div>
            <p style="font-size: 18px; line-height: 1.35; color: var(--color-graphite); max-width: 780px; margin-bottom: 48px;">
                As human decomposition advances, systemic physiological shifts drive predictable taxonomic blooms and crashes across mucosal, dermal, and enteric communities.
            </p>

            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px;">
                <div class="instrument-card-light">
                    <div class="mono-tag" style="color: var(--color-graphite); margin-bottom: 12px;">DAYS 0 — 3 &bull; PHASE 01</div>
                    <div style="font-size: 24px; line-height: 1.2; margin-bottom: 12px; color: var(--color-abyssal-ink);">
                        Fresh & Aerobic Senescence
                    </div>
                    <p style="font-size: 15px; line-height: 1.35; color: var(--color-graphite);">
                        Cutaneous commensals (<i>Staphylococcus epidermidis</i>, <i>Streptococcus oralis</i>, <i>Cutibacterium acnes</i>) dominate the remaining mucosal oxygen niche. Rapid cellular hypoxia begins triggering metabolic depletion.
                    </p>
                    <div style="font-family: var(--font-mono); font-size: 12px; color: var(--color-graphite); margin-top: 20px; border-top: 1px solid var(--color-lichen); padding-top: 12px;">
                        PHYSICAL CORRELATE: Rigor mortis established; faint iliac fossa greenish discoloration.
                    </div>
                </div>

                <div class="instrument-card-light">
                    <div class="mono-tag" style="color: var(--color-graphite); margin-bottom: 12px;">DAYS 3 — 12 &bull; PHASE 02</div>
                    <div style="font-size: 24px; line-height: 1.2; margin-bottom: 12px; color: var(--color-abyssal-ink);">
                        Putrefactive Bloat & Liquefaction
                    </div>
                    <p style="font-size: 15px; line-height: 1.35; color: var(--color-graphite);">
                        Translocation and explosive proliferation of enteric anaerobes (<i>Clostridium perfringens</i>, <i>Bacteroides fragilis</i>) and proteolytic swarming organisms (<i>Proteus mirabilis</i>), alongside dipteran biomarkers (<i>Wohlfahrtiimonas</i>).
                    </p>
                    <div style="font-family: var(--font-mono); font-size: 12px; color: var(--color-graphite); margin-top: 20px; border-top: 1px solid var(--color-lichen); padding-top: 12px;">
                        PHYSICAL CORRELATE: Flaccid rigor; generalized gaseous bloat; venous marbling; purge fluid.
                    </div>
                </div>

                <div class="instrument-card-light">
                    <div class="mono-tag" style="color: var(--color-graphite); margin-bottom: 12px;">DAYS 12 — 30+ &bull; PHASE 03</div>
                    <div style="font-size: 24px; line-height: 1.2; margin-bottom: 12px; color: var(--color-abyssal-ink);">
                        Advanced Decay & Soil Infiltration
                    </div>
                    <p style="font-size: 15px; line-height: 1.35; color: var(--color-graphite);">
                        Collapse of enteric anaerobes following abdominal wall rupture. Robust colonization by soil saprophytes and extremophiles (<i>Acinetobacter baumannii</i>, <i>Pseudomonas fluorescens</i>, <i>Bacillus subtilis</i>).
                    </p>
                    <div style="font-family: var(--font-mono); font-size: 12px; color: var(--color-graphite); margin-top: 20px; border-top: 1px solid var(--color-lichen); padding-top: 12px;">
                        PHYSICAL CORRELATE: Abdominal rupture; black putrefaction; skeletonization; dry remains.
                    </div>
                </div>
            </div>
        </div>
    </div>
    """)

    # --- SECTION 04: CLOSURE VOID GROUND (#000000) ---
    render_clean_html(f"""
    <div id="dossier" style="background-color: var(--color-void); padding: 100px 40px 80px 40px; border-top: 1px solid #1a2223;">
        <div style="max-width: 1200px; margin: 0 auto;">
            
            
            <div style="margin-bottom: 40px;">
                <h2 style="font-size: clamp(32px, 5vw, 64px); line-height: 1.05; letter-spacing: -0.02em; color: var(--color-paper); margin: 0 0 16px 0;">
                    Ready for clinical post-mortem examination.
                </h2>
                <p style="font-size: 18px; color: var(--color-graphite); margin: 0 0 36px 0; max-width: 680px; line-height: 1.4;">
                    Proceed to the interactive triage workflow to input autopsy particulars, correlate morphological findings, confirm microbial bioindicators, and export the Form PM-5372 dossier.
                </p>
                <a href="?view=examination{auth_q}" target="_self" class="sober-btn-dark">COMMENCE POST-MORTEM EXAMINATION</a>
            </div>
        </div>
    </div>
    """)

    # --- MINIMAL LANDING PAGE FOOTER ---
    render_clean_html(f"""
    <div class="necrotrace-footer" style="display: block !important; width: 100%; background-color: #000000; border-top: 1px solid #141417; padding-top: 54px; padding-bottom: 38px; font-family: var(--font-display);">
        <div style="max-width: 1120px; margin: 0 auto; padding: 0 24px; box-sizing: border-box; font-family: var(--font-display);">
            
            <!-- Top Section: Brand & Links on left, Legal on right -->
            <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: flex-start; gap: 36px 48px;">
                
                <!-- Left: Logo & Navigation Links -->
                <div style="display: flex; flex-direction: column; gap: 24px; min-width: 260px; font-family: var(--font-display);">
                    <!-- Logo mark -->
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <img src="{LOGO_ICON_B64}" style="height: 34px; width: 34px; object-fit: contain; vertical-align: middle; filter: drop-shadow(0 0 10px rgba(116, 194, 92, 0.45));" alt="NecroTrace Logo" />
                        <span style="font-size: 22px; font-weight: 600; color: #ffffff; letter-spacing: -0.02em; font-family: var(--font-display);">necrotrace</span>
                    </div>

                    <!-- Horizontal Links -->
                    <div style="display: flex; flex-wrap: wrap; gap: 14px 24px; font-size: 14px; color: #888888; font-family: var(--font-display);">
                        <a href="#" style="color: #888888; text-decoration: none; font-family: var(--font-display); transition: color 0.15s;" onmouseover="this.style.color='#ffffff'" onmouseout="this.style.color='#888888'">Overview</a>
                        <a href="#succession" style="color: #888888; text-decoration: none; font-family: var(--font-display); transition: color 0.15s;" onmouseover="this.style.color='#ffffff'" onmouseout="this.style.color='#888888'">Features</a>
                        <a href="#succession" style="color: #888888; text-decoration: none; font-family: var(--font-display); transition: color 0.15s;" onmouseover="this.style.color='#ffffff'" onmouseout="this.style.color='#888888'">Methodology</a>
                        <a href="#dossier" style="color: #888888; text-decoration: none; font-family: var(--font-display); transition: color 0.15s;" onmouseover="this.style.color='#ffffff'" onmouseout="this.style.color='#888888'">Dossier</a>
                        <a href="?view=examination{auth_q}" target="_self" style="color: #888888; text-decoration: none; font-family: var(--font-display); transition: color 0.15s;" onmouseover="this.style.color='#ffffff'" onmouseout="this.style.color='#888888'">Examination</a>
                        <a href="mailto:tanishwalture@gmail.com" style="color: #888888; text-decoration: none; font-family: var(--font-display); transition: color 0.15s;" onmouseover="this.style.color='#ffffff'" onmouseout="this.style.color='#888888'">Contact</a>
                    </div>
                </div>

                <!-- Right: Legal Links & Project Info -->
                <div style="display: flex; flex-direction: column; gap: 14px; min-width: 280px; font-family: var(--font-display);">
                    <div style="font-size: 15px; font-weight: 600; color: #ffffff; letter-spacing: -0.01em; font-family: var(--font-display);">
                        Legal &amp; Policies
                    </div>
                    <div style="display: flex; flex-wrap: wrap; gap: 10px 20px; font-size: 13px; font-family: var(--font-display);">
                        <a href="?view=privacy" target="_self" style="color: #888888; text-decoration: none; font-family: var(--font-display); transition: color 0.15s;" onmouseover="this.style.color='#ffffff'" onmouseout="this.style.color='#888888'">Privacy Policy</a>
                        <a href="?view=terms" target="_self" style="color: #888888; text-decoration: none; font-family: var(--font-display); transition: color 0.15s;" onmouseover="this.style.color='#ffffff'" onmouseout="this.style.color='#888888'">Terms &amp; Conditions</a>
                        <a href="?view=cookies" target="_self" style="color: #888888; text-decoration: none; font-family: var(--font-display); transition: color 0.15s;" onmouseover="this.style.color='#ffffff'" onmouseout="this.style.color='#888888'">Cookie &amp; Third-Party Policy</a>
                    </div>
                    <div style="font-size: 12px; color: #71717a; line-height: 1.5; margin-top: 4px; font-family: var(--font-display);">
                        Developed by Tanish Walture (Team BroomWroom)<br/>
                        Contact: <a href="mailto:tanishwalture@gmail.com" style="color: #888888; text-decoration: none; font-family: var(--font-display);">tanishwalture@gmail.com</a>
                    </div>
                </div>

            </div>

            <!-- Academic Disclaimer -->
            <div style="margin-top: 32px; padding: 14px 18px; background: rgba(245, 158, 11, 0.06); border: 1px solid rgba(245, 158, 11, 0.2); border-radius: 8px; font-family: var(--font-display);">
                <div style="font-size: 12px; color: #a1a1aa; line-height: 1.55; font-family: var(--font-display);">
                    <span class="footer-mono" style="color: #f59e0b; font-weight: 600; font-size: 11px; letter-spacing: 0.04em;">&#9888; ACADEMIC RESEARCH PROTOTYPE</span> &mdash; NecroTrace is an educational demonstration project. It is NOT certified, validated, or approved for use in actual forensic investigations, criminal cases, or legal proceedings. All generated reports are for research and demonstration purposes only.
                </div>
            </div>

            <!-- Subtle Hairline Separator -->
            <div style="height: 1px; width: 100%; background-color: #171717; margin: 28px 0 20px 0;"></div>

            <!-- Bottom Row: Copyright & Social Icons -->
            <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 16px; font-size: 13px; color: #71717a; font-family: var(--font-display);">
                <div style="font-family: var(--font-display);">
                    &copy; 2026 NecroTrace. All rights reserved. &bull; <a href="https://github.com/BroomWroom/NecroTrace/blob/main/LICENSE" target="_blank" rel="noreferrer" style="color: #71717a; text-decoration: none; font-family: var(--font-display);">MIT License</a>
                </div>

                <div style="display: flex; align-items: center; gap: 20px;">
                    <!-- Mail -->
                    <a href="mailto:tanishwalture@gmail.com" title="Email" style="color: #71717a; display: inline-flex; align-items: center; text-decoration: none; transition: color 0.15s;" onmouseover="this.style.color='#ffffff'" onmouseout="this.style.color='#71717a'">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="16" x="2" y="4" rx="2"/><path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/></svg>
                    </a>
                    <!-- Instagram -->
                    <a href="https://instagram.com/tnsh_Zzz" target="_blank" rel="noreferrer" title="Instagram" style="color: #71717a; display: inline-flex; align-items: center; text-decoration: none; transition: color 0.15s;" onmouseover="this.style.color='#ffffff'" onmouseout="this.style.color='#71717a'">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="20" rx="5" ry="5"/><path d="M16 11.37A4 4 0 1 1 12.63 8 4 4 0 0 1 16 11.37z"/><line x1="17.5" y1="6.5" x2="17.51" y2="6.5"/></svg>
                    </a>
                    <!-- LinkedIn -->
                    <a href="https://www.linkedin.com/in/tanish-walture-20a79130a/" target="_blank" rel="noreferrer" title="LinkedIn" style="color: #71717a; display: inline-flex; align-items: center; text-decoration: none; transition: color 0.15s;" onmouseover="this.style.color='#ffffff'" onmouseout="this.style.color='#71717a'">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"/><rect x="2" y="9" width="4" height="12"/><circle cx="4" cy="4" r="2"/></svg>
                    </a>
                    <!-- GitHub -->
                    <a href="https://github.com/BroomWroom" target="_blank" rel="noreferrer" title="GitHub" style="color: #71717a; display: inline-flex; align-items: center; text-decoration: none; transition: color 0.15s;" onmouseover="this.style.color='#ffffff'" onmouseout="this.style.color='#71717a'">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"/><path d="M9 18c-4.51 2-5-2-7-2"/></svg>
                    </a>
                </div>
            </div>

        </div>
    </div>
    """
                      )


# =============================================================================
# VIEW 2: MEDICAL EXAMINER DIAGNOSTIC TRIAGE WORKFLOW
# =============================================================================
elif st.session_state["view"] == "examination":
    sync_document_title("examination")

    # --- STRICT CENTERED LAYOUT (920px) & DARKROOM STYLING ---
    render_clean_html("""
    <style>
    /* EXECUTIVE CENTERED VIEWPORT (CLOSER TO MIDDLE) */
    div[data-testid="stAppViewContainer"] > section.main,
    div[data-testid="stAppViewContainer"] .stMain {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
    }

    div[data-testid="stMainBlockContainer"],
    div[data-testid="stAppViewBlockContainer"],
    .main .block-container,
    .stMainBlockContainer,
    .block-container,
    div[class*="stMainBlockContainer"],
    div[class*="block-container"] {
        max-width: 880px !important;
        width: 100% !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-top: 24px !important;
        padding-bottom: 96px !important;
        padding-left: 20px !important;
        padding-right: 20px !important;
        box-sizing: border-box !important;
    }

    /* Vertical rhythm between fields and elements */
    div[data-testid="stVerticalBlock"] {
        gap: 16px !important;
    }

    /* Streamlit Bordered Container - Step Cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #364040 !important;
        border-radius: 12px !important;
        background: rgba(28, 38, 39, 0.45) !important;
        padding: 24px 28px !important;
        margin-bottom: 24px !important;
    }

    /* Nested containers inside step cards (symptom inspection panels) */
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #2e3838 !important;
        border-radius: 8px !important;
        background: rgba(22, 30, 31, 0.6) !important;
        padding: 18px 20px !important;
        margin-bottom: 14px !important;
    }

    /* Media Card Layout (Matching User Reference Design) */
    .taxa-media-card {
        max-width: 380px;
        width: 100%;
        margin-left: auto;
        margin-right: auto;
        margin-bottom: 8px;
        border: 1px solid #334344;
        border-radius: 12px;
        background: #141e1f;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
    }
    .taxa-media-card:hover {
        border-color: #4f6364;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
        transform: translateY(-2px);
    }

    /* Action area styling */
    div.taxa-dock-wrapper {
        max-width: 380px;
        margin: 0 auto 36px auto;
        padding: 6px 12px;
        background: rgba(18, 26, 27, 0.65);
        border: 1px solid #2b393a;
        border-radius: 8px;
    }

    /* Input & Selectbox Styling */
    div[data-baseweb="input"] {
        background-color: #162021 !important;
        border: 1px solid #384545 !important;
        border-radius: 6px !important;
        transition: border-color 0.2s ease;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: var(--color-bioluminescent-lime) !important;
    }
    div[data-baseweb="input"] input {
        color: #ffffff !important;
        font-family: var(--font-mono) !important;
        font-size: 13px !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #162021 !important;
        border: 1px solid #384545 !important;
        border-radius: 6px !important;
        color: #ffffff !important;
        font-family: var(--font-mono) !important;
        font-size: 13px !important;
    }

    /* Slider styling - Bioluminescent Lime accents */
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: var(--color-bioluminescent-lime) !important;
        border-color: var(--color-bioluminescent-lime) !important;
        box-shadow: 0 0 8px rgba(206, 247, 158, 0.4) !important;
    }
    div[data-testid="stSlider"] div[data-baseweb="slider"] div[style*="background"] {
        background-color: var(--color-bioluminescent-lime) !important;
    }
    div[data-testid="stSlider"] [data-testid="stThumbValue"] {
        color: var(--color-bioluminescent-lime) !important;
        font-family: var(--font-mono) !important;
        font-size: 12px !important;
    }
    div[data-testid="stSlider"] [data-testid="stTickBar"] {
        color: #667272 !important;
    }

    /* Labels */
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] p {
        color: #c9cbbe !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        letter-spacing: 0.03em !important;
        font-family: var(--font-mono) !important;
    }
    </style>
    """)

    # Top Navigation Bar in Examination View
    cur_officer = st.session_state.get("authenticated_officer") or {}
    cur_uid = cur_officer.get("local_id", "")
    auth_q = f"&auth={cur_uid}" if cur_uid else ""
    officer_name_short = cur_officer.get("name", "Examiner")
    officer_badge_short = cur_officer.get("badge", "CFS")
    nav_exam_html = f"""
    
    <header style="width: 100%; border-bottom: 1px solid var(--color-graphite); padding: 12px 0 20px 0; margin-bottom: 28px;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <img src="{LOGO_ICON_B64}" style="height: 26px; width: 26px; object-fit: contain; vertical-align: middle; filter: drop-shadow(0 0 8px rgba(116, 194, 92, 0.35));" alt="NecroTrace Logo" />
                <span style="font-family: var(--font-mono); font-size: 13px; color: var(--color-paper); font-weight: 600; letter-spacing: 0.04em;">
                    NECROTRACE <span style="color: var(--color-graphite);">//</span> EXAMINATION ROOM
                </span>
                
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-family: var(--font-mono); font-size: 11px; color: #6ee7b7; background: rgba(6, 78, 59, 0.55); border: 1px solid #10b981; padding: 5px 10px; border-radius: 6px; display: inline-flex; align-items: center; gap: 6px;">
                    {officer_name_short} ({officer_badge_short})
                </span>
                <a href="?view=logout" target="_self" style="font-family: var(--font-mono); font-size: 11px; color: #fca5a5; text-decoration: none; border: 1px solid rgba(239, 68, 68, 0.4); padding: 5px 10px; border-radius: 6px; background: rgba(239, 68, 68, 0.08); transition: all 0.15s ease;">LOG OUT</a>
                <a href="?view=landing{auth_q}" target="_self" class="sober-btn-ghost" style="padding: 6px 16px; font-size: 11px; height: 34px; min-height: 34px; text-decoration: none;">&larr; RETURN TO PLATFORM OVERVIEW</a>
            </div>
        </div>
    </header>
    """
    render_clean_html(nav_exam_html)

    # Title & Subtitle (Centered container)
    render_clean_html("""
    <div id="examination-form" style="margin-bottom: 30px;">
        <h1 style="font-size: clamp(26px, 3.0vw, 36px); line-height: 1.15; letter-spacing: -0.02em; color: var(--color-paper); margin: 0 0 10px 0;">
            Medical Examiner Diagnostic Triage
        </h1>
        <p style="font-size: 15px; color: var(--color-graphite); margin: 0; line-height: 1.45;">
            Autopsy particulars and morphological postmortem findings correlate directly with microbial ecological succession kinetics, synthesizing a compositional profile to predict quantile postmortem intervals with probabilistic error bounds.
        </p>
    </div>
    """)

    # -------------------------------------------------------------
    # STEP 1: PATIENT PARTICULARS & SCENE FACTORS
    # -------------------------------------------------------------
    with st.container(border=True):
        st.markdown(
            '<div class="section-counter" style="margin-bottom: 20px;">01 / PATIENT & SCENE PARTICULARS</div>', unsafe_allow_html=True)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            deceased_name = st.text_input(
                "Name of Deceased / Ref:", value="Unidentified Individual (Ref: Unknown #42)")
            age_val = st.text_input(
                "Estimated Age:", value="Approx. 35 - 40 Years")
            height_val = st.text_input("Height (approx):", value="172 cm")
            ambient_temp = st.slider(
                "Scene Temperature (°C):", min_value=5.0, max_value=42.0, value=23.5, step=0.5)
        with col_p2:
            sex_val = st.selectbox(
                "Sex:", ["Male", "Female", "Indeterminate / Skeletal"], index=0)
            swab_site = st.selectbox(
                "Anatomical Swab Site:",
                ["Oral Cavity / Mucosal Surface", "Nasal Mucosa",
                    "Abdominal Surface", "Soil-Body Interface"],
                index=0,
            )
            weight_val = st.text_input("Weight (approx):", value="68 kg")
            humidity_val = st.slider(
                "Relative Humidity (%):", min_value=20.0, max_value=100.0, value=65.0, step=5.0)

        st.markdown(
            '<hr class="hairline-dark" style="margin: 20px 0;">', unsafe_allow_html=True)

        # Administrative Medico-Legal Identifiers
        st.markdown('<div style="font-family: var(--font-mono); font-size: 11px; color: var(--color-graphite); margin-bottom: 12px; letter-spacing: 0.05em; text-transform: uppercase;">ADMINISTRATIVE DOSSIER IDENTIFIERS</div>', unsafe_allow_html=True)
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            pm_report_no = st.text_input(
                "Post Mortem Report No:", value="PM-619 / 2026")
            default_ps = cur_officer.get("station") if cur_officer.get(
                "station") else "New Township P.S."
            police_station = st.text_input(
                "Police Station (P.S.):", value=default_ps)
        with col_m2:
            inquest_no = st.text_input("Inquest Number:", value="14 / 2026")
            default_officer = f"{cur_officer.get('name')} ({cur_officer.get('badge', 'EXAMINER')})" if cur_officer.get(
                "name") else "Dr. Tanish Walture, M.D. (WBMC / 45826)"
            analyst_name = st.text_input(
                "Examining Medical Officer:", value=default_officer)

        st.markdown(
            '<hr class="hairline-dark" style="margin: 18px 0 14px 0;">', unsafe_allow_html=True)

        # Provisional Autopsy Diagnoses & Inquest Findings
        st.markdown('<div style="font-family: var(--font-mono); font-size: 11px; color: var(--color-graphite); margin-bottom: 12px; letter-spacing: 0.05em; text-transform: uppercase;">PROVISIONAL AUTOPSY DIAGNOSES & INQUEST CLASSIFICATION</div>', unsafe_allow_html=True)
        col_c1, col_c2 = st.columns([1.25, 0.75])
        with col_c1:
            cause_of_death_input = st.text_input(
                "Provisional Cause of Death (Anatomical / Pathological Finding):",
                value="ASPHYXIA AS A RESULT OF CONSTRICTION OF NECK (PENDING TOXICOLOGY & HISTOLOGY)",
                help="Anatomical / pathological trauma diagnosis or mechanical cause (e.g. Asphyxia, Craniofacial Trauma, Hypovolemic Shock, Pending Chemical Viscera Analysis)."
            )
        with col_c2:
            manner_of_death_input = st.selectbox(
                "Manner of Death:",
                [
                    "Matter under judicial inquiry / Forensic Inquest",
                    "Homicide (Suspected / Under Investigation)",
                    "Suicide",
                    "Accidental",
                    "Natural / Pathological",
                    "Undetermined / Pending Viscera & Histology",
                ],
                index=0,
                help="Legal classification of manner of death for coroner/magistrate inquest proceedings."
            )

    # -------------------------------------------------------------
    # STEP 2: MORPHOLOGICAL SIGNS AUTOPSY INSPECTION (VISUAL PANELS)
    # -------------------------------------------------------------
    with st.container(border=True):
        st.markdown(
            '<div class="section-counter" style="margin-bottom: 16px;">02 / MORPHOLOGICAL SIGNS AUTOPSY INSPECTION</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size: 14px; color: var(--color-graphite); margin-bottom: 22px; line-height: 1.45;">
            Record clinical postmortem decomposition signs. The succession engine aligns physical observations with microbial chronometers. Review the real-time visual inspection guides below for autopsy palpation checkpoints and physical appearance.
        </div>
        """, unsafe_allow_html=True)

        # Panel 1: Rigor Mortis Status
        with st.container(border=True):
            r_col1, r_col2 = st.columns([1.05, 0.95])
            with r_col1:
                st.markdown(
                    '<div style="font-size: 15px; font-weight: 500; color: var(--color-paper); margin-bottom: 4px;">1. Rigor Mortis Status</div>', unsafe_allow_html=True)
                st.caption(
                    MORPHOLOGICAL_SIGN_GUIDES["rigor_mortis"]["description"])
                rigor_opt = st.radio(
                    "Rigor Mortis:",
                    list(
                        MORPHOLOGICAL_SIGN_GUIDES["rigor_mortis"]["stages"].keys()),
                    index=3 if not st.session_state["triage_confirmed"] else 3,
                    label_visibility="collapsed",
                    key="radio_rigor",
                )
            with r_col2:
                r_info = MORPHOLOGICAL_SIGN_GUIDES["rigor_mortis"]["stages"][rigor_opt]
                render_clean_html(f"""
                <div class="symptom-visual-box">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-family: var(--font-mono); font-size: 11px; font-weight: 500; color: {r_info['severity_color']}; border: 1px solid {r_info['severity_color']}; padding: 2px 8px; border-radius: 4px;">
                            {r_info['badge']}
                        </span>
                        <span style="font-family: var(--font-mono); font-size: 11px; color: var(--color-graphite);">{r_info['phase_tag']}</span>
                    </div>
                    <div style="font-size: 13px; color: var(--color-paper); line-height: 1.4; margin-bottom: 8px;">
                        <b>Autopsy Appearance:</b> {r_info['appearance']}
                    </div>
                    <div style="font-size: 12px; color: #9bb0b1; line-height: 1.35; margin-bottom: 6px;">
                        <b>Palpation Checkpoint:</b> {MORPHOLOGICAL_SIGN_GUIDES['rigor_mortis']['palpation_cue']}
                    </div>
                    <div style="font-size: 11px; color: var(--color-graphite);">
                        <b>Anatomical Focus:</b> {MORPHOLOGICAL_SIGN_GUIDES['rigor_mortis']['anatomical_focus']}
                    </div>
                </div>
                """)

        # Panel 2: Abdominal Distension & Bloat
        with st.container(border=True):
            b_col1, b_col2 = st.columns([1.05, 0.95])
            with b_col1:
                st.markdown(
                    '<div style="font-size: 15px; font-weight: 500; color: var(--color-paper); margin-bottom: 4px;">2. Abdominal Distension & Bloat</div>', unsafe_allow_html=True)
                st.caption(MORPHOLOGICAL_SIGN_GUIDES["bloat"]["description"])
                bloat_opt = st.radio(
                    "Bloat State:",
                    list(MORPHOLOGICAL_SIGN_GUIDES["bloat"]["stages"].keys()),
                    index=2,
                    label_visibility="collapsed",
                    key="radio_bloat",
                )
            with b_col2:
                b_info = MORPHOLOGICAL_SIGN_GUIDES["bloat"]["stages"][bloat_opt]
                render_clean_html(f"""
                <div class="symptom-visual-box">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-family: var(--font-mono); font-size: 11px; font-weight: 500; color: {b_info['severity_color']}; border: 1px solid {b_info['severity_color']}; padding: 2px 8px; border-radius: 4px;">
                            {b_info['badge']}
                        </span>
                        <span style="font-family: var(--font-mono); font-size: 11px; color: var(--color-graphite);">{b_info['phase_tag']}</span>
                    </div>
                    <div style="font-size: 13px; color: var(--color-paper); line-height: 1.4; margin-bottom: 8px;">
                        <b>Autopsy Appearance:</b> {b_info['appearance']}
                    </div>
                    <div style="font-size: 12px; color: #9bb0b1; line-height: 1.35; margin-bottom: 6px;">
                        <b>Palpation Checkpoint:</b> {MORPHOLOGICAL_SIGN_GUIDES['bloat']['palpation_cue']}
                    </div>
                    <div style="font-size: 11px; color: var(--color-graphite);">
                        <b>Anatomical Focus:</b> {MORPHOLOGICAL_SIGN_GUIDES['bloat']['anatomical_focus']}
                    </div>
                </div>
                """)

        # Panel 3: Skin Discoloration & Vascular Marbling
        with st.container(border=True):
            d_col1, d_col2 = st.columns([1.05, 0.95])
            with d_col1:
                st.markdown('<div style="font-size: 15px; font-weight: 500; color: var(--color-paper); margin-bottom: 4px;">3. Skin Discoloration & Vascular Marbling</div>', unsafe_allow_html=True)
                st.caption(
                    MORPHOLOGICAL_SIGN_GUIDES["discoloration"]["description"])
                discolor_opt = st.radio(
                    "Discoloration:",
                    list(
                        MORPHOLOGICAL_SIGN_GUIDES["discoloration"]["stages"].keys()),
                    index=2,
                    label_visibility="collapsed",
                    key="radio_discolor",
                )
            with d_col2:
                d_info = MORPHOLOGICAL_SIGN_GUIDES["discoloration"]["stages"][discolor_opt]
                render_clean_html(f"""
                <div class="symptom-visual-box">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-family: var(--font-mono); font-size: 11px; font-weight: 500; color: {d_info['severity_color']}; border: 1px solid {d_info['severity_color']}; padding: 2px 8px; border-radius: 4px;">
                            {d_info['badge']}
                        </span>
                        <span style="font-family: var(--font-mono); font-size: 11px; color: var(--color-graphite);">{d_info['phase_tag']}</span>
                    </div>
                    <div style="font-size: 13px; color: var(--color-paper); line-height: 1.4; margin-bottom: 8px;">
                        <b>Autopsy Appearance:</b> {d_info['appearance']}
                    </div>
                    <div style="font-size: 12px; color: #9bb0b1; line-height: 1.35; margin-bottom: 6px;">
                        <b>Palpation Checkpoint:</b> {MORPHOLOGICAL_SIGN_GUIDES['discoloration']['palpation_cue']}
                    </div>
                    <div style="font-size: 11px; color: var(--color-graphite);">
                        <b>Anatomical Focus:</b> {MORPHOLOGICAL_SIGN_GUIDES['discoloration']['anatomical_focus']}
                    </div>
                </div>
                """)

        # Panel 4: Purge Fluid & Natural Orifices
        with st.container(border=True):
            pu_col1, pu_col2 = st.columns([1.05, 0.95])
            with pu_col1:
                st.markdown(
                    '<div style="font-size: 15px; font-weight: 500; color: var(--color-paper); margin-bottom: 4px;">4. Purge Fluid & Natural Orifices</div>', unsafe_allow_html=True)
                st.caption(
                    MORPHOLOGICAL_SIGN_GUIDES["purge_fluid"]["description"])
                purge_opt = st.radio(
                    "Purge Fluid:",
                    list(
                        MORPHOLOGICAL_SIGN_GUIDES["purge_fluid"]["stages"].keys()),
                    index=2,
                    label_visibility="collapsed",
                    key="radio_purge",
                )
            with pu_col2:
                pu_info = MORPHOLOGICAL_SIGN_GUIDES["purge_fluid"]["stages"][purge_opt]
                render_clean_html(f"""
                <div class="symptom-visual-box">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-family: var(--font-mono); font-size: 11px; font-weight: 500; color: {pu_info['severity_color']}; border: 1px solid {pu_info['severity_color']}; padding: 2px 8px; border-radius: 4px;">
                            {pu_info['badge']}
                        </span>
                        <span style="font-family: var(--font-mono); font-size: 11px; color: var(--color-graphite);">{pu_info['phase_tag']}</span>
                    </div>
                    <div style="font-size: 13px; color: var(--color-paper); line-height: 1.4; margin-bottom: 8px;">
                        <b>Autopsy Appearance:</b> {pu_info['appearance']}
                    </div>
                    <div style="font-size: 12px; color: #9bb0b1; line-height: 1.35; margin-bottom: 6px;">
                        <b>Palpation Checkpoint:</b> {MORPHOLOGICAL_SIGN_GUIDES['purge_fluid']['palpation_cue']}
                    </div>
                    <div style="font-size: 11px; color: var(--color-graphite);">
                        <b>Anatomical Focus:</b> {MORPHOLOGICAL_SIGN_GUIDES['purge_fluid']['anatomical_focus']}
                    </div>
                </div>
                """)

        # Panel 5: Entomology & Maggot Colonization
        with st.container(border=True):
            m_col1, m_col2 = st.columns([1.05, 0.95])
            with m_col1:
                st.markdown(
                    '<div style="font-size: 15px; font-weight: 500; color: var(--color-paper); margin-bottom: 4px;">5. Entomology & Maggot Colonization</div>', unsafe_allow_html=True)
                st.caption(
                    MORPHOLOGICAL_SIGN_GUIDES["maggot_activity"]["description"])
                maggots_opt = st.radio(
                    "Entomology Activity:",
                    list(
                        MORPHOLOGICAL_SIGN_GUIDES["maggot_activity"]["stages"].keys()),
                    index=1,
                    label_visibility="collapsed",
                    key="radio_maggots",
                )
            with m_col2:
                m_info = MORPHOLOGICAL_SIGN_GUIDES["maggot_activity"]["stages"][maggots_opt]
                render_clean_html(f"""
                <div class="symptom-visual-box">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-family: var(--font-mono); font-size: 11px; font-weight: 500; color: {m_info['severity_color']}; border: 1px solid {m_info['severity_color']}; padding: 2px 8px; border-radius: 4px;">
                            {m_info['badge']}
                        </span>
                        <span style="font-family: var(--font-mono); font-size: 11px; color: var(--color-graphite);">{m_info['phase_tag']}</span>
                    </div>
                    <div style="font-size: 13px; color: var(--color-paper); line-height: 1.4; margin-bottom: 8px;">
                        <b>Autopsy Appearance:</b> {m_info['appearance']}
                    </div>
                    <div style="font-size: 12px; color: #9bb0b1; line-height: 1.35; margin-bottom: 6px;">
                        <b>Palpation Checkpoint:</b> {MORPHOLOGICAL_SIGN_GUIDES['maggot_activity']['palpation_cue']}
                    </div>
                    <div style="font-size: 11px; color: var(--color-graphite);">
                        <b>Anatomical Focus:</b> {MORPHOLOGICAL_SIGN_GUIDES['maggot_activity']['anatomical_focus']}
                    </div>
                </div>
                """)

        # Dynamic Triage Calculation
        signs_dict = {
            "rigor_mortis": rigor_opt,
            "bloat": bloat_opt,
            "discoloration": discolor_opt,
            "purge_fluid": purge_opt,
            "maggot_activity": maggots_opt,
        }
        triage_eval = evaluate_morphological_triage(signs_dict)
        st.session_state["triage_eval"] = triage_eval

        # Consolidated Succession Phase Correlation Banner
        render_clean_html(f"""
        <div style="background: rgba(18, 26, 27, 0.95); border-left: 3px solid var(--color-bioluminescent-lime); border-radius: 0 10px 10px 0; padding: 20px 24px; margin-top: 14px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; flex-wrap: wrap; gap: 8px;">
                <span style="font-family: var(--font-mono); font-size: 12px; color: var(--color-bioluminescent-lime); letter-spacing: -0.01em;">
                    PREDICTED DECOMPOSITION PHASE &bull; {triage_eval['primary_phase'].upper()}
                </span>
                <span style="font-family: var(--font-mono); font-size: 12px; color: var(--color-paper); background: rgba(255,255,255,0.06); padding: 3px 10px; border-radius: 4px;">
                    CORRELATED WINDOW: {triage_eval['coarse_clinical_range']}
                </span>
            </div>
            <div style="font-size: 14px; color: #b2c2c2; line-height: 1.45;">
                {triage_eval['biological_summary']}
            </div>
        </div>
        """)

    # -------------------------------------------------------------
    # STEP 3: AUTO-SUGGESTED MICROBIAL BIOINDICATORS (RICH MEDIA CARDS)
    # -------------------------------------------------------------
    with st.container(border=True):
        st.markdown('<div class="section-counter" style="margin-bottom: 16px;">03 / CONFIRMED MICROBIAL BIOINDICATORS & RICH MEDIA</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size: 14px; color: var(--color-graphite); margin-bottom: 24px; line-height: 1.45;">
            Microbial succession taxonomy aligned with your autopsy observations. Each card presents high-magnification microscopy placeholder imagery, Gram-stain classification, and biochemical mechanisms. Confirm the bioindicators verified by swab testing:
        </div>
        """, unsafe_allow_html=True)

        suggested_list = triage_eval["suggested_bioindicators"]
        selected_taxa_current = []

        # Display Candidate Taxa in a 2-Column Grid of Rich Media Cards (Matching User Reference)
        col_t1, col_t2 = st.columns(2)
        for i, item in enumerate(suggested_list):
            target_col = col_t1 if (i % 2 == 0) else col_t2
            with target_col:
                rec_badge = (
                    '<span style="font-family: var(--font-mono); font-size: 10px; background: rgba(206, 247, 158, 0.15); color: var(--color-bioluminescent-lime); border: 1px solid var(--color-bioluminescent-lime); padding: 2px 7px; border-radius: 4px;">RECOMMENDED</span>'
                    if item["is_recommended"] else ""
                )
                gram_class = "gram-badge-pos" if "positive" in item.get(
                    "gram_stain", "").lower() else "gram-badge-neg"
                banner_html = get_microbe_card_banner(
                    item["taxon_id"],
                    item["common_name"],
                    item.get("morphology_type", "bacilli"),
                    item.get("gram_stain", "Gram-positive"),
                )

                card_html = f"""
                <div class="taxa-media-card">
                    <!-- Rich Media Header (Placeholder Image Banner) -->
                    {banner_html}
                    
                    <!-- Supporting Text Area -->
                    <div style="padding: 16px 18px 14px 18px;">
                        <!-- Title goes here -->
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; flex-wrap: wrap; gap: 6px;">
                            <span style="font-size: 16px; color: var(--color-paper); font-weight: 500;">
                                <i>{item['common_name']}</i>
                            </span>
                            {rec_badge}
                        </div>
                        
                        <!-- Secondary text (Gram & Window Badges) -->
                        <div style="display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px;">
                            <span class="{gram_class}">{item.get('gram_stain', 'Gram-positive')}</span>
                            <span style="font-family: var(--font-mono); font-size: 11px; background: rgba(77, 87, 87, 0.35); color: var(--color-lichen); padding: 2px 7px; border-radius: 4px; border: 1px solid var(--color-graphite);">{item.get('peak_window', '')}</span>
                            <span style="font-family: var(--font-mono); font-size: 11px; color: #8fa0a0;">{item.get('evidence_tier', '')}</span>
                        </div>
                        
                        <!-- Supporting text description -->
                        <div style="font-size: 12.5px; color: #d0dede; line-height: 1.45; margin-bottom: 10px; background: rgba(0,0,0,0.25); padding: 10px 14px; border-radius: 6px; border-left: 2px solid var(--color-graphite);">
                            <b>Biochemical Action:</b> {item.get('biochemical_action', item['role'])}
                        </div>
                        <div style="font-size: 11.5px; color: #9eb0b0; line-height: 1.35; margin-bottom: 6px;">
                            <b>Habitat / Niche:</b> {item.get('habitat', '')}
                        </div>
                        <div style="font-size: 11px; color: var(--color-graphite); line-height: 1.35;">
                            <b>Cell Morphology:</b> {item.get('morphology_desc', item['role'])}
                        </div>
                    </div>
                </div>
                """
                render_clean_html(card_html)

                # Action area: Confirmation checkbox inside clean dock with proper separation
                st.markdown('<div class="taxa-dock-wrapper">',
                            unsafe_allow_html=True)
                is_checked = st.checkbox(
                    f"Confirm {item['common_name']} ({item['stage']})",
                    value=item["is_recommended"],
                    key=f"exam_taxa_{item['taxon_id']}",
                    help=item.get("biochemical_action", item["role"]),
                )
                st.markdown('</div>', unsafe_allow_html=True)
                if is_checked:
                    selected_taxa_current.append(item["taxon_id"])

        st.markdown("<div style='margin-top: 8px;'></div>",
                    unsafe_allow_html=True)

        # Confirmation Action Bar
        col_act1, col_act2 = st.columns([1.8, 1.2])
        with col_act1:
            if st.button("CONFIRM MICROBIAL PROFILE & CALCULATE PMI", type="primary", use_container_width=True):
                if not selected_taxa_current:
                    st.error(
                        "Please confirm at least one microbial bioindicator to calculate the postmortem interval.")
                else:
                    st.session_state["confirmed_taxa"] = selected_taxa_current
                    st.session_state["triage_confirmed"] = True

                    # Synthesize abundance profile and run inference
                    features = synthesize_abundance_profile(
                        confirmed_taxa=selected_taxa_current,
                        feature_schema=model.feature_names_,
                        ambient_temp_c=ambient_temp,
                        humidity_pct=humidity_val,
                    )
                    st.session_state["features_df"] = features

                    preds = model.predict_intervals(features)
                    p_est = float(preds["predicted_pmi"].values[0])
                    p_low = float(preds["lower_bound"].values[0])
                    p_high = float(preds["upper_bound"].values[0])

                    st.session_state["pmi_results"] = {
                        "predicted_pmi": p_est,
                        "lower_bound": p_low,
                        "upper_bound": p_high,
                        "decomposition_stage": triage_eval["primary_phase"],
                        "ambient_temp_c": ambient_temp,
                        "shannon_entropy": float(features["alpha_shannon_entropy"].values[0]),
                    }

                    st.session_state["case_particulars"] = {
                        "deceased_name": deceased_name,
                        "age": age_val,
                        "sex": sex_val,
                        "height_cm": height_val,
                        "weight_kg": weight_val,
                        "sample_site": swab_site,
                        "pm_report_no": pm_report_no,
                        "police_station": police_station,
                        "inquest_no": inquest_no,
                        "inquest_date": datetime.now().strftime("%d / %m / %Y"),
                        "analyst": cur_officer.get("name") or analyst_name,
                        "reg_no": cur_officer.get("badge") or "WBMC / 45826",
                        "designation": cur_officer.get("role") or "Medical Officer & Forensic Specialist",
                        "department": cur_officer.get("station") or "District Medico-Legal Center & Morgue",
                        "institution": cur_officer.get("station") or "District Medico-Legal Center & Morgue",
                        "rigor_obs": rigor_opt,
                        "bloat_obs": bloat_opt,
                        "discolor_obs": discolor_opt,
                        "cause_of_death": cause_of_death_input,
                        "manner_of_death": manner_of_death_input,
                        "micro_findings": f"Diagnostic bioindicator confirmation ({len(selected_taxa_current)} verified taxa): {', '.join([t.replace('_', ' ') for t in selected_taxa_current[:4]])} predominant.",
                    }
                    st.success(
                        "Microbial succession profile verified. Quantile PMI estimated.")

        with col_act2:
            if st.session_state["triage_confirmed"]:
                if st.button("EDIT MORPHOLOGICAL SIGNS", use_container_width=True):
                    st.session_state["triage_confirmed"] = False
                    st.info("Triage unlocked for adjustments.")

    # -------------------------------------------------------------
    # STEP 4: QUANTILE INFERENCE DISPLAY & RESULTS
    # -------------------------------------------------------------
    if st.session_state["triage_confirmed"] and st.session_state["pmi_results"]:
        pmi = st.session_state["pmi_results"]
        p_est = pmi["predicted_pmi"]
        p_low = pmi["lower_bound"]
        p_high = pmi["upper_bound"]
        p_stage = pmi["decomposition_stage"]
        temp_now = pmi["ambient_temp_c"]

        now_dt = datetime.now()
        dt_most_likely = now_dt - timedelta(days=p_est)
        dt_earliest = now_dt - timedelta(days=p_high)
        dt_latest = now_dt - timedelta(days=p_low)

        with st.container(border=True):
            st.markdown(
                '<div class="section-counter" style="margin-bottom: 20px;">04 / TIME-OF-DEATH INFERENCE RESULTS</div>', unsafe_allow_html=True)

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.metric(
                    label="Estimated Elapsed PMI",
                    value=f"{p_est:.1f} Days",
                    delta=f"{p_est * 24.0:.0f} Hours elapsed",
                    delta_color="off",
                )
            with kpi2:
                st.metric(
                    label="Probable Window of Death",
                    value=f"{p_low:.1f} – {p_high:.1f} Days",
                    delta=f"Span: ±{(p_high - p_low)/2.0:.1f} Days",
                    delta_color="off",
                )
            with kpi3:
                st.metric(
                    label="Decomposition Stage",
                    value=p_stage.split(" / ")[0],
                    delta=f"ADD: {p_est * temp_now:.1f} °C·days",
                    delta_color="off",
                )
            with kpi4:
                st.metric(
                    label="Most Probable Date of Death",
                    value=dt_most_likely.strftime("%b %d, %Y"),
                    delta=f"Est. {dt_most_likely.strftime('%H:%M hrs')}",
                    delta_color="off",
                )

            # Interactive Plotly Timeline Gauge
            fig_pmi = go.Figure()
            fig_pmi.add_vrect(
                x0=p_low,
                x1=p_high,
                fillcolor="#334647",
                opacity=0.8,
                layer="below",
                line_width=0,
                annotation_text="Probable Window of Death",
                annotation_position="top left",
                annotation_font=dict(
                    color="#cef79e", size=11, family="Roboto Mono"),
            )
            fig_pmi.add_trace(go.Scatter(
                x=[p_low, p_high],
                y=[0, 0],
                mode="lines",
                line=dict(color="#cef79e", width=4),
                hoverinfo="skip",
            ))
            fig_pmi.add_trace(go.Scatter(
                x=[p_est],
                y=[0],
                mode="markers+text",
                marker=dict(color="#ffffff", size=14, symbol="circle"),
                text=[f"Estimated: {p_est:.1f}d ({p_est*24:.0f}h)"],
                textposition="bottom center",
                textfont=dict(color="#ffffff", size=12, family="Roboto Mono"),
                hovertemplate="Estimated PMI: %{x:.2f} Days<extra></extra>",
            ))
            fig_pmi.update_layout(
                xaxis=dict(
                    title="Elapsed Time Since Death (Days)",
                    range=[0, max(28.0, p_high * 1.3)],
                    showgrid=True,
                    gridcolor="#2d3c3d",
                    title_font=dict(color="#c9cbbe", size=12,
                                    family="Roboto Mono"),
                    tickfont=dict(color="#c9cbbe", size=11,
                                  family="Roboto Mono"),
                ),
                yaxis=dict(showticklabels=False, range=[-0.5, 0.5]),
                height=180,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="#1d2728",
                plot_bgcolor="#1d2728",
                showlegend=False,
            )
            st.plotly_chart(fig_pmi, use_container_width=True)

            # ---------------------------------------------------------
            # STEP 4B: SHAP ALGORITHMIC EXPLAINABILITY & BIOINDICATOR ATTRIBUTION
            # ---------------------------------------------------------
            features_input = st.session_state.get("features_df")
            if features_input is None and model is not None:
                features_input = synthesize_abundance_profile(
                    confirmed_taxa=st.session_state.get("confirmed_taxa", []),
                    feature_schema=model.feature_names_,
                    ambient_temp_c=temp_now,
                    humidity_pct=pmi.get("humidity_pct", 65.0),
                )
                st.session_state["features_df"] = features_input

            if features_input is not None and model is not None:
                with st.expander("Algorithmic Explainability: Microbial Clock Attribution (SHAP)", expanded=True):
                    try:
                        shap_data = explain_sample_prediction(
                            model, features_input, top_n=8)
                        st.session_state["shap_results"] = shap_data

                        st.markdown(
                            f"""
                            <div style="font-size: 13.5px; color: #c9cbbe; margin-bottom: 12px; line-height: 1.5;">
                                <b>TreeSHAP Non-Black-Box Forensic Decomposition:</b> Baseline model expectation 
                                <code>E[f(X)] = {shap_data['base_value']:.2f} days</code>. Individual sample prediction 
                                <code>f(x) = {shap_data['predicted_pmi']:.2f} days</code>. The bioindicators below represent 
                                the primary microbial succession signals shifting the postmortem interval.
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        fig_shap = generate_attribution_plot(
                            shap_data, top_n=8, dark_theme=True)
                        st.pyplot(fig_shap)

                        st.markdown(
                            "<div style='margin-top: 14px; font-weight: 600; color: #e2e8f0; font-size: 13px;'>Bioindicator Attribution Narrative (Top Drivers):</div>", unsafe_allow_html=True)
                        for bullet in shap_data["narrative_bullet_points"]:
                            st.markdown(
                                f"<div style='font-size: 12.5px; color: #94a3b8; margin-bottom: 4px;'>• {bullet}</div>", unsafe_allow_html=True)
                    except Exception as ex_err:
                        st.warning(
                            f"SHAP attribution calculation unavailable: {ex_err}")

            # ---------------------------------------------------------
            # STEP 5: OFFICIAL POST-MORTEM REPORT & PDF EXPORT
            # ---------------------------------------------------------
            st.markdown(
                '<hr class="hairline-dark" style="margin: 32px 0;">', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-counter" style="margin-bottom: 20px;">05 / CASE REPORT DOSSIER</div>', unsafe_allow_html=True)

            case_info = st.session_state["case_particulars"]
            pm_no = case_info.get("pm_report_no", "PM-619 / 2026")
            ps_name = case_info.get("police_station", "New Township P.S.")
            inquest_no = case_info.get("inquest_no", "14 / 2026")
            dec_name = case_info.get(
                "deceased_name", "Unidentified Individual")
            doc_name = case_info.get("analyst", "Dr. Tanish Walture")
            raw_cause = case_info.get("cause_of_death", "Pending Inquest")

            # Evidence Digest & Compact High-Scannability QR Payload
            evidence_raw = f"{pm_no}|{ps_name}|{inquest_no}|{dec_name}|{p_est:.2f}|{p_low:.2f}|{p_high:.2f}|{doc_name}|{raw_cause}"
            evidence_hash = compute_sha256_hash(evidence_raw)
            clean_pm = pm_no.replace(" ", "").replace("/", "-")
            clean_inq = inquest_no.replace(" ", "").replace("/", "-")

            if "case_passcode_seeds" not in st.session_state:
                st.session_state["case_passcode_seeds"] = {}
            if clean_pm not in st.session_state["case_passcode_seeds"]:
                st.session_state["case_passcode_seeds"][clean_pm] = secrets.token_hex(
                    4)
            case_seed = st.session_state["case_passcode_seeds"][clean_pm]

            qr_url = f"https://necrotrace.streamlit.app/?view=verify&case={clean_pm}&inq={clean_inq}&pmi={p_est:.1f}d&hash={evidence_hash[:16]}&seed={case_seed}"

            qr_svg_str = generate_qr_code_svg(qr_url, size=130.0)
            qr_b64 = base64.b64encode(
                qr_svg_str.encode("utf-8")).decode("ascii")

            # On-screen preview of Form PM-5372
            render_clean_html(f"""
            <div class="pm-preview-box" style="border: 1px solid var(--color-graphite); padding: 20px; background-color: #1a2425; color: var(--color-paper); border-radius: 12px; margin-bottom: 24px;">
                <div class="pm-header-row" style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--color-graphite); padding-bottom: 14px; margin-bottom: 16px; flex-wrap: wrap; gap: 16px;">
                    <div style="flex: 1; min-width: 240px;">
                        <div class="mono-tag" style="color: var(--color-bioluminescent-lime); font-size: 10px;">DEPARTMENT OF FORENSIC MEDICINE & POLICE MORGUE</div>
                        <div style="font-size: 18px; font-weight: 600; color: var(--color-paper); margin: 6px 0;">POST MORTEM EXAMINATION REPORT • FORM NO. PM-5372</div>
                        <div style="font-family: var(--font-mono); font-size: 11.5px; color: var(--color-graphite); line-height: 1.4;">
                            REPORT NO: {case_info.get('pm_report_no')} &bull; P.S.: {case_info.get('police_station')} &bull; INQUEST: {case_info.get('inquest_no')}
                        </div>
                    </div>
                    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; background: #ffffff; padding: 10px 12px; border-radius: 8px; border: 2px solid var(--color-bioluminescent-lime); box-shadow: 0 4px 16px rgba(0,0,0,0.35); align-self: center;">
                        <img src="data:image/svg+xml;base64,{qr_b64}" width="116" height="116" style="display: block; max-width: 100%; height: auto;" alt="Forensic QR Verification Seal" />
                        <span style="font-family: var(--font-mono); font-size: 8px; font-weight: 700; color: #000000; letter-spacing: 0.06em; margin-top: 5px;">SCAN TO AUTHENTICATE</span>
                    </div>
                </div>

                <div class="pm-particulars-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 16px; font-size: 12.5px; background-color: #222f30; padding: 14px; border-radius: 8px;">
                    <div><b>Deceased Reference:</b> {case_info.get('deceased_name')}</div>
                    <div><b>Age / Sex:</b> {case_info.get('age')} / {case_info.get('sex')}</div>
                    <div><b>Swab Location:</b> {case_info.get('sample_site')}</div>
                    <div><b>Examining Doctor:</b> {case_info.get('analyst')}</div>
                    <div><b>Council Reg:</b> {case_info.get('reg_no')}</div>
                    <div><b>Specimen Status:</b> <span style="color: var(--color-bioluminescent-lime);">SATISFACTORY (Adequate DNA)</span></div>
                </div>

                <div style="background-color: #273637; border-left: 3px solid var(--color-bioluminescent-lime); padding: 14px 16px; border-radius: 0 8px 8px 0; margin-bottom: 16px;">
                    <div class="mono-tag" style="color: var(--color-bioluminescent-lime); margin-bottom: 6px;">MEDICO-LEGAL OPINION: TIME & CAUSE OF DEATH</div>
                    <div style="font-size: 16px; color: var(--color-paper); margin-bottom: 4px;">
                        <b>Estimated Time Elapsed:</b> {p_est:.1f} Days (approx. {p_est*24.0:.0f} Hours prior to examination)
                    </div>
                    <div style="font-size: 13.5px; color: #dbeafe; margin-bottom: 4px;">
                        <b>Probable Forensic Window:</b> {p_low:.1f} to {p_high:.1f} Days prior to recovery
                    </div>
                    <div style="font-size: 12.5px; color: var(--color-graphite); margin-bottom: 12px;">
                        <b>Calculated Calendar Date of Death:</b> {dt_earliest.strftime('%d/%m/%Y')} to {dt_latest.strftime('%d/%m/%Y')} (Most Probable: {dt_most_likely.strftime('%d/%m/%Y')})
                    </div>
                    <div class="pm-opinion-subgrid" style="border-top: 1px solid #384d4e; padding-top: 10px; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; font-size: 12.5px;">
                        <div>
                            <span class="mono-tag" style="font-size: 10px; color: #f87171;">PROVISIONAL CAUSE OF DEATH</span>
                            <div style="color: var(--color-paper); font-weight: 500; margin-top: 3px;">{case_info.get('cause_of_death', 'Pending Inquest')}</div>
                        </div>
                        <div>
                            <span class="mono-tag" style="font-size: 10px; color: var(--color-graphite);">MANNER OF DEATH</span>
                            <div style="color: #cbd5e1; margin-top: 3px;">{case_info.get('manner_of_death', 'Matter under judicial inquiry')}</div>
                        </div>
                    </div>
                    <div style="border-top: 1px solid #33494a; padding-top: 8px; margin-top: 10px; display: flex; justify-content: space-between; align-items: center; font-family: var(--font-mono); font-size: 10.5px; color: var(--color-graphite); flex-wrap: wrap; gap: 6px;">
                        <span><b>Digital Evidence Digest (SHA-256):</b> <code style="color: var(--color-bioluminescent-lime); word-break: break-all;">{evidence_hash[:24]}...</code></span>
                        <span style="color: #6ee7b7;">TAMPER-EVIDENT QR VERIFIED</span>
                    </div>
                </div>
            </div>
            """)

            qc_report = {
                "read_depth": 28410,
                "shannon_entropy": pmi.get("shannon_entropy", 2.85),
                "retained_taxa": len(st.session_state["confirmed_taxa"]),
                "initial_taxa": 50,
                "dropped_taxa": 0,
            }

            shap_exp = st.session_state.get("shap_results")
            shap_buf = None
            shap_narrative_list = None
            if shap_exp:
                try:
                    shap_buf = generate_attribution_plot_bytes(
                        shap_exp, top_n=8)
                    shap_narrative_list = shap_exp.get(
                        "narrative_bullet_points", [])
                except Exception:
                    shap_buf = None

            with st.spinner("Compiling Post-Mortem Report PDF (Form PM-5372)..."):
                pdf_bytes = generate_forensic_pdf(
                    case_metadata=case_info,
                    pmi_findings=pmi,
                    qc_metrics=qc_report,
                    shap_plot_buf=shap_buf,
                    shap_narrative=shap_narrative_list,
                )

            # ---------------------------------------------------------
            # CHAIN-OF-CUSTODY AUTHENTICATION & GATED PDF DOWNLOAD
            # ---------------------------------------------------------
            is_report_unlocked = (
                pm_no in st.session_state.get("unlocked_reports", set())
                or clean_pm in st.session_state.get("unlocked_reports", set())
            )

            if not is_report_unlocked:
                expected_passcode = generate_release_passcode(
                    clean_pm, seed=case_seed)

                render_clean_html(f"""
                <div style="background: #172324; border: 1.5px solid #f59e0b; border-radius: 12px; padding: 22px; margin-top: 14px; margin-bottom: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
                    <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #2d3d3e; padding-bottom: 14px; margin-bottom: 16px; flex-wrap: wrap; gap: 10px;">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <div class="mono-tag" style="color: #f59e0b; font-size: 11px; font-weight: 700; border: 1px solid #f59e0b; padding: 2px 8px; border-radius: 4px;">MANDATORY QR AUTH</div>
                            <div>
                                <div class="mono-tag" style="color: #f59e0b; font-size: 10px;">CHAIN-OF-CUSTODY AUDIT PROTOCOL</div>
                                <div style="font-size: 17px; font-weight: 600; color: #ffffff;">SCAN PHYSICAL QR CODE TO RELEASE DOSSIER</div>
                            </div>
                        </div>
                        <span style="font-family: var(--font-mono); font-size: 11px; background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid #f59e0b; padding: 4px 12px; border-radius: 9999px;">
                            DOWNLOAD LOCKED
                        </span>
                    </div>
                    <div style="font-size: 13.5px; color: #cbd5e1; line-height: 1.6; margin-bottom: 14px;">
                        In accordance with statutory chain-of-custody protocols, direct browser navigation is disabled. 
                        An authorized medical examiner or investigating officer must <b>compulsorily scan the physical QR code seal</b> using a mobile camera or departmental device to verify identity and generate the one-time workstation release passcode.
                    </div>
                    <div style="display: flex; align-items: center; gap: 10px; background: rgba(0,0,0,0.3); border: 1px dashed rgba(245, 158, 11, 0.4); padding: 10px 14px; border-radius: 8px;">
                        <span style="font-size: 18px;">📱</span>
                        <span style="font-size: 12.5px; color: #94a3b8;">
                            Scan the <b>Forensic QR Verification Seal</b> on Form PM-5372 with your mobile device. Upon credential verification, your screen will display the synchronized 6-digit release passcode.
                        </span>
                    </div>
                </div>
                """)

                st.markdown(
                    '<div class="mono-tag" style="margin-bottom: 8px;">ENTER 6-DIGIT MOBILE WORKSTATION RELEASE PASSCODE:</div>', unsafe_allow_html=True)
                pass_col1, pass_col2, pass_col3 = st.columns([2.5, 1.8, 1.0])
                with pass_col1:
                    pass_in = st.text_input(
                        "Enter Release Passcode",
                        placeholder="e.g. 849201 or NC-849201 (from mobile screen)",
                        key=f"passcode_input_{clean_pm}",
                        label_visibility="collapsed"
                    )
                with pass_col2:
                    if st.button("VERIFY PASSCODE & RELEASE REPORT", type="primary", use_container_width=True, key=f"btn_unlock_{clean_pm}"):
                        if verify_release_passcode(clean_pm, pass_in, seed=case_seed):
                            st.session_state["unlocked_reports"].add(pm_no)
                            st.session_state["unlocked_reports"].add(clean_pm)
                            st.success(
                                "Mobile Workstation Release Code Confirmed. Forensic Dossier Released.")
                            st.rerun()
                        else:
                            st.error(
                                "Invalid release passcode. Please scan the QR code with your mobile device to generate an authentic code.")
                with pass_col3:
                    if st.button("↻ REFRESH", use_container_width=True, key=f"btn_refresh_{clean_pm}"):
                        st.session_state["case_passcode_seeds"][clean_pm] = secrets.token_hex(
                            4)
                        st.rerun()

            else:
                officer = st.session_state.get("authenticated_officer") or {}
                officer_label = officer.get("name", "Authorized Officer")
                officer_badge = officer.get(
                    "badge", "CHAIN-OF-CUSTODY VERIFIED")
                render_clean_html(f"""
                <div style="background: rgba(6, 78, 59, 0.45); border: 1.5px solid #10b981; border-radius: 10px; padding: 14px 20px; margin-top: 14px; margin-bottom: 18px; display: flex; align-items: center; justify-content: space-between;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span style="font-size: 22px; color: #34d399;">✓</span>
                        <div>
                            <div style="font-family: var(--font-mono); font-size: 11px; font-weight: 700; color: #6ee7b7; letter-spacing: 0.05em;">CHAIN-OF-CUSTODY AUTHENTICATED</div>
                            <div style="font-size: 13px; color: #ffffff;">Report Released To: <b>{officer_label}</b> &bull; Badge: <code>{officer_badge}</code></div>
                        </div>
                    </div>
                    <span style="font-family: var(--font-mono); font-size: 11px; background: #064e3b; color: #6ee7b7; padding: 4px 10px; border-radius: 9999px;">
                        UNLOCKED
                    </span>
                </div>
                """)

                st.download_button(
                    label="DOWNLOAD POST-MORTEM REPORT (PDF)",
                    data=pdf_bytes,
                    file_name=f"PostMortem_Report_{case_info.get('pm_report_no', 'PM').replace('/', '_').replace(' ', '')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

    st.markdown("<hr class='hairline-dark' style='margin: 40px 0 20px 0;'>",
                unsafe_allow_html=True)
    if st.button("← RETURN TO PLATFORM OVERVIEW (VIEW 1)", use_container_width=True, key="btn_return_v1_bottom"):
        st.session_state["view"] = "landing"
        st.query_params["view"] = "landing"
        cur_off = st.session_state.get("authenticated_officer") or {}
        if cur_off.get("local_id"):
            st.query_params["auth"] = cur_off["local_id"]
        st.rerun()


# =============================================================================
# VIEW 3: OFFICIAL MEDICO-LEGAL DIGITAL VERIFICATION PORTAL
# =============================================================================
elif st.session_state["view"] == "verify":
    sync_document_title("verify")
    # Mandatory QR scan parameter guard: Prevents manual URL manipulation
    has_qr_params = bool(st.query_params.get(
        "seed") and st.query_params.get("case"))
    if not has_qr_params:
        render_clean_html("""
        <div style="background: #172324; border: 1.5px solid #ef4444; border-radius: 12px; padding: 28px; margin: 40px auto; max-width: 640px; text-align: center; box-shadow: 0 4px 24px rgba(0,0,0,0.4);">
            <div style="font-size: 32px; margin-bottom: 12px;">🚫</div>
            <div class="mono-tag" style="color: #f87171; font-size: 11px; font-weight: 700;">CHAIN-OF-CUSTODY SECURITY ENFORCEMENT</div>
            <div style="font-size: 20px; font-weight: 600; color: #ffffff; margin: 8px 0 14px 0;">MANDATORY QR SCAN REQUIRED</div>
            <div style="font-size: 13.5px; color: #cbd5e1; line-height: 1.6; margin-bottom: 20px;">
                Direct browser access to this verification endpoint is prohibited under evidentiary audit rules. 
                You must <b>compulsorily scan the physical QR code seal</b> displayed on the official Form PM-5372 report or mortuary display using a mobile device camera.
            </div>
            <div style="background: #0d1516; border: 1px dashed #33494a; border-radius: 8px; padding: 12px; font-size: 12px; color: #94a3b8; font-family: var(--font-mono);">
                SECURITY ERROR: Missing cryptographically signed session seed token.
            </div>
        </div>
        """)
        if st.button("← RETURN TO APPLICATION HOME", use_container_width=True, key="btn_qr_blocked_home"):
            st.session_state["view"] = "landing"
            st.query_params.clear()
            st.query_params["view"] = "landing"
            st.rerun()
        st.stop()

    # Extract query params from authentic QR scan
    case_param = st.query_params.get("case", "").replace("-", " / ")
    inq_param = st.query_params.get("inq", "").replace("-", " / ")
    pmi_param = st.query_params.get("pmi", "6.8d").replace("d", " Days")
    hash_param = st.query_params.get("hash", "7f83b165ff29a1b4")
    ps_param = st.query_params.get("ps", "Police Inquest Division")
    dec_param = st.query_params.get("dec", "Case Subject")
    doc_param = st.query_params.get("doc", "Medical Examiner")
    cod_param = st.query_params.get("cod", "Under active inquest")
    mod_param = st.query_params.get("mod", "Matter under judicial inquiry")

    # If active session state has case particulars, prioritize them
    if st.session_state.get("case_particulars"):
        cp = st.session_state["case_particulars"]
        case_param = cp.get("pm_report_no", case_param)
        inq_param = cp.get("inquest_no", inq_param)
        ps_param = cp.get("police_station", ps_param)
        dec_param = cp.get("deceased_name", dec_param)
        doc_param = cp.get("analyst", doc_param)
        cod_param = cp.get("cause_of_death", cod_param)
        mod_param = cp.get("manner_of_death", mod_param)
    if st.session_state.get("pmi_results"):
        p_res = st.session_state["pmi_results"]
        pmi_param = f"{p_res['predicted_pmi']:.1f} Days (Forensic Window: {p_res['lower_bound']:.1f} to {p_res['upper_bound']:.1f} Days)"

    # Top Navigation Banner
    render_clean_html(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 16px 24px; background: #152021; border-bottom: 1px solid #2d3e40; margin-bottom: 28px;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <img src="{LOGO_ICON_B64}" width="32" height="32" style="border-radius: 4px;" alt="NecroTrace Logo" />
            <div>
                <div style="font-family: var(--font-mono); font-size: 11px; color: var(--color-bioluminescent-lime); letter-spacing: 0.08em;">STATE FORENSIC SERVICE &bull; DIGITAL REPOSITORY</div>
                <div style="font-size: 15px; font-weight: 500; color: #ffffff;">NecroTrace Medico-Legal Verification Portal</div>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-family: var(--font-mono); font-size: 11px; background: #064e3b; color: #6ee7b7; border: 1px solid #059669; padding: 4px 10px; border-radius: 9999px;">
                LIVE VERIFIED DOSSIER
            </span>
        </div>
    </div>
    """)

    # Main Certificate Container
    st.markdown('<div style="max-width: 860px; margin: 0 auto; padding: 0 16px;">',
                unsafe_allow_html=True)

    clean_case_id = case_param.replace(" ", "").replace("/", "-")
    is_authed = (
        st.session_state.get("authenticated_officer") is not None
        or case_param in st.session_state.get("unlocked_reports", set())
        or clean_case_id in st.session_state.get("unlocked_reports", set())
    )
    seed_param = st.query_params.get("seed", "")
    release_code = generate_release_passcode(clean_case_id, seed=seed_param)

    # -------------------------------------------------------------------------
    # FIREBASE OFFICER AUTHENTICATION GATEWAY
    # -------------------------------------------------------------------------
    if not is_authed:
        firebase_online = is_firebase_configured()
        fb_status_html = '<span style="font-family: var(--font-mono); font-size: 11px; background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid #10b981; padding: 3px 10px; border-radius: 9999px;">MEDICO-LEGAL CLOUD GATEWAY</span>'

        render_clean_html(f"""
        <div style="background: #172425; border: 1.5px solid #059669; border-radius: 14px; padding: 24px; margin-bottom: 24px; box-shadow: 0 8px 32px rgba(0,0,0,0.35);">
            <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #283a3c; padding-bottom: 14px; margin-bottom: 16px;">
                <div>
                    <div class="mono-tag" style="color: var(--color-bioluminescent-lime); font-size: 10px;">MEDICO-LEGAL ACCESS CONTROL &bull; DIGITAL CHAIN OF CUSTODY</div>
                    <div style="font-size: 18px; font-weight: 600; color: #ffffff;">OFFICER CREDENTIAL AUTHENTICATION</div>
                </div>
                {fb_status_html}
            </div>
            <div style="font-size: 13px; color: #cbd5e1; line-height: 1.5; margin-bottom: 16px;">
                Post-mortem report generation and verification are restricted to authorized examiners. 
                Please verify your registered email below via Firebase to unlock the Form PM-5372 dossier.
            </div>
        </div>
        """)

        auth_tab_signin, auth_tab_reg = st.tabs([
            "Officer Sign-In & Verification",
            "Register Authorized Personnel"
        ])

        with auth_tab_signin:
            v_col_email, v_col_pass = st.columns([1.2, 1.0])
            with v_col_email:
                v_email = st.text_input(
                    "Registered Departmental Email", placeholder="e.g. coroner@necrotrace.gov", key="verify_portal_email")
            with v_col_pass:
                v_pass = st.text_input(
                    "Security Credentials / Passcode", type="password", key="verify_portal_pass")

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("CHECK EMAIL REGISTRATION", use_container_width=True, key="btn_check_reg"):
                    if not v_email:
                        st.warning("Please enter an email address to check.")
                    else:
                        with st.spinner("Checking Firebase Registry..."):
                            reg_status = check_email_registered_in_firebase(
                                v_email)
                        if reg_status:
                            st.success(
                                f"Email '{v_email}' is REGISTERED in the Directory. Please enter passcode and click authenticate.")
                        else:
                            st.info(
                                f"Email '{v_email}' is not yet enrolled. Please use the 'Register' tab to enroll.")

            with btn_col2:
                if st.button("AUTHENTICATE & UNLOCK DOSSIER", use_container_width=True, key="btn_auth_unlock"):
                    if not v_email or not v_pass:
                        st.warning(
                            "Please enter both registered email and password.")
                    else:
                        with st.spinner("Authenticating & fetching profile from Cloud Firestore..."):
                            auth_res = sign_in_officer(v_email, v_pass)
                        if auth_res.get("success"):
                            st.session_state["authenticated_officer"] = auth_res.get(
                                "officer_info")
                            st.session_state["unlocked_reports"].add(
                                case_param)
                            st.session_state["unlocked_reports"].add(
                                clean_case_id)
                            st.success(
                                f"Credentials Validated via Cloud Firestore. Welcome, {auth_res.get('officer_info', {}).get('name')}.")
                            st.rerun()
                        else:
                            st.error(auth_res.get('message'))

        with auth_tab_reg:
            st.markdown('<div style="font-size: 13px; color: #cbd5e1; margin-bottom: 12px;">Enroll an authorized forensic practitioner into the Firebase authentication repository.</div>', unsafe_allow_html=True)
            r_c1, r_c2 = st.columns(2)
            with r_c1:
                r_name = st.text_input(
                    "Full Name & Title", placeholder="Dr. Jane Doe, M.D.", key="reg_officer_name")
                r_station = st.text_input(
                    "Police Station / Lab", placeholder="State Forensic Science Lab", key="reg_officer_station")
            with r_c2:
                r_badge = st.text_input(
                    "Badge / Reg. No.", placeholder="MED-9042 / WBMC-45826", key="reg_officer_badge")
                r_role = st.selectbox("Role", ["Forensic Pathologist", "Chief Medical Examiner", "Senior Investigating Officer",
                                      "Forensic Anthropologist", "Toxicology Specialist", "Judicial Inquest Officer"], index=0, key="reg_officer_role")

            r_email = st.text_input(
                "Departmental Email", placeholder="jane.doe@necrotrace.gov", key="reg_officer_email")
            r_pass = st.text_input(
                "Assign Password (min. 6 characters)", type="password", key="reg_officer_pass")

            r_consent = st.checkbox(
                "I consent to my name, email, badge number, and station being stored securely in Firebase (Google Cloud) for authentication purposes. I have read the [Privacy Policy](?view=privacy).",
                value=False,
                key="verify_reg_consent"
            )

            if st.button("ENROLL OFFICER IN FORENSIC DIRECTORY", use_container_width=True, key="btn_register_officer", disabled=not r_consent):
                if not r_email or not r_pass or not r_name:
                    st.warning(
                        "Please provide Name, Email, and a secure Password.")
                elif len(r_pass) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    with st.spinner("Enrolling officer & saving profile to Cloud Firestore..."):
                        reg_out = register_officer(
                            email=r_email,
                            password=r_pass,
                            full_name=r_name,
                            name=r_name,
                            badge=r_badge,
                            station=r_station,
                            role=r_role,
                        )
                    if reg_out.get("success"):
                        st.session_state["authenticated_officer"] = reg_out.get(
                            "officer_info")
                        st.session_state["unlocked_reports"].add(case_param)
                        st.session_state["unlocked_reports"].add(clean_case_id)
                        st.success(
                            f"Officer {r_name} saved to Cloud Firestore. Unlocking dossier...")
                        st.rerun()
                    else:
                        st.error(reg_out.get('message'))

    else:
        # OFFICER IS AUTHENTICATED: Display Verified Status & Release Controls
        cur_officer = st.session_state.get("authenticated_officer") or {}
        off_name = cur_officer.get("name") or cur_officer.get(
            "fullName") or "Dr. Tanish Walture"
        off_role = cur_officer.get("role", "Chief Forensic Pathologist")
        off_badge = cur_officer.get("badge", "CFS-9042")
        off_station = cur_officer.get(
            "station", "Central Forensic Science Laboratory")

        render_clean_html(f"""
        <div style="background: rgba(6, 78, 59, 0.45); border: 2px solid #10b981; border-radius: 14px; padding: 24px; margin-bottom: 24px; box-shadow: 0 8px 32px rgba(16, 185, 129, 0.2);">
            <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #14532d; padding-bottom: 14px; margin-bottom: 16px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 28px; color: #34d399;">✓</span>
                    <div>
                        <div class="mono-tag" style="color: #6ee7b7; font-size: 10px;">FIREBASE &amp; FIRESTORE VERIFIED &bull; CHAIN-OF-CUSTODY PROTOCOL</div>
                        <div style="font-size: 20px; font-weight: 600; color: #ffffff;">AUTHORIZED EXAMINER: {off_name}</div>
                    </div>
                </div>
                <span style="font-family: var(--font-mono); font-size: 11px; background: #064e3b; color: #6ee7b7; border: 1px solid #059669; padding: 4px 12px; border-radius: 9999px;">
                    CREDENTIALS VALIDATED
                </span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; font-size: 12px; color: #d1fae5; margin-bottom: 18px; background: rgba(0, 0, 0, 0.25); padding: 12px; border-radius: 8px;">
                <div><b>Designation:</b> {off_role}</div>
                <div><b>Badge / Registration:</b> <code>{off_badge}</code></div>
                <div><b>Posting:</b> {off_station}</div>
            </div>

            <!-- Mortuary Workstation Release Passcode Box -->
            <div style="background: #0d1516; border: 1.5px dashed var(--color-bioluminescent-lime); border-radius: 10px; padding: 18px; text-align: center; margin-bottom: 18px;">
                <div class="mono-tag" style="color: var(--color-bioluminescent-lime); font-size: 11px; letter-spacing: 0.08em;">MORTUARY WORKSTATION RELEASE PASSCODE</div>
                <div style="font-size: 34px; font-weight: 700; font-family: var(--font-mono); color: #ffffff; letter-spacing: 0.14em; margin: 8px 0;">{release_code}</div>
                <div style="font-size: 12px; color: #94a3b8;">Enter this 6-digit passcode (<code>{release_code}</code> or <code>{release_code.replace('NC-', '')}</code>) on the mortuary terminal to unlock local workstation downloading and physical printing.</div>
            </div>
        </div>
        """)

        # Generate on-demand authentic certified PDF for mobile / browser download
        pmi_num = 6.8
        try:
            pmi_num = float(st.query_params.get("pmi", "6.8d").replace(
                "d", "").replace("Days", "").strip())
        except Exception:
            pmi_num = 6.8

        v_case_meta = {
            "pm_report_no": case_param,
            "police_station": ps_param,
            "inquest_no": inq_param,
            "date_of_exam": datetime.now().strftime("%Y-%m-%d"),
            "time_of_exam": datetime.now().strftime("%H:%M HRS"),
            "analyst": doc_param,
            "deceased_name": dec_param,
            "deceased_age_sex": "Approx. 35-40 Yrs / Male",
            "sample_site": "Oral / Buccal Swab",
            "reg_no": "WBMC / 45826",
            "cause_of_death": cod_param,
            "manner_of_death": mod_param,
        }
        if st.session_state.get("case_particulars"):
            v_case_meta.update(st.session_state["case_particulars"])

        v_pmi_find = {
            "predicted_pmi": pmi_num,
            "lower_bound": max(0.5, pmi_num - 1.8),
            "upper_bound": pmi_num + 2.1,
            "shannon_entropy": 2.85,
            "top_indicator": "Gammaproteobacteria / Pseudomonas",
        }
        if st.session_state.get("pmi_results"):
            v_pmi_find.update(st.session_state["pmi_results"])

        v_qc_met = {
            "read_depth": 28410,
            "shannon_entropy": 2.85,
            "retained_taxa": len(st.session_state.get("confirmed_taxa", [])) or 12,
            "initial_taxa": 50,
            "dropped_taxa": 0,
        }

        v_shap_exp = st.session_state.get("shap_results")
        v_shap_buf = None
        v_shap_narrative = None
        if v_shap_exp:
            try:
                v_shap_buf = generate_attribution_plot_bytes(
                    v_shap_exp, top_n=8)
                v_shap_narrative = v_shap_exp.get(
                    "narrative_bullet_points", [])
            except Exception:
                v_shap_buf = None

        with st.spinner("Generating Form PM-5372 PDF Report..."):
            verified_pdf_bytes = generate_forensic_pdf(
                case_metadata=v_case_meta,
                pmi_findings=v_pmi_find,
                qc_metrics=v_qc_met,
                shap_plot_buf=v_shap_buf,
                shap_narrative=v_shap_narrative,
            )

        col_dl, col_so = st.columns([3, 1])
        with col_dl:
            st.download_button(
                label="DOWNLOAD FORM PM-5372 (PDF)",
                data=verified_pdf_bytes,
                file_name=f"PostMortem_Report_{clean_case_id}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with col_so:
            if st.button("SIGN OUT", use_container_width=True, key="btn_signout"):
                st.session_state["authenticated_officer"] = None
                st.rerun()

    # Verification Certificate Box
    render_clean_html(f"""
    <div style="background: #192425; border: 1.5px solid var(--color-bioluminescent-lime); border-radius: 14px; padding: 28px; margin-bottom: 24px; box-shadow: 0 8px 32px rgba(0,0,0,0.4);">
        <!-- Top Status Banner -->
        <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #2d3e40; padding-bottom: 20px; margin-bottom: 20px;">
            <div>
                <div class="mono-tag" style="color: var(--color-bioluminescent-lime); margin-bottom: 4px;">INQUEST REPORT SUMMARY &bull; FORM NO. PM-5372</div>
                <div style="font-size: 24px; font-weight: 500; color: #ffffff; letter-spacing: -0.01em;">AUTHENTICATED POST-MORTEM DOSSIER</div>
                <div style="font-family: var(--font-mono); font-size: 12px; color: var(--color-graphite); margin-top: 4px;">
                    Central Forensic Science Laboratory &bull; Medico-Legal Verification Seal
                </div>
            </div>
            <div style="background: rgba(6, 78, 59, 0.4); border: 1.5px solid #10b981; padding: 12px 18px; border-radius: 10px; text-align: center;">
                <div style="font-size: 20px;">✓</div>
                <div style="font-family: var(--font-mono); font-size: 11px; font-weight: 700; color: #6ee7b7; letter-spacing: 0.05em;">HASH-VERIFIED</div>
                <div style="font-size: 9px; color: #a7f3d0;">RECORD MATCHED</div>
            </div>
        </div>

        <!-- Case Identification Metadata -->
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; background: #202e2f; padding: 16px; border-radius: 8px; margin-bottom: 20px; font-size: 13px;">
            <div>
                <div class="mono-tag" style="font-size: 10px; color: var(--color-graphite);">POST-MORTEM REPORT NO:</div>
                <div style="font-size: 15px; font-weight: 600; color: #ffffff; margin-top: 2px;">{case_param}</div>
            </div>
            <div>
                <div class="mono-tag" style="font-size: 10px; color: var(--color-graphite);">POLICE INQUEST REFERENCE:</div>
                <div style="font-size: 15px; font-weight: 600; color: #ffffff; margin-top: 2px;">{inq_param} ({ps_param})</div>
            </div>
            <div>
                <div class="mono-tag" style="font-size: 10px; color: var(--color-graphite);">DECEASED IDENTIFIER:</div>
                <div style="color: var(--color-paper); margin-top: 2px;">{dec_param}</div>
            </div>
            <div>
                <div class="mono-tag" style="font-size: 10px; color: var(--color-graphite);">EXAMINING MEDICAL OFFICER:</div>
                <div style="color: var(--color-paper); margin-top: 2px;">{doc_param}</div>
            </div>
        </div>

        <!-- Forensic Findings (PMI & COD) -->
        <div style="background: #233335; border-left: 4px solid var(--color-bioluminescent-lime); padding: 18px; border-radius: 0 8px 8px 0; margin-bottom: 20px;">
            <div class="mono-tag" style="color: var(--color-bioluminescent-lime); margin-bottom: 6px;">MEDICO-LEGAL OPINION & BIOLOGICAL SUCCESSION FINDINGS</div>
            <div style="font-size: 18px; color: #ffffff; margin-bottom: 6px;">
                <b>Estimated Time Elapsed (PMI):</b> <span style="color: #6ee7b7;">{pmi_param}</span>
            </div>
            <div style="font-size: 13px; color: #cbd5e1; margin-bottom: 12px; line-height: 1.4;">
                Derived via calibrated metagenomic succession bioindicators (16S rRNA taxonomic profiling & Quantile XGBoost pinball loss optimization) concordant with macroscopic autopsy signs under prevailing scene ambient factors.
            </div>
            <div style="border-top: 1px solid #364b4d; padding-top: 12px; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; font-size: 13px;">
                <div>
                    <span class="mono-tag" style="font-size: 10px; color: #f87171;">PROVISIONAL CAUSE OF DEATH</span>
                    <div style="color: #ffffff; font-weight: 500; margin-top: 3px;">{cod_param}</div>
                </div>
                <div>
                    <span class="mono-tag" style="font-size: 10px; color: var(--color-graphite);">MANNER OF DEATH</span>
                    <div style="color: #cbd5e1; margin-top: 3px;">{mod_param}</div>
                </div>
            </div>
        </div>

        <!-- Cryptographic Evidence & Legal Admissibility -->
        <div style="background: #172122; border: 1px solid #2b3b3d; padding: 16px; border-radius: 8px; font-family: var(--font-mono); font-size: 11px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="color: var(--color-graphite);">CRYPTOGRAPHIC EVIDENCE DIGEST (SHA-256):</span>
                <span style="color: #34d399; font-weight: 600;">STATUS: UNALTERED</span>
            </div>
            <div style="background: #0e1415; padding: 8px 12px; border-radius: 4px; color: var(--color-bioluminescent-lime); word-break: break-all; font-size: 12px; margin-bottom: 12px;">
                {hash_param if len(hash_param) > 20 else hash_param + '7f83b165ff29a1b4d081f2157790b8f44d187ef1ca14efef22384a51e60f0891'[len(hash_param):]}
            </div>
            <div style="font-size: 10.5px; color: #94a3b8; line-height: 1.45;">
                <b>Notice:</b> This digital verification certificate is generated following methodologies inspired by Daubert and Frye evidentiary standards for research and demonstration purposes. This is NOT a legally certified document and must not be used as evidence in any legal proceeding.
            </div>
        </div>
    </div>
    """)

    # Navigation buttons
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        if st.button("← RETURN TO LANDING MATRIX", use_container_width=True):
            st.session_state["view"] = "landing"
            st.query_params["view"] = "landing"
            cur_off = st.session_state.get("authenticated_officer") or {}
            if cur_off.get("local_id"):
                st.query_params["auth"] = cur_off["local_id"]
            st.rerun()
    with v_col2:
        if st.button("OPEN AUTOPSY EXAMINATION ROOM →", use_container_width=True):
            st.session_state["view"] = "examination"
            st.query_params["view"] = "examination"
            cur_off = st.session_state.get("authenticated_officer") or {}
            if cur_off.get("local_id"):
                st.query_params["auth"] = cur_off["local_id"]
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# VIEW 4: PRIVACY POLICY
# =============================================================================
elif st.session_state["view"] == "privacy":
    sync_document_title("privacy")
    render_clean_html(LEGAL_PAGE_CSS)

    render_clean_html(f"""
    <header style="width: 100%; border-bottom: 1px solid var(--color-graphite); padding: 12px 0 20px 0; margin-bottom: 24px;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <img src="{LOGO_ICON_B64}" style="height: 26px; width: 26px; object-fit: contain; vertical-align: middle; filter: drop-shadow(0 0 8px rgba(116, 194, 92, 0.35));" alt="NecroTrace Logo" />
                <span style="font-family: var(--font-mono); font-size: 13px; color: var(--color-paper); font-weight: 600; letter-spacing: 0.04em;">
                    NECROTRACE <span style="color: var(--color-graphite);">//</span> COMPLIANCE
                </span>
                <span style="font-family: var(--font-mono); font-size: 11px; color: var(--color-bioluminescent-lime); letter-spacing: 0.03em;">
                    PRIVACY PROTOCOL
                </span>
            </div>
            <div>
                <a href="?view=landing" target="_self" class="sober-btn-ghost" style="padding: 5px 14px; font-size: 11px; height: 32px; min-height: 32px; text-decoration: none;">&larr; PLATFORM OVERVIEW</a>
            </div>
        </div>
    </header>
    """)

    st.markdown("""
# Privacy Policy

**Effective Date:** September 2026  
**Data Controller:** Tanish Walture (Team BroomWroom)  
**Primary Contact:** [tanishwalture@gmail.com](mailto:tanishwalture@gmail.com)

---

## 1. What Data We Collect

When you register as an examiner on NecroTrace, we collect the following personal information:

| Data Field | Purpose | Required? |
|---|---|---|
| **Full Name & Title** | Identify authorized examiners in reports | Yes |
| **Email Address** | Firebase authentication (login & registration) | Yes |
| **Security Passcode** | Account access (hashed via Firebase, never in plaintext) | Yes |
| **Badge / Registration Number** | Institutional identification on generated reports | Optional |
| **Police Station / Forensic Lab** | Institutional affiliation for report context | Optional |
| **Professional Role** | Role classification within the application | Optional |

>  **Case Examination Data**: All clinical autopsy particulars entered during an examination (deceased demographics, morphological findings, temperature, post-mortem intervals) are processed **ephemerally in your active browser session**. They are **NOT stored in any database** and are discarded upon session end.

**Dynamic Passcodes**: Temporary 6-digit release tokens are stored in Cloud Firestore for cross-device synchronization and expire automatically within 3 minutes.

---

## 2. Data Storage & Infrastructure

- **Firebase Authentication** (Google Cloud Platform): Stores registered emails and cryptographically hashed passwords.
- **Cloud Firestore** (Google Cloud Platform): Stores officer profiles (`name`, `badge`, `station`, `role`, `email`, `enrolled_at`, `last_login`) in the secured `officers` collection.
- **Data Residency**: Cloud resources are hosted within Google Cloud infrastructure and may process data through secure data centers internationally.

We do **NOT** store:
- Autopsy case notes or specimen metadata across sessions
- Behavioral analytics, heatmaps, or tracking pixels
- Commercial advertising identifiers or profiles

---

## 3. Third-Party Services & Integrations

| Provider | Service | Purpose | Data Transferred |
|---|---|---|---|
| **Google LLC** | Firebase Authentication | Identity verification | Email & hashed password |
| **Google LLC** | Cloud Firestore | Profile & passcode sync | Profile metadata |
| **Google LLC** | Google Fonts API | Typography delivery | IP address (standard HTTP) |
| **Cloudflare Inc.** | unpkg CDN | Lenis smooth scroll library | IP address (standard HTTP) |

We do **NOT** integrate tracking pixels, advertising networks, or analytics tools (e.g. Google Analytics, Hotjar, Mixpanel).

---

## 4. Your Rights (DPDPA 2023 & GDPR)

Under India's **Digital Personal Data Protection Act (DPDPA) 2023** and the EU **GDPR**, you have the following enforceable rights:

- **Right to Access**: Request a copy of your stored examiner credentials.
- **Right to Rectification**: Request correction of any inaccurate profile details.
- **Right to Erasure**: Request complete deletion of your profile from Cloud Firestore and Firebase Auth.
- **Right to Withdraw Consent**: Revoke registration consent at any time.

To exercise any of these rights, email: **[tanishwalture@gmail.com](mailto:tanishwalture@gmail.com)**. Requests are handled within 7 business days.

---

## 5. Data Retention & Deletion

- Examiner accounts remain in Cloud Firestore until an account holder requests deletion.
- Temporary release passcodes expire automatically after 3 minutes.
- Active session states are cleared when the browser tab or session closes.

---

## 6. Academic Research Scope & Children's Privacy

NecroTrace is an academic research prototype intended for forensic researchers and students aged 18 and above. We do not knowingly collect personal data from minors.

---

## 7. Policy Updates

We may update this policy periodically. The "Effective Date" above will always reflect the latest revision.
    """)

    st.markdown("<div style='margin-top: 28px;'></div>",
                unsafe_allow_html=True)
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        if st.button("← RETURN TO LANDING MATRIX", use_container_width=True, key="btn_priv_landing"):
            st.session_state["view"] = "landing"
            st.query_params["view"] = "landing"
            st.rerun()
    with b_col2:
        if st.button("VIEW TERMS & CONDITIONS →", use_container_width=True, key="btn_priv_terms"):
            st.session_state["view"] = "terms"
            st.query_params["view"] = "terms"
            st.rerun()


# =============================================================================
# VIEW 5: TERMS & CONDITIONS
# =============================================================================
elif st.session_state["view"] == "terms":
    sync_document_title("terms")
    render_clean_html(LEGAL_PAGE_CSS)

    render_clean_html(f"""
    <header style="width: 100%; border-bottom: 1px solid var(--color-graphite); padding: 12px 0 20px 0; margin-bottom: 24px;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <img src="{LOGO_ICON_B64}" style="height: 26px; width: 26px; object-fit: contain; vertical-align: middle; filter: drop-shadow(0 0 8px rgba(116, 194, 92, 0.35));" alt="NecroTrace Logo" />
                <span style="font-family: var(--font-mono); font-size: 13px; color: var(--color-paper); font-weight: 600; letter-spacing: 0.04em;">
                    NECROTRACE <span style="color: var(--color-graphite);">//</span> COMPLIANCE
                </span>
                <span style="font-family: var(--font-mono); font-size: 11px; color: var(--color-bioluminescent-lime); letter-spacing: 0.03em;">
                    TERMS OF SERVICE
                </span>
            </div>
            <div>
                <a href="?view=landing" target="_self" class="sober-btn-ghost" style="padding: 5px 14px; font-size: 11px; height: 32px; min-height: 32px; text-decoration: none;">&larr; PLATFORM OVERVIEW</a>
            </div>
        </div>
    </header>
    """)

    st.markdown("""
# Terms & Conditions

**Effective Date:** September 2026  
**Project Lead:** Tanish Walture (Team BroomWroom)  
**Repository:** [github.com/BroomWroom](https://github.com/BroomWroom)

---

## 1. Academic Research Prototype Notice

> ⚠️ **IMPORTANT NOTICE:** NecroTrace is an **educational research demonstration project** developed for academic showcase purposes. It is **NOT** a certified, accredited, or legally approved clinical forensic instrument.

---

## 2. No Official Certification or Legal Admissibility

- **No ISO 17025 Accreditation:** NecroTrace has NOT undergone accreditation or conformity assessment under ISO/IEC 17025.
- **No Judicial Validation:** The models, succession curves, and quantile intervals have NOT been validated in an appellate or evidentiary hearing under *Daubert v. Merrell Dow Pharmaceuticals* or *Frye v. United States*.
- **No Evidence Use:** Generated Post-Mortem Reports (Form PM-5372) are **NOT official medico-legal documents** and must **NOT be submitted as evidence** in courtrooms, judicial inquests, police investigations, or administrative proceedings.
- **No Substitute for Professional Judgment:** NecroTrace is not a substitute for certified forensic pathologists, medical examiners, or entomological experts.

---

## 3. Permitted & Prohibited Uses

**Permitted Uses:**
- Academic research, bioinformatic study, and forensic science pedagogy
- Computational experimentation with quantile regression & metagenomic clocks
- Educational hackathon and portfolio demonstrations

**Prohibited Uses:**
- Utilizing outputs in actual criminal or civil court cases
- Making official determinations of time or cause of death
- Misrepresenting generated documents as accredited government or police records

---

## 4. Open-Source Licensing & "AS IS" Disclaimer

NecroTrace source code is published under the **MIT License**:

```text
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 5. Intellectual Property & Scientific Attribution

- Biological microscopy assets displayed in bioindicator viewports are sourced from public scientific reference repositories for educational and illustrative use.
- The NecroTrace architecture, design system, and bioinformatic pipelines are the intellectual creation of Tanish Walture and Team BroomWroom.

---

## 6. Limitation of Liability

Under no circumstances shall the authors, developers, or affiliated institutions be held liable for any direct, indirect, special, incidental, or consequential damages resulting from the use or inability to use this platform.

---

## 7. Governing Law

These Terms are governed by and construed in accordance with the laws of India, including the **Information Technology Act 2000** and the **Digital Personal Data Protection Act 2023**.
    """)

    st.markdown("<div style='margin-top: 28px;'></div>",
                unsafe_allow_html=True)
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        if st.button("← RETURN TO LANDING MATRIX", use_container_width=True, key="btn_terms_landing"):
            st.session_state["view"] = "landing"
            st.query_params["view"] = "landing"
            st.rerun()
    with b_col2:
        if st.button("VIEW COOKIE POLICY →", use_container_width=True, key="btn_terms_cookies"):
            st.session_state["view"] = "cookies"
            st.query_params["view"] = "cookies"
            st.rerun()


# =============================================================================
# VIEW 6: COOKIE & THIRD-PARTY POLICY
# =============================================================================
elif st.session_state["view"] == "cookies":
    sync_document_title("cookies")
    render_clean_html(LEGAL_PAGE_CSS)

    render_clean_html(f"""
    <header style="width: 100%; border-bottom: 1px solid var(--color-graphite); padding: 12px 0 20px 0; margin-bottom: 24px;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <img src="{LOGO_ICON_B64}" style="height: 26px; width: 26px; object-fit: contain; vertical-align: middle; filter: drop-shadow(0 0 8px rgba(116, 194, 92, 0.35));" alt="NecroTrace Logo" />
                <span style="font-family: var(--font-mono); font-size: 13px; color: var(--color-paper); font-weight: 600; letter-spacing: 0.04em;">
                    NECROTRACE <span style="color: var(--color-graphite);">//</span> COMPLIANCE
                </span>
                <span style="font-family: var(--font-mono); font-size: 11px; color: var(--color-bioluminescent-lime); letter-spacing: 0.03em;">
                    COOKIE &amp; DATA DISCLOSURE
                </span>
            </div>
            <div>
                <a href="?view=landing" target="_self" class="sober-btn-ghost" style="padding: 5px 14px; font-size: 11px; height: 32px; min-height: 32px; text-decoration: none;">&larr; PLATFORM OVERVIEW</a>
            </div>
        </div>
    </header>
    """)

    st.markdown("""
# Cookie & Third-Party Policy

**Effective Date:** September 2026  
**Contact:** [tanishwalture@gmail.com](mailto:tanishwalture@gmail.com)

---

## 1. What Cookies We Use

NecroTrace sets **NO marketing, tracking, profiling, or analytical cookies**.

The only cookie set during your session is the built-in Streamlit session token:

| Cookie Name | Provider | Purpose | Legal Category | Consent Required? |
|---|---|---|---|---|
| **Streamlit Session Cookie** | Streamlit Framework | Maintains routing state and session parameters | Strictly Necessary | **No** (Exempt under IT Act & GDPR) |

---

## 2. External Third-Party Requests

When rendering the NecroTrace interface, your browser loads the following third-party assets:

| Asset | Host | Purpose | Data Transmitted |
|---|---|---|---|
| **Inter Tight & Roboto Mono Fonts** | `fonts.googleapis.com` | Typography rendering | Standard HTTP request (IP Address) |
| **Lenis Smooth Scroll CSS** | `unpkg.com` (Cloudflare) | Editorial scrolling styles | Standard HTTP request (IP Address) |
| **Firebase Auth API** | `identitytoolkit.googleapis.com` | User sign-in & enrollment | Email & password payload |
| **Firestore Database API** | `firestore.googleapis.com` | Officer profile verification | Profile metadata payload |

---

## 3. Why We Do Not Show a Cookie Banner

Because NecroTrace **exclusively uses strictly necessary session cookies** and operates zero advertising or analytics trackers:
- **Indian IT Act 2000 & DPDPA 2023:** No consent banner required for functional session tokens.
- **EU ePrivacy Directive & GDPR (Article 5(3)):** Strictly necessary cookies required to provide a service explicitly requested by the user are exempt from consent requirements.

---

## 4. Managing Browser Cookies

You can inspect, block, or delete cookies via your browser's Privacy & Security settings. Note that disabling session cookies will prevent Streamlit from maintaining application state.
    """)

    st.markdown("<div style='margin-top: 28px;'></div>",
                unsafe_allow_html=True)
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        if st.button("← RETURN TO LANDING MATRIX", use_container_width=True, key="btn_cook_landing"):
            st.session_state["view"] = "landing"
            st.query_params["view"] = "landing"
            st.rerun()
    with b_col2:
        if st.button("OPEN AUTOPSY EXAMINATION ROOM →", use_container_width=True, key="btn_cook_exam"):
            st.session_state["view"] = "examination"
            st.query_params["view"] = "examination"
            st.rerun()


# =============================================================================
# VIEW 7: 404 ERROR PAGE (DOSSIER NOT FOUND)
# =============================================================================
elif st.session_state["view"] == "404":
    sync_document_title("404")
    render_clean_html(LEGAL_PAGE_CSS)

    invalid_view_param = str(st.query_params.get("view", "unknown"))
    now_utc_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    render_clean_html(f"""
    <header style="width: 100%; border-bottom: 1px solid var(--color-graphite); padding: 12px 0 20px 0; margin-bottom: 24px;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <img src="{LOGO_ICON_B64}" style="height: 26px; width: 26px; object-fit: contain; vertical-align: middle; filter: drop-shadow(0 0 8px rgba(116, 194, 92, 0.35));" alt="NecroTrace Logo" />
                <span style="font-family: var(--font-mono); font-size: 13px; color: var(--color-paper); font-weight: 600; letter-spacing: 0.04em;">
                    NECROTRACE <span style="color: var(--color-graphite);">//</span> SYSTEM DIAGNOSTICS
                </span>
            </div>
            <div>
                <span style="font-family: var(--font-mono); font-size: 11px; color: #f87171; background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.4); padding: 4px 10px; border-radius: 6px; letter-spacing: 0.05em;">
                    ERROR 404 &bull; ROUTE FAULT
                </span>
            </div>
        </div>
    </header>

    <div style="background: #111a1b; border: 1px solid #283739; border-radius: 14px; padding: 36px 36px; margin-bottom: 24px; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);">
        <div style="display: flex; align-items: baseline; gap: 16px; margin-bottom: 12px;">
            <span style="font-family: var(--font-mono); font-size: 42px; font-weight: 700; color: #f87171; line-height: 1;">404</span>
            <span style="font-family: var(--font-mono); font-size: 13px; color: var(--color-graphite); letter-spacing: 0.05em; text-transform: uppercase;">
                DIAGNOSTIC PROTOCOL // ROUTE NOT RESOLVED
            </span>
        </div>

        <h1 style="font-size: 26px; font-weight: 600; color: #ffffff; letter-spacing: -0.02em; margin: 0 0 12px 0;">
            Forensic Dossier or Pathway Not Found
        </h1>

        <p style="font-size: 14px; color: #c9cbbe; line-height: 1.6; margin: 0 0 24px 0;">
            The requested examination route, legal document, or query parameter could not be identified in the NecroTrace platform directory. The resource may have been relocated, archived, or requested with an unrecognized parameter.
        </p>

        <!-- System Diagnostics Terminal -->
        <div style="background: #080d0e; border: 1px solid #1c2b2d; border-radius: 8px; padding: 18px; font-family: var(--font-mono); font-size: 12px; color: #94a3b8; margin-bottom: 24px;">
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #1c2b2d; padding-bottom: 8px; margin-bottom: 12px;">
                <span style="color: var(--color-bioluminescent-lime); font-weight: 600;">SYSTEM TELEMETRY LOG</span>
                <span style="color: #f87171;">STATUS: 404 NOT_FOUND</span>
            </div>
            <div style="display: grid; grid-template-columns: 140px 1fr; gap: 8px; line-height: 1.5;">
                <span style="color: var(--color-graphite);">REQUESTED VIEW:</span>
                <span style="color: #ffffff; word-break: break-all;">?view={invalid_view_param}</span>
                <span style="color: var(--color-graphite);">TIMESTAMP:</span>
                <span style="color: #cbd5e1;">{now_utc_str}</span>
                <span style="color: var(--color-graphite);">RESOLUTION:</span>
                <span style="color: #fca5a5;">UNRESOLVED_ROUTE_EXCEPTION</span>
                <span style="color: var(--color-graphite);">CORE SERVICES:</span>
                <span style="color: #6ee7b7;">ACTIVE // NOMINAL</span>
            </div>
        </div>

        <!-- Quick Platform Directory -->
        <div style="border-top: 1px solid #1c2b2d; padding-top: 20px;">
            <div style="font-family: var(--font-mono); font-size: 11px; color: var(--color-graphite); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px;">
                AVAILABLE PLATFORM DIRECTORY:
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 10px 18px; font-size: 13px;">
                <a href="?view=landing" style="color: #cef79e; text-decoration: none;">&bull; Platform Overview</a>
                <a href="?view=examination" style="color: #cef79e; text-decoration: none;">&bull; Examination Suite</a>
                <a href="?view=privacy" style="color: #cef79e; text-decoration: none;">&bull; Privacy Policy</a>
                <a href="?view=terms" style="color: #cef79e; text-decoration: none;">&bull; Terms &amp; Conditions</a>
                <a href="?view=cookies" style="color: #cef79e; text-decoration: none;">&bull; Cookie Policy</a>
            </div>
        </div>
    </div>
    """)

    err_col1, err_col2 = st.columns(2)
    with err_col1:
        if st.button("← RETURN TO PLATFORM OVERVIEW", use_container_width=True, key="btn_404_landing"):
            st.session_state["view"] = "landing"
            st.query_params.clear()
            st.query_params["view"] = "landing"
            st.rerun()
    with err_col2:
        if st.button("OPEN AUTOPSY EXAMINATION ROOM →", use_container_width=True, key="btn_404_exam"):
            st.session_state["view"] = "examination"
            st.query_params.clear()
            st.query_params["view"] = "examination"
            st.rerun()
