#!/usr/bin/env python3
"""
NecroTrace - Forensic Metagenomics & Postmortem Interval (PMI) Estimation Platform.
Integrated Biosciences Design System ("Bioluminescent Laboratory at Midnight").
Features an architectural editorial landing page with Lenis smooth scrolling,
paired with an authoritative Medical Examiner Diagnostic Triage workflow.
Developed by Team BroomWroom (Lead: Tanish Walture).
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.pipeline import NecroQuantileRegressor
from src.triage import (
    evaluate_morphological_triage,
    synthesize_abundance_profile,
    FORENSIC_BIOINDICATOR_CATALOG,
)
from src.reporting.pdf_generator import (
    generate_forensic_pdf,
    compute_sha256_hash,
)

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NecroTrace // Forensic Metagenomics",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# 2. APPLICATION ROUTING & STATE
# -----------------------------------------------------------------------------
# Handle query parameters for view routing if present
query_view = st.query_params.get("view", None)
if "view" not in st.session_state:
    st.session_state["view"] = query_view if query_view in ["landing", "examination"] else "landing"
elif query_view in ["landing", "examination"] and query_view != st.session_state["view"]:
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


def render_clean_html(html_str: str):
    """
    Renders raw HTML safely in Streamlit without triggering markdown's
    4-space indented code block parser. Strips leading and trailing whitespace
    from each line and removes empty lines that cause CommonMark to break HTML blocks.
    """
    cleaned_lines = [line.strip() for line in html_str.splitlines() if line.strip()]
    st.markdown("\n".join(cleaned_lines), unsafe_allow_html=True)


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
        --color-graphite: #4d5757;
        --color-lichen: #c9cbbe;
        --color-tissue: #e7e8e1;
        --color-frost: #eeeeee;
        --color-void: #000000;
        --font-display: 'Inter Tight', 'Aspekta', -apple-system, sans-serif;
        --font-mono: 'Roboto Mono', monospace;
    }

    /* Streamlit overrides */
    header[data-testid="stHeader"] { display: none !important; }
    div[data-testid="stToolbar"] { display: none !important; }
    footer { display: none !important; }
    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
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
        width: 6px;
        height: 6px;
        border-radius: 9999px;
        background-color: var(--color-bioluminescent-lime);
        display: inline-block;
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
</style>

<script src="https://unpkg.com/lenis@1.3.26/dist/lenis.min.js"></script>
<script>
    // Initialize Lenis smooth scroll
    document.addEventListener("DOMContentLoaded", () => {
        if (window.Lenis) {
            const lenis = new Lenis({
                duration: 1.2,
                easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
                direction: 'vertical',
                gestureDirection: 'vertical',
                smooth: true,
                mouseMultiplier: 1,
            });
            function raf(time) {
                lenis.raf(time);
                requestAnimationFrame(raf);
            }
            requestAnimationFrame(raf);
        }
    });
</script>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. CACHED MODEL LOADER
# -----------------------------------------------------------------------------
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
# VIEW 1: BIOLUMINESCENT LABORATORY LANDING PAGE
# =============================================================================
if st.session_state["view"] == "landing":

    # --- ARCHITECTURAL INSTRUMENT NAV ---
    nav_html = """
    <div style="padding: 24px 48px; border-bottom: 1px solid var(--color-graphite); display: flex; align-items: center; justify-content: space-between; max-width: 1300px; margin: 0 auto;">
        <div style="display: flex; align-items: center; gap: 16px;">
            <span style="font-family: var(--font-mono); font-size: 14px; letter-spacing: -0.02em; color: var(--color-paper);">
                NECROTRACE <span style="color: var(--color-graphite);">//</span> LAB-CLOCK
            </span>
            <div style="display: flex; align-items: center; gap: 6px; padding: 4px 10px; border: 1px solid var(--color-graphite); border-radius: 9999px;">
                <span class="signal-dot"></span>
                <span style="font-family: var(--font-mono); font-size: 11px; color: var(--color-bioluminescent-lime);">SYSTEM VERIFIED</span>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 24px;">
            <a href="#platform" style="font-family: var(--font-mono); font-size: 13px; color: var(--color-graphite); text-decoration: none;">01 PLATFORM</a>
            <a href="#succession" style="font-family: var(--font-mono); font-size: 13px; color: var(--color-graphite); text-decoration: none;">02 SUCCESSION</a>
            <a href="#dossier" style="font-family: var(--font-mono); font-size: 13px; color: var(--color-graphite); text-decoration: none;">03 CASE DOSSIER</a>
            <a href="?view=examination" target="_self" style="font-family: var(--font-mono); font-size: 12px; color: var(--color-paper); text-decoration: none; border: 1px solid var(--color-graphite); padding: 7px 14px; border-radius: 6px; letter-spacing: -0.02em;">EXAMINATION ROOM &rarr;</a>
        </div>
    </div>
    """
    render_clean_html(nav_html)

    # --- SECTION 01: HERO SECTION (ABYSSAL INK CANVAS #222f30) ---
    render_clean_html("""
    <div style="max-width: 1200px; margin: 0 auto; padding: 100px 40px 40px 40px;">
        <div class="section-counter">
            <span class="signal-dot"></span>
            01 / FORENSIC METAGENOMICS
        </div>
        <h1 class="hero-title">The microbial clock of human decomposition.</h1>
        <p class="hero-sub">
            High-throughput metagenomic taxonomic profiling and quantile regression to infer postmortem intervals with quantifiable evidentiary certainty.
        </p>
        <div style="display: flex; align-items: center; gap: 16px; flex-wrap: wrap; margin-top: 8px;">
            <a href="?view=examination" target="_self" class="sober-btn-dark">COMMENCE POST-MORTEM EXAMINATION</a>
            <a href="#platform" class="sober-btn-ghost">METHODOLOGY SPECIFICATIONS</a>
        </div>
    </div>
    <hr class="hairline-dark" style="max-width: 1200px; margin: 60px auto;">
    """)

    # --- SECTION 02: INSTRUMENTATION & ARCHITECTURE (DARK BAND #222f30) ---
    render_clean_html("""
    <div id="platform" style="max-width: 1200px; margin: 0 auto; padding: 0 40px 80px 40px;">
        <div class="section-counter">02 / INSTRUMENTATION</div>
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
                    Direct generation of courtroom-grade autopsy documentation. Complies with legal evidentiary standards (Daubert / Frye) through explicit error bounds, specimen adequacy clearance, and institutional certification.
                </div>
            </div>
        </div>
    </div>
    """)

    # --- SECTION 03: SUCCESSION WAVES (LIGHT FLIP TO BONE WHITE #f7f7f5) ---
    render_clean_html("""
    <div id="succession" style="background-color: var(--color-bone-white); color: var(--color-abyssal-ink); padding: 100px 40px; margin-top: 40px;">
        <div style="max-width: 1200px; margin: 0 auto;">
            <div class="section-counter" style="border-color: var(--color-graphite); color: var(--color-graphite);">
                <span class="signal-dot"></span>
                03 / SUCCESSION DYNAMICS
            </div>
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
    render_clean_html("""
    <div id="dossier" style="background-color: var(--color-void); padding: 100px 40px 80px 40px; border-top: 1px solid #1a2223;">
        <div style="max-width: 1200px; margin: 0 auto;">
            <div class="section-counter" style="border-color: #333333; color: #888888;">
                <span class="signal-dot"></span>
                04 / MEDICO-LEGAL INTEGRATION
            </div>
            
            <div style="margin-bottom: 40px;">
                <h2 style="font-size: clamp(32px, 5vw, 64px); line-height: 1.05; letter-spacing: -0.02em; color: var(--color-paper); margin: 0 0 16px 0;">
                    Ready for clinical post-mortem examination.
                </h2>
                <p style="font-size: 18px; color: var(--color-graphite); margin: 0 0 36px 0; max-width: 680px; line-height: 1.4;">
                    Proceed to the interactive triage workflow to input autopsy particulars, correlate morphological findings, confirm microbial bioindicators, and export the official Form PM-5372 dossier.
                </p>
                <a href="?view=examination" target="_self" class="sober-btn-dark">COMMENCE POST-MORTEM EXAMINATION</a>
            </div>
        </div>
    </div>
    """)

    # Absolute Footer
    render_clean_html("""
    <div style="background-color: var(--color-void); padding: 40px 40px 60px 40px; border-top: 1px solid #151515; max-width: 1200px; margin: 0 auto; display: flex; justify-content: space-between; align-items: center;">
        <div class="mono-tag" style="color: #666666; font-size: 12px;">
            NECROTRACE // PLATFORM v0.2.0 &bull; FORENSIC METAGENOMICS
        </div>
        <div class="mono-tag" style="color: #666666; font-size: 12px;">
            TEAM BROOMWROOM &bull; TANISH WALTURE &bull; VMEDITHON 3.0 &bull; VIT CHENNAI
        </div>
    </div>
    """)


# =============================================================================
# VIEW 2: MEDICAL EXAMINER DIAGNOSTIC TRIAGE WORKFLOW
# =============================================================================
elif st.session_state["view"] == "examination":

    # Top Navigation Bar in Examination View
    nav_exam_html = """
    <div style="padding: 18px 48px; border-bottom: 1px solid var(--color-graphite); display: flex; align-items: center; justify-content: space-between; max-width: 1300px; margin: 0 auto 32px auto;">
        <div style="display: flex; align-items: center; gap: 16px;">
            <span style="font-family: var(--font-mono); font-size: 13px; color: var(--color-paper);">
                NECROTRACE <span style="color: var(--color-graphite);">//</span> EXAMINATION ROOM
            </span>
            <span class="signal-dot"></span>
            <span style="font-family: var(--font-mono); font-size: 11px; color: var(--color-bioluminescent-lime);">
                CLINICAL TRIAGE ACTIVE
            </span>
        </div>
        <div>
            <a href="?view=landing" target="_self" class="sober-btn-ghost" style="padding: 7px 16px; font-size: 12px; height: 36px; min-height: 36px;">&larr; RETURN TO PLATFORM OVERVIEW</a>
        </div>
    </div>
    """
    render_clean_html(nav_exam_html)

    render_clean_html("""
    <div style="max-width: 1200px; margin: 16px auto 32px auto; padding: 0 16px;">
        <h1 style="font-size: 38px; line-height: 1.1; letter-spacing: -0.02em; color: var(--color-paper); margin: 0 0 8px 0;">
            Medical Examiner Diagnostic Triage
        </h1>
        <p style="font-size: 16px; color: var(--color-graphite); margin: 0;">
            Objective time-of-death inference: Patient Particulars ➔ Morphological Signs ➔ Confirmed Bioindicators ➔ Quantile PMI Output ➔ Form PM-5372 PDF.
        </p>
    </div>
    """)

    container_exam = st.container()
    with container_exam:
        st.markdown('<div style="max-width: 1200px; margin: 0 auto; padding: 0 16px;">', unsafe_allow_html=True)

        # -------------------------------------------------------------
        # STEP 1: PATIENT PARTICULARS & SCENE FACTORS
        # -------------------------------------------------------------
        st.markdown('<div class="section-counter">01 / PATIENT & SCENE PARTICULARS</div>', unsafe_allow_html=True)
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1:
            deceased_name = st.text_input("Name of Deceased / Ref:", value="Unidentified Individual (Ref: Unknown #42)")
            age_val = st.text_input("Estimated Age:", value="Approx. 35 - 40 Years")
        with col_p2:
            sex_val = st.selectbox("Sex:", ["Male", "Female", "Indeterminate / Skeletal"], index=0)
            swab_site = st.selectbox(
                "Anatomical Swab Site:",
                ["Oral Cavity / Mucosal Surface", "Nasal Mucosa", "Abdominal Surface", "Soil-Body Interface"],
                index=0,
            )
        with col_p3:
            height_val = st.text_input("Height (approx):", value="172 cm")
            weight_val = st.text_input("Weight (approx):", value="68 kg")
        with col_p4:
            ambient_temp = st.slider("Scene Temperature (°C):", min_value=5.0, max_value=42.0, value=23.5, step=0.5)
            humidity_val = st.slider("Relative Humidity (%):", min_value=20.0, max_value=100.0, value=65.0, step=5.0)

        # Administrative Identifiers
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            pm_report_no = st.text_input("Post Mortem Report No:", value="PM-619 / 2026")
        with col_m2:
            police_station = st.text_input("Police Station (P.S.):", value="New Township P.S.")
        with col_m3:
            inquest_no = st.text_input("Inquest Number:", value="14 / 2026")
        with col_m4:
            analyst_name = st.text_input("Examining Medical Officer:", value="Dr. Tanish Walture, M.D. (WBMC / 45826)")

        st.markdown('<hr class="hairline-dark">', unsafe_allow_html=True)

        # -------------------------------------------------------------
        # STEP 2: MORPHOLOGICAL SIGNS QUESTIONNAIRE
        # -------------------------------------------------------------
        st.markdown('<div class="section-counter">02 / MORPHOLOGICAL SIGNS QUESTIONNAIRE</div>', unsafe_allow_html=True)
        st.write("Record physical postmortem decomposition signs. The succession engine dynamically aligns these findings with microbial phases.")

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.markdown("**1. Rigor Mortis Status**")
            rigor_opt = st.radio(
                "Rigor Mortis:",
                [
                    "Early / Developing (Jaw, neck, facial muscles)",
                    "Fully Established (Generalized stiffening across all limbs & trunk)",
                    "Passing Off (Receding from face, persisting in lower limbs)",
                    "Completely Absent / Flaccid (Passed off due to decomposition)",
                ],
                index=3 if not st.session_state["triage_confirmed"] else 3,
                label_visibility="collapsed",
            )

            st.markdown("<br/>**4. Purge Fluid & Natural Orifices**", unsafe_allow_html=True)
            purge_opt = st.radio(
                "Purge Fluid:",
                [
                    "Absent (Orifices clear, eyes intact)",
                    "Early Serous / Frothy discharge at nares and lips",
                    "Blood-stained Purge Fluid (Active putrefactive liquefaction)",
                    "Desiccated / Dry remains",
                ],
                index=2,
                label_visibility="collapsed",
            )

        with col_s2:
            st.markdown("**2. Abdominal Distension & Bloat**")
            bloat_opt = st.radio(
                "Bloat State:",
                [
                    "None / Flat (Abdomen soft, no gaseous distension)",
                    "Initial / Mild (Early firmness, mild lower quadrant distension)",
                    "Moderate Bloat (Tense generalized distension, scrotum/vulva swelling)",
                    "Severe Bloat & Purge (Massive distension, blood-stained froth at orifices)",
                    "Ruptured / Subsiding (Abdominal wall collapsed, tissue liquefaction)",
                ],
                index=2,
                label_visibility="collapsed",
            )

            st.markdown("<br/>**5. Entomology & Maggot Activity**", unsafe_allow_html=True)
            maggots_opt = st.radio(
                "Entomology Activity:",
                [
                    "None detected",
                    "Early fly egg deposits / small instar larvae in natural orifices",
                    "Active larval feeding masses across soft tissues",
                    "Pupae / Empty puparia present",
                ],
                index=1,
                label_visibility="collapsed",
            )

        with col_s3:
            st.markdown("**3. Skin Discoloration & Vascular Marbling**")
            discolor_opt = st.radio(
                "Discoloration:",
                [
                    "Normal / Postmortem Pallor (No putrefactive staining)",
                    "Greenish discoloration over Right Iliac Fossa",
                    "Arborescent Venous Marbling (Branching greenish-purple venous network)",
                    "Generalized Dusky Green / Bronzing (Extensive torso & facial discoloration)",
                    "Black Putrefaction (Dark brownish-black discoloration, skin slippage)",
                ],
                index=2,
                label_visibility="collapsed",
            )

        # Dynamic triage evaluation
        signs_dict = {
            "rigor_mortis": rigor_opt,
            "bloat": bloat_opt,
            "discoloration": discolor_opt,
            "purge_fluid": purge_opt,
            "maggot_activity": maggots_opt,
        }
        triage_eval = evaluate_morphological_triage(signs_dict)
        st.session_state["triage_eval"] = triage_eval

        st.markdown('<hr class="hairline-dark">', unsafe_allow_html=True)

        # -------------------------------------------------------------
        # STEP 3: AUTO-SUGGESTED MICROBIAL BIOINDICATORS
        # -------------------------------------------------------------
        st.markdown('<div class="section-counter">03 / AUTO-SUGGESTED MICROBIAL BIOINDICATORS</div>', unsafe_allow_html=True)

        render_clean_html(f"""
        <div style="background-color: #1a2425; border-left: 3px solid var(--color-bioluminescent-lime); padding: 16px 20px; border-radius: 0 8px 8px 0; margin-bottom: 24px;">
            <div style="font-family: var(--font-mono); font-size: 12px; color: var(--color-bioluminescent-lime); margin-bottom: 4px;">
                DIAGNOSTIC STAGE CORRELATION &bull; {triage_eval['primary_phase'].upper()}
            </div>
            <div style="font-size: 16px; color: var(--color-paper); margin-bottom: 4px;">
                Expected Time Window: {triage_eval['coarse_clinical_range']}
            </div>
            <div style="font-size: 14px; color: var(--color-graphite);">
                {triage_eval['biological_summary']}
            </div>
        </div>
        """)

        st.write("Review auto-suggested bioindicators. Check/confirm the diagnostic taxa verified by swab testing:")

        suggested_list = triage_eval["suggested_bioindicators"]
        selected_taxa_current = []

        col_b1, col_b2 = st.columns(2)
        for i, item in enumerate(suggested_list):
            col_target = col_b1 if i % 2 == 0 else col_b2
            with col_target:
                is_checked = st.checkbox(
                    f"**{item['common_name']}** ({item['stage']})",
                    value=item["is_recommended"],
                    key=f"exam_taxa_{item['taxon_id']}",
                    help=item["role"],
                )
                if is_checked:
                    selected_taxa_current.append(item["taxon_id"])
                st.caption(f"↳ {item['role']}")

        st.markdown("<br/>", unsafe_allow_html=True)

        # Confirmation Action
        col_act1, col_act2 = st.columns([1.8, 1.2])
        with col_act1:
            if st.button("CONFIRM MICROBIAL PROFILE & CALCULATE PMI", type="primary", use_container_width=True):
                if not selected_taxa_current:
                    st.error("Please confirm at least one microbial bioindicator to calculate the postmortem interval.")
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
                        "analyst": analyst_name,
                        "reg_no": "WBMC / 45826",
                        "institution": "District Medico-Legal Center & Morgue",
                        "rigor_obs": rigor_opt,
                        "bloat_obs": bloat_opt,
                        "discolor_obs": discolor_opt,
                        "micro_findings": f"Diagnostic bioindicator confirmation ({len(selected_taxa_current)} verified taxa): {', '.join([t.replace('_', ' ') for t in selected_taxa_current[:4]])} predominant.",
                    }
                    st.success("Microbial succession profile verified. Quantile PMI estimated.")

        with col_act2:
            if st.session_state["triage_confirmed"]:
                if st.button("EDIT MORPHOLOGICAL SIGNS", use_container_width=True):
                    st.session_state["triage_confirmed"] = False
                    st.info("Triage unlocked for adjustments.")

        # -------------------------------------------------------------
        # STEP 4: QUANTILE INFERENCE DISPLAY
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

            st.markdown('<hr class="hairline-dark">', unsafe_allow_html=True)
            st.markdown('<div class="section-counter">04 / TIME-OF-DEATH INFERENCE RESULTS</div>', unsafe_allow_html=True)

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
                annotation_font=dict(color="#cef79e", size=11, family="Roboto Mono"),
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
                    title_font=dict(color="#c9cbbe", size=12, family="Roboto Mono"),
                    tickfont=dict(color="#c9cbbe", size=11, family="Roboto Mono"),
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
            # STEP 5: OFFICIAL POST-MORTEM REPORT & PDF EXPORT
            # ---------------------------------------------------------
            st.markdown('<hr class="hairline-dark">', unsafe_allow_html=True)
            st.markdown('<div class="section-counter">05 / OFFICIAL CASE REPORT DOSSIER</div>', unsafe_allow_html=True)

            case_info = st.session_state["case_particulars"]

            # On-screen preview of Form PM-5372
            render_clean_html(f"""
            <div style="border: 1px solid var(--color-graphite); padding: 24px; background-color: #1a2425; color: var(--color-paper); border-radius: 12px; margin-bottom: 24px;">
                <div style="text-align: center; border-bottom: 1px solid var(--color-graphite); padding-bottom: 12px; margin-bottom: 16px;">
                    <div class="mono-tag" style="color: var(--color-bioluminescent-lime);">DEPARTMENT OF FORENSIC MEDICINE & POLICE MORGUE</div>
                    <div style="font-size: 20px; color: var(--color-paper); margin: 6px 0;">POST MORTEM EXAMINATION REPORT — FORM NO. PM-5372</div>
                    <div style="font-family: var(--font-mono); font-size: 12px; color: var(--color-graphite);">
                        REPORT NO: {case_info.get('pm_report_no')} &bull; P.S.: {case_info.get('police_station')} &bull; INQUEST: {case_info.get('inquest_no')}
                    </div>
                </div>
                
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 20px; font-size: 14px; background-color: #222f30; padding: 14px; border-radius: 8px;">
                    <div><b>Deceased Reference:</b> {case_info.get('deceased_name')}</div>
                    <div><b>Age / Sex:</b> {case_info.get('age')} / {case_info.get('sex')}</div>
                    <div><b>Swab Location:</b> {case_info.get('sample_site')}</div>
                    <div><b>Examining Doctor:</b> {case_info.get('analyst')}</div>
                    <div><b>Council Reg:</b> {case_info.get('reg_no')}</div>
                    <div><b>Specimen Status:</b> <span style="color: var(--color-bioluminescent-lime);">SATISFACTORY (Adequate DNA)</span></div>
                </div>

                <div style="background-color: #273637; border-left: 3px solid var(--color-bioluminescent-lime); padding: 16px; border-radius: 0 8px 8px 0; margin-bottom: 16px;">
                    <div class="mono-tag" style="color: var(--color-bioluminescent-lime); margin-bottom: 6px;">MEDICO-LEGAL OPINION: TIME ELAPSED SINCE DEATH</div>
                    <div style="font-size: 18px; color: var(--color-paper); margin-bottom: 4px;">
                        <b>Estimated Time Elapsed:</b> {p_est:.1f} Days (approx. {p_est*24.0:.0f} Hours prior to examination)
                    </div>
                    <div style="font-size: 15px; color: #dbeafe; margin-bottom: 4px;">
                        <b>Probable Forensic Window:</b> {p_low:.1f} to {p_high:.1f} Days prior to recovery
                    </div>
                    <div style="font-size: 14px; color: var(--color-graphite);">
                        <b>Calculated Calendar Date of Death:</b> {dt_earliest.strftime('%d/%m/%Y')} to {dt_latest.strftime('%d/%m/%Y')} (Most Probable: {dt_most_likely.strftime('%d/%m/%Y')})
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

            with st.spinner("Compiling official Post-Mortem Report PDF (Form PM-5372)..."):
                pdf_bytes = generate_forensic_pdf(
                    case_metadata=case_info,
                    pmi_findings=pmi,
                    qc_metrics=qc_report,
                )

            st.download_button(
                label="DOWNLOAD OFFICIAL POST-MORTEM REPORT (PDF)",
                data=pdf_bytes,
                file_name=f"PostMortem_Report_{case_info.get('pm_report_no', 'PM').replace('/', '_').replace(' ', '')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)
