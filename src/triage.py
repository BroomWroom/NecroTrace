#!/usr/bin/env python3
"""
NecroTrace - Medico-Legal Diagnostic Triage Engine.
Maps clinical postmortem morphological signs (rigor, bloat, marbling, purge, discoloration)
to microbial ecological succession stages, auto-suggests forensic bioindicators, and
synthesizes compositional abundance profiles for Quantile XGBoost inference.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 1. BIOLOGICAL KNOWLEDGE BASE: TAXA SUCCESSION ARCHETYPES
# -----------------------------------------------------------------------------
FORENSIC_BIOINDICATOR_CATALOG = {
    # Early Postmortem Flora (Fresh / Algor Mortis / Established Rigor: Days 0 - 3)
    "Staphylococcus_epidermidis": {
        "common_name": "Staphylococcus epidermidis",
        "stage": "Fresh / Early Rigor",
        "role": "Cutaneous & mucosal aerobe; normal commensal that dominates immediately postmortem before tissue anoxia.",
        "typical_weight": 0.22,
        "indicator_group": "early",
    },
    "Streptococcus_oralis": {
        "common_name": "Streptococcus oralis",
        "stage": "Fresh / Early Rigor",
        "role": "Oral mucosal pioneer commensal; abundant in early swabs prior to enteric migration.",
        "typical_weight": 0.18,
        "indicator_group": "early",
    },
    "Cutibacterium_acnes": {
        "common_name": "Cutibacterium acnes",
        "stage": "Fresh / Early Rigor",
        "role": "Sebaceous skin commensal; rapidly senesces as decomposition fluid washes over tissues.",
        "typical_weight": 0.15,
        "indicator_group": "early",
    },
    "Corynebacterium_striatum": {
        "common_name": "Corynebacterium striatum",
        "stage": "Fresh / Early Rigor",
        "role": "Aerobic skin surface colonizer; marker of un-decomposed epidermal barrier.",
        "typical_weight": 0.12,
        "indicator_group": "early",
    },
    "Rothia_dentocariosa": {
        "common_name": "Rothia dentocariosa",
        "stage": "Fresh / Early Rigor",
        "role": "Oropharyngeal commensal; declines steeply within 48-72 hours postmortem.",
        "typical_weight": 0.08,
        "indicator_group": "early",
    },
    "Micrococcus_luteus": {
        "common_name": "Micrococcus luteus",
        "stage": "Fresh / Early Rigor",
        "role": "Obligate aerobe; highly sensitive to postmortem tissue hypoxia.",
        "typical_weight": 0.06,
        "indicator_group": "early",
    },

    # Active Putrefaction Flora (Bloat / Marbling / Purge / Autolysis: Days 3 - 12)
    "Clostridium_perfringens": {
        "common_name": "Clostridium perfringens",
        "stage": "Bloat / Active Putrefaction",
        "role": "Potent anaerobic gas-producer (alpha-toxin); driver of postmortem abdominal distension, subcutaneous emphysema, and tissue liquefaction.",
        "typical_weight": 0.28,
        "indicator_group": "mid",
    },
    "Bacteroides_fragilis": {
        "common_name": "Bacteroides fragilis",
        "stage": "Bloat / Active Putrefaction",
        "role": "Predominant obligate anaerobe; translocates from lower intestine across mucosal barriers during active bloat.",
        "typical_weight": 0.22,
        "indicator_group": "mid",
    },
    "Proteus_mirabilis": {
        "common_name": "Proteus mirabilis",
        "stage": "Bloat / Active Putrefaction",
        "role": "Swarming proteolytic gammaproteobacterium; causes venous marbling and rapid protein liquefaction.",
        "typical_weight": 0.14,
        "indicator_group": "mid",
    },
    "Morganella_morganii": {
        "common_name": "Morganella morganii",
        "stage": "Bloat / Active Putrefaction",
        "role": "Opportunistic enteric organism; associated with bloody purge fluid and putrefactive odor.",
        "typical_weight": 0.12,
        "indicator_group": "mid",
    },
    "Enterococcus_faecalis": {
        "common_name": "Enterococcus faecalis",
        "stage": "Bloat / Active Putrefaction",
        "role": "Facultative anaerobe; proliferates rapidly throughout decomposing abdominal organs and blood vessels.",
        "typical_weight": 0.10,
        "indicator_group": "mid",
    },
    "Wohlfahrtiimonas_chitiniclastica": {
        "common_name": "Wohlfahrtiimonas chitiniclastica",
        "stage": "Bloat / Active Putrefaction",
        "role": "Symbiont of blowfly (Calliphoridae) larvae; definitive biological marker of insect colonization during active decay.",
        "typical_weight": 0.08,
        "indicator_group": "mid",
    },
    "Ignatzschineria_larvae": {
        "common_name": "Ignatzschineria larvae",
        "stage": "Bloat / Active Putrefaction",
        "role": "Fly-larva associated necrobiome biomarker; spikes during peak maggot feeding activity.",
        "typical_weight": 0.08,
        "indicator_group": "mid",
    },

    # Advanced Decay Flora (Tissue Rupture / Black Putrefaction / Skeletonization: Days > 12)
    "Acinetobacter_baumannii": {
        "common_name": "Acinetobacter baumannii",
        "stage": "Advanced Decay / Skeletonization",
        "role": "Hardy environmental saprophyte; withstands tissue desiccation and outcompetes enteric flora in advanced decay.",
        "typical_weight": 0.25,
        "indicator_group": "late",
    },
    "Pseudomonas_fluorescens": {
        "common_name": "Pseudomonas fluorescens",
        "stage": "Advanced Decay / Skeletonization",
        "role": "Soil and moisture saprophyte; colonizes post-rupture remains and skeletonizing bone surfaces.",
        "typical_weight": 0.20,
        "indicator_group": "late",
    },
    "Bacillus_subtilis": {
        "common_name": "Bacillus subtilis",
        "stage": "Advanced Decay / Skeletonization",
        "role": "Endospore-forming soil decomposer; hydrolyzes complex fibrous and cartilaginous proteins.",
        "typical_weight": 0.18,
        "indicator_group": "late",
    },
    "Sphingobacterium_multivorum": {
        "common_name": "Sphingobacterium multivorum",
        "stage": "Advanced Decay / Skeletonization",
        "role": "Proteolytic soil bacterium; marker of soil-body interface and late-stage skeletal breakdown.",
        "typical_weight": 0.14,
        "indicator_group": "late",
    },
    "Streptomyces_albus": {
        "common_name": "Streptomyces albus",
        "stage": "Advanced Decay / Skeletonization",
        "role": "Actinomycete; proliferates on desiccated, mummified, or dry skeletal remains.",
        "typical_weight": 0.12,
        "indicator_group": "late",
    },
    "Planococcus_halocryophilus": {
        "common_name": "Planococcus halocryophilus",
        "stage": "Advanced Decay / Skeletonization",
        "role": "Extremotolerant bacterium capable of surviving cold soil temperatures during late postmortem decay.",
        "typical_weight": 0.08,
        "indicator_group": "late",
    },
}


# -----------------------------------------------------------------------------
# 2. MORPHOLOGICAL TRIAGE EVALUATION
# -----------------------------------------------------------------------------
def evaluate_morphological_triage(signs: Dict[str, str]) -> Dict[str, Any]:
    """
    Evaluates clinical autopsy signs entered by the medical examiner:
    - Rigor mortis state
    - Abdominal distension / bloat
    - Skin discoloration & vascular marbling
    - Purge fluid & natural orifices
    - Entomology / maggot mass

    Returns a diagnostic synthesis:
    - Likely decomposition phase
    - Coarse clinical time window estimate
    - Auto-suggested microbial bioindicators with rationale
    """
    rigor = signs.get("rigor_mortis", "Completely Absent (Passed off)")
    bloat = signs.get("bloat", "None / Flat abdomen")
    discoloration = signs.get("discoloration", "Normal / Postmortem Pallor")
    purge = signs.get("purge_fluid", "Absent")
    maggots = signs.get("maggot_activity", "None detected")

    score_early = 0
    score_mid = 0
    score_late = 0

    # 1. Rigor Mortis Evaluation
    if "Early" in rigor or "Developing" in rigor:
        score_early += 3
    elif "Fully Established" in rigor:
        score_early += 4
    elif "Passing Off" in rigor:
        score_early += 2
        score_mid += 2
    else:  # Completely absent / passed off
        score_mid += 2
        score_late += 2

    # 2. Bloat Evaluation
    if "None" in bloat or "Flat" in bloat:
        score_early += 3
    elif "Initial / Mild" in bloat:
        score_early += 1
        score_mid += 3
    elif "Moderate" in bloat:
        score_mid += 5
    elif "Severe" in bloat:
        score_mid += 4
        score_late += 2
    elif "Ruptured" in bloat or "Subsiding" in bloat:
        score_late += 5

    # 3. Discoloration & Marbling
    if "Normal" in discoloration or "Pallor" in discoloration:
        score_early += 3
    elif "Right Iliac Fossa" in discoloration:
        score_early += 1
        score_mid += 2
    elif "Marbling" in discoloration:
        score_mid += 4
    elif "Dusky Green" in discoloration:
        score_mid += 4
        score_late += 1
    elif "Black Putrefaction" in discoloration:
        score_late += 5

    # 4. Purge Fluid
    if "Blood-stained" in purge or "Frothy" in purge:
        score_mid += 3
    elif "Subsiding / Dry" in purge:
        score_late += 3

    # 5. Entomology
    if "Active larval mass" in maggots:
        score_mid += 3
        score_late += 2
    elif "Pupae" in maggots or "Empty puparia" in maggots:
        score_late += 4

    # Determine predominant phase
    if score_early >= score_mid and score_early >= score_late:
        primary_phase = "Fresh / Early Postmortem"
        coarse_clinical_range = "0.5 to 3.0 Days (12 to 72 Hours)"
        target_group = "early"
        biological_summary = (
            "Morphological signs (active/persisting rigor, minimal bloat, absence of advanced marbling) "
            "indicate an early postmortem period. Aerobic skin and mucosal commensals are preserved, with "
            "minimal enteric translocation."
        )
    elif score_mid >= score_late:
        primary_phase = "Bloat / Active Putrefaction"
        coarse_clinical_range = "3.0 to 12.0 Days"
        target_group = "mid"
        biological_summary = (
            "Morphological signs (cessation of rigor, prominent abdominal bloat, venous marbling, and purge fluid) "
            "corroborate active microbial putrefaction. Anaerobic enteric bacteria have proliferated, producing "
            "characteristic putrefactive gases and soft tissue autolysis."
        )
    else:
        primary_phase = "Advanced Decay & Skeletonization"
        coarse_clinical_range = "12.0 to 30.0+ Days"
        target_group = "late"
        biological_summary = (
            "Morphological signs (abdominal rupture, extensive dark tissue liquefaction, bone exposure, "
            "or advanced maggot pupation) reflect late postmortem succession. Normal microflora have collapsed, "
            "and resilient soil saprophytes predominate."
        )

    # Filter and rank candidate bioindicators
    suggested_bioindicators = []
    for taxon_id, info in FORENSIC_BIOINDICATOR_CATALOG.items():
        # High relevance if matching the primary phase
        is_primary = (info["indicator_group"] == target_group)
        suggested_bioindicators.append({
            "taxon_id": taxon_id,
            "common_name": info["common_name"],
            "stage": info["stage"],
            "role": info["role"],
            "is_recommended": is_primary,
            "typical_weight": info["typical_weight"],
        })

    # Sort recommended to top
    suggested_bioindicators.sort(key=lambda x: (not x["is_recommended"], -x["typical_weight"]))

    return {
        "primary_phase": primary_phase,
        "coarse_clinical_range": coarse_clinical_range,
        "target_group": target_group,
        "biological_summary": biological_summary,
        "suggested_bioindicators": suggested_bioindicators,
    }


# -----------------------------------------------------------------------------
# 3. COMPOSITIONAL ABUNDANCE PROFILE SYNTHESIS
# -----------------------------------------------------------------------------
def synthesize_abundance_profile(
    confirmed_taxa: List[str],
    feature_schema: List[str],
    ambient_temp_c: float = 23.5,
    humidity_pct: float = 65.0,
    target_depth: int = 28500
) -> pd.DataFrame:
    """
    Constructs a model-ready feature vector for the pre-trained Quantile XGBoost models.
    
    1. Distributes read counts according to the confirmed forensic bioindicators.
    2. Adds realistic biological background commensal counts to unconfirmed taxa.
    3. Computes Centered Log-Ratio (CLR) transformation with +1 pseudocount.
    4. Calculates Shannon Diversity Index (H').
    5. Incorporates environmental covariates (ambient_temp_c, humidity_pct).
    6. Aligns strictly to feature_schema.
    """
    # Extract microbial taxa names in schema (excluding environmental columns)
    taxa_in_schema = [
        f for f in feature_schema
        if f not in ["alpha_shannon_entropy", "ambient_temp_c", "humidity_pct"]
    ]

    # Baseline synthetic count vector
    raw_counts = np.full(len(taxa_in_schema), 20.0, dtype=float)

    # Elevate confirmed bioindicators based on their biological weighting
    for i, taxon in enumerate(taxa_in_schema):
        if taxon in confirmed_taxa:
            info = FORENSIC_BIOINDICATOR_CATALOG.get(taxon, {})
            weight = info.get("typical_weight", 0.15)
            # Confirmed bioindicators receive substantial read allocation
            raw_counts[i] += weight * target_depth * np.random.uniform(0.9, 1.1)
        else:
            # Low background noise
            raw_counts[i] += np.random.uniform(5.0, 45.0)

    # Normalize to total sequencing depth
    raw_counts = (raw_counts / raw_counts.sum()) * target_depth
    counts_df = pd.DataFrame([raw_counts], columns=taxa_in_schema, index=["SPECIMEN_EVAL"])

    # 1. Centered Log-Ratio (CLR)
    counts_offset = counts_df.astype(float) + 1.0
    log_counts = np.log(counts_offset)
    geometric_mean_log = log_counts.mean(axis=1)
    clr_df = log_counts.sub(geometric_mean_log, axis=0)

    # 2. Shannon Entropy (H')
    row_sums = counts_df.sum(axis=1).replace(0, 1.0)
    props = counts_df.div(row_sums, axis=0).values[0]
    p_nz = props[props > 0]
    shannon_h = -np.sum(p_nz * np.log(p_nz))

    # 3. Assemble complete feature vector aligned to schema
    features_df = pd.DataFrame(index=["SPECIMEN_EVAL"])
    for col in feature_schema:
        if col == "alpha_shannon_entropy":
            features_df[col] = round(shannon_h, 3)
        elif col == "ambient_temp_c":
            features_df[col] = float(ambient_temp_c)
        elif col == "humidity_pct":
            features_df[col] = float(humidity_pct)
        elif col in clr_df.columns:
            features_df[col] = clr_df[col].values[0]
        else:
            features_df[col] = 0.0

    return features_df
