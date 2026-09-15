#!/usr/bin/env python3
"""
NecroTrace - Synthetic Postmortem Metagenomic Data Generator.
Simulates realistic 16S rRNA taxonomic abundance profiles and environmental
metadata across decomposition stages (0 to 30 days PMI).

Based on empirical necrobiome succession dynamics (Metcalf et al., Belk et al.).
Generates 1,000 samples with negative-binomial overdispersed read counts.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def generate_necrobiome_dataset(
    n_samples: int = 1000,
    random_seed: int = 42,
    output_dir: str = "data"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic forensic metagenomic count matrices and metadata.

    Args:
        n_samples: Number of postmortem samples to simulate (default: 1000).
        random_seed: Seed for reproducibility.
        output_dir: Directory to save generated CSV files.

    Returns:
        tuple of (counts_df, metadata_df)
    """
    np.random.seed(random_seed)

    # -------------------------------------------------------------
    # 1. Metadata Generation
    # -------------------------------------------------------------
    sample_ids = [f"SAMP_{i+1:04d}" for i in range(n_samples)]

    # True PMI distributed across 0.1 to 30.0 days
    # Skew slightly toward earlier days (0-15) where forensic cases most commonly concentrate
    pmi_days = np.random.exponential(scale=10.0, size=n_samples)
    pmi_days = np.clip(pmi_days, 0.1, 30.0)

    # Ambient temperature (Mean 22°C, std 4°C, truncated 12°C - 32°C)
    ambient_temp = np.random.normal(loc=22.0, scale=4.0, size=n_samples)
    ambient_temp = np.clip(ambient_temp, 12.0, 32.0).round(1)

    # Relative humidity (40% - 90%)
    humidity = np.random.uniform(low=40.0, high=90.0, size=n_samples).round(1)

    # Anatomical sampling site
    sites = np.random.choice(
        ["oral_cavity", "skin_surface", "soil_interface"],
        size=n_samples,
        p=[0.40, 0.40, 0.20]
    )

    # Accumulated Degree Days (ADD = PMI * Temp)
    add_metric = (pmi_days * ambient_temp).round(2)

    # Categorical decomposition stage based on ADD
    def classify_stage(add_val: float) -> str:
        if add_val < 35:
            return "Fresh"
        elif add_val < 150:
            return "Bloat"
        elif add_val < 350:
            return "Active Decay"
        elif add_val < 600:
            return "Advanced Decay"
        else:
            return "Dry Remains"

    decay_stages = [classify_stage(add) for add in add_metric]

    metadata_df = pd.DataFrame({
        "sample_id": sample_ids,
        "true_pmi_days": pmi_days.round(2),
        "ambient_temp_c": ambient_temp,
        "humidity_pct": humidity,
        "accumulated_degree_days": add_metric,
        "sample_site": sites,
        "decomposition_stage": decay_stages
    })

    # -------------------------------------------------------------
    # 2. Forensic Taxa Archetypes & Succession Parameters
    # -------------------------------------------------------------
    # Biological groups with characteristic curves:
    # 1. Early Aerobic Bloomers (Sigmoid decay): High initially, crash by Day 3-6
    # 2. Mid-Decomposition Anaerobes (Gaussian pulse): Peak during bloat & active decay (Days 4-14)
    # 3. Late Soil Recruits / Saprophytes (Sigmoid growth): Low initially, expand in advanced decay (Days 12-30)
    # 4. Ubiquitous Commensals: Present steadily throughout
    # 5. Low-prevalence / Rare Taxa: Very low incidence (<5% prevalence) to test QC filtering

    taxa_definitions = {
        # Group 1: Early Aerobic Bloomers
        "Staphylococcus_epidermidis": ("early", 0.08, 3.0),
        "Streptococcus_oralis": ("early", 0.07, 2.5),
        "Corynebacterium_striatum": ("early", 0.06, 4.0),
        "Cutibacterium_acnes": ("early", 0.05, 3.5),
        "Rothia_dentocariosa": ("early", 0.04, 2.0),
        "Neisseria_sicca": ("early", 0.03, 1.8),
        "Micrococcus_luteus": ("early", 0.03, 3.0),
        "Streptococcus_mitis": ("early", 0.04, 2.2),
        "Haemophilus_parainfluenzae": ("early", 0.02, 1.5),
        "Lactobacillus_gasseri": ("early", 0.02, 2.5),

        # Group 2: Mid-Decomposition Anaerobes & Enteric Bacteria (Bloat / Active Decay)
        "Clostridium_perfringens": ("mid", 0.12, 6.0, 3.0),
        "Clostridium_difficile": ("mid", 0.08, 7.5, 3.5),
        "Bacteroides_fragilis": ("mid", 0.09, 8.0, 3.0),
        "Parvimonas_micra": ("mid", 0.05, 6.5, 2.5),
        "Peptostreptococcus_anaerobius": ("mid", 0.05, 7.0, 2.8),
        "Escherichia_coli": ("mid", 0.06, 5.5, 3.0),
        "Klebsiella_pneumoniae": ("mid", 0.05, 6.0, 2.5),
        "Proteus_mirabilis": ("mid", 0.06, 8.5, 3.2),
        "Morganella_morganii": ("mid", 0.05, 9.0, 3.5),
        "Enterococcus_faecalis": ("mid", 0.07, 7.0, 3.0),
        "Wohlfahrtiimonas_chitiniclastica": ("mid", 0.06, 11.0, 4.0),  # Insect-associated
        "Ignatzschineria_larvae": ("mid", 0.07, 10.0, 3.8),          # Fly-larvae associated
        "Fusobacterium_nucleatum": ("mid", 0.04, 5.0, 2.2),
        "Prevotella_melaninogenica": ("mid", 0.04, 6.0, 2.5),

        # Group 3: Late Decomposers & Soil Saprophytes (Advanced Decay / Skeletonization)
        "Acinetobacter_baumannii": ("late", 0.09, 12.0),
        "Pseudomonas_fluorescens": ("late", 0.08, 14.0),
        "Pseudomonas_putida": ("late", 0.07, 15.0),
        "Bacillus_subtilis": ("late", 0.06, 13.0),
        "Bacillus_cereus": ("late", 0.06, 12.5),
        "Flavobacterium_succinicans": ("late", 0.05, 16.0),
        "Sphingobacterium_multivorum": ("late", 0.05, 17.0),
        "Chryseobacterium_gleum": ("late", 0.04, 18.0),
        "Streptomyces_albus": ("late", 0.05, 19.0),
        "Nocardia_asteroides": ("late", 0.04, 20.0),
        "Planococcus_halocryophilus": ("late", 0.03, 17.5),
        "Pedobacter_heparinus": ("late", 0.04, 16.5),
        "Cellulomonas_flavigena": ("late", 0.03, 21.0),
        "Arthrobacter_globiformis": ("late", 0.04, 18.5),
        "Burkholderia_cepacia": ("late", 0.03, 15.5),
        "Rhizobium_radiobacter": ("late", 0.03, 22.0),

        # Group 4: Commensal Background (Low steady presence)
        "Brevibacterium_casei": ("commensal", 0.02),
        "Corynebacterium_jeikeium": ("commensal", 0.02),
        "Streptococcus_constellatus": ("commensal", 0.015),
        "Propionibacterium_avidum": ("commensal", 0.015),

        # Group 5: Rare / Transient Taxa (<5% prevalence, designed to be pruned by QC filter)
        "Serratia_marcescens": ("rare", 0.003, 0.04),
        "Staphylococcus_aureus": ("rare", 0.002, 0.03),
        "Treponema_denticola": ("rare", 0.002, 0.02),
        "Campylobacter_rectus": ("rare", 0.001, 0.03),
        "Filifactor_alocis": ("rare", 0.001, 0.02),
        "Prevotella_intermedia": ("rare", 0.002, 0.04),
    }

    n_taxa = len(taxa_definitions)
    taxa_names = list(taxa_definitions.keys())

    # -------------------------------------------------------------
    # 3. Compute Base Biological Relative Abundances
    # -------------------------------------------------------------
    base_abundances = np.zeros((n_samples, n_taxa))

    # Calculate effective progression using temperature acceleration (Q10 effect approximation)
    # Higher ambient temperature accelerates biological decay
    temp_factor = (ambient_temp - 20.0) / 10.0
    effective_pmi = pmi_days * (1.0 + 0.3 * temp_factor)
    effective_pmi = np.maximum(effective_pmi, 0.05)

    for j, (taxon, params) in enumerate(taxa_definitions.items()):
        curve_type = params[0]
        max_frac = params[1]

        if curve_type == "early":
            # Sigmoid decay: 1 / (1 + exp((pmi - t0) * k))
            t0 = params[2]
            k = 1.0
            curve = 1.0 / (1.0 + np.exp((effective_pmi - t0) * k))
            base_abundances[:, j] = max_frac * curve

        elif curve_type == "mid":
            # Gaussian peak: exp(-0.5 * ((pmi - peak) / width)^2)
            peak = params[2]
            width = params[3]
            curve = np.exp(-0.5 * ((effective_pmi - peak) / width) ** 2)
            base_abundances[:, j] = max_frac * curve

        elif curve_type == "late":
            # Sigmoid surge: 1 / (1 + exp(-(pmi - t0) * k))
            t0 = params[2]
            k = 0.45
            curve = 1.0 / (1.0 + np.exp(-(effective_pmi - t0) * k))
            base_abundances[:, j] = max_frac * curve

        elif curve_type == "commensal":
            # Steady baseline with slight variation
            base_abundances[:, j] = max_frac * (0.8 + 0.4 * np.random.uniform(0.8, 1.2, size=n_samples))

        elif curve_type == "rare":
            # Zero for most samples, small burst in only a few samples (< 5% prevalence)
            prevalence_p = params[2]
            mask = np.random.binomial(n=1, p=prevalence_p, size=n_samples)
            base_abundances[:, j] = mask * max_frac * np.random.uniform(0.5, 1.5, size=n_samples)

    # Normalize base proportions so row sum = 1.0
    row_sums = base_abundances.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    norm_proportions = base_abundances / row_sums

    # -------------------------------------------------------------
    # 4. Generate Overdispersed NGS Read Counts
    # -------------------------------------------------------------
    # Total sequencing depth per sample follows a log-normal distribution
    # Median ~ 25,000 reads; 98% of samples have 5,000 - 65,000 reads
    depths = np.random.lognormal(mean=10.0, sigma=0.45, size=n_samples).astype(int)

    # Intentionally insert ~15 low-depth samples (<1,000 reads) to validate the QC filter
    low_depth_idx = np.random.choice(n_samples, size=15, replace=False)
    depths[low_depth_idx] = np.random.randint(450, 950, size=15)

    # Negative-binomial count simulation (gamma-poisson mixture) to model biological overdispersion
    # Poisson lambda has gamma noise: shape k, scale theta = mu / k
    # Dispersion factor k: smaller k = higher overdispersion
    dispersion_k = 15.0

    counts = np.zeros((n_samples, n_taxa), dtype=int)
    for i in range(n_samples):
        expected_counts = norm_proportions[i, :] * depths[i]
        # Gamma rate multiplier for each taxon
        gamma_rates = np.random.gamma(shape=dispersion_k, scale=1.0 / dispersion_k, size=n_taxa)
        noisy_lambdas = expected_counts * gamma_rates
        counts[i, :] = np.random.poisson(lam=noisy_lambdas)

    counts_df = pd.DataFrame(counts, index=sample_ids, columns=taxa_names)
    counts_df.index.name = "sample_id"

    # -------------------------------------------------------------
    # 5. Output to Disk
    # -------------------------------------------------------------
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    counts_file = out_path / "synthetic_counts.csv"
    metadata_file = out_path / "metadata.csv"

    counts_df.to_csv(counts_file)
    metadata_df.to_csv(metadata_file, index=False)

    print(f"[+] Successfully generated {n_samples} synthetic necrobiome samples.")
    print(f"    - Matrix shape: {counts_df.shape[0]} samples x {counts_df.shape[1]} taxa")
    print(f"    - Low-depth samples (<1000 reads): {(depths < 1000).sum()}")
    print(f"    - Output: {counts_file} and {metadata_file}")

    return counts_df, metadata_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic necrobiome metagenomic dataset.")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--outdir", type=str, default="data", help="Output directory (default: data)")

    args = parser.parse_args()
    generate_necrobiome_dataset(n_samples=args.samples, random_seed=args.seed, output_dir=args.outdir)
