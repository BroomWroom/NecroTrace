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
        "stage": "Fresh / Early Rigor (0–3d)",
        "role": "Cutaneous & mucosal aerobe; normal commensal that dominates immediately postmortem before tissue anoxia.",
        "typical_weight": 0.22,
        "indicator_group": "early",
        "gram_stain": "Gram-positive",
        "morphology_type": "cocci_clusters",
        "morphology_desc": "Spherical cocci (0.5–1.5 µm) in grape-like clusters; non-motile, non-sporing facultative anaerobe.",
        "habitat": "Cutaneous stratum corneum & mucosal epithelial barrier",
        "biochemical_action": "Consumes remaining superficial oxygen; maintains high initial baseline prior to systemic putrefactive anoxia.",
        "evidence_tier": "Tier 1: Fresh Phase Commensal",
        "peak_window": "0 – 48 Hours",
    },
    "Streptococcus_oralis": {
        "common_name": "Streptococcus oralis",
        "stage": "Fresh / Early Rigor (0–3d)",
        "role": "Oral mucosal pioneer commensal; abundant in early swabs prior to enteric migration.",
        "typical_weight": 0.18,
        "indicator_group": "early",
        "gram_stain": "Gram-positive",
        "morphology_type": "cocci_chains",
        "morphology_desc": "Ovoid to spherical cocci in pairs and short chains; alpha-hemolytic mucosal pioneer.",
        "habitat": "Oral cavity, buccal mucosa, saliva",
        "biochemical_action": "Dominates oral swabs in fresh corpses; rapidly suppressed once putrefactive gastrointestinal purge fluid ascends.",
        "evidence_tier": "Tier 1: Oral Commensal Pioneer",
        "peak_window": "0 – 36 Hours",
    },
    "Cutibacterium_acnes": {
        "common_name": "Cutibacterium acnes",
        "stage": "Fresh / Early Rigor (0–3d)",
        "role": "Sebaceous skin commensal; rapidly senesces as decomposition fluid washes over tissues.",
        "typical_weight": 0.15,
        "indicator_group": "early",
        "gram_stain": "Gram-positive",
        "morphology_type": "pleomorphic_rods",
        "morphology_desc": "Diphtheroid-like, slow-growing, pleomorphic non-sporing rod with branching tendencies.",
        "habitat": "Sebaceous glands, hair follicles, dermal pores",
        "biochemical_action": "Lipase-mediated triglyceride hydrolysis; relative abundance plummets during putrefactive skin slippage.",
        "evidence_tier": "Tier 1: Dermal Integrity Marker",
        "peak_window": "0 – 48 Hours",
    },
    "Corynebacterium_striatum": {
        "common_name": "Corynebacterium striatum",
        "stage": "Fresh / Early Rigor (0–3d)",
        "role": "Aerobic skin surface colonizer; marker of un-decomposed epidermal barrier.",
        "typical_weight": 0.12,
        "indicator_group": "early",
        "gram_stain": "Gram-positive",
        "morphology_type": "club_rods",
        "morphology_desc": "Slender, slightly curved or club-shaped rods with distinctive banded metachromatic granules.",
        "habitat": "Skin surface, axillary & inguinal epidermis",
        "biochemical_action": "Marker of an intact stratum corneum; rapidly disappears once epidermal detachment and bullae emerge.",
        "evidence_tier": "Tier 2: Intact Cutaneous Marker",
        "peak_window": "12 – 60 Hours",
    },
    "Rothia_dentocariosa": {
        "common_name": "Rothia dentocariosa",
        "stage": "Fresh / Early Rigor (0–3d)",
        "role": "Oropharyngeal commensal; declines steeply within 48-72 hours postmortem.",
        "typical_weight": 0.08,
        "indicator_group": "early",
        "gram_stain": "Gram-positive",
        "morphology_type": "coccoid_rods",
        "morphology_desc": "Pleomorphic rod-to-coccoid organisms forming rudimentary branching filaments.",
        "habitat": "Oropharyngeal mucosa & dental supragingival plaque",
        "biochemical_action": "Steep exponential decay curve serves as a biological chronometer for the first 72 hours postmortem.",
        "evidence_tier": "Tier 2: Oropharyngeal Timekeeper",
        "peak_window": "0 – 72 Hours",
    },
    "Micrococcus_luteus": {
        "common_name": "Micrococcus luteus",
        "stage": "Fresh / Early Rigor (0–3d)",
        "role": "Obligate aerobe; highly sensitive to postmortem tissue hypoxia.",
        "typical_weight": 0.06,
        "indicator_group": "early",
        "gram_stain": "Gram-positive",
        "morphology_type": "tetrad_cocci",
        "morphology_desc": "Spherical cocci (0.9–1.8 µm) occurring in tetrads and sarcinae; intense yellow carotenoid pigmentation.",
        "habitat": "Superficial cutaneous layer & airborne dust deposition",
        "biochemical_action": "Strict obligate aerobe; serves as a sensitive bioindicator that tissue hypoxia is not yet generalized.",
        "evidence_tier": "Tier 2: Aerobic Extinction Marker",
        "peak_window": "0 – 48 Hours",
    },

    # Active Putrefaction Flora (Bloat / Marbling / Purge / Autolysis: Days 3 - 12)
    "Clostridium_perfringens": {
        "common_name": "Clostridium perfringens",
        "stage": "Bloat / Active Putrefaction (3–12d)",
        "role": "Potent anaerobic gas-producer (alpha-toxin); driver of postmortem abdominal distension, subcutaneous emphysema, and tissue liquefaction.",
        "typical_weight": 0.28,
        "indicator_group": "mid",
        "gram_stain": "Gram-positive",
        "morphology_type": "spore_bacilli",
        "morphology_desc": "Large, blunt-ended encapsulated bacilli (3–8 µm); subterminal oval endospores, obligate anaerobe.",
        "habitat": "Colonic lumen & mesenteric lymphatic network",
        "biochemical_action": "Lecithinase C & alpha-toxin release; ferments tissue glycogen into massive volumes of CO2 and H2, inflating the abdomen.",
        "evidence_tier": "Tier 1: Primary Putrefaction Driver",
        "peak_window": "3 – 9 Days",
    },
    "Bacteroides_fragilis": {
        "common_name": "Bacteroides fragilis",
        "stage": "Bloat / Active Putrefaction (3–12d)",
        "role": "Predominant obligate anaerobe; translocates from lower intestine across mucosal barriers during active bloat.",
        "typical_weight": 0.22,
        "indicator_group": "mid",
        "gram_stain": "Gram-negative",
        "morphology_type": "anaerobic_rods",
        "morphology_desc": "Pleomorphic non-sporing rods with rounded ends; bile-resistant capsulated obligate anaerobe.",
        "habitat": "Lower gastrointestinal tract & peritoneal cavity",
        "biochemical_action": "Enzymatic degradation of mucosal basement membranes; translocates systemically through portal veins during gut barrier collapse.",
        "evidence_tier": "Tier 1: Enteric Translocation Chronometer",
        "peak_window": "4 – 10 Days",
    },
    "Proteus_mirabilis": {
        "common_name": "Proteus mirabilis",
        "stage": "Bloat / Active Putrefaction (3–12d)",
        "role": "Swarming proteolytic gammaproteobacterium; causes venous marbling and rapid protein liquefaction.",
        "typical_weight": 0.14,
        "indicator_group": "mid",
        "gram_stain": "Gram-negative",
        "morphology_type": "swarming_rods",
        "morphology_desc": "Straight motile rods with peritrichous flagella demonstrating vigorous cyclic swarming waves.",
        "habitat": "Intestinal tract & intravascular venous blood",
        "biochemical_action": "Urease and gelatinase secretion; produces hydrogen sulfide (H2S) reacting with hemoglobin to create green sulfhemoglobin marbling.",
        "evidence_tier": "Tier 1: Venous Marbling Bioindicator",
        "peak_window": "3 – 8 Days",
    },
    "Morganella_morganii": {
        "common_name": "Morganella morganii",
        "stage": "Bloat / Active Putrefaction (3–12d)",
        "role": "Opportunistic enteric organism; associated with bloody purge fluid and putrefactive odor.",
        "typical_weight": 0.12,
        "indicator_group": "mid",
        "gram_stain": "Gram-negative",
        "morphology_type": "enteric_rods",
        "morphology_desc": "Straight motile rods (0.6–0.7 µm); facultatively anaerobic, ornithine decarboxylase-positive.",
        "habitat": "Enteric canal & pulmonary vasculature",
        "biochemical_action": "Decarboxylates lysine and ornithine into cadaverine and putrescine; hallmark organism of serosanguinous oral/nasal purge fluid.",
        "evidence_tier": "Tier 2: Purge Fluid Bioindicator",
        "peak_window": "4 – 11 Days",
    },
    "Enterococcus_faecalis": {
        "common_name": "Enterococcus faecalis",
        "stage": "Bloat / Active Putrefaction (3–12d)",
        "role": "Facultative anaerobe; proliferates rapidly throughout decomposing abdominal organs and blood vessels.",
        "typical_weight": 0.10,
        "indicator_group": "mid",
        "gram_stain": "Gram-positive",
        "morphology_type": "ovoid_pairs",
        "morphology_desc": "Ovoid cocci (0.6–2.0 µm) in pairs and short chains; hardy facultative anaerobe tolerant of high bile/salt concentrations.",
        "habitat": "Lower bowel & retroperitoneal tissues",
        "biochemical_action": "Rapid colonization of post-arrest ischemic organs; thrives in hypoxic visceral environments where commensals fail.",
        "evidence_tier": "Tier 2: Systemic Colonization Marker",
        "peak_window": "3 – 10 Days",
    },
    "Wohlfahrtiimonas_chitiniclastica": {
        "common_name": "Wohlfahrtiimonas chitiniclastica",
        "stage": "Bloat / Active Putrefaction (3–12d)",
        "role": "Symbiont of blowfly (Calliphoridae) larvae; definitive biological marker of insect colonization during active decay.",
        "typical_weight": 0.08,
        "indicator_group": "mid",
        "gram_stain": "Gram-negative",
        "morphology_type": "dipteran_symbiont",
        "morphology_desc": "Short straight rods (0.5–1.0 µm), non-motile; obligately aerobic, strongly chitinase-positive.",
        "habitat": "Salivary glands and alimentary tract of necrophagous blowflies (*Lucilia sericata*, *Wohlfahrtia*)",
        "biochemical_action": "Directly inoculated into cadaveric orifices by ovipositing flies; acts as an unforgeable biological signature of active entomological feeding.",
        "evidence_tier": "Tier 1: Entomological Succession Correlate",
        "peak_window": "3 – 9 Days",
    },
    "Ignatzschineria_larvae": {
        "common_name": "Ignatzschineria larvae",
        "stage": "Bloat / Active Putrefaction (3–12d)",
        "role": "Fly-larva associated necrobiome biomarker; spikes during peak maggot feeding activity.",
        "typical_weight": 0.08,
        "indicator_group": "mid",
        "gram_stain": "Gram-negative",
        "morphology_type": "dipteran_symbiont",
        "morphology_desc": "Non-motile, non-sporing aerobic rods; specifically associated with flesh-fly (*Sarcophagidae*) larval development.",
        "habitat": "Maggot feeding masses & necrotic wound cavities",
        "biochemical_action": "Spikes to high relative abundance in tandem with second- and third-instar larval masses; accelerates soft tissue autolysis.",
        "evidence_tier": "Tier 1: Maggot Mass Peak Marker",
        "peak_window": "4 – 10 Days",
    },

    # Advanced Decay Flora (Tissue Rupture / Black Putrefaction / Skeletonization: Days > 12)
    "Acinetobacter_baumannii": {
        "common_name": "Acinetobacter baumannii",
        "stage": "Advanced Decay / Skeletonization (12–30d+)",
        "role": "Hardy environmental saprophyte; withstands tissue desiccation and outcompetes enteric flora in advanced decay.",
        "typical_weight": 0.25,
        "indicator_group": "late",
        "gram_stain": "Gram-negative",
        "morphology_type": "coccobacilli",
        "morphology_desc": "Short, plump, almost spherical coccobacilli (1.0–1.5 µm); strictly aerobic, non-motile, highly desiccation-resistant.",
        "habitat": "Desiccating tendon, periosteum, and arid soil interface",
        "biochemical_action": "Forms dense biofilms on dry connective tissues following abdominal collapse; outcompetes dying enteric anaerobes.",
        "evidence_tier": "Tier 1: Post-Rupture Desiccation Marker",
        "peak_window": "12 – 25 Days",
    },
    "Pseudomonas_fluorescens": {
        "common_name": "Pseudomonas fluorescens",
        "stage": "Advanced Decay / Skeletonization (12–30d+)",
        "role": "Soil and moisture saprophyte; colonizes post-rupture remains and skeletonizing bone surfaces.",
        "typical_weight": 0.20,
        "indicator_group": "late",
        "gram_stain": "Gram-negative",
        "morphology_type": "motile_rods",
        "morphology_desc": "Motile rods with multiple polar flagella; psychrotrophic, produces water-soluble greenish pyoverdine fluorophore.",
        "habitat": "Cadaveric Decomposition Island (CDI) soil & bone marrow",
        "biochemical_action": "Extracellular collagenase and lipase hydrolysis of liquefied cadaveric fluids draining into soil substrate.",
        "evidence_tier": "Tier 1: Soil-Cadaver Interface Marker",
        "peak_window": "14 – 28 Days",
    },
    "Bacillus_subtilis": {
        "common_name": "Bacillus subtilis",
        "stage": "Advanced Decay / Skeletonization (12–30d+)",
        "role": "Endospore-forming soil decomposer; hydrolyzes complex fibrous and cartilaginous proteins.",
        "typical_weight": 0.18,
        "indicator_group": "late",
        "gram_stain": "Gram-positive",
        "morphology_type": "endospore_former",
        "morphology_desc": "Straight rod-shaped cells (2.0–4.0 µm) with central/subterminal oval endospores that resist heat and desiccation.",
        "habitat": "Humic topsoil & drying skeletal remains",
        "biochemical_action": "Subtilisin and neutral protease secretion; digests recalcitrant fibrous ligaments and cartilage clinging to skeletonized bone.",
        "evidence_tier": "Tier 1: Cartilage & Bone Breakdown Marker",
        "peak_window": "15 – 30+ Days",
    },
    "Sphingobacterium_multivorum": {
        "common_name": "Sphingobacterium multivorum",
        "stage": "Advanced Decay / Skeletonization (12–30d+)",
        "role": "Proteolytic soil bacterium; marker of soil-body interface and late-stage skeletal breakdown.",
        "typical_weight": 0.14,
        "indicator_group": "late",
        "gram_stain": "Gram-negative",
        "morphology_type": "non_motile_rods",
        "morphology_desc": "Non-motile, non-sporing yellow-pigmented rods containing high cellular concentrations of sphingophospholipids.",
        "habitat": "Decomposition soil matrix & bone surfaces",
        "biochemical_action": "Flourishes during advanced black putrefaction and dry decay in terrestrial environments.",
        "evidence_tier": "Tier 2: Late Decomposition Soil Flora",
        "peak_window": "16 – 30+ Days",
    },
    "Streptomyces_albus": {
        "common_name": "Streptomyces albus",
        "stage": "Advanced Decay / Skeletonization (12–30d+)",
        "role": "Actinomycete; proliferates on desiccated, mummified, or dry skeletal remains.",
        "typical_weight": 0.12,
        "indicator_group": "late",
        "gram_stain": "Gram-positive",
        "morphology_type": "branching_actinomycete",
        "morphology_desc": "Extensively branching vegetative hyphae (0.5–1.0 µm) developing aerial mycelium with spiral spore chains.",
        "habitat": "Dry skeletal remains, mummified skin remnants, desiccated tissue",
        "biochemical_action": "Degrades keratinous and fibrous remnants; serves as an evidentiary marker of dry decay or mummified postmortem states.",
        "evidence_tier": "Tier 2: Mummification & Dry Remains Bioindicator",
        "peak_window": "20 – 35+ Days",
    },
    "Planococcus_halocryophilus": {
        "common_name": "Planococcus halocryophilus",
        "stage": "Advanced Decay / Skeletonization (12–30d+)",
        "role": "Extremotolerant bacterium capable of surviving cold soil temperatures during late postmortem decay.",
        "typical_weight": 0.08,
        "indicator_group": "late",
        "gram_stain": "Gram-positive",
        "morphology_type": "psychrotolerant_cocci",
        "morphology_desc": "Spherical cocci in pairs or small clusters; psychrotolerant and halotolerant with orange-yellow colonies.",
        "habitat": "Cold climate soils, sub-surface skeletal contacts",
        "biochemical_action": "Maintains metabolic activity under low temperatures (sub-zero survival); calibrates PMI models in chilly or refrigerated scenes.",
        "evidence_tier": "Tier 3: Environmental Extremotolerant Marker",
        "peak_window": "18 – 40+ Days",
    },
}


# -----------------------------------------------------------------------------
# 2. MORPHOLOGICAL AUTOPSY INSPECTION GUIDES
# -----------------------------------------------------------------------------
MORPHOLOGICAL_SIGN_GUIDES = {
    "rigor_mortis": {
        "title": "Rigor Mortis Progression",
        "description": "Postmortem muscle rigidity caused by ATP depletion preventing actin-myosin detachment. Follows Nysten's law (cranio-caudal stiffening).",
        "anatomical_focus": "Temporomandibular joints, cervical spine, phalanges, elbows, knees, ankles.",
        "palpation_cue": "Apply firm passive flexion pressure to mandible and joints. Note mechanical resistance versus flaccidity.",
        "stages": {
            "Early / Developing (Jaw, neck, facial muscles)": {
                "badge": "DEVELOPING // 2-6 HOURS",
                "appearance": "Rigidity confined to small muscle groups (mandible, eyelids, neck). Limbs remain completely flaccid.",
                "severity_color": "#cef79e",
                "phase_tag": "Fresh (Phase 01)",
            },
            "Fully Established (Generalized stiffening across all limbs & trunk)": {
                "badge": "PEAK RIGIDITY // 12-24 HOURS",
                "appearance": "Generalized fixation across all joints. Cadaver can often be lifted by the head or heels like a board.",
                "severity_color": "#cef79e",
                "phase_tag": "Fresh / Algor (Phase 01)",
            },
            "Passing Off (Receding from face, persisting in lower limbs)": {
                "badge": "DISSIPATING // 24-36 HOURS",
                "appearance": "Mandible and upper limbs relaxed; residual rigidity lingering only in lower limbs and knees.",
                "severity_color": "#e7c95e",
                "phase_tag": "Transition (Phase 01-02)",
            },
            "Completely Absent / Flaccid (Passed off due to decomposition)": {
                "badge": "SECONDARY FLACCIDITY // >36-48 HOURS",
                "appearance": "All muscles completely flaccid due to autolytic breakdown of myofilaments. Distinct from fresh flaccidity.",
                "severity_color": "#e77e5e",
                "phase_tag": "Putrefaction (Phase 02-03)",
            },
        }
    },
    "bloat": {
        "title": "Abdominal Distension & Bloat",
        "description": "Gaseous inflation driven by bacterial anaerobic fermentation (CO2, H2, CH4, H2S) producing high internal pressure.",
        "anatomical_focus": "Umbilical region, hypogastrium, scrotum/vulva, facial tissues.",
        "palpation_cue": "Percuss abdomen for tympanitic resonance; observe scrotal/labial ballooning and eyeball protrusion.",
        "stages": {
            "None / Flat (Abdomen soft, no gaseous distension)": {
                "badge": "FLAT / NORMAL CONTOUR // 0-48H",
                "appearance": "Normal body contours, soft abdominal wall, no subcutaneous emphysema.",
                "severity_color": "#cef79e",
                "phase_tag": "Fresh (Phase 01)",
            },
            "Initial / Mild (Early firmness, mild lower quadrant distension)": {
                "badge": "MILD TENSION // 2-4 DAYS",
                "appearance": "Abdomen tense upon palpation; mild swelling over cecal region and right iliac fossa.",
                "severity_color": "#cef79e",
                "phase_tag": "Early Bloat (Phase 02)",
            },
            "Moderate Bloat (Tense generalized distension, scrotum/vulva swelling)": {
                "badge": "GENERALIZED BLOAT // 4-7 DAYS",
                "appearance": "Tense drum-like abdomen, scrotal ballooning, tongue protruding between swollen lips.",
                "severity_color": "#e7c95e",
                "phase_tag": "Active Bloat (Phase 02)",
            },
            "Severe Bloat & Purge (Massive distension, blood-stained froth at orifices)": {
                "badge": "MAXIMUM DISTENSION // 6-10 DAYS",
                "appearance": "Massive distension, extreme venous congestion, eyes protruding, cutaneous blebs/bullae.",
                "severity_color": "#e77e5e",
                "phase_tag": "Peak Putrefaction (Phase 02)",
            },
            "Ruptured / Subsiding (Abdominal wall collapsed, tissue liquefaction)": {
                "badge": "WALL RUPTURE & COLLAPSE // >10-14 DAYS",
                "appearance": "Abdominal wall ruptures, releasing putrefactive gas and purge fluid; visceral exposure.",
                "severity_color": "#d65050",
                "phase_tag": "Advanced Decay (Phase 03)",
            },
        }
    },
    "discoloration": {
        "title": "Skin Discoloration & Vascular Marbling",
        "description": "Hemolysis and hemoglobin breakdown reacting with bacterially produced H2S to form dark green sulfhemoglobin.",
        "anatomical_focus": "Right iliac fossa, superficial veins of chest, shoulders, abdomen, and thighs.",
        "palpation_cue": "Inspect anterior trunk under direct illumination. Trace branching patterns along superficial venous trunks.",
        "stages": {
            "Normal / Postmortem Pallor (No putrefactive staining)": {
                "badge": "POSTMORTEM PALLOR // 0-24H",
                "appearance": "Pale skin without greenish discoloration or venous arborization. Hypostatic lividity may be fixed.",
                "severity_color": "#cef79e",
                "phase_tag": "Fresh (Phase 01)",
            },
            "Greenish discoloration over Right Iliac Fossa": {
                "badge": "CECAL SULFHEMOGLOBIN // 24-48H",
                "appearance": "First visible sign of putrefaction: 4–6 cm greenish macule over cecum in the right lower quadrant.",
                "severity_color": "#cef79e",
                "phase_tag": "Early Putrefaction (Phase 01-02)",
            },
            "Arborescent Venous Marbling (Branching greenish-purple venous network)": {
                "badge": "VASCULAR MARBLING // 2-5 DAYS",
                "appearance": "Mosaic tree-like arborization of superficial veins stained dark green/purple by sulfhemoglobin.",
                "severity_color": "#e7c95e",
                "phase_tag": "Active Bloat (Phase 02)",
            },
            "Generalized Dusky Green / Bronzing (Extensive torso & facial discoloration)": {
                "badge": "GENERALIZED BRONZING // 5-10 DAYS",
                "appearance": "Discoloration spreads over entire trunk, neck, face, and extremities; skin slippage begins.",
                "severity_color": "#e77e5e",
                "phase_tag": "Advanced Putrefaction (Phase 02)",
            },
            "Black Putrefaction (Dark brownish-black discoloration, skin slippage)": {
                "badge": "BLACK PUTREFACTION // >12 DAYS",
                "appearance": "Exposed tissues turn dark brownish-black to pitch; large epidermal slippage sheets and autolysis.",
                "severity_color": "#d65050",
                "phase_tag": "Black Decay (Phase 03)",
            },
        }
    },
    "purge_fluid": {
        "title": "Purge Fluid & Orifices",
        "description": "Forced expulsion of autolytic, serosanguinous liquefaction fluid driven out of natural orifices by intrathoracic pressure.",
        "anatomical_focus": "Nares, oral cavity, ear canals, rectum.",
        "palpation_cue": "Check patency of mouth and nose; note presence of frothy, foul-smelling, dark reddish-brown fluid.",
        "stages": {
            "Absent (Orifices clear, eyes intact)": {
                "badge": "CLEAR ORIFICES // 0-48H",
                "appearance": "No fluid discharge from nose or mouth. Facial features sharp.",
                "severity_color": "#cef79e",
                "phase_tag": "Fresh (Phase 01)",
            },
            "Early Serous / Frothy discharge at nares and lips": {
                "badge": "EARLY SEROUS FROTH // 2-4 DAYS",
                "appearance": "Thin pinkish froth pooling at external nares and oral commissures.",
                "severity_color": "#cef79e",
                "phase_tag": "Early Bloat (Phase 02)",
            },
            "Blood-stained Purge Fluid (Active putrefactive liquefaction)": {
                "badge": "ACTIVE PURGE // 4-10 DAYS",
                "appearance": "Dark red-brown, foul-smelling purge fluid bubbling from mouth and nose; high intrathoracic pressure.",
                "severity_color": "#e77e5e",
                "phase_tag": "Active Bloat (Phase 02)",
            },
            "Desiccated / Dry remains": {
                "badge": "DRIED RESIDUE // >12 DAYS",
                "appearance": "Purge fluid dried into brown crusts; facial skin leathery and desiccated.",
                "severity_color": "#d65050",
                "phase_tag": "Dry Decay (Phase 03)",
            },
        }
    },
    "maggot_activity": {
        "title": "Entomology & Maggot Colonization",
        "description": "Chronological colonization by necrophagous insects (Calliphoridae, Sarcophagidae) tracking soft tissue decay.",
        "anatomical_focus": "Eyelids, nostrils, mouth, folds of neck, wounds, clothing margins.",
        "palpation_cue": "Inspect orifices and folds under illumination; check for fly egg clutches, active larvae, or brown puparia.",
        "stages": {
            "None detected": {
                "badge": "NO INSECT OVIPOSITION // 0-24H",
                "appearance": "No fly eggs or larvae present on exposed mucosal or dermal surfaces.",
                "severity_color": "#cef79e",
                "phase_tag": "Fresh (Phase 01)",
            },
            "Early fly egg deposits / small instar larvae in natural orifices": {
                "badge": "EGG CLUTCHES / 1ST INSTAR // 1-3 DAYS",
                "appearance": "Yellowish-white clusters of fly eggs in corners of eyes, nostrils, or teeth; tiny active larvae.",
                "severity_color": "#cef79e",
                "phase_tag": "Early Colonization (Phase 01-02)",
            },
            "Active larval feeding masses across soft tissues": {
                "badge": "ACTIVE LARVAL MASS // 4-10 DAYS",
                "appearance": "Dense, voracious maggot masses tunneling into tissues; elevated local temperatures and rapid mass loss.",
                "severity_color": "#e77e5e",
                "phase_tag": "Peak Larval Decay (Phase 02)",
            },
            "Pupae / Empty puparia present": {
                "badge": "PUPATION / POST-FEEDING // >12 DAYS",
                "appearance": "Dark reddish-brown barrel-shaped puparia in surrounding soil or clothing folds; skeletonization starts.",
                "severity_color": "#d65050",
                "phase_tag": "Post-Feeding / Late (Phase 03)",
            },
        }
    }
}


# -----------------------------------------------------------------------------
# 3. MORPHOLOGICAL TRIAGE EVALUATION
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
    - Auto-suggested microbial bioindicators with full visual and biochemical profiles
    - Active morphological visual inspection guides
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

    # Filter and rank candidate bioindicators with rich visual metadata
    suggested_bioindicators = []
    for taxon_id, info in FORENSIC_BIOINDICATOR_CATALOG.items():
        is_primary = (info["indicator_group"] == target_group)
        suggested_bioindicators.append({
            "taxon_id": taxon_id,
            "common_name": info["common_name"],
            "stage": info["stage"],
            "role": info["role"],
            "is_recommended": is_primary,
            "typical_weight": info["typical_weight"],
            "gram_stain": info.get("gram_stain", "Gram-positive"),
            "morphology_type": info.get("morphology_type", "bacilli"),
            "morphology_desc": info.get("morphology_desc", ""),
            "habitat": info.get("habitat", ""),
            "biochemical_action": info.get("biochemical_action", ""),
            "evidence_tier": info.get("evidence_tier", "Tier 1: Bioindicator"),
            "peak_window": info.get("peak_window", "Decomposition Interval"),
        })

    # Sort recommended to top, then by biological weight
    suggested_bioindicators.sort(key=lambda x: (not x["is_recommended"], -x["typical_weight"]))

    return {
        "primary_phase": primary_phase,
        "coarse_clinical_range": coarse_clinical_range,
        "target_group": target_group,
        "biological_summary": biological_summary,
        "suggested_bioindicators": suggested_bioindicators,
        "active_guides": {
            "rigor": MORPHOLOGICAL_SIGN_GUIDES["rigor_mortis"]["stages"].get(rigor, {}),
            "bloat": MORPHOLOGICAL_SIGN_GUIDES["bloat"]["stages"].get(bloat, {}),
            "discoloration": MORPHOLOGICAL_SIGN_GUIDES["discoloration"]["stages"].get(discoloration, {}),
            "purge": MORPHOLOGICAL_SIGN_GUIDES["purge_fluid"]["stages"].get(purge, {}),
            "maggots": MORPHOLOGICAL_SIGN_GUIDES["maggot_activity"]["stages"].get(maggots, {}),
        }
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
