"""
IEI Diagnostic Engine - Core Mathematical Framework
Based on Shannon's Information Theory + Clinical Pattern Recognition
Author: Saul (with Claude's assistance)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

# ============================================================================
# IEI CATEGORY DEFINITIONS (IUIS 2024 Classification - Broad Groups)
# ============================================================================

IEI_CATEGORIES = [
    'Combined_ID',              # SCID, CID, syndromic combined (WAS, DiGeorge, Omenn)
    'Antibody_Deficiency',      # CVID, XLA, SAD, IgAD, IgG subclass
    'Phagocyte_Defect',         # CGD, LAD, HIES, cyclic neutropenia, Chédiak-Higashi
    'Complement_Deficiency',    # Classical, alternative, lectin pathways
    'Autoinflammatory',         # FMF, CAPS, TRAPS, Blau, PAPA
    'Immune_Dysregulation',     # IPEX, APECED, HLH, ALPS, cytokine defects, autoimmune lymphoproliferative
    'Innate_Immunity',          # MSMD, CMC, viral susceptibility, TLR defects, NEMO
    'Bone_Marrow_Failure'       # Congenital neutropenia, true marrow failure (NOT WAS - that's Combined_ID)
]

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PathognomicPattern:
    """Represents a constellation of findings highly specific for a diagnosis"""
    name: str
    triggers: List[str]  # Format: "Q23:Ataxia+Telangiectasia"
    probability: float
    category: str
    confirm_with: List[str]  # Question IDs to ask for confirmation
    
@dataclass
class Question:
    """Represents a diagnostic question"""
    id: str
    text: str
    answer_options: List[str]
    base_information_gain: float
    is_nodal: bool = False
    nodal_weight: float = 1.0

# ============================================================================
# PATHOGNOMONIC PATTERNS - Clinical Pearl Recognition
# ============================================================================

PATHOGNOMONIC_PATTERNS = [
    PathognomicPattern(
        name="Ataxia-Telangiectasia",
        triggers=["Q23:Yes"],  # Both ataxia AND telangiectasia
        probability=0.95,
        category="Immune_Dysregulation",
        confirm_with=["Q35", "Q27", "Q12"]  # Lymphoma? Autoimmunity? Low IgA?
    ),
    PathognomicPattern(
        name="Wiskott-Aldrich Syndrome",
        triggers=["Q7:Yes_severe", "Q2:Yes_severe", "Q17:Thrombocytopenia"],
        probability=0.90,
        category="Combined_ID",  # WAS is combined immunodeficiency, not marrow failure
        confirm_with=["Q6", "Q15", "Q36"]  # Male? Microbes? Consanguinity?
    ),
    PathognomicPattern(
        name="LAD-1 (Leukocyte Adhesion Deficiency)",
        triggers=["Q39:Yes"],  # Late umbilical stump detachment
        probability=0.85,
        category="Phagocyte_Defect",
        confirm_with=["Q40", "Q8", "Q36"]  # Tooth retention? Abscesses? Consanguinity?
    ),
    PathognomicPattern(
        name="Chronic Granulomatous Disease",
        triggers=["Q15:Fungi", "Q8:Yes", "Q45:Yes"],  # Fungi + Abscesses + Granulomas
        probability=0.88,
        category="Phagocyte_Defect",
        confirm_with=["Q6", "Q25", "Q41"]  # Sex? Complicated pneumonia? Pneumatoceles?
    ),
    PathognomicPattern(
        name="SCID (Severe Combined Immunodeficiency)",
        triggers=["Q1:<6mo", "Q5:Yes", "Q33:Yes"],  # Early onset + vaccine reaction + absent thymus
        probability=0.92,
        category="Combined_ID",
        confirm_with=["Q43", "Q46", "Q44"]  # FTT? Opportunistic? Diarrhea?
    ),
    PathognomicPattern(
        name="APECED (APS-1)",
        triggers=["Q10:Yes", "Q27:Yes", "Q31:Yes"],  # CMC + Autoimmunity + organ-specific
        probability=0.87,
        category="Immune_Dysregulation",
        confirm_with=["Q36", "Q13", "Q49"]  # Consanguinity? Malformations? Ethnicity?
    ),
    PathognomicPattern(
        name="Hyper-IgE Syndrome (STAT3)",
        triggers=["Q7:Yes_severe", "Q8:Yes", "Q24:Yes", "Q13:Yes"],  # Eczema + Abscesses + Bronchiectasis + Facies
        probability=0.86,
        category="Combined_ID",  # HIES is Th17 deficiency, not primary phagocyte defect
        confirm_with=["Q14", "Q20", "Q40"]  # Dysmorphism? Nails? Teeth retention?
    ),
]

# ============================================================================
# QUESTION DEFINITIONS WITH NODAL WEIGHTS
# ============================================================================

QUESTIONS = {
    # NODAL QUESTIONS - Major category routers
    "Q3": Question(
        id="Q3",
        text="Recurrent infections?",
        answer_options=["Yes_single_pathogen", "Yes_multiple_pathogens", "Non_infectious_manifestations"],
        base_information_gain=0.92,
        is_nodal=True,
        nodal_weight=2.5
    ),
    "Q9": Question(
        id="Q9",
        text="Recurrent fever?",
        answer_options=["No", "Yes"],
        base_information_gain=1.15,
        is_nodal=True,
        nodal_weight=2.8  # High weight for autoinflammatory
    ),
    "Q15": Question(
        id="Q15",
        text="What is the PRIMARY type of pathogen causing infections?",
        answer_options=["Fungi", "Bacteria", "Virus", "Mycobacteria", "Parasite", "None/Non_infectious"],
        base_information_gain=2.95,
        is_nodal=True,
        nodal_weight=3.5  # Highest nodal weight
    ),
    "Q1": Question(
        id="Q1",
        text="Age at onset?",
        answer_options=["<6mo", "6mo-5yr", "5-12yr", "12+_years"],
        base_information_gain=2.35,
        is_nodal=True,
        nodal_weight=3.2
    ),
    "Q12": Question(
        id="Q12",
        text="Dysgammaglobulinemia?",
        answer_options=["Normal", "Hypogammaglobulinemia", "Hypergammaglobulinemia", "Specific_Deficiency"],
        base_information_gain=2.05,
        is_nodal=True,
        nodal_weight=3.0
    ),
    "Q5": Question(
        id="Q5",
        text="Adverse reaction to vaccine(s)? BCG, MMR, VZV, Polio",
        answer_options=["No", "Yes_BCG", "Yes_Viral", "Yes_Multiple"],
        base_information_gain=1.55,
        is_nodal=True,
        nodal_weight=2.9
    ),
    "Q17": Question(
        id="Q17",
        text="Cytopenias?",
        answer_options=["None", "Thrombocytopenia", "Neutropenia", "Lymphopenia", "Multiple"],
        base_information_gain=1.65,
        is_nodal=True,
        nodal_weight=2.6
    ),
    "Q16": Question(
        id="Q16",
        text="Multiple different pathogen types (e.g., bacteria AND fungi)?",
        answer_options=["No_single_type", "Yes_two_types", "Yes_three_or_more"],
        base_information_gain=1.25,
        is_nodal=False
    ),
    
    # Batch 1: High-value questions from original 35
    "Q2": Question(
        id="Q2",
        text="Bleeding tendency? (Petechiae, epistaxis, easy bruising, bloody diarrhea, melena)",
        answer_options=["No", "Yes_mild", "Yes_severe"],
        base_information_gain=0.88,
        is_nodal=False
    ),
    "Q6": Question(
        id="Q6",
        text="Sex of the patient?",
        answer_options=["Male", "Female"],
        base_information_gain=0.18,
        is_nodal=False
    ),
    "Q7": Question(
        id="Q7",
        text="History of eczema or rash?",
        answer_options=["No", "Yes_mild", "Yes_severe"],
        base_information_gain=0.98,
        is_nodal=False
    ),
    "Q11": Question(
        id="Q11",
        text="Lymphoproliferation? (Hepatomegaly, splenomegaly, lymphadenopathy)",
        answer_options=["No", "Yes_one_site", "Yes_multiple_sites"],
        base_information_gain=1.25,
        is_nodal=False
    ),
    "Q13": Question(
        id="Q13",
        text="Congenital malformation(s)?",
        answer_options=["No", "Yes_cardiac", "Yes_skeletal", "Yes_other", "Yes_multiple"],
        base_information_gain=0.78,
        is_nodal=False
    ),
    "Q14": Question(
        id="Q14",
        text="Dysmorphism or peculiar facies?",
        answer_options=["No", "Yes"],
        base_information_gain=0.82,
        is_nodal=False
    ),
    "Q18": Question(
        id="Q18",
        text="Polycythemia/hypercellularity? (Leukocytosis, eosinophilia, neutrophilia, thrombocytosis)",
        answer_options=["No", "Yes_eosinophilia", "Yes_leukocytosis", "Yes_other"],
        base_information_gain=0.72,
        is_nodal=False
    ),
    "Q19": Question(
        id="Q19",
        text="Silver hair or hypopigmentation?",
        answer_options=["No", "Yes"],
        base_information_gain=0.58,
        is_nodal=False
    ),
    "Q20": Question(
        id="Q20",
        text="Dystrophic nails?",
        answer_options=["No", "Yes"],
        base_information_gain=0.42,
        is_nodal=False
    ),
    "Q21": Question(
        id="Q21",
        text="Alopecia or vitiligo?",
        answer_options=["No", "Yes"],
        base_information_gain=0.52,
        is_nodal=False
    ),
    
    # Batch 2: Pathognomonic markers and severity indicators from original 35
    "Q22": Question(
        id="Q22",
        text="Warts or molluscum contagiosum?",
        answer_options=["No", "Yes"],
        base_information_gain=0.68,
        is_nodal=False
    ),
    "Q23": Question(
        id="Q23",
        text="Ataxia? Telangiectasia?",
        answer_options=["No", "Yes_ataxia", "Yes_telangiectasia", "Yes_both"],
        base_information_gain=0.52,
        is_nodal=False
    ),
    "Q24": Question(
        id="Q24",
        text="Bronchiectases?",
        answer_options=["No", "Yes"],
        base_information_gain=0.98,
        is_nodal=False
    ),
    "Q25": Question(
        id="Q25",
        text="Complicated pneumonia? (Empyema, lung abscess, necrotizing)",
        answer_options=["No", "Yes"],
        base_information_gain=1.20,
        is_nodal=False
    ),
    "Q26": Question(
        id="Q26",
        text="Arthritis?",
        answer_options=["No", "Yes"],
        base_information_gain=0.48,
        is_nodal=False
    ),
    "Q27": Question(
        id="Q27",
        text="Autoimmunity? (Lupus, vasculitis, serum autoantibodies)",
        answer_options=["No", "Yes_organ_specific", "Yes_systemic"],
        base_information_gain=0.92,
        is_nodal=False
    ),
    "Q28": Question(
        id="Q28",
        text="HLH (hemophagocytic lymphohistiocytosis)?",
        answer_options=["No", "Yes"],
        base_information_gain=0.72,
        is_nodal=False
    ),
    "Q29": Question(
        id="Q29",
        text="Intensive care unit admission?",
        answer_options=["No", "Yes"],
        base_information_gain=0.62,
        is_nodal=False
    ),
    "Q30": Question(
        id="Q30",
        text="Neurologic deficit? Developmental delay?",
        answer_options=["No", "Yes"],
        base_information_gain=0.58,
        is_nodal=False
    ),
    "Q31": Question(
        id="Q31",
        text="Inflammatory bowel disease?",
        answer_options=["No", "Yes"],
        base_information_gain=0.78,
        is_nodal=False
    ),
    "Q32": Question(
        id="Q32",
        text="Anhidrotic ectodermal dysplasia?",
        answer_options=["No", "Yes"],
        base_information_gain=0.38,
        is_nodal=False
    ),
    "Q33": Question(
        id="Q33",
        text="Absent thymic shadow? Hypoplastic thymus?",
        answer_options=["No", "Yes"],
        base_information_gain=1.35,
        is_nodal=False
    ),
    "Q34": Question(
        id="Q34",
        text="Severe atopy?",
        answer_options=["No", "Yes"],
        base_information_gain=0.88,
        is_nodal=False
    ),
    "Q35": Question(
        id="Q35",
        text="Lymphoma? Malignancy?",
        answer_options=["No", "Yes"],
        base_information_gain=0.62,
        is_nodal=False
    ),
    
    # Additional high-IG questions (top 10)
    "Q4": Question(
        id="Q4",
        text="Infection site(s)?",
        answer_options=["Sinopulmonary", "Skin_soft_tissue", "Invasive_deep", "Disseminated_multiple"],
        base_information_gain=1.85,
        is_nodal=False
    ),
    "Q10": Question(
        id="Q10",
        text="Chronic mucocutaneous candidiasis?",
        answer_options=["No", "Yes"],
        base_information_gain=1.50,
        is_nodal=False
    ),
    "Q8": Question(
        id="Q8",
        text="History of Abscesses?",
        answer_options=["No", "Yes"],
        base_information_gain=1.40,
        is_nodal=False
    ),
}

# ============================================================================
# FERMI ESTIMATIONS - Probability Distributions
# These are expert-derived estimates based on IUIS 2024 + ESID registry
# ============================================================================

def initialize_prior_probabilities() -> Dict[str, float]:
    """
    Initialize prior probabilities for each IEI category
    
    MAXIMALLY FLATTENED PRIORS: In a specialized IEI clinic, we start truly agnostic.
    The top three categories (Antibody, Combined, Phagocyte) are equal at 15%.
    This ensures questions drive the diagnosis, not epidemiological priors.
    
    These priors reflect a patient already referred for IEI evaluation with
    sufficient clinical suspicion to warrant comprehensive diagnostic workup.
    """
    return {
        'Antibody_Deficiency': 0.15,   # DOWN from 0.25 - equal footing
        'Combined_ID': 0.15,           # Same - critical early detection
        'Phagocyte_Defect': 0.15,      # Same - common presentations
        'Immune_Dysregulation': 0.13,  # UP from 0.12 - increasingly recognized
        'Autoinflammatory': 0.12,      # UP from 0.10 - common in clinic
        'Innate_Immunity': 0.12,       # UP from 0.10 - regionally important
        'Complement_Deficiency': 0.10, # UP from 0.08 - underdiagnosed
        'Bone_Marrow_Failure': 0.08    # UP from 0.05 - rare but critical
    }

# Conditional probabilities: P(Answer | Category)
# Format: QUESTION_ID -> Answer -> {Category: Probability}

CONDITIONAL_PROBABILITIES = {
    # Q15: Microbe types - THE MOST INFORMATIVE QUESTION
    "Q15": {
        "Fungi": {
            'Phagocyte_Defect': 0.50,      # CGD, CARD9
            'Combined_ID': 0.25,           # SCID with fungal infections
            'Innate_Immunity': 0.15,       # CARD9, STAT1 GOF
            'Immune_Dysregulation': 0.05,
            'Antibody_Deficiency': 0.03,
            'Autoinflammatory': 0.01,
            'Complement_Deficiency': 0.005,
            'Bone_Marrow_Failure': 0.005
        },
        "Bacteria": {
            'Antibody_Deficiency': 0.48,   # Softened from 0.55 - Classic sinopulmonary
            'Phagocyte_Defect': 0.22,      # Boosted - Abscesses, invasive infections
            'Complement_Deficiency': 0.12, # Encapsulated bacteria
            'Combined_ID': 0.10,
            'Immune_Dysregulation': 0.05,
            'Innate_Immunity': 0.02,
            'Autoinflammatory': 0.005,
            'Bone_Marrow_Failure': 0.005
        },
        "Virus": {
            'Combined_ID': 0.55,           # T-cell deficiency (increased from 0.45)
            'Innate_Immunity': 0.15,       # Viral susceptibility syndromes
            'Immune_Dysregulation': 0.12,  # Viral-triggered HLH
            'Antibody_Deficiency': 0.08,   # Reduced from 0.10
            'Bone_Marrow_Failure': 0.05,   # WAS, XLP (reduced)
            'Phagocyte_Defect': 0.03,
            'Autoinflammatory': 0.01,
            'Complement_Deficiency': 0.01
        },
        "Mycobacteria": {
            'Innate_Immunity': 0.60,       # MSMD pathway defects
            'Phagocyte_Defect': 0.20,      # CGD
            'Combined_ID': 0.10,
            'Immune_Dysregulation': 0.05,
            'Antibody_Deficiency': 0.03,
            'Autoinflammatory': 0.01,
            'Complement_Deficiency': 0.005,
            'Bone_Marrow_Failure': 0.005
        },
        "Parasite": {
            'Combined_ID': 0.40,
            'Antibody_Deficiency': 0.25,
            'Immune_Dysregulation': 0.15,
            'Innate_Immunity': 0.10,
            'Phagocyte_Defect': 0.05,
            'Autoinflammatory': 0.025,
            'Complement_Deficiency': 0.015,
            'Bone_Marrow_Failure': 0.01
        },
        "None": {
            'Autoinflammatory': 0.50,      # Non-infectious manifestations
            'Immune_Dysregulation': 0.25,
            'Complement_Deficiency': 0.10,
            'Antibody_Deficiency': 0.05,
            'Combined_ID': 0.05,
            'Phagocyte_Defect': 0.03,
            'Innate_Immunity': 0.01,
            'Bone_Marrow_Failure': 0.01
        }
    },
    
    # Q1: Age at onset - Critical for narrowing differential
    "Q1": {
        "<6mo": {
            'Combined_ID': 0.65,           # SCID typically <6mo
            'Bone_Marrow_Failure': 0.15,   # WAS, congenital neutropenia
            'Phagocyte_Defect': 0.10,      # LAD, CGD can present early
            'Immune_Dysregulation': 0.05,  # IPEX, early HLH
            'Innate_Immunity': 0.03,
            'Antibody_Deficiency': 0.01,   # Maternal IgG protects
            'Complement_Deficiency': 0.005,
            'Autoinflammatory': 0.005
        },
        "6mo-5yr": {
            'Antibody_Deficiency': 0.40,   # Agammaglobulinemia, CVID onset
            'Phagocyte_Defect': 0.25,      # CGD, HIES typical age
            'Combined_ID': 0.15,           # CID variants
            'Immune_Dysregulation': 0.10,
            'Innate_Immunity': 0.05,
            'Autoinflammatory': 0.03,
            'Bone_Marrow_Failure': 0.01,
            'Complement_Deficiency': 0.01
        },
        "5-12yr": {
            'Antibody_Deficiency': 0.50,   # CVID peak onset
            'Immune_Dysregulation': 0.15,  # APECED, ALPS
            'Autoinflammatory': 0.12,
            'Complement_Deficiency': 0.10,
            'Phagocyte_Defect': 0.08,
            'Combined_ID': 0.03,
            'Innate_Immunity': 0.01,
            'Bone_Marrow_Failure': 0.01
        },
        "12+_years": {
            'Antibody_Deficiency': 0.50,   # Softened from 0.60 - CVID, SAD late onset
            'Complement_Deficiency': 0.18,
            'Immune_Dysregulation': 0.12,
            'Autoinflammatory': 0.10,
            'Phagocyte_Defect': 0.06,
            'Combined_ID': 0.03,
            'Innate_Immunity': 0.008,
            'Bone_Marrow_Failure': 0.002
        }
    },
    
    # Q12: Dysgammaglobulinemia - Direct lab correlation
    "Q12": {
        "Hypogammaglobulinemia": {
            'Antibody_Deficiency': 0.55,   # Softened from 0.70
            'Combined_ID': 0.25,           # Boosted from 0.20
            'Immune_Dysregulation': 0.10,  # Boosted from 0.05
            'Phagocyte_Defect': 0.05,
            'Bone_Marrow_Failure': 0.03,
            'Innate_Immunity': 0.015,
            'Autoinflammatory': 0.003,
            'Complement_Deficiency': 0.002
        },
        "Hypergammaglobulinemia": {
            'Immune_Dysregulation': 0.35,  # ALPS, autoimmune lymphoproliferative
            'Combined_ID': 0.25,           # WAS with high IgE/IgA/IgG, some SCID variants
            'Autoinflammatory': 0.20,
            'Antibody_Deficiency': 0.10,   # CVID with inflammation (reduced)
            'Innate_Immunity': 0.05,
            'Phagocyte_Defect': 0.03,
            'Bone_Marrow_Failure': 0.015,
            'Complement_Deficiency': 0.005
        },
        "Specific_Deficiency": {
            'Antibody_Deficiency': 0.75,   # IgA deficiency, IgG subclass
            'Immune_Dysregulation': 0.12,
            'Combined_ID': 0.05,
            'Autoinflammatory': 0.04,
            'Complement_Deficiency': 0.02,
            'Phagocyte_Defect': 0.01,
            'Innate_Immunity': 0.005,
            'Bone_Marrow_Failure': 0.005
        },
        "Normal": {
            'Phagocyte_Defect': 0.28,
            'Combined_ID': 0.22,           # WAS, some combined IDs have normal total Ig
            'Autoinflammatory': 0.20,
            'Innate_Immunity': 0.15,
            'Complement_Deficiency': 0.10,
            'Immune_Dysregulation': 0.04,
            'Antibody_Deficiency': 0.005,  # Essentially rules out antibody deficiency
            'Bone_Marrow_Failure': 0.005
        }
    },
    
    # Q3: Recurrent infections - Major partitioner
    "Q3": {
        "Yes_multiple_pathogens": {
            'Antibody_Deficiency': 0.45,   # Softened from 0.50
            'Phagocyte_Defect': 0.25,
            'Combined_ID': 0.18,           # Boosted from 0.15
            'Complement_Deficiency': 0.06,
            'Immune_Dysregulation': 0.04,
            'Innate_Immunity': 0.015,
            'Bone_Marrow_Failure': 0.003,
            'Autoinflammatory': 0.002
        },
        "Yes_single_pathogen": {
            'Phagocyte_Defect': 0.35,
            'Antibody_Deficiency': 0.25,
            'Combined_ID': 0.15,
            'Complement_Deficiency': 0.12,
            'Innate_Immunity': 0.08,
            'Immune_Dysregulation': 0.03,
            'Bone_Marrow_Failure': 0.015,
            'Autoinflammatory': 0.005
        },
        "Non_infectious_manifestations": {
            'Autoinflammatory': 0.50,      # Softened from 0.55
            'Immune_Dysregulation': 0.28,
            'Complement_Deficiency': 0.12,
            'Antibody_Deficiency': 0.05,
            'Phagocyte_Defect': 0.03,
            'Combined_ID': 0.015,
            'Innate_Immunity': 0.003,
            'Bone_Marrow_Failure': 0.002
        }
    },
    
    # Q9: Recurrent fever - Autoinflammatory flag
    "Q9": {
        "Yes": {
            'Autoinflammatory': 0.60,
            'Immune_Dysregulation': 0.20,
            'Innate_Immunity': 0.08,
            'Phagocyte_Defect': 0.05,
            'Antibody_Deficiency': 0.04,
            'Combined_ID': 0.02,
            'Bone_Marrow_Failure': 0.005,
            'Complement_Deficiency': 0.005
        },
        "No": {
            'Antibody_Deficiency': 0.40,
            'Phagocyte_Defect': 0.20,
            'Combined_ID': 0.15,
            'Immune_Dysregulation': 0.10,
            'Complement_Deficiency': 0.08,
            'Innate_Immunity': 0.05,
            'Bone_Marrow_Failure': 0.015,
            'Autoinflammatory': 0.005
        }
    },
    
    # Q5: Vaccine reactions - T-cell/Combined deficiency marker
    "Q5": {
        "Yes_BCG": {
            'Combined_ID': 0.55,
            'Innate_Immunity': 0.25,       # MSMD
            'Phagocyte_Defect': 0.10,
            'Immune_Dysregulation': 0.05,
            'Bone_Marrow_Failure': 0.03,
            'Antibody_Deficiency': 0.01,
            'Autoinflammatory': 0.005,
            'Complement_Deficiency': 0.005
        },
        "Yes_Viral": {
            'Combined_ID': 0.70,
            'Innate_Immunity': 0.15,
            'Immune_Dysregulation': 0.08,
            'Bone_Marrow_Failure': 0.04,
            'Antibody_Deficiency': 0.02,
            'Phagocyte_Defect': 0.005,
            'Autoinflammatory': 0.003,
            'Complement_Deficiency': 0.002
        },
        "Yes_Multiple": {
            'Combined_ID': 0.80,
            'Immune_Dysregulation': 0.10,
            'Innate_Immunity': 0.05,
            'Bone_Marrow_Failure': 0.03,
            'Antibody_Deficiency': 0.01,
            'Phagocyte_Defect': 0.005,
            'Autoinflammatory': 0.003,
            'Complement_Deficiency': 0.002
        },
        "No": {
            'Antibody_Deficiency': 0.45,
            'Phagocyte_Defect': 0.20,
            'Autoinflammatory': 0.12,
            'Complement_Deficiency': 0.10,
            'Immune_Dysregulation': 0.08,
            'Innate_Immunity': 0.03,
            'Bone_Marrow_Failure': 0.015,
            'Combined_ID': 0.005
        }
    },
    
    # Q17: Cytopenias - Bone marrow and dysregulation
    "Q17": {
        "Thrombocytopenia": {
            'Combined_ID': 0.40,           # WAS primarily
            'Immune_Dysregulation': 0.30,  # Autoimmune cytopenias (ALPS, CVID with ITP)
            'Bone_Marrow_Failure': 0.15,   # True marrow failure
            'Antibody_Deficiency': 0.08,   # CVID with ITP
            'Phagocyte_Defect': 0.04,
            'Autoinflammatory': 0.02,
            'Innate_Immunity': 0.005,
            'Complement_Deficiency': 0.005
        },
        "Neutropenia": {
            'Phagocyte_Defect': 0.50,      # Cyclic, severe congenital
            'Bone_Marrow_Failure': 0.20,
            'Immune_Dysregulation': 0.15,
            'Combined_ID': 0.08,
            'Antibody_Deficiency': 0.04,
            'Autoinflammatory': 0.02,
            'Innate_Immunity': 0.005,
            'Complement_Deficiency': 0.005
        },
        "Lymphopenia": {
            'Combined_ID': 0.60,
            'Immune_Dysregulation': 0.20,
            'Bone_Marrow_Failure': 0.10,
            'Antibody_Deficiency': 0.05,
            'Phagocyte_Defect': 0.03,
            'Innate_Immunity': 0.01,
            'Autoinflammatory': 0.005,
            'Complement_Deficiency': 0.005
        },
        "Multiple": {
            'Bone_Marrow_Failure': 0.40,
            'Immune_Dysregulation': 0.35,  # HLH, ALPS
            'Combined_ID': 0.15,
            'Antibody_Deficiency': 0.05,
            'Phagocyte_Defect': 0.03,
            'Autoinflammatory': 0.01,
            'Innate_Immunity': 0.005,
            'Complement_Deficiency': 0.005
        },
        "None": {
            'Antibody_Deficiency': 0.40,
            'Autoinflammatory': 0.20,
            'Phagocyte_Defect': 0.15,
            'Complement_Deficiency': 0.12,
            'Innate_Immunity': 0.08,
            'Immune_Dysregulation': 0.03,
            'Combined_ID': 0.015,
            'Bone_Marrow_Failure': 0.005
        }
    },
    
    # Q16: Multiple pathogen types - Combined vs specific defects
    "Q16": {
        "Yes_three_or_more": {
            'Combined_ID': 0.70,           # Multiple pathogens = broad defect
            'Immune_Dysregulation': 0.15,  # HLH, severe CVID
            'Antibody_Deficiency': 0.08,   # Severe CVID
            'Phagocyte_Defect': 0.04,
            'Innate_Immunity': 0.02,
            'Bone_Marrow_Failure': 0.005,
            'Autoinflammatory': 0.003,
            'Complement_Deficiency': 0.002
        },
        "Yes_two_types": {
            'Phagocyte_Defect': 0.45,      # CGD = bacteria + fungi
            'Combined_ID': 0.30,           # Moderate combined deficiency
            'Antibody_Deficiency': 0.15,   # CVID with complications
            'Immune_Dysregulation': 0.05,
            'Innate_Immunity': 0.03,
            'Bone_Marrow_Failure': 0.01,
            'Complement_Deficiency': 0.005,
            'Autoinflammatory': 0.005
        },
        "No_single_type": {
            'Antibody_Deficiency': 0.50,
            'Phagocyte_Defect': 0.20,
            'Innate_Immunity': 0.12,
            'Autoinflammatory': 0.08,
            'Complement_Deficiency': 0.05,
            'Immune_Dysregulation': 0.03,
            'Combined_ID': 0.015,
            'Bone_Marrow_Failure': 0.005
        }
    },
    
    # BATCH 1: Additional questions from original 35
    
    # Q2: Bleeding tendency - Platelet/coagulation issues
    "Q2": {
        "Yes_severe": {
            'Combined_ID': 0.45,           # WAS is here (thrombocytopenia + immunodeficiency)
            'Immune_Dysregulation': 0.25,  # Severe autoimmune cytopenias
            'Bone_Marrow_Failure': 0.15,   # True marrow failure (congenital thrombocytopenia)
            'Complement_Deficiency': 0.08,
            'Antibody_Deficiency': 0.04,
            'Phagocyte_Defect': 0.02,
            'Innate_Immunity': 0.005,
            'Autoinflammatory': 0.005
        },
        "Yes_mild": {
            'Immune_Dysregulation': 0.35,  # CVID with ITP
            'Antibody_Deficiency': 0.25,
            'Combined_ID': 0.15,           # Mild WAS variants
            'Bone_Marrow_Failure': 0.12,
            'Complement_Deficiency': 0.08,
            'Autoinflammatory': 0.03,
            'Phagocyte_Defect': 0.015,
            'Innate_Immunity': 0.005
        },
        "No": {
            'Antibody_Deficiency': 0.48,
            'Phagocyte_Defect': 0.18,
            'Autoinflammatory': 0.12,
            'Innate_Immunity': 0.10,
            'Complement_Deficiency': 0.06,
            'Combined_ID': 0.03,
            'Immune_Dysregulation': 0.02,
            'Bone_Marrow_Failure': 0.01
        }
    },
    
    # Q6: Sex - X-linked disorders
    "Q6": {
        "Male": {
            'Antibody_Deficiency': 0.40,   # XLA more likely
            'Phagocyte_Defect': 0.20,      # X-CGD
            'Combined_ID': 0.15,           # X-SCID
            'Bone_Marrow_Failure': 0.10,   # WAS, XLP
            'Immune_Dysregulation': 0.08,  # IPEX, XLP
            'Innate_Immunity': 0.04,
            'Autoinflammatory': 0.02,
            'Complement_Deficiency': 0.01
        },
        "Female": {
            'Antibody_Deficiency': 0.50,   # CVID, IgAD (no X-linked bias)
            'Autoinflammatory': 0.15,      # Many AR
            'Phagocyte_Defect': 0.10,      # AR forms
            'Immune_Dysregulation': 0.10,
            'Combined_ID': 0.08,           # AR forms
            'Complement_Deficiency': 0.04,
            'Innate_Immunity': 0.02,
            'Bone_Marrow_Failure': 0.01    # Lower (many X-linked)
        }
    },
    
    # Q7: Eczema/rash - T-cell, allergic, HIES
    "Q7": {
        "Yes_severe": {
            'Combined_ID': 0.50,           # WAS, Omenn, HIES (Th17 defect) - boosted
            'Phagocyte_Defect': 0.20,      # True phagocyte (reduced from 0.30)
            'Immune_Dysregulation': 0.15,  # IPEX
            'Bone_Marrow_Failure': 0.08,   # True marrow syndromes
            'Innate_Immunity': 0.04,
            'Antibody_Deficiency': 0.02,
            'Autoinflammatory': 0.005,
            'Complement_Deficiency': 0.005
        },
        "Yes_mild": {
            'Antibody_Deficiency': 0.40,   # CVID, IgAD with atopy
            'Immune_Dysregulation': 0.20,
            'Phagocyte_Defect': 0.15,
            'Combined_ID': 0.12,
            'Innate_Immunity': 0.08,
            'Bone_Marrow_Failure': 0.03,
            'Autoinflammatory': 0.015,
            'Complement_Deficiency': 0.005
        },
        "No": {
            'Antibody_Deficiency': 0.48,
            'Autoinflammatory': 0.15,
            'Complement_Deficiency': 0.12,
            'Innate_Immunity': 0.10,
            'Phagocyte_Defect': 0.08,
            'Immune_Dysregulation': 0.04,
            'Combined_ID': 0.02,
            'Bone_Marrow_Failure': 0.01
        }
    },
    
    # Q11: Lymphoproliferation - Dysregulation, lymphoma risk
    "Q11": {
        "Yes_multiple_sites": {
            'Immune_Dysregulation': 0.50,  # ALPS, autoimmune lymphoproliferative
            'Antibody_Deficiency': 0.25,   # CVID with granulomas
            'Combined_ID': 0.12,           # Some CID
            'Bone_Marrow_Failure': 0.05,   # XLP
            'Innate_Immunity': 0.04,
            'Phagocyte_Defect': 0.02,
            'Autoinflammatory': 0.015,
            'Complement_Deficiency': 0.005
        },
        "Yes_one_site": {
            'Antibody_Deficiency': 0.45,
            'Immune_Dysregulation': 0.25,
            'Combined_ID': 0.12,
            'Innate_Immunity': 0.08,
            'Phagocyte_Defect': 0.05,
            'Bone_Marrow_Failure': 0.03,
            'Autoinflammatory': 0.015,
            'Complement_Deficiency': 0.005
        },
        "No": {
            'Autoinflammatory': 0.30,
            'Antibody_Deficiency': 0.25,
            'Phagocyte_Defect': 0.18,
            'Complement_Deficiency': 0.12,
            'Innate_Immunity': 0.08,
            'Combined_ID': 0.04,
            'Immune_Dysregulation': 0.02,
            'Bone_Marrow_Failure': 0.01
        }
    },
    
    # Q13: Congenital malformations - Syndromic IEI
    "Q13": {
        "Yes_multiple": {
            'Combined_ID': 0.60,           # DiGeorge, other syndromic
            'Bone_Marrow_Failure': 0.20,   # Some syndromes
            'Immune_Dysregulation': 0.10,
            'Phagocyte_Defect': 0.05,
            'Complement_Deficiency': 0.03,
            'Antibody_Deficiency': 0.015,
            'Innate_Immunity': 0.003,
            'Autoinflammatory': 0.002
        },
        "Yes_cardiac": {
            'Combined_ID': 0.70,           # DiGeorge primarily
            'Complement_Deficiency': 0.12,
            'Bone_Marrow_Failure': 0.08,
            'Immune_Dysregulation': 0.05,
            'Antibody_Deficiency': 0.03,
            'Phagocyte_Defect': 0.015,
            'Innate_Immunity': 0.003,
            'Autoinflammatory': 0.002
        },
        "Yes_skeletal": {
            'Combined_ID': 0.45,           # Some SCID variants
            'Bone_Marrow_Failure': 0.25,
            'Phagocyte_Defect': 0.15,      # Some phagocyte syndromes
            'Immune_Dysregulation': 0.08,
            'Antibody_Deficiency': 0.04,
            'Complement_Deficiency': 0.02,
            'Innate_Immunity': 0.005,
            'Autoinflammatory': 0.005
        },
        "Yes_other": {
            'Combined_ID': 0.40,
            'Immune_Dysregulation': 0.20,
            'Bone_Marrow_Failure': 0.15,
            'Phagocyte_Defect': 0.12,
            'Antibody_Deficiency': 0.08,
            'Complement_Deficiency': 0.03,
            'Innate_Immunity': 0.015,
            'Autoinflammatory': 0.005
        },
        "No": {
            'Antibody_Deficiency': 0.50,
            'Autoinflammatory': 0.15,
            'Phagocyte_Defect': 0.12,
            'Innate_Immunity': 0.10,
            'Immune_Dysregulation': 0.08,
            'Complement_Deficiency': 0.03,
            'Combined_ID': 0.015,
            'Bone_Marrow_Failure': 0.005
        }
    },
    
    # Q14: Dysmorphism/peculiar facies - Syndromic
    "Q14": {
        "Yes": {
            'Combined_ID': 0.55,           # DiGeorge, syndromic SCID
            'Bone_Marrow_Failure': 0.20,   # Some syndromes
            'Phagocyte_Defect': 0.12,      # HIES
            'Immune_Dysregulation': 0.08,
            'Antibody_Deficiency': 0.03,
            'Complement_Deficiency': 0.015,
            'Innate_Immunity': 0.003,
            'Autoinflammatory': 0.002
        },
        "No": {
            'Antibody_Deficiency': 0.50,
            'Autoinflammatory': 0.15,
            'Phagocyte_Defect': 0.12,
            'Innate_Immunity': 0.10,
            'Immune_Dysregulation': 0.08,
            'Complement_Deficiency': 0.03,
            'Combined_ID': 0.015,
            'Bone_Marrow_Failure': 0.005
        }
    },
    
    # Q18: Polycythemia/hypercellularity - Inflammatory, allergic
    "Q18": {
        "Yes_eosinophilia": {
            'Combined_ID': 0.45,           # HIES (Th17 defect with eosinophilia)
            'Phagocyte_Defect': 0.25,      # Other phagocyte conditions
            'Immune_Dysregulation': 0.15,  # IPEX, some dysregulation
            'Innate_Immunity': 0.08,
            'Antibody_Deficiency': 0.04,
            'Autoinflammatory': 0.02,
            'Bone_Marrow_Failure': 0.005,
            'Complement_Deficiency': 0.005
        },
        "Yes_leukocytosis": {
            'Autoinflammatory': 0.40,
            'Phagocyte_Defect': 0.25,      # LAD
            'Immune_Dysregulation': 0.15,
            'Innate_Immunity': 0.10,
            'Antibody_Deficiency': 0.05,
            'Combined_ID': 0.03,
            'Bone_Marrow_Failure': 0.015,
            'Complement_Deficiency': 0.005
        },
        "Yes_other": {
            'Autoinflammatory': 0.35,
            'Immune_Dysregulation': 0.25,
            'Phagocyte_Defect': 0.15,
            'Antibody_Deficiency': 0.12,
            'Innate_Immunity': 0.08,
            'Combined_ID': 0.03,
            'Complement_Deficiency': 0.015,
            'Bone_Marrow_Failure': 0.005
        },
        "No": {
            'Antibody_Deficiency': 0.50,
            'Phagocyte_Defect': 0.15,
            'Combined_ID': 0.12,
            'Innate_Immunity': 0.10,
            'Immune_Dysregulation': 0.08,
            'Complement_Deficiency': 0.03,
            'Autoinflammatory': 0.015,
            'Bone_Marrow_Failure': 0.005
        }
    },
    
    # Q19: Silver hair/hypopigmentation - Chédiak-Higashi, Griscelli
    "Q19": {
        "Yes": {
            'Phagocyte_Defect': 0.65,      # Chédiak-Higashi
            'Immune_Dysregulation': 0.20,  # Griscelli type 2 (HLH)
            'Bone_Marrow_Failure': 0.10,
            'Combined_ID': 0.03,
            'Innate_Immunity': 0.015,
            'Antibody_Deficiency': 0.003,
            'Autoinflammatory': 0.001,
            'Complement_Deficiency': 0.001
        },
        "No": {
            'Antibody_Deficiency': 0.47,
            'Autoinflammatory': 0.15,
            'Phagocyte_Defect': 0.12,
            'Innate_Immunity': 0.10,
            'Immune_Dysregulation': 0.09,
            'Combined_ID': 0.04,
            'Complement_Deficiency': 0.02,
            'Bone_Marrow_Failure': 0.01
        }
    },
    
    # Q20: Dystrophic nails - Ectodermal dysplasia, HIES
    "Q20": {
        "Yes": {
            'Combined_ID': 0.50,           # HIES (Th17 defect)
            'Innate_Immunity': 0.25,       # NEMO, ectodermal dysplasia with immunodeficiency
            'Phagocyte_Defect': 0.12,      # Other phagocyte issues
            'Immune_Dysregulation': 0.08,
            'Antibody_Deficiency': 0.03,
            'Bone_Marrow_Failure': 0.015,
            'Autoinflammatory': 0.003,
            'Complement_Deficiency': 0.002
        },
        "No": {
            'Antibody_Deficiency': 0.48,
            'Autoinflammatory': 0.15,
            'Innate_Immunity': 0.10,
            'Combined_ID': 0.10,
            'Immune_Dysregulation': 0.09,
            'Phagocyte_Defect': 0.05,
            'Complement_Deficiency': 0.02,
            'Bone_Marrow_Failure': 0.01
        }
    },
    
    # Q21: Alopecia or vitiligo - Autoimmune/dysregulation
    "Q21": {
        "Yes": {
            'Immune_Dysregulation': 0.60,  # APECED, IPEX
            'Autoinflammatory': 0.20,
            'Antibody_Deficiency': 0.12,   # CVID with autoimmunity
            'Innate_Immunity': 0.04,
            'Combined_ID': 0.02,
            'Phagocyte_Defect': 0.015,
            'Complement_Deficiency': 0.003,
            'Bone_Marrow_Failure': 0.002
        },
        "No": {
            'Antibody_Deficiency': 0.48,
            'Phagocyte_Defect': 0.18,
            'Autoinflammatory': 0.12,
            'Innate_Immunity': 0.10,
            'Combined_ID': 0.06,
            'Complement_Deficiency': 0.03,
            'Immune_Dysregulation': 0.02,
            'Bone_Marrow_Failure': 0.01
        }
    },
    
    # Q4: Infection sites - Grouped by clinical significance
    "Q4": {
        "Sinopulmonary": {
            'Antibody_Deficiency': 0.50,   # Softened from 0.60
            'Complement_Deficiency': 0.18,
            'Phagocyte_Defect': 0.12,
            'Combined_ID': 0.10,
            'Immune_Dysregulation': 0.05,
            'Innate_Immunity': 0.03,
            'Bone_Marrow_Failure': 0.015,
            'Autoinflammatory': 0.005
        },
        "Skin_soft_tissue": {
            'Phagocyte_Defect': 0.48,      # Skin, abscesses, lymph nodes
            'Antibody_Deficiency': 0.20,
            'Innate_Immunity': 0.15,
            'Combined_ID': 0.10,
            'Immune_Dysregulation': 0.04,
            'Bone_Marrow_Failure': 0.02,
            'Autoinflammatory': 0.005,
            'Complement_Deficiency': 0.005
        },
        "Invasive_deep": {
            'Complement_Deficiency': 0.35,  # CNS, blood, bone
            'Combined_ID': 0.28,
            'Antibody_Deficiency': 0.18,
            'Phagocyte_Defect': 0.10,
            'Innate_Immunity': 0.05,
            'Immune_Dysregulation': 0.03,
            'Bone_Marrow_Failure': 0.008,
            'Autoinflammatory': 0.002
        },
        "Disseminated_multiple": {
            'Combined_ID': 0.45,           # Multiple sites = broad defect
            'Antibody_Deficiency': 0.25,
            'Phagocyte_Defect': 0.15,
            'Immune_Dysregulation': 0.08,
            'Complement_Deficiency': 0.04,
            'Innate_Immunity': 0.02,
            'Bone_Marrow_Failure': 0.008,
            'Autoinflammatory': 0.002
        }
    },
    
    # Q10: Chronic mucocutaneous candidiasis
    "Q10": {
        "Yes": {
            'Innate_Immunity': 0.50,       # APECED, STAT1 GOF, CARD9
            'Immune_Dysregulation': 0.25,  # APECED overlap
            'Combined_ID': 0.15,
            'Phagocyte_Defect': 0.05,
            'Antibody_Deficiency': 0.03,
            'Bone_Marrow_Failure': 0.01,
            'Autoinflammatory': 0.005,
            'Complement_Deficiency': 0.005
        },
        "No": {
            'Antibody_Deficiency': 0.45,
            'Phagocyte_Defect': 0.20,
            'Autoinflammatory': 0.12,
            'Combined_ID': 0.10,
            'Immune_Dysregulation': 0.08,
            'Complement_Deficiency': 0.03,
            'Bone_Marrow_Failure': 0.015,
            'Innate_Immunity': 0.005
        }
    },
    
    # Q8: History of Abscesses
    "Q8": {
        "Yes": {
            'Phagocyte_Defect': 0.65,      # CGD, HIES, LAD
            'Innate_Immunity': 0.15,
            'Combined_ID': 0.10,
            'Antibody_Deficiency': 0.05,
            'Immune_Dysregulation': 0.03,
            'Bone_Marrow_Failure': 0.01,
            'Autoinflammatory': 0.005,
            'Complement_Deficiency': 0.005
        },
        "No": {
            'Antibody_Deficiency': 0.50,
            'Autoinflammatory': 0.15,
            'Combined_ID': 0.12,
            'Complement_Deficiency': 0.10,
            'Immune_Dysregulation': 0.08,
            'Innate_Immunity': 0.03,
            'Bone_Marrow_Failure': 0.015,
            'Phagocyte_Defect': 0.005
        }
    },
    
    # BATCH 2: Pathognomonic markers and severity indicators
    
    # Q22: Warts/molluscum - T-cell and NK deficiency
    "Q22": {
        "Yes": {
            'Combined_ID': 0.50,           # WHIM, DOCK8, T-cell defects
            'Innate_Immunity': 0.25,       # GATA2, EVER mutations
            'Immune_Dysregulation': 0.12,
            'Antibody_Deficiency': 0.08,
            'Phagocyte_Defect': 0.03,
            'Bone_Marrow_Failure': 0.015,
            'Autoinflammatory': 0.003,
            'Complement_Deficiency': 0.002
        },
        "No": {
            'Antibody_Deficiency': 0.18,
            'Phagocyte_Defect': 0.17,
            'Autoinflammatory': 0.15,
            'Immune_Dysregulation': 0.14,
            'Combined_ID': 0.13,
            'Innate_Immunity': 0.11,
            'Complement_Deficiency': 0.08,
            'Bone_Marrow_Failure': 0.04
        }
    },
    
    # Q23: Ataxia/telangiectasia - PATHOGNOMONIC for A-T
    "Q23": {
        "Yes_both": {
            'Immune_Dysregulation': 0.95,  # Ataxia-telangiectasia
            'Combined_ID': 0.03,
            'Antibody_Deficiency': 0.015,
            'Phagocyte_Defect': 0.002,
            'Innate_Immunity': 0.001,
            'Autoinflammatory': 0.001,
            'Complement_Deficiency': 0.0005,
            'Bone_Marrow_Failure': 0.0005
        },
        "Yes_telangiectasia": {
            'Immune_Dysregulation': 0.70,  # Likely A-T
            'Combined_ID': 0.15,
            'Antibody_Deficiency': 0.08,
            'Phagocyte_Defect': 0.04,
            'Innate_Immunity': 0.02,
            'Complement_Deficiency': 0.005,
            'Autoinflammatory': 0.003,
            'Bone_Marrow_Failure': 0.002
        },
        "Yes_ataxia": {
            'Immune_Dysregulation': 0.45,  # Could be A-T
            'Combined_ID': 0.25,
            'Antibody_Deficiency': 0.15,
            'Innate_Immunity': 0.08,
            'Phagocyte_Defect': 0.04,
            'Complement_Deficiency': 0.02,
            'Autoinflammatory': 0.005,
            'Bone_Marrow_Failure': 0.005
        },
        "No": {
            'Antibody_Deficiency': 0.18,
            'Phagocyte_Defect': 0.17,
            'Combined_ID': 0.16,
            'Autoinflammatory': 0.14,
            'Immune_Dysregulation': 0.11,
            'Innate_Immunity': 0.12,
            'Complement_Deficiency': 0.08,
            'Bone_Marrow_Failure': 0.04
        }
    },
    
    # Q24: Bronchiectases - Chronic infection/inflammation
    "Q24": {
        "Yes": {
            'Antibody_Deficiency': 0.45,   # CVID, XLA with chronic infection
            'Phagocyte_Defect': 0.25,      # CGD, HIES
            'Combined_ID': 0.15,           # Some combined with chronic infection
            'Innate_Immunity': 0.08,
            'Immune_Dysregulation': 0.04,
            'Complement_Deficiency': 0.02,
            'Autoinflammatory': 0.005,
            'Bone_Marrow_Failure': 0.005
        },
        "No": {
            'Autoinflammatory': 0.18,
            'Combined_ID': 0.17,
            'Immune_Dysregulation': 0.15,
            'Innate_Immunity': 0.14,
            'Phagocyte_Defect': 0.13,
            'Antibody_Deficiency': 0.11,
            'Complement_Deficiency': 0.08,
            'Bone_Marrow_Failure': 0.04
        }
    },
    
    # Q25: Complicated pneumonia - Severe defects
    "Q25": {
        "Yes": {
            'Phagocyte_Defect': 0.40,      # CGD, severe neutrophil defects
            'Combined_ID': 0.25,           # Severe T-cell defects
            'Antibody_Deficiency': 0.18,   # Severe hypogammaglobulinemia
            'Complement_Deficiency': 0.10,
            'Immune_Dysregulation': 0.04,
            'Innate_Immunity': 0.02,
            'Bone_Marrow_Failure': 0.008,
            'Autoinflammatory': 0.002
        },
        "No": {
            'Antibody_Deficiency': 0.18,
            'Autoinflammatory': 0.17,
            'Immune_Dysregulation': 0.15,
            'Innate_Immunity': 0.14,
            'Combined_ID': 0.13,
            'Phagocyte_Defect': 0.11,
            'Complement_Deficiency': 0.08,
            'Bone_Marrow_Failure': 0.04
        }
    },
    
    # Q26: Arthritis - Autoinflammatory and dysregulation
    "Q26": {
        "Yes": {
            'Autoinflammatory': 0.45,
            'Immune_Dysregulation': 0.25,
            'Antibody_Deficiency': 0.15,   # CVID with arthritis
            'Complement_Deficiency': 0.08,
            'Combined_ID': 0.04,
            'Innate_Immunity': 0.02,
            'Phagocyte_Defect': 0.008,
            'Bone_Marrow_Failure': 0.002
        },
        "No": {
            'Antibody_Deficiency': 0.18,
            'Phagocyte_Defect': 0.17,
            'Combined_ID': 0.16,
            'Innate_Immunity': 0.14,
            'Immune_Dysregulation': 0.12,
            'Autoinflammatory': 0.11,
            'Complement_Deficiency': 0.08,
            'Bone_Marrow_Failure': 0.04
        }
    },
    
    # Q27: Autoimmunity - Dysregulation marker
    "Q27": {
        "Yes_systemic": {
            'Immune_Dysregulation': 0.50,  # ALPS, CTLA4, LRBA
            'Complement_Deficiency': 0.20,  # Lupus-like
            'Antibody_Deficiency': 0.15,   # CVID with autoimmunity
            'Autoinflammatory': 0.08,
            'Combined_ID': 0.04,
            'Innate_Immunity': 0.02,
            'Phagocyte_Defect': 0.008,
            'Bone_Marrow_Failure': 0.002
        },
        "Yes_organ_specific": {
            'Immune_Dysregulation': 0.60,  # APECED, IPEX
            'Autoinflammatory': 0.15,
            'Antibody_Deficiency': 0.12,
            'Combined_ID': 0.08,
            'Complement_Deficiency': 0.03,
            'Innate_Immunity': 0.015,
            'Phagocyte_Defect': 0.003,
            'Bone_Marrow_Failure': 0.002
        },
        "No": {
            'Antibody_Deficiency': 0.18,
            'Phagocyte_Defect': 0.17,
            'Combined_ID': 0.16,
            'Innate_Immunity': 0.14,
            'Autoinflammatory': 0.13,
            'Immune_Dysregulation': 0.10,
            'Complement_Deficiency': 0.08,
            'Bone_Marrow_Failure': 0.04
        }
    },
    
    # Q28: HLH - Immune dysregulation hallmark
    "Q28": {
        "Yes": {
            'Immune_Dysregulation': 0.75,  # Primary HLH, FHL
            'Combined_ID': 0.15,           # Some T-cell defects
            'Bone_Marrow_Failure': 0.05,   # XLP
            'Innate_Immunity': 0.03,
            'Antibody_Deficiency': 0.015,
            'Phagocyte_Defect': 0.003,
            'Autoinflammatory': 0.001,
            'Complement_Deficiency': 0.001
        },
        "No": {
            'Antibody_Deficiency': 0.18,
            'Phagocyte_Defect': 0.17,
            'Combined_ID': 0.16,
            'Innate_Immunity': 0.14,
            'Autoinflammatory': 0.13,
            'Immune_Dysregulation': 0.10,
            'Complement_Deficiency': 0.08,
            'Bone_Marrow_Failure': 0.04
        }
    },
    
    # Q29: ICU admission - Severity marker
    "Q29": {
        "Yes": {
            'Combined_ID': 0.40,           # Severe presentations
            'Phagocyte_Defect': 0.25,      # Severe infections
            'Immune_Dysregulation': 0.15,  # HLH, severe autoimmune
            'Antibody_Deficiency': 0.10,
            'Complement_Deficiency': 0.05,
            'Innate_Immunity': 0.03,
            'Bone_Marrow_Failure': 0.015,
            'Autoinflammatory': 0.005
        },
        "No": {
            'Antibody_Deficiency': 0.20,
            'Autoinflammatory': 0.17,
            'Innate_Immunity': 0.15,
            'Immune_Dysregulation': 0.14,
            'Phagocyte_Defect': 0.13,
            'Combined_ID': 0.10,
            'Complement_Deficiency': 0.08,
            'Bone_Marrow_Failure': 0.03
        }
    },
    
    # Q30: Neurologic deficit - Syndromic IEI
    "Q30": {
        "Yes": {
            'Immune_Dysregulation': 0.40,  # A-T, other neurologic IEI
            'Combined_ID': 0.30,           # Some syndromic SCID
            'Innate_Immunity': 0.15,       # NEMO with neurologic features
            'Bone_Marrow_Failure': 0.08,
            'Phagocyte_Defect': 0.04,
            'Antibody_Deficiency': 0.02,
            'Complement_Deficiency': 0.008,
            'Autoinflammatory': 0.002
        },
        "No": {
            'Antibody_Deficiency': 0.19,
            'Phagocyte_Defect': 0.17,
            'Combined_ID': 0.15,
            'Autoinflammatory': 0.14,
            'Innate_Immunity': 0.13,
            'Immune_Dysregulation': 0.10,
            'Complement_Deficiency': 0.08,
            'Bone_Marrow_Failure': 0.04
        }
    },
    
    # Q31: IBD - Dysregulation and some combined
    "Q31": {
        "Yes": {
            'Immune_Dysregulation': 0.50,  # IPEX, IL-10 pathway
            'Combined_ID': 0.25,           # Some T-cell defects
            'Antibody_Deficiency': 0.12,   # CVID with IBD
            'Autoinflammatory': 0.08,
            'Innate_Immunity': 0.03,
            'Phagocyte_Defect': 0.015,
            'Complement_Deficiency': 0.003,
            'Bone_Marrow_Failure': 0.002
        },
        "No": {
            'Antibody_Deficiency': 0.18,
            'Phagocyte_Defect': 0.17,
            'Combined_ID': 0.15,
            'Autoinflammatory': 0.14,
            'Innate_Immunity': 0.14,
            'Immune_Dysregulation': 0.10,
            'Complement_Deficiency': 0.08,
            'Bone_Marrow_Failure': 0.04
        }
    },
    
    # Q32: Anhidrotic ectodermal dysplasia - NEMO/IKBKG
    "Q32": {
        "Yes": {
            'Innate_Immunity': 0.85,       # NEMO deficiency
            'Combined_ID': 0.10,
            'Phagocyte_Defect': 0.03,
            'Immune_Dysregulation': 0.015,
            'Antibody_Deficiency': 0.003,
            'Autoinflammatory': 0.001,
            'Complement_Deficiency': 0.0005,
            'Bone_Marrow_Failure': 0.0005
        },
        "No": {
            'Antibody_Deficiency': 0.18,
            'Phagocyte_Defect': 0.17,
            'Combined_ID': 0.16,
            'Autoinflammatory': 0.14,
            'Immune_Dysregulation': 0.13,
            'Innate_Immunity': 0.10,
            'Complement_Deficiency': 0.08,
            'Bone_Marrow_Failure': 0.04
        }
    },
    
    # Q33: Absent/hypoplastic thymus - SEVERE marker for Combined ID
    "Q33": {
        "Yes": {
            'Combined_ID': 0.90,           # SCID, DiGeorge
            'Immune_Dysregulation': 0.05,
            'Bone_Marrow_Failure': 0.03,
            'Antibody_Deficiency': 0.015,
            'Phagocyte_Defect': 0.003,
            'Innate_Immunity': 0.001,
            'Autoinflammatory': 0.0005,
            'Complement_Deficiency': 0.0005
        },
        "No": {
            'Antibody_Deficiency': 0.19,
            'Phagocyte_Defect': 0.17,
            'Autoinflammatory': 0.14,
            'Immune_Dysregulation': 0.14,
            'Innate_Immunity': 0.13,
            'Combined_ID': 0.11,
            'Complement_Deficiency': 0.08,
            'Bone_Marrow_Failure': 0.04
        }
    },
    
    # Q34: Severe atopy - T-cell and dysregulation
    "Q34": {
        "Yes": {
            'Combined_ID': 0.40,           # Omenn, DOCK8, HIES
            'Immune_Dysregulation': 0.25,  # IPEX
            'Phagocyte_Defect': 0.15,
            'Antibody_Deficiency': 0.12,
            'Innate_Immunity': 0.05,
            'Bone_Marrow_Failure': 0.02,
            'Autoinflammatory': 0.008,
            'Complement_Deficiency': 0.002
        },
        "No": {
            'Antibody_Deficiency': 0.18,
            'Phagocyte_Defect': 0.17,
            'Autoinflammatory': 0.15,
            'Innate_Immunity': 0.14,
            'Immune_Dysregulation': 0.13,
            'Combined_ID': 0.11,
            'Complement_Deficiency': 0.08,
            'Bone_Marrow_Failure': 0.04
        }
    },
    
    # Q35: Lymphoma/malignancy - Dysregulation and some combined
    "Q35": {
        "Yes": {
            'Immune_Dysregulation': 0.50,  # A-T, ALPS, XLP
            'Combined_ID': 0.25,           # Some T-cell defects
            'Antibody_Deficiency': 0.12,   # CVID with lymphoma risk
            'Bone_Marrow_Failure': 0.08,   # XLP
            'Innate_Immunity': 0.03,
            'Phagocyte_Defect': 0.015,
            'Complement_Deficiency': 0.003,
            'Autoinflammatory': 0.002
        },
        "No": {
            'Antibody_Deficiency': 0.18,
            'Phagocyte_Defect': 0.17,
            'Combined_ID': 0.15,
            'Autoinflammatory': 0.14,
            'Innate_Immunity': 0.14,
            'Immune_Dysregulation': 0.10,
            'Complement_Deficiency': 0.08,
            'Bone_Marrow_Failure': 0.04
        }
    }
}

# ============================================================================
# SHANNON ENTROPY CALCULATIONS
# ============================================================================

def calculate_entropy(probabilities: Dict[str, float]) -> float:
    """
    Calculate Shannon entropy: H(X) = -Σ p(x) * log2(p(x))
    
    Args:
        probabilities: Dictionary of {category: probability}
    
    Returns:
        Entropy in bits
    """
    entropy = 0.0
    for prob in probabilities.values():
        if prob > 0:  # Avoid log(0)
            entropy -= prob * np.log2(prob)
    return entropy

def calculate_information_gain(
    current_probs: Dict[str, float],
    question_id: str,
    conditional_probs: Dict[str, Dict[str, Dict[str, float]]]
) -> float:
    """
    Calculate expected information gain for asking a question
    
    IG(Q) = H(current) - Σ P(answer) * H(posterior | answer)
    
    Args:
        current_probs: Current probability distribution over categories
        question_id: ID of question to evaluate
        conditional_probs: Conditional probability tables
    
    Returns:
        Information gain in bits
    """
    current_entropy = calculate_entropy(current_probs)
    
    if question_id not in conditional_probs:
        return 0.0
    
    # Calculate expected posterior entropy
    expected_posterior_entropy = 0.0
    
    for answer, answer_probs in conditional_probs[question_id].items():
        # P(answer) = Σ P(answer|category) * P(category)
        p_answer = sum(
            answer_probs.get(cat, 0) * current_probs.get(cat, 0)
            for cat in IEI_CATEGORIES
        )
        
        if p_answer > 0:
            # Calculate posterior: P(category|answer)
            posterior = {}
            for cat in IEI_CATEGORIES:
                # Bayes: P(cat|ans) = P(ans|cat) * P(cat) / P(ans)
                posterior[cat] = (
                    answer_probs.get(cat, 0) * current_probs.get(cat, 0) / p_answer
                )
            
            # Weight by probability of this answer
            expected_posterior_entropy += p_answer * calculate_entropy(posterior)
    
    return current_entropy - expected_posterior_entropy

# ============================================================================
# PATTERN MATCHING
# ============================================================================

def check_pathognomonic_patterns(
    answers: Dict[str, str],
    patterns: List[PathognomicPattern]
) -> Optional[Tuple[PathognomicPattern, float]]:
    """
    Check if current answers match any pathognomonic patterns
    
    Args:
        answers: Dictionary of {question_id: answer}
        patterns: List of pathognomonic patterns to check
    
    Returns:
        Tuple of (matched_pattern, confidence) or None
    """
    for pattern in patterns:
        match = True
        for trigger in pattern.triggers:
            # Parse trigger format: "Q23:Yes" or "Q7:Yes"
            parts = trigger.split(':')
            if len(parts) != 2:
                continue
            
            q_id, expected_answer = parts
            
            # Handle compound triggers like "Ataxia+Telangiectasia"
            if '+' in expected_answer:
                # For Q23, we need BOTH ataxia and telangiectasia
                # This requires special answer format handling
                actual = answers.get(q_id, '')
                if expected_answer not in actual and actual != "Yes":
                    match = False
                    break
            else:
                if answers.get(q_id) != expected_answer:
                    match = False
                    break
        
        if match:
            return (pattern, pattern.probability)
    
    return None

# ============================================================================
# BAYESIAN UPDATE ENGINE
# ============================================================================

def update_probabilities_bayesian(
    current_probs: Dict[str, float],
    question_id: str,
    answer: str,
    conditional_probs: Dict[str, Dict[str, Dict[str, float]]]
) -> Dict[str, float]:
    """
    Update probability distribution using Bayes' theorem
    
    P(category | answer) = P(answer | category) * P(category) / P(answer)
    
    Includes probability floor (0.5%) to prevent premature elimination of categories.
    This prevents overconfident early convergence and keeps differential alive.
    
    Args:
        current_probs: Current probability distribution
        question_id: Question that was answered
        answer: The answer given
        conditional_probs: Conditional probability tables
    
    Returns:
        Updated probability distribution
    """
    if question_id not in conditional_probs:
        return current_probs
    
    if answer not in conditional_probs[question_id]:
        return current_probs
    
    answer_probs = conditional_probs[question_id][answer]
    
    # Calculate P(answer) = Σ P(answer|category) * P(category)
    p_answer = sum(
        answer_probs.get(cat, 0) * current_probs.get(cat, 0)
        for cat in IEI_CATEGORIES
    )
    
    if p_answer == 0:
        return current_probs
    
    # Calculate posterior for each category
    updated = {}
    for cat in IEI_CATEGORIES:
        # Bayes' theorem
        posterior = (
            answer_probs.get(cat, 0) * current_probs.get(cat, 0) / p_answer
        )
        # Apply floor: minimum 0.5% to keep all categories alive
        updated[cat] = max(posterior, 0.005)
    
    # Re-normalize to ensure sum = 1.0
    total = sum(updated.values())
    if total > 0:
        updated = {cat: prob / total for cat, prob in updated.items()}
    
    return updated

# ============================================================================
# WEIGHTED QUESTION SELECTION
# ============================================================================

def select_next_question(
    current_probs: Dict[str, float],
    available_questions: List[str],
    questions_dict: Dict[str, Question],
    conditional_probs: Dict[str, Dict[str, Dict[str, float]]]
) -> str:
    """
    Select next question using weighted information gain
    
    Weighted_IG = base_IG * relevance_weight * nodal_weight
    
    Relevance weight increases for questions targeting leading categories
    
    Args:
        current_probs: Current probability distribution
        available_questions: List of question IDs not yet asked
        questions_dict: Question definitions
        conditional_probs: Conditional probability tables
    
    Returns:
        Question ID with highest weighted information gain
    """
    # Find leading categories (top 3 by probability)
    sorted_cats = sorted(current_probs.items(), key=lambda x: x[1], reverse=True)
    leading_categories = {cat for cat, _ in sorted_cats[:3]}
    
    best_question = None
    best_weighted_ig = -1.0
    
    for q_id in available_questions:
        if q_id not in questions_dict:
            continue
        
        question = questions_dict[q_id]
        
        # Calculate base information gain
        base_ig = calculate_information_gain(current_probs, q_id, conditional_probs)
        
        # Calculate relevance weight
        # Questions that discriminate well among leading categories get bonus
        relevance_weight = calculate_relevance_weight(
            q_id, leading_categories, conditional_probs
        )
        
        # Apply nodal weight if applicable
        nodal_weight = question.nodal_weight if question.is_nodal else 1.0
        
        # Combined weighted information gain
        weighted_ig = base_ig * relevance_weight * nodal_weight
        
        if weighted_ig > best_weighted_ig:
            best_weighted_ig = weighted_ig
            best_question = q_id
    
    return best_question

def calculate_relevance_weight(
    question_id: str,
    leading_categories: set,
    conditional_probs: Dict[str, Dict[str, Dict[str, float]]]
) -> float:
    """
    Calculate how relevant a question is to discriminating among leading categories
    
    Higher weight if the question has different probability distributions
    for the leading categories
    
    Args:
        question_id: Question to evaluate
        leading_categories: Set of currently leading category names
        conditional_probs: Conditional probability tables
    
    Returns:
        Relevance weight (1.0 to 2.0)
    """
    if question_id not in conditional_probs:
        return 1.0
    
    # Calculate variance in probabilities across leading categories
    # for each possible answer
    max_variance = 0.0
    
    for answer, answer_probs in conditional_probs[question_id].items():
        leading_probs = [
            answer_probs.get(cat, 0) for cat in leading_categories
        ]
        
        if leading_probs:
            variance = np.var(leading_probs)
            max_variance = max(max_variance, variance)
    
    # Convert variance to weight (1.0 to 2.0 range)
    # High variance = high discrimination = high weight
    relevance_weight = 1.0 + min(max_variance * 10, 1.0)
    
    return relevance_weight

# ============================================================================
# MAIN DIAGNOSTIC ENGINE
# ============================================================================

class IEIDiagnosticEngine:
    """
    Main diagnostic engine combining pattern recognition and Shannon entropy
    """
    
    def __init__(self):
        self.current_probs = initialize_prior_probabilities()
        self.answers = {}
        self.asked_questions = []
        self.pathognomonic_match = None
        
    def process_answer(self, question_id: str, answer: str) -> Dict:
        """
        Process an answer and update probabilities
        
        Returns diagnostic state and next question
        """
        # Store answer
        self.answers[question_id] = answer
        self.asked_questions.append(question_id)
        
        # MINIMUM QUESTIONS THRESHOLD: Must ask at least 15 questions
        # Check this BEFORE pattern detection to ensure comprehensive assessment
        min_questions_met = len(self.asked_questions) >= 15
        
        # Check for pathognomonic patterns ONLY if minimum questions met
        if min_questions_met:
            pattern_match = check_pathognomonic_patterns(
                self.answers, 
                PATHOGNOMONIC_PATTERNS
            )
            
            if pattern_match:
                pattern, confidence = pattern_match
                self.pathognomonic_match = pattern
                
                # If very high confidence, suggest confirmation questions
                if confidence >= 0.90:
                    return {
                        'status': 'pattern_detected',
                        'suspected_diagnosis': pattern.name,
                        'confidence': confidence,
                        'category': pattern.category,
                        'confirm_with': pattern.confirm_with,
                        'current_probabilities': self.current_probs
                    }
        
        # Update probabilities using Bayes' theorem
        self.current_probs = update_probabilities_bayesian(
            self.current_probs,
            question_id,
            answer,
            CONDITIONAL_PROBABILITIES
        )
        
        # Check stopping criteria
        max_prob = max(self.current_probs.values())
        current_entropy = calculate_entropy(self.current_probs)
        
        # Stop if VERY high confidence OR very low entropy
        # BUT only after asking minimum questions
        # Raised threshold to 99.5% to prevent overconfidence
        if min_questions_met and (max_prob >= 0.995 or current_entropy < 0.15):
            return {
                'status': 'diagnosis_reached',
                'top_diagnosis': self.get_top_diagnoses(n=1)[0],
                'differential': self.get_top_diagnoses(n=8),  # Show all 8 categories
                'confidence': max_prob,
                'entropy': current_entropy,
                'current_probabilities': self.current_probs
            }
        
        # Select next question
        available = [
            q_id for q_id in QUESTIONS.keys() 
            if q_id not in self.asked_questions
        ]
        
        if not available:
            # No more questions, return top differential
            return {
                'status': 'questions_exhausted',
                'differential': self.get_top_diagnoses(n=8),  # Show all 8 categories
                'entropy': current_entropy,
                'current_probabilities': self.current_probs
            }
        
        next_q = select_next_question(
            self.current_probs,
            available,
            QUESTIONS,
            CONDITIONAL_PROBABILITIES
        )
        
        return {
            'status': 'continue',
            'next_question': next_q,
            'question_text': QUESTIONS[next_q].text,
            'answer_options': QUESTIONS[next_q].answer_options,
            'current_entropy': current_entropy,
            'top_categories': self.get_top_diagnoses(n=3),
            'current_probabilities': self.current_probs
        }
    
    def get_top_diagnoses(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top N diagnoses by probability"""
        sorted_probs = sorted(
            self.current_probs.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_probs[:n]
    
    def reset(self):
        """Reset the engine for a new patient"""
        self.current_probs = initialize_prior_probabilities()
        self.answers = {}
        self.asked_questions = []
        self.pathognomonic_match = None

# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

if __name__ == "__main__":
    # Quick test of the engine
    print("=" * 70)
    print("IEI DIAGNOSTIC ENGINE - CORE MATHEMATICAL FRAMEWORK")
    print("=" * 70)
    
    # Initialize engine
    engine = IEIDiagnosticEngine()
    
    print("\nInitial probability distribution:")
    for cat, prob in engine.get_top_diagnoses(n=8):
        print(f"  {cat:.<30} {prob:.3f}")
    
    print(f"\nInitial entropy: {calculate_entropy(engine.current_probs):.3f} bits")
    
    # Simulate answering Q15 with "Fungi"
    print("\n" + "=" * 70)
    print("SIMULATION: Patient with fungal infections")
    print("=" * 70)
    
    result = engine.process_answer("Q15", "Fungi")
    
    print("\nAfter Q15 (Fungi):")
    print(f"Status: {result['status']}")
    print(f"Current entropy: {result.get('current_entropy', 'N/A'):.3f} bits")
    print("\nTop categories:")
    for cat, prob in result['top_categories']:
        print(f"  {cat:.<30} {prob:.3f}")
    
    if result['status'] == 'continue':
        print(f"\nNext question: {result['next_question']}")
        print(f"  {result['question_text']}")
        print(f"  Options: {', '.join(result['answer_options'])}")
    
    print("\n" + "=" * 70)
    print("Core engine ready for Streamlit integration!")
    print("=" * 70)
