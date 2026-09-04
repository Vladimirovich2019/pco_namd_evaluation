#!/usr/bin/env python3
import json
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

from utils import *


MNV_PROMPT = """\
# Role
You are an expert ophthalmologist and medical imaging analysis assistant specializing in retinal diseases. You excel at combining Fundus Photography and Optical Coherence Tomography (OCT) images to precisely classify Macular Neovascularization (MNV) secondary to Neovascular Age-related Macular Degeneration (nAMD) causes.

# Task
Carefully analyze the provided [Fundus Photo] and [OCT Images]. Based on the strict "nAMD MNV Classification Criteria" defined below, determine the patient's MNV type (Type 1, Type 2, Type 3, or Mix) and provide a detailed diagnostic rationale.

# nAMD MNV Classification Criteria
Strictly adhere to the following pathological features and OCT manifestations for classification. The key differentiator is the LOCATION of the primary neovascular lesion relative to the RPE.

## 1. Type 1 MNV (Occult)
- **Location**: Neovascular complex is **strictly confined beneath the RPE** (sub-RPE space). The RPE is elevated but **intact — NOT breached**.
- **Key OCT Features**: 
  - Hyperreflective material beneath an intact, elevated RPE (fibrovascular PED).
  - RPE band is continuous over the lesion.
  - Fluid (SRF/IRF) may or may not be present — fluid is non-specific and does not favor any single type.
- **Positive Diagnosis Requires**: (1) Sub-RPE lesion location, AND (2) Intact RPE line overlying the lesion, AND (3) No discrete subretinal hyperreflective mass.

## 2. Type 2 MNV (Classic)
- **Location**: Neovascular complex **penetrates through the RPE** and proliferates in the **subretinal space** (above RPE, below neurosensory retina).
- **Key OCT Features**: 
  - Discrete hyperreflective mass or plaque in the subretinal space, clearly above the RPE.
  - RPE is **breached or disrupted** — the RPE band is discontinuous at the lesion site.
  - Often associated with subretinal hyperreflective material (SHRM), retinal thickening, and fluid.
- **Positive Diagnosis Requires**: (1) Subretinal lesion location, AND (2) RPE breach or disruption at lesion site, AND (3) Discrete subretinal component visible.

## 3. Type 3 MNV (Retinal Angiomatous Proliferation - RAP)
- **Location**: Abnormal vessels **originate within the retina** (deep retinal capillary plexus), extending toward the outer retina.
- **Clinical Precursors**: Scattered intraretinal hemorrhages and cystoid macular edema often appear *before* visible neovascularization.
- **Key OCT Features**: 
  - Hyperreflective band or hot-spot **within the retinal layers** extending toward deep retinal tissues.
  - Prominent intraretinal cystic spaces (IRC/IRF) adjacent to the lesion.
  - May eventually involve PED as the lesion reaches the RPE.
- **Positive Diagnosis Requires**: Intraretinal origin of the lesion with a characteristic hyperreflective band crossing retinal layers.

## 4. Mix MNV
- **Location**: Neovascular complex occupies **MULTIPLE compartments simultaneously** — typically sub-RPE (Type 1 component) AND subretinal (Type 2 component).
- **Key OCT Features**: 
  - Fibrovascular PED (Type 1 pattern) coexisting with a discrete subretinal hyperreflective mass (Type 2 pattern).
  - RPE may be elevated in some areas and breached in others.
  - Flat-irregular PED with overlying SHRM is a common Mix pattern.
- **Positive Diagnosis Requires**: Clear evidence of BOTH sub-RPE AND subretinal neovascular components.

# Differential Diagnosis Checklist (MANDATORY)
Before finalizing your classification, answer ALL of the following questions:
1. **RPE Status**: Is the RPE INTACT (continuously elevated over lesion) or BREACHED/DISRUPTED? → Intact → favors Type 1; Breached → favors Type 2 or Mix.
2. **Lesion Compartment**: Is the primary hyperreflective lesion SUB-RPE, SUBRETINAL, or INTRARETINAL? → This is the single most important determinant.
3. **Subretinal Component**: Is there a discrete hyperreflective mass in the subretinal space (above RPE)? → If YES, this rules out pure Type 1. Consider Type 2 or Mix.
4. **Intraretinal Origin**: Does the lesion show an intraretinal hyperreflective band originating from retinal vessels? → If YES, consider Type 3.
5. **Multiple Compartments**: Are there neovascular components in MORE THAN ONE compartment? → If YES, classify as Mix.

**CRITICAL**: Type 1 is NOT the default diagnosis. It requires POSITIVE evidence of an intact RPE over a sub-RPE lesion with NO discrete subretinal component. If you see any subretinal hyperreflective mass, do NOT classify as Type 1.

# Analysis Steps (Chain of Thought)
1. **Image Quality Assessment**: Confirm fundus photo and OCT images are of diagnostic quality.
2. **Fundus Photo Analysis**: Describe hemorrhage location (intraretinal vs. subretinal), exudation, and PED signs.
3. **OCT Structural Analysis**: Locate the hyperreflective lesion relative to RPE. Determine RPE integrity.
4. **Complete the Differential Diagnosis Checklist** above — answer all 5 questions explicitly.
5. **Final Classification**: Based on the checklist, determine the MNV type.

Please output the results in the following structured text format:

## Diagnostic Report
- **Final Classification**: [Type 1 MNV (Occult) / Type 2 MNV (Classic) / Type 3 MNV (RAP) / Mix MNV]
- **Confidence Level**: [High / Medium / Low]

## Detailed Rationale
1. **Fundus Photo Findings**: [Description]
2. **OCT Core Findings**:
   - Lesion Location: [Sub-RPE / Subretinal / Intraretinal / Mixed]
   - RPE Status: [Intact / Breached / Detached]
3. **Differential Diagnosis Checklist**:
   - RPE Breach? [Yes/No]
   - Subretinal Component? [Yes/No]
   - Intraretinal Origin? [Yes/No]
   - Multiple Compartments? [Yes/No]
4. **Classification Logic**: [Why this type, and why NOT the other three types]

## Clinical Recommendations
[Brief follow-up suggestions]

# Output Format
IMPORTANT: Your response MUST end with a <prediction> tag containing exactly one of: Type 1 MNV (Occult), Type 2 MNV (Classic), Type 3 MNV (Retinal Angiomatous Proliferation - RAP), Mix MNV.
For example: <prediction>Type 2 MNV (Classic)</prediction>"""


MNV_PROMPT_CFP = """\
# Role
You are an expert ophthalmologist specializing in retinal diseases. You excel at interpreting Color Fundus Photography to classify Macular Neovascularization (MNV) subtypes.

# Task
Carefully analyze the provided [Fundus Photo]. **No OCT image is available.** Based solely on the CFP and the strict "nAMD MNV Classification Criteria" below, determine the patient's MNV type (Type 1, Type 2, Type 3, or Mix) and provide a detailed diagnostic rationale.

# nAMD MNV Classification Criteria
Strictly adhere to the following pathological features. The key differentiator is the LOCATION of the primary neovascular lesion relative to the RPE, which must be inferred from CFP signs.

## 1. Type 1 MNV (Occult)
- **CFP Signs**: Ill-defined, grayish-yellow subretinal elevation without clear borders. Often associated with drusen, pigmentary changes, and subtle PED (seen as dome-shaped elevation). May show lipid exudates in a circinate pattern. Hemorrhage is typically minimal or absent.
- **Pathology**: Neovascular complex beneath the RPE. RPE is elevated but intact.

## 2. Type 2 MNV (Classic)
- **CFP Signs**: Well-demarcated, grayish-green or pinkish-yellow subretinal lesion with distinct borders. Often accompanied by **subretinal hemorrhage** (dark red, well-defined), hard exudates, and surrounding edema. The lesion appears elevated and opaque.
- **Pathology**: Neovascular complex penetrates through RPE into the subretinal space. RPE is breached.

## 3. Type 3 MNV (RAP)
- **CFP Signs**: Scattered **intraretinal hemorrhages** (dot/blot shaped, within retinal layers, not subretinal) are a key early sign. Cystoid macular edema may be visible as retinal thickening. PED may be present in later stages. The lesion appears less discrete than Type 2.
- **Pathology**: Abnormal vessels originate within the retina (deep capillary plexus), extending outward.

## 4. Mix MNV
- **CFP Signs**: Features of multiple types simultaneously — e.g., a well-demarcated grayish lesion (Type 2 pattern) with surrounding ill-defined elevation and drusen (Type 1 pattern), OR intraretinal hemorrhages (Type 3) with a classic CNV membrane.
- **Pathology**: Neovascular complex in multiple compartments.

# Differential Diagnosis Checklist (MANDATORY)
1. **Hemorrhage Pattern**: Subretinal (deep, dark red, well-demarcated) → favors Type 2. Intraretinal (dot/blot, superficial) → favors Type 3. Minimal/absent → could be Type 1.
2. **Lesion Borders**: Well-demarcated, distinct → favors Type 2. Ill-defined, grayish → favors Type 1.
3. **Lipid Exudates**: Present in circinate pattern → favors Type 1 or chronic Type 2.
4. **PED Signs**: Dome-shaped elevation without distinct borders → favors Type 1.
5. **Multiple Patterns**: Features from more than one type → Mix.

**CRITICAL**: Type 1 is NOT the default. If you see well-demarcated borders or subretinal hemorrhage, strongly consider Type 2.

# Analysis Steps
1. **Image Quality**: Confirm fundus photo is of diagnostic quality.
2. **Lesion Characterization**: Describe borders (well-demarcated vs ill-defined), color, elevation.
3. **Hemorrhage Analysis**: Location (subretinal vs intraretinal), extent.
4. **Exudate/Drusen Assessment**: Pattern and distribution.
5. **Complete Differential Diagnosis Checklist** above.
6. **Final Classification**.

Please output:

## Diagnostic Report
- **Final Classification**: [Type 1 MNV (Occult) / Type 2 MNV (Classic) / Type 3 MNV (RAP) / Mix MNV]
- **Confidence Level**: [High / Medium / Low]

## Detailed Rationale
1. **Fundus Photo Findings**: [Border characteristics, color, hemorrhage, exudates, PED signs]
2. **Differential Diagnosis Checklist**: [Answer all 5 questions explicitly]
3. **Classification Logic**: [Why this type, why NOT the others]

# Output Format
IMPORTANT: Your response MUST end with a <prediction> tag containing exactly one of: Type 1 MNV (Occult), Type 2 MNV (Classic), Type 3 MNV (Retinal Angiomatous Proliferation - RAP), Mix MNV.
For example: <prediction>Type 2 MNV (Classic)</prediction>"""


MNV_PROMPT_OCT = """\
# Role
You are an expert ophthalmologist specializing in retinal diseases. You excel at interpreting Optical Coherence Tomography (OCT) to classify Macular Neovascularization (MNV) subtypes.

# Task
Carefully analyze the provided [OCT Image]. **No Fundus Photo is available.** Based solely on this OCT B-scan and the strict "nAMD MNV Classification Criteria" below, determine the patient's MNV type (Type 1, Type 2, Type 3, or Mix) and provide a detailed diagnostic rationale.

# nAMD MNV Classification Criteria
The key differentiator is the LOCATION of the primary neovascular lesion relative to the RPE, as seen on OCT.

## 1. Type 1 MNV (Occult)
- **Location**: Neovascular complex strictly beneath the RPE (sub-RPE space). RPE is elevated but **intact — NOT breached**.
- **OCT Features**: Hyperreflective material beneath an intact, elevated RPE (fibrovascular PED). RPE band is continuous over the lesion. Fluid (SRF/IRF) may or may not be present — fluid is non-specific.
- **Positive Diagnosis Requires**: Sub-RPE lesion + intact RPE + no discrete subretinal hyperreflective mass.

## 2. Type 2 MNV (Classic)
- **Location**: Neovascular complex penetrates through RPE into the **subretinal space** (above RPE).
- **OCT Features**: Discrete hyperreflective mass or plaque above the RPE. RPE is **breached or disrupted** (discontinuous band). Often with SHRM, retinal thickening, fluid.
- **Positive Diagnosis Requires**: Subretinal lesion + RPE breach + discrete subretinal component.

## 3. Type 3 MNV (RAP)
- **Location**: Abnormal vessels **originate within the retina** extending toward outer retina.
- **OCT Features**: Hyperreflective band within retinal layers extending toward deep tissues. Prominent intraretinal cystic spaces adjacent to lesion. May eventually involve PED.
- **Positive Diagnosis Requires**: Intraretinal origin with hyperreflective band crossing retinal layers.

## 4. Mix MNV
- **Location**: Neovascular complex in **multiple compartments** (sub-RPE + subretinal).
- **OCT Features**: Fibrovascular PED (Type 1 pattern) coexisting with discrete subretinal hyperreflective mass (Type 2 pattern). RPE may be elevated in some areas and breached in others.
- **Positive Diagnosis Requires**: Evidence of BOTH sub-RPE AND subretinal components.

# Differential Diagnosis Checklist (MANDATORY)
1. **RPE Status**: INTACT (continuously elevated) or BREACHED/DISRUPTED? → Key differentiator.
2. **Lesion Compartment**: Primary lesion is SUB-RPE, SUBRETINAL, or INTRARETINAL?
3. **Subretinal Component**: Is there a discrete hyperreflective mass above RPE? → If YES, rules out pure Type 1.
4. **Intraretinal Origin**: Is there a hyperreflective band originating within the retina? → If YES, consider Type 3.
5. **Multiple Compartments**: Components in more than one compartment? → If YES, Mix.

**CRITICAL**: Type 1 is NOT the default. If you see any subretinal hyperreflective mass or RPE breach, do NOT classify as Type 1.

# Analysis Steps
1. **Image Quality**: Confirm OCT is of diagnostic quality.
2. **RPE Integrity**: Is the RPE band continuous or breached?
3. **Lesion Location**: Where is the primary hyperreflective lesion?
4. **Complete Differential Diagnosis Checklist** above.
5. **Final Classification**.

Please output:

## Diagnostic Report
- **Final Classification**: [Type 1 MNV (Occult) / Type 2 MNV (Classic) / Type 3 MNV (RAP) / Mix MNV]
- **Confidence Level**: [High / Medium / Low]

## Detailed Rationale
1. **OCT Findings**: [Lesion location, RPE status, fluid presence, SHRM]
2. **Differential Diagnosis Checklist**: [Answer all 5 questions explicitly]
3. **Classification Logic**: [Why this type, why NOT the others]

# Output Format
IMPORTANT: Your response MUST end with a <prediction> tag containing exactly one of: Type 1 MNV (Occult), Type 2 MNV (Classic), Type 3 MNV (Retinal Angiomatous Proliferation - RAP), Mix MNV.
For example: <prediction>Type 2 MNV (Classic)</prediction>"""


DIRECT_RESPONSE_PROMPT = """\
# Role
You are an expert ophthalmologist and retinal specialist. Your task is to analyze baseline Color Fundus Photography (CFP) and Optical Coherence Tomography (OCT) images of patients with Neovascular Age-Related Macular Degeneration (nAMD) and predict their structural response to a standard anti-VEGF loading dose therapy (3 monthly injections).

# Task
Based on the baseline CFP and OCT images, predict the patient's anatomic outcome after three consecutive monthly intravitreal anti-VEGF injections. Use the definitive anatomical criteria below.

# Treatment Response Criteria
## Good Response: Complete disappearance of Subretinal Fluid (SRF), Intraretinal Fluid (IRF), and intraretinal cysts (IRC); OR a >75% reduction in Central Retinal Thickness (CRT) compared to baseline.
## Partial Response: Persistent SRF, IRF, and IRC; AND a 25%–75% reduction in CRT relative to baseline.
## Poor Response: Persistent SRF, IRF, and IRC; AND a <25% reduction in CRT compared to baseline.
## Non-response: No change or worsening of CRT, SRF, IRF, and Pigment Epithelial Detachment (PED) relative to baseline.

# Analysis Steps
1. **Image Quality Assessment**: Confirm images are of diagnostic quality.
2. **Fundus Photo Analysis**: Describe hemorrhage location (subretinal vs intraretinal), exudation, and PED signs.
3. **OCT Structural Analysis**: Evaluate lesion location, RPE integrity, fluid distribution (SRF/IRF/PED), and presence of SHRM.
4. **Response Prediction**: Based on the baseline structural features, reason about the expected 3-month treatment response.
5. **Final Prediction**: Conclude with the predicted response category.

Please output in this format:

## Diagnostic Report
- **Predicted Outcome**: [Good / Partial / Poor / Non-response]
- **Confidence Level**: [High / Medium / Low]

## Detailed Rationale
1. **Fundus Photo Findings**: [Description]
2. **OCT Core Findings**:
   - Lesion Location: [Description]
   - RPE Status: [Intact / Breached / Detached]
   - Fluid Distribution: [SRF/IRF/PED specifics]
3. **Response Prediction Logic**:
   - Supporting Evidence: [Key features supporting the predicted response]
   - Risk Factors: [Features suggesting poorer prognosis]

## Clinical Recommendations
[Brief suggestions based on predicted response]

# Output Format
IMPORTANT: Your response MUST end with a <prediction> tag containing exactly one of: Good response, Partial response, Poor response, Non-response.
For example: <prediction>Good response</prediction>"""


GIVEN_BIOMARKER_PROMPT = """\
# Role
You are an expert ophthalmologist and retinal specialist. Your task is to predict the structural response to anti-VEGF loading dose therapy (3 monthly injections) in patients with nAMD.

# Task
You are given the patient's baseline CFP and OCT images, AND the following exact biomarker measurements (graded by a retinal expert). Use these precise measurements together with the images to predict the anatomic outcome after three consecutive monthly intravitreal anti-VEGF injections.

# Biomarker Data
The following biomarkers were measured at baseline:

{bio_data}

**Biomarker Value Legend:**
- CFP-Hemorrhage: 0 = No hemorrhage, 1 = Hemorrhage present
- OCT-SRF / OCT-IRF / OCT-PED / OCT-SHRM: 0 = Absent, 1 = Present, not involving fovea, 2 = Present, involving fovea
- CRT: Central Retinal Thickness in μm

# Treatment Response Criteria
## Good Response: Complete disappearance of Subretinal Fluid (SRF), Intraretinal Fluid (IRF), and intraretinal cysts (IRC); OR a >75% reduction in Central Retinal Thickness (CRT) compared to baseline.
## Partial Response: Persistent SRF, IRF, and IRC; AND a 25%-75% reduction in CRT relative to baseline.
## Poor Response: Persistent SRF, IRF, and IRC; AND a <25% reduction in CRT compared to baseline.
## Non-response: No change or worsening of CRT, SRF, IRF, and Pigment Epithelial Detachment (PED) relative to baseline.

# Analysis Steps
1. **Image Verification**: Compare the CFP and OCT images with the provided biomarker measurements. Do the images confirm these measurements?
2. **Biomarker Integration**: Systematically evaluate each biomarker in relation to the Treatment Response Criteria. Consider which biomarkers suggest potential for CRT reduction vs which suggest chronicity.
3. **Response Prediction**: Based on the combined biomarker profile and image findings, predict the most likely 3-month treatment response.

Please output in this format:

## Diagnostic Report
- **Predicted Outcome**: [Good / Partial / Poor / Non-response]
- **Confidence Level**: [High / Medium / Low]

## Detailed Rationale
1. **Image-Biomarker Correlation**: [Do the images support the provided measurements?]
2. **Biomarker Integration**:
   - Key Findings: [Analysis of each biomarker]
   - CRT Assessment: [CRT value and its role in response definition]
3. **Response Prediction Logic**: [Synthesis of all factors leading to the predicted response]

## Clinical Recommendations
[Brief suggestions based on predicted response]

# Output Format
IMPORTANT: Your response MUST end with a <prediction> tag containing exactly one of: Good response, Partial response, Poor response, Non-response.
For example: <prediction>Good response</prediction>"""


GIVEN_BIOMARKER_NOIMG_PROMPT = """\
# Role
You are an expert ophthalmologist and retinal specialist.

# Task
You are given the following exact biomarker measurements from a patient with nAMD at baseline. Based solely on these structured data values, predict the anatomic outcome after three consecutive monthly intravitreal anti-VEGF injections. **You do not have access to images — rely only on the biomarker values provided below.**

# Biomarker Data
The following biomarkers were measured at baseline:

{bio_data}

**Biomarker Value Legend:**
- CFP-Hemorrhage: 0 = No hemorrhage, 1 = Hemorrhage present
- OCT-SRF / OCT-IRF / OCT-PED / OCT-SHRM: 0 = Absent, 1 = Present, not involving fovea, 2 = Present, involving fovea
- CRT: Central Retinal Thickness in μm

# Treatment Response Criteria
## Good Response: Complete disappearance of SRF, IRF, and IRC; OR a >75% reduction in CRT compared to baseline.
## Partial Response: Persistent SRF, IRF, IRC; AND a 25%-75% reduction in CRT relative to baseline.
## Poor Response: Persistent SRF, IRF, IRC; AND a <25% reduction in CRT compared to baseline.
## Non-response: No change or worsening of CRT, SRF, IRF, and PED relative to baseline.

# Analysis Steps
1. **Biomarker Assessment**: Systematically evaluate each biomarker in relation to the Treatment Response Criteria. Note which biomarkers indicate potential for fluid resolution (and thus CRT reduction) vs which suggest chronic structural changes.
2. **Response Prediction**: Synthesize all biomarker findings to predict the most likely response category.

Please output in this format:

## Diagnostic Report
- **Predicted Outcome**: [Good / Partial / Poor / Non-response]
- **Confidence Level**: [High / Medium / Low]

## Detailed Rationale
1. **Biomarker Profile Analysis**:
   - Key Findings: [Analysis of each biomarker in relation to response criteria]
   - CRT Assessment: [CRT value and its role in response definition]
2. **Response Prediction Logic**: [Synthesis of all factors leading to the predicted response]

# Output Format
IMPORTANT: Your response MUST end with a <prediction> tag containing exactly one of: Good response, Partial response, Poor response, Non-response.
For example: <prediction>Poor response</prediction>"""


# ── Label Mapping Functions ───────────────────────────────────────────────────

def _map_mnv_label(x):
    if pd.isna(x) or x is None:
        return None
    x_str = str(x).strip().strip('[]')
    if x_str in {'1', '2', '3', 'Mix'}:
        return x_str
    return {
        'Type 1 MNV (Occult)': '1',
        'Type 2 MNV (Classic)': '2',
        'Type 3 MNV (Retinal Angiomatous Proliferation - RAP)': '3',
        'Mix MNV': 'Mix',
    }.get(x_str, "")


def _map_response_label(x):
    if pd.isna(x) or x is None:
        return None
    x_str = str(x).strip().lower()
    if x_str in {'1', '2', '3', '4'}:
        return x_str
    if 'good' in x_str:
        return '1'
    if 'partial' in x_str:
        return '2'
    if 'poor' in x_str:
        return '3'
    if 'non' in x_str or 'no response' in x_str:
        return '4'
    return ""


# ── Task Configuration ────────────────────────────────────────────────────────

TASKS = {
    "mnv": {
        "name": "MNV Classification (CFP+OCT)",
        "data_source": "mnv",
        "gt_column": "MNV Type",
        "prompt": MNV_PROMPT,
        "map_label": _map_mnv_label,
        "eval_labels": ['1', '2', '3', 'Mix'],
        "label_names": {'1': 'Type 1', '2': 'Type 2', '3': 'Type 3', 'Mix': 'Mix'},
        "file_prefix": "predictions",
    },
    "mnv_cfp_only": {
        "name": "MNV Classification (CFP Only)",
        "data_source": "mnv",
        "gt_column": "MNV Type",
        "prompt": MNV_PROMPT_CFP,
        "image_mode": "cfp_only",
        "map_label": _map_mnv_label,
        "eval_labels": ['1', '2', '3', 'Mix'],
        "label_names": {'1': 'Type 1', '2': 'Type 2', '3': 'Type 3', 'Mix': 'Mix'},
        "file_prefix": "predictions_mnv_cfp",
    },
    "mnv_oct_only": {
        "name": "MNV Classification (OCT Only)",
        "data_source": "mnv",
        "gt_column": "MNV Type",
        "prompt": MNV_PROMPT_OCT,
        "image_mode": "oct_only",
        "map_label": _map_mnv_label,
        "eval_labels": ['1', '2', '3', 'Mix'],
        "label_names": {'1': 'Type 1', '2': 'Type 2', '3': 'Type 3', 'Mix': 'Mix'},
        "file_prefix": "predictions_mnv_oct",
    },
    "response_direct": {
        "name": "Treatment Response (Images Only)",
        "data_source": "response",
        "gt_column": "Response",
        "prompt": DIRECT_RESPONSE_PROMPT,
        "map_label": _map_response_label,
        "eval_labels": ['1', '2', '3', '4'],
        "label_names": {'1': 'Good', '2': 'Partial', '3': 'Poor', '4': 'Non-response'},
        "file_prefix": "predictions_response_direct",
    },
    "given_biomarker_noimg": {
        "name": "Treatment Response (w/o Images)",
        "data_source": "response",
        "gt_column": "Response",
        "prompt": GIVEN_BIOMARKER_NOIMG_PROMPT,
        "aux_columns": ["MNV Type", "CFP-Hemorrhage", "OCT-SRF", "OCT-IRF", "OCT-PED", "OCT-SHRM", "CRT/um"],
        "include_images": False,
        "map_label": _map_response_label,
        "eval_labels": ['1', '2', '3', '4'],
        "label_names": {'1': 'Good', '2': 'Partial', '3': 'Poor', '4': 'Non-response'},
        "file_prefix": "predictions_givenBiomarker_noimg",
    },
    "given_biomarker": {
        "name": "Treatment Response (Combination)",
        "data_source": "response",
        "gt_column": "Response",
        "prompt": GIVEN_BIOMARKER_PROMPT,
        "aux_columns": ["MNV Type", "CFP-Hemorrhage", "OCT-SRF", "OCT-IRF", "OCT-PED", "OCT-SHRM", "CRT/um"],
        "map_label": _map_response_label,
        "eval_labels": ['1', '2', '3', '4'],
        "label_names": {'1': 'Good', '2': 'Partial', '3': 'Poor', '4': 'Non-response'},
        "file_prefix": "predictions_givenBiomarker",
    },
}

MNV_TYPE_MAP = {'1': 'Type 1 MNV (Occult)', '2': 'Type 2 MNV (Classic)', '3': 'Type 3 MNV (RAP)', 'Mix': 'Mix MNV'}


def _get_data_paths(data_source: str):
    if data_source == "mnv":
        return MNV_EXCEL, MNV_IMG_DIR
    elif data_source == "response":
        return RESP_EXCEL, RESP_IMG_DIR
    else:
        raise ValueError(f"Unknown data source: {data_source}")


def _format_single_biomarker(row, col, col_label):
    val = row.get(col)
    if col == 'CRT/um':
        val_str = f"{int(val)} μm" if pd.notna(val) else 'Missing'
    else:
        val_str = str(int(val)) if pd.notna(val) else 'Missing'
    return f"- {col_label}: {val_str}"


def _format_biomarkers(row, aux_columns):
    lines = []
    mnv_col, *bio_cols = aux_columns
    mnv_val = row.get(mnv_col)
    mnv_str = MNV_TYPE_MAP.get(str(mnv_val).strip(), str(mnv_val)) if pd.notna(mnv_val) else 'Unknown'
    lines.append(f"- MNV Subtype: {mnv_str}")

    col_labels = {
        'CFP-Hemorrhage': 'CFP Hemorrhage (CFP-Hemorrhage)', 'OCT-SRF': 'Subretinal Fluid (OCT-SRF)',
        'OCT-IRF': 'Intraretinal Fluid (OCT-IRF)', 'OCT-PED': 'Pigment Epithelial Detachment (OCT-PED)',
        'OCT-SHRM': 'Subretinal Hyperreflective Material (OCT-SHRM)', 'CRT/um': 'Central Retinal Thickness (CRT/um)'
    }
    for col in bio_cols:
        val = row[col]
        label = col_labels.get(col, col)
        if pd.isna(val):
            val_str = 'Missing'
        elif col == 'CRT/um':
            val_str = f"{int(val)} μm"
        else:
            val_str = f"{int(val)}"
        lines.append(f"- {label}: {val_str}")
    return "\n".join(lines)


# ── Core Processing ───────────────────────────────────────────────────────────

def process_single_case(case_id: str, prompt_template: str, ground_truth: str,
                        config: dict, img_dir: str, aux_context: str = None,
                        include_images: bool = True,
                        image_mode: str = "both") -> dict:
    print(f"Processing case {case_id}...")

    full_prompt = prompt_template
    if aux_context:
        if '{bio_data}' in prompt_template:
            full_prompt = prompt_template.replace('{bio_data}', aux_context)
        else:
            full_prompt = prompt_template.strip() + f"\n\nAdditional Information:\nMNV Subtype: {aux_context}"

    if not include_images:
        content = [{"type": "text", "text": full_prompt}]
    else:
        images = prepare_case_data(case_id, img_dir)
        content = [{"type": "text", "text": full_prompt}]
        if image_mode in ("both", "cfp_only"):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{images['cfp'][1]};base64,{images['cfp'][0]}"}
            })
        if image_mode in ("both", "oct_only"):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{images['oct'][1]};base64,{images['oct'][0]}"}
            })
    _, api_result = call_llm_api(case_id, content, **config)

    if not api_result["success"]:
        return {
            "case_id": case_id,
            "ground_truth": ground_truth,
            "prompt": full_prompt,
            "success": False,
            "error": api_result["error"],
            "full_response": None,
            "extracted_prediction": None,
            "mnv_type": aux_context,
        }

    full_response = api_result["full_response"]
    extracted_pred = extract_from_response(full_response, "<prediction>")

    return {
        "case_id": case_id,
        "ground_truth": ground_truth,
        "prompt": full_prompt,
        "success": True,
        "full_response": full_response,
        "extracted_prediction": extracted_pred,
        "error": None,
        "mnv_type": aux_context,
    }


def evaluate(final_results: list, task: dict):
    df = pd.DataFrame([{
        "case_id": r["case_id"],
        "ground_truth": r.get("ground_truth"),
        "predicted": r.get("extracted_prediction"),
        "success": r["success"],
    } for r in final_results])

    n_missing = (~df['success'] | df['predicted'].isna() | (df['predicted'] == '')).sum()
    if n_missing > 0:
        print(f"\n{n_missing} cases still need predictions. Run without --eval to fill them.")
        return

    map_label = task["map_label"]
    df['gt'] = df['ground_truth'].apply(map_label)
    df['pred'] = df['predicted'].apply(map_label)

    valid = df[df['gt'].notna()]
    if len(valid) == 0:
        print("No valid cases with ground truth.")
        return

    accuracy = (valid['gt'] == valid['pred']).sum() / len(valid) * 100
    labels = task["eval_labels"]
    cm = compute_confusion_matrix(valid['gt'].tolist(), valid['pred'].tolist(), labels=labels)
    metrics = compute_metrics_from_confusion(cm)
    for label in metrics:
        metrics[label]['support'] = int((valid['gt'] == label).sum())

    macro = compute_average_metrics(metrics, 'macro')
    weighted = compute_average_metrics(metrics, 'weighted')
    names = task["label_names"]
    print_classification_report(metrics, {'macro': macro, 'weighted': weighted}, names, accuracy=accuracy)
    print_confusion_matrix(cm, names)


def main():
    parser = argparse.ArgumentParser(description='nAMD MNV Classification and Treatment Response')
    parser.add_argument("-m", type=str, required=True,
                       help='Model key (GPT, Gemini, Qwen)')
    parser.add_argument('--task', type=str, default='mnv', choices=list(TASKS),
                       help=f'Task to run: {", ".join(TASKS)}')
    parser.add_argument('--fresh', action='store_true',
                       help='Start fresh, ignore existing results')
    parser.add_argument('--eval', action='store_true',
                       help='Evaluate existing results only')
    parser.add_argument('--run', type=int, default=None,
                       help='Run number for repeated experiments')
    parser.add_argument('--temperature', type=float, default=None,
                       help='API temperature (default: API default, ~1.0)')
    args = parser.parse_args()

    task = TASKS[args.task]
    base_url, model_name, api_key = get_model_config(args.m)
    config = {"base_url": base_url, "model_name": model_name, "api_key": api_key}
    if args.temperature is not None:
        config["temperature"] = args.temperature

    print("=" * 80)
    print(f"nAMD {task['name']}")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_short = model_name.replace('/', '_').replace(':', '_')
    prefix = task["file_prefix"]
    run_suffix = f"_run{args.run}" if args.run else ""
    filename_base = f"{model_short}_tokens{MAX_TOKENS}{run_suffix}"
    results_file = os.path.join(OUTPUT_DIR, f"{prefix}_detailed_{filename_base}.json")

    # ── Evaluation Mode ──
    if args.eval:
        if not os.path.exists(results_file):
            print(f"\nError: No results file at {results_file}")
            print(f"Run prediction first: python run.py -m {args.m} --task {args.task}")
            return
        print(f"\nLoading results from {results_file}...")
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} results")
        evaluate(results, task)
        return

    # ── Prediction Mode ──
    print(f"Model: {model_name}")
    print(f"Results will be saved to: {results_file}")

    resume_file = None
    if os.path.exists(results_file) and not args.fresh:
        resume_file = results_file
        print(f"\nFound existing results: {results_file}")
        print("Will resume. (Use --fresh to overwrite)")
    elif args.fresh and os.path.exists(results_file):
        print(f"\n--fresh: overwriting {os.path.basename(results_file)}")
    else:
        print(f"\nStarting fresh run.")

    prompt_template = task["prompt"]
    print(f"\nPrompt template loaded ({len(prompt_template)} characters)")

    # ── Get data paths ──
    excel_file, img_dir = _get_data_paths(task["data_source"])

    # ── Resume: load previous results ──
    if resume_file:
        print(f"\n{'='*80}\nRESUME MODE\n{'='*80}")
        with open(resume_file, 'r', encoding='utf-8') as f:
            previous_results = json.load(f)
        print(f"Loaded {len(previous_results)} previous results")

        prev_dict = {r["case_id"]: r for r in previous_results}
        cases_to_retry = []
        for r in previous_results:
            if not r.get("success", False) or r.get("extracted_prediction") is None or r.get("extracted_prediction") == "":
                cases_to_retry.append((r["case_id"], r.get("ground_truth"), r.get("mnv_type")))

        if len(cases_to_retry) == 0:
            print("\nAll cases valid! Run with --eval to evaluate.")
            return

        print(f"\n{len(cases_to_retry)} cases to reprocess")
        case_data = cases_to_retry
        prev_dict = prev_dict

    # ── Fresh: load from Excel ──
    else:
        print(f"\nLoading data from {excel_file}...")
        df = pd.read_excel(excel_file)
        print(f"Loaded {len(df)} cases")

        gt_col = task["gt_column"]
        aux_col = task.get("aux_column")
        aux_cols = task.get("aux_columns")
        aux_col_label = task.get("aux_column_label")
        case_data = []
        for _, row in df.iterrows():
            cid = f"{int(row['Code']):03d}"
            gt = str(row[gt_col]) if pd.notna(row.get(gt_col)) else None
            if aux_cols:
                aux = _format_biomarkers(row, aux_cols)
            elif aux_col and aux_col_label:
                aux = _format_single_biomarker(row, aux_col, aux_col_label)
            elif aux_col:
                aux = str(row[aux_col]) if pd.notna(row.get(aux_col)) else None
            else:
                aux = None
            case_data.append((cid, gt, aux))

        prev_dict = {}

    # ── Process ──
    print(f"\nProcessing {len(case_data)} cases...")
    print(f"base_url={base_url}, model={model_name}")

    include_images = task.get("include_images", True)
    image_mode = task.get("image_mode", "both")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single_case, cid, prompt_template, gt, config, img_dir, aux, include_images, image_mode): cid
            for cid, gt, aux in case_data
        }

        completed = 0
        for future in as_completed(futures):
            cid = futures[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                status = "✓" if result["success"] else "✗"
                pred = result.get("extracted_prediction", "N/A")
                aux = result.get("mnv_type")
                aux_str = ""
                if aux:
                    aux_short = aux.split('\n')[0] if '\n' in aux else aux
                    aux_str = f" [{aux_short}]"
                print(f"[{completed}/{len(case_data)}] {status} Case {cid}: {pred}{aux_str}")
            except Exception as e:
                print(f"[{completed}/{len(case_data)}] ✗ Case {cid}: Exception - {e}")
                _, gt, aux = next((x for x in case_data if x[0] == cid), (cid, None, None))
                results.append({
                    "case_id": cid, "ground_truth": gt, "prompt": prompt_template,
                    "success": False, "error": str(e), "full_response": None,
                    "extracted_prediction": None, "mnv_type": aux,
                })
                completed += 1

    # ── Merge & Save ──
    if resume_file:
        print(f"\nMerging {len(results)} new results with {len(previous_results)} existing...")
        for r in results:
            prev_dict[r["case_id"]] = r
        final_results = list(prev_dict.values())
    else:
        final_results = results

    final_results.sort(key=lambda x: x["case_id"])

    output_json = os.path.join(OUTPUT_DIR, f"{prefix}_detailed_{filename_base}.json")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to {output_json}")

    summary = []
    for r in final_results:
        row = {
            "case_id": r["case_id"],
            "case_number": int(r["case_id"]),
            "ground_truth": r.get("ground_truth"),
            "predicted": r["extracted_prediction"],
            "success": r["success"],
            "error": r.get("error"),
        }
        if r.get("mnv_type"):
            row["mnv_type"] = r["mnv_type"]
        summary.append(row)

    summary_df = pd.DataFrame(summary)
    output_csv = os.path.join(OUTPUT_DIR, f"{prefix}_summary_{filename_base}.csv")
    summary_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Summary CSV saved to {output_csv}")

    missing = sum(1 for r in final_results
                  if not r.get("success", False)
                  or r.get("extracted_prediction") is None
                  or r.get("extracted_prediction") == "")
    if missing > 0:
        print(f"\n{missing} cases still need predictions. Run again to fill them.")
    else:
        print(f"\nAll cases have predictions! Run evaluation: python run.py -m {args.m} --task {args.task} --eval")


if __name__ == "__main__":
    main()
