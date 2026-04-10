"""
backend/llm_report.py

Generates patient-level reports using Claude API.
Two modes:
  - "clinical"    : plain language for oncologists
  - "researcher"  : technical, references GAT attention + UQ metrics
"""

import os
import anthropic
from .data_loader import get_patient, get_patient_attention, get_metrics

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL  = "claude-sonnet-4-20250514"


# ─── System prompts ───────────────────────────────────────────────────────────

SYSTEM_CLINICAL = """You are a clinical AI assistant helping oncologists interpret 
single-cell RNA sequencing data from melanoma patients undergoing anti-PD-1 immunotherapy.

Your reports must:
- Use plain, accessible language — avoid jargon where possible
- Explain what immune escape means in simple terms when first mentioned
- Focus on clinically relevant findings: likely response to immunotherapy, 
  dominant immune cell populations, and key warning signals
- Be structured: start with a 2-sentence summary, then expand on findings
- Never make definitive clinical recommendations — frame everything as 
  "the model suggests" or "the data indicates"
- Keep reports concise: 200-300 words

Tone: Professional, clear, empathetic."""


SYSTEM_RESEARCHER = """You are a computational biology AI assistant interpreting 
results from TME-VI (Tumor Microenvironment Variational Inference), a Graph Attention 
Network Decoder Virtual Instrument built within the AI Virtual Cell (AIVC) framework.

The pipeline:
1. scVI encodes 45,884 genes into 50-dimensional Cellular Universal Representations (URs)
2. A GAT Decoder VI (2-layer, 4-head attention, 15-neighbor graph) predicts immune escape
3. Monte Carlo Dropout (T=50 passes) quantifies epistemic uncertainty per cell

Your reports must:
- Reference specific model outputs: mc_mean_prob, mc_std, GAT attention weights
- Interpret attention weight patterns in terms of cell-cell communication and TME biology
- Discuss uncertainty: high mc_std cells = model is less confident, biological ambiguity
- Connect findings to the Sade-Feldman et al. 2018 melanoma immunotherapy literature
- Note pre vs post-treatment differences when available
- Use precise biological terminology: TME niche, immune escape, exhaustion, checkpoint blockade
- Structure: Abstract (3 sentences) → Cell composition → Escape analysis → 
  Attention patterns → Uncertainty profile → Biological interpretation
- Target length: 400-500 words

Tone: Scientific, precise, hypothesis-generating."""


# ─── Prompt builders ──────────────────────────────────────────────────────────

def _build_clinical_prompt(patient: dict, attn: dict) -> str:
    comp = patient["cell_type_composition"]
    top3 = list(patient["top_escape_celltypes"].items())

    comp_str = "\n".join(
        f"  - {ct}: {pct*100:.1f}% of cells"
        for ct, pct in sorted(comp.items(), key=lambda x: -x[1])
    )
    top3_str = "\n".join(
        f"  - {ct}: mean escape score {score:.2f}"
        for ct, score in top3
    )
    pre_post = ""
    if patient["mean_escape_pre"] and patient["mean_escape_post"]:
        pre_post = (
            f"\nPre-treatment mean escape: {patient['mean_escape_pre']:.3f}"
            f"\nPost-treatment mean escape: {patient['mean_escape_post']:.3f}"
        )

    return f"""Patient: {patient['patient_id']}
Known clinical response: {patient['response']}
Total cells analyzed: {patient['n_cells']:,}
{pre_post}

IMMUNE ESCAPE SUMMARY
Overall mean escape probability: {patient['mean_escape']:.3f} (scale 0-1, higher = more escape)
Percentage of cells with high escape (>0.5): {patient['pct_high_escape']*100:.1f}%
Model confidence (lower uncertainty = more confident): mean std = {patient['mean_uncertainty']:.4f}

IMMUNE CELL COMPOSITION
{comp_str}

CELL TYPES WITH HIGHEST ESCAPE ACTIVITY
{top3_str}

Please write a clinical report for this patient's oncologist explaining:
1. What this immune profile suggests about likely immunotherapy response
2. Which immune cell populations are most concerning
3. The overall confidence level in these predictions
"""


def _build_researcher_prompt(patient: dict, attn: dict, metrics: dict) -> str:
    comp = patient["cell_type_composition"]
    escape_by_ct = patient["escape_by_celltype"]
    top_pairs = attn["pair_summary"][:5]

    comp_str = "\n".join(
        f"  {ct}: {pct*100:.1f}%"
        for ct, pct in sorted(comp.items(), key=lambda x: -x[1])
    )
    escape_str = "\n".join(
        f"  {ct}: {score:.4f}"
        for ct, score in sorted(escape_by_ct.items(), key=lambda x: -x[1])
    )
    attn_str = "\n".join(
        f"  {p['src_celltype']} → {p['dst_celltype']}: "
        f"mean_weight={p['mean_weight']:.4f}, n={p['n_edges']}"
        for p in top_pairs
    ) if top_pairs else "  No intra-patient attention edges in top 5%"

    pre_post = ""
    if patient["mean_escape_pre"] and patient["mean_escape_post"]:
        delta = patient["mean_escape_post"] - patient["mean_escape_pre"]
        pre_post = (
            f"\nPre-treatment mc_mean_prob: {patient['mean_escape_pre']:.4f}"
            f"\nPost-treatment mc_mean_prob: {patient['mean_escape_post']:.4f}"
            f"\nΔ (post - pre): {delta:+.4f}"
        )

    return f"""PATIENT: {patient['patient_id']}
Clinical response: {patient['response']}
Cells: {patient['n_cells']:,} (pre={patient['n_pre']}, post={patient['n_post']})

MODEL PERFORMANCE CONTEXT
Global cell-level AUROC: {metrics['cell_auroc']} (confidence-filtered top-20%: {metrics['conf20_auroc']})
Global patient-level AUROC: {metrics['patient_auroc']}
MC Dropout passes: {metrics['mc_passes']}

PATIENT-LEVEL MC DROPOUT OUTPUTS
mc_mean_prob (mean): {patient['mean_escape']:.4f}
mc_std (mean epistemic uncertainty): {patient['mean_uncertainty']:.4f}
pct cells > 0.5 escape threshold: {patient['pct_high_escape']*100:.1f}%
{pre_post}

CELL TYPE COMPOSITION (fraction of total TME)
{comp_str}

MC_MEAN_PROB BY CELL TYPE
{escape_str}

GAT LAYER-1 ATTENTION — TOP CELL-TYPE PAIRS (intra-patient edges only)
{attn_str}

Please write a technical researcher report covering:
1. Patient TME cell composition and what it implies about immune contexture
2. Escape probability profile per cell type — which populations drive escape signal
3. GAT attention patterns — what the inter-cell attention weights reveal about 
   TME communication and potential immunosuppressive interactions
4. Uncertainty profile — are high-escape cells also high-uncertainty?
5. Biological interpretation connecting to checkpoint blockade biology and 
   the Sade-Feldman et al. 2018 findings
"""


# ─── Main entry point ─────────────────────────────────────────────────────────

def generate_report(patient_id: str, mode: str = "clinical") -> str:
    """
    Generate a report for a patient.

    Args:
        patient_id: e.g. "P3"
        mode: "clinical" or "researcher"

    Returns:
        Report text (streaming assembled into a single string)
    """
    patient = get_patient(patient_id)
    if patient is None:
        return f"Patient {patient_id} not found."

    attn    = get_patient_attention(patient_id)
    metrics = get_metrics()

    if mode == "clinical":
        system = SYSTEM_CLINICAL
        prompt = _build_clinical_prompt(patient, attn)
    else:
        system = SYSTEM_RESEARCHER
        prompt = _build_researcher_prompt(patient, attn, metrics)

    # Stream response
    full_text = ""
    with client.messages.stream(
        model=MODEL,
        max_tokens=1000,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            full_text += text

    return full_text


def stream_report(patient_id: str, mode: str = "clinical"):
    """
    Generator version for Streamlit st.write_stream().
    Yields text chunks as they arrive from Claude API.
    """
    patient = get_patient(patient_id)
    if patient is None:
        yield f"Patient {patient_id} not found."
        return

    attn    = get_patient_attention(patient_id)
    metrics = get_metrics()

    if mode == "clinical":
        system = SYSTEM_CLINICAL
        prompt = _build_clinical_prompt(patient, attn)
    else:
        system = SYSTEM_RESEARCHER
        prompt = _build_researcher_prompt(patient, attn, metrics)

    with client.messages.stream(
        model=MODEL,
        max_tokens=1000,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text
