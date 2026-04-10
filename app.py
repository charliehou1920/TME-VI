"""
app.py — TME-VI Streamlit Frontend

Pages:
  1. Overview    — model performance metrics + ROC + UQ curves
  2. Cell Atlas  — global UMAP, interactive coloring
  3. Patients    — 32-patient table with response status
  4. Report      — per-patient detail + LLM report (clinical / researcher)
"""

import time
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Config ───────────────────────────────────────────────────────────────────
import os
API = os.environ.get("API_URL", "http://localhost:8000/api")

st.set_page_config(
    page_title="TME-VI · Tumor Immune Escape Decoder",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Design system ────────────────────────────────────────────────────────────
COLORS = {
    "bg":          "#0A0E1A",
    "surface":     "#111827",
    "surface2":    "#1C2333",
    "border":      "#2A3547",
    "teal":        "#2DD4BF",
    "teal_dim":    "#134E4A",
    "amber":       "#F59E0B",
    "amber_dim":   "#451A03",
    "red":         "#F87171",
    "green":       "#4ADE80",
    "text":        "#E2E8F0",
    "text_muted":  "#64748B",
    "responder":   "#4ADE80",
    "nonresponder":"#F87171",
}

CELL_TYPE_COLORS = {
    "Cytotoxic CD8 T":   "#2DD4BF",
    "CD4 T":             "#818CF8",
    "Naive/Memory CD8 T":"#60A5FA",
    "B cell":            "#F472B6",
    "Macrophage":        "#FB923C",
    "γδ T":              "#A78BFA",
    "Plasma cell":       "#34D399",
    "pDC":               "#FCD34D",
}

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: {COLORS['bg']};
    color: {COLORS['text']};
  }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
    background-color: {COLORS['surface']};
    border-right: 1px solid {COLORS['border']};
  }}
  [data-testid="stSidebar"] * {{ color: {COLORS['text']} !important; }}

  /* Metric cards */
  [data-testid="stMetric"] {{
    background: {COLORS['surface2']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 16px 20px;
  }}
  [data-testid="stMetricLabel"] {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: {COLORS['text_muted']} !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }}
  [data-testid="stMetricValue"] {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 28px;
    color: {COLORS['teal']} !important;
    font-weight: 500;
  }}

  /* Tables */
  [data-testid="stDataFrame"] {{ border: 1px solid {COLORS['border']}; border-radius: 8px; }}

  /* Buttons */
  .stButton > button {{
    background: {COLORS['teal_dim']};
    border: 1px solid {COLORS['teal']};
    color: {COLORS['teal']};
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.05em;
    border-radius: 6px;
    padding: 8px 20px;
    transition: all 0.2s;
  }}
  .stButton > button:hover {{
    background: {COLORS['teal']};
    color: {COLORS['bg']};
  }}

  /* Radio buttons */
  .stRadio > label {{ color: {COLORS['text_muted']} !important; font-size: 12px; }}

  /* Headings */
  h1 {{ font-family: 'IBM Plex Mono', monospace; font-size: 22px; font-weight: 500;
         color: {COLORS['text']}; letter-spacing: -0.02em; margin-bottom: 4px; }}
  h2 {{ font-family: 'IBM Plex Mono', monospace; font-size: 15px; font-weight: 500;
         color: {COLORS['teal']}; letter-spacing: 0.05em; text-transform: uppercase;
         margin-top: 32px; margin-bottom: 12px; }}
  h3 {{ font-family: 'IBM Plex Sans', sans-serif; font-size: 14px; font-weight: 600;
         color: {COLORS['text']}; }}

  /* Divider */
  hr {{ border-color: {COLORS['border']}; margin: 24px 0; }}

  /* Report text */
  .report-box {{
    background: {COLORS['surface2']};
    border: 1px solid {COLORS['border']};
    border-left: 3px solid {COLORS['teal']};
    border-radius: 8px;
    padding: 24px 28px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 14px;
    line-height: 1.8;
    color: {COLORS['text']};
  }}

  /* Badge */
  .badge-r  {{ background:{COLORS['responder']}22; color:{COLORS['responder']};
               border:1px solid {COLORS['responder']}55; border-radius:4px;
               padding:2px 8px; font-family:'IBM Plex Mono',monospace; font-size:11px; }}
  .badge-nr {{ background:{COLORS['nonresponder']}22; color:{COLORS['nonresponder']};
               border:1px solid {COLORS['nonresponder']}55; border-radius:4px;
               padding:2px 8px; font-family:'IBM Plex Mono',monospace; font-size:11px; }}

  /* Selectbox / radio */
  .stSelectbox label, .stRadio label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: {COLORS['text_muted']} !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }}

  /* Hide Streamlit branding */
  #MainMenu, footer, header {{ visibility: hidden; }}

  /* Main content padding */
  .block-container {{ padding-top: 2rem; padding-bottom: 2rem; }}
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono", color=COLORS["text"], size=11),
    xaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
    yaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
    margin=dict(l=40, r=20, t=40, b=40),
)


# ─── API helpers ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def api_get(endpoint: str):
    try:
        r = requests.get(f"{API}{endpoint}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(endpoint: str, payload: dict):
    try:
        r = requests.post(f"{API}{endpoint}", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ─── Sidebar navigation ───────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 24px 0;'>
      <div style='font-family: IBM Plex Mono; font-size:13px; font-weight:500;
                  color:#2DD4BF; letter-spacing:0.05em;'>TME-VI</div>
      <div style='font-family: IBM Plex Sans; font-size:11px; color:#64748B;
                  margin-top:2px;'>Tumor Immune Escape Decoder</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["Overview", "Cell Atlas", "Patients", "Report"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(f"""
    <div style='font-family:IBM Plex Mono; font-size:10px; color:{COLORS["text_muted"]};
                line-height:1.8;'>
      GSE120575 · n=16,290 cells<br>
      32 melanoma patients<br>
      anti-PD-1 immunotherapy<br>
      scVI + GAT · MC Dropout T=50
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "Overview":
    st.markdown("# Model Performance")
    st.markdown(f"<div style='font-family:IBM Plex Mono; font-size:11px; color:{COLORS['text_muted']};'>Decoder Virtual Instrument · AI Virtual Cell Framework</div>", unsafe_allow_html=True)
    st.markdown("---")

    metrics = api_get("/metrics")
    if not metrics:
        st.stop()

    # ── Headline metrics ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Cell AUROC", f"{metrics['cell_auroc']:.4f}",
                  delta=f"+{metrics['cell_auroc']-metrics['cell_auroc_v1']:.4f} vs baseline")
    with c2:
        st.metric("Conf-Filtered AUROC", f"{metrics['conf20_auroc']:.4f}",
                  delta="Top 20% confident cells")
    with c3:
        st.metric("Patient AUROC", f"{metrics['patient_auroc']:.4f}",
                  delta="Response prediction")
    with c4:
        st.metric("MC Passes", f"{metrics['mc_passes']}",
                  delta=f"σ̄={metrics['mean_uncertainty']:.4f}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # ── ROC Curve ──
    with col_left:
        st.markdown("## ROC Curve · Cell-Level")
        roc = metrics["roc_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=roc["fpr"], y=roc["tpr"],
            mode="lines",
            line=dict(color=COLORS["teal"], width=2),
            name=f"AUC = {roc['auroc']:.4f}",
            fill="tozeroy",
            fillcolor=f"rgba(45,212,191,0.07)",
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(color=COLORS["border"], width=1, dash="dash"),
            showlegend=False,
        ))
        fig.update_layout(
            **PLOTLY_THEME,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.6, y=0.1, font=dict(size=11)),
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Confidence-filtered AUROC ──
    with col_right:
        st.markdown("## AUROC vs Confidence Filter")
        pct    = metrics["filtered_auroc_pct"]
        values = metrics["filtered_auroc_values"]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=pct, y=values,
            mode="lines+markers",
            line=dict(color=COLORS["amber"], width=2),
            marker=dict(size=6, color=COLORS["amber"]),
            name="Filtered AUROC",
        ))
        fig2.add_hline(
            y=metrics["cell_auroc_v1"],
            line_dash="dash",
            line_color=COLORS["border"],
            annotation_text=f"Baseline {metrics['cell_auroc_v1']}",
            annotation_font_size=10,
            annotation_font_color=COLORS["text_muted"],
        )
        fig2.update_layout(
            **PLOTLY_THEME,
            xaxis_title="% Cells Retained (low uncertainty first)",
            yaxis_title="AUROC",
            height=320,
            yaxis_range=[0.80, 1.0],
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── AUROC by uncertainty bin ──
    st.markdown("## AUROC by Uncertainty Quantile")
    bins = metrics.get("auroc_by_uncertainty_bin", [])
    if bins:
        fig3 = go.Figure(go.Bar(
            x=[f"Q{i+1}" for i in range(len(bins))],
            y=[b for b in bins if b is not None],
            marker_color=COLORS["teal"],
            marker_line_color=COLORS["teal_dim"],
            marker_line_width=1,
        ))
        fig3.add_hline(
            y=metrics["cell_auroc_v1"],
            line_dash="dash", line_color=COLORS["border"],
            annotation_text="Baseline", annotation_font_size=10,
            annotation_font_color=COLORS["text_muted"],
        )
        fig3.update_layout(
            **PLOTLY_THEME,
            xaxis_title="Uncertainty quantile (Q1=low, Q5=high)",
            yaxis_title="AUROC",
            height=260,
            yaxis_range=[0.7, 1.0],
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown(f"""
    <div style='font-family:IBM Plex Mono; font-size:11px; color:{COLORS["text_muted"]};
                background:{COLORS["surface2"]}; border:1px solid {COLORS["border"]};
                border-radius:6px; padding:14px 18px; margin-top:8px; line-height:1.9;'>
      <b style='color:{COLORS["teal"]}'>Key finding:</b>
      Filtering to the 20% most confident cells (lowest MC std) improves AUROC from
      {metrics["cell_auroc"]:.4f} → {metrics["conf20_auroc"]:.4f}.
      The model is calibrated: it correctly identifies when it doesn't know.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CELL ATLAS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Cell Atlas":
    st.markdown("# Cell Atlas")
    st.markdown(f"<div style='font-family:IBM Plex Mono; font-size:11px; color:{COLORS['text_muted']};'>16,290 CD45⁺ immune cells · melanoma TME · scVI UMAP</div>", unsafe_allow_html=True)
    st.markdown("---")

    ctrl_col, _ = st.columns([1, 3])
    with ctrl_col:
        color_by = st.selectbox(
            "Color by",
            ["cell_type", "escape_prob", "uncertainty", "response"],
            format_func=lambda x: {
                "cell_type":   "Cell Type",
                "escape_prob": "Escape Probability",
                "uncertainty": "MC Uncertainty (std)",
                "response":    "Patient Response",
            }[x]
        )

    umap_data = api_get(f"/umap?color_by={color_by}&downsample=6000")
    if not umap_data:
        st.stop()

    df = pd.DataFrame(umap_data["points"])

    if color_by == "cell_type":
        fig = px.scatter(
            df, x="umap_x", y="umap_y",
            color="cell_type",
            color_discrete_map=CELL_TYPE_COLORS,
            hover_data={"patient_id": True, "escape_prob": ":.3f",
                        "uncertainty": ":.4f", "umap_x": False, "umap_y": False},
            labels={"umap_x": "UMAP 1", "umap_y": "UMAP 2"},
        )
    elif color_by == "response":
        fig = px.scatter(
            df, x="umap_x", y="umap_y",
            color="response",
            color_discrete_map={
                "Responder":     COLORS["responder"],
                "Non-responder": COLORS["nonresponder"],
            },
            hover_data={"patient_id": True, "cell_type": True, "umap_x": False, "umap_y": False},
            labels={"umap_x": "UMAP 1", "umap_y": "UMAP 2"},
        )
    else:
        cmap = "RdYlGn_r" if color_by == "escape_prob" else "plasma"
        fig = px.scatter(
            df, x="umap_x", y="umap_y",
            color=color_by,
            color_continuous_scale=cmap,
            hover_data={"patient_id": True, "cell_type": True,
                        "escape_prob": ":.3f", "umap_x": False, "umap_y": False},
            labels={"umap_x": "UMAP 1", "umap_y": "UMAP 2"},
        )

    fig.update_traces(marker=dict(size=2.5, opacity=0.75))
    fig.update_layout(
        **PLOTLY_THEME,
        height=560,
        legend=dict(
            font=dict(size=11),
            itemsizing="constant",
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Showing {umap_data['n_shown']:,} of {umap_data['n_total']:,} cells (random sample for performance)")

    # ── Cell type stats ──
    st.markdown("## Cell Type Composition")
    ct_data = api_get("/celltypes")
    if ct_data:
        stats = ct_data["stats"]
        rows = []
        for ct, s in stats.items():
            rows.append({
                "Cell Type":       ct,
                "Cells":           s["n_cells"],
                "% of TME":        f"{s['pct']:.1f}%",
                "Mean Escape":     f"{s['mean_escape']:.4f}",
                "Mean Uncertainty":f"{s['mean_uncertainty']:.4f}",
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PATIENTS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Patients":
    st.markdown("# Patient Cohort")
    st.markdown(f"<div style='font-family:IBM Plex Mono; font-size:11px; color:{COLORS['text_muted']};'>32 melanoma patients · anti-PD-1 immunotherapy · Sade-Feldman et al. 2018</div>", unsafe_allow_html=True)
    st.markdown("---")

    data = api_get("/patients")
    if not data:
        st.stop()

    df = pd.DataFrame(data["patients"])

    # ── Summary stats ──
    n_r  = (df["response"] == "Responder").sum()
    n_nr = (df["response"] == "Non-responder").sum()
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Patients", 32)
    with c2: st.metric("Responders",     n_r)
    with c3: st.metric("Non-responders", n_nr)
    with c4: st.metric("Median Escape",  f"{df['mean_escape'].median():.3f}")

    st.markdown("---")

    # ── Filter ──
    fc1, fc2 = st.columns([1, 3])
    with fc1:
        resp_filter = st.selectbox("Filter by response",
                                   ["All", "Responder", "Non-responder"])
    if resp_filter != "All":
        df = df[df["response"] == resp_filter]

    # Sort by escape score
    df = df.sort_values("mean_escape", ascending=False)

    # ── Table ──
    display_df = df[["patient_id", "response", "n_cells",
                      "mean_escape", "pct_high_escape", "mean_uncertainty"]].copy()
    display_df.columns = ["Patient", "Response", "Cells",
                          "Mean Escape", "% High Escape", "Mean Uncertainty"]
    display_df["% High Escape"] = (display_df["% High Escape"] * 100).round(1)
    display_df["Mean Escape"]   = display_df["Mean Escape"].round(4)
    display_df["Mean Uncertainty"] = display_df["Mean Uncertainty"].round(4)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Mean Escape": st.column_config.ProgressColumn(
                "Mean Escape", min_value=0, max_value=1, format="%.4f"
            ),
            "% High Escape": st.column_config.NumberColumn("% High Escape", format="%.1f%%"),
        }
    )

    # ── Escape distribution by response ──
    st.markdown("## Escape Score Distribution by Response")
    all_data = api_get("/patients")
    if all_data:
        full_df = pd.DataFrame(all_data["patients"])
        fig = go.Figure()
        for resp, color in [("Responder", COLORS["responder"]),
                             ("Non-responder", COLORS["nonresponder"])]:
            subset = full_df[full_df["response"] == resp]["mean_escape"]
            fig.add_trace(go.Box(
                y=subset,
                name=resp,
                marker_color=color,
                boxmean=True,
                line_color=color,
                fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.2)",
            ))
        fig.update_layout(
            **PLOTLY_THEME,
            yaxis_title="Mean Escape Probability",
            height=320,
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div style='font-family:IBM Plex Mono; font-size:11px; color:{COLORS["text_muted"]};
                margin-top:8px;'>
      → Click <b style='color:{COLORS["teal"]}'>Report</b> in the sidebar to generate
      a patient-specific immune escape report.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — REPORT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Report":
    st.markdown("# Patient Report")
    st.markdown(f"<div style='font-family:IBM Plex Mono; font-size:11px; color:{COLORS['text_muted']};'>LLM-generated immune escape analysis · Claude {'{'}sonnet{'}'}</div>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Controls ──
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        patient_id = st.selectbox(
            "Patient",
            [f"P{i}" for i in [1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,
                                 18,19,20,21,22,23,24,25,26,27,28,29,30,31,33,35]]
        )
    with c2:
        mode = st.radio("Report mode", ["Clinical", "Researcher"],
                        horizontal=True)
        mode_key = mode.lower()

    # ── Load patient data ──
    patient = api_get(f"/patient/{patient_id}")
    attn    = api_get(f"/patient/{patient_id}/attention")

    if not patient:
        st.stop()

    # ── Patient header ──
    resp = patient["response"]
    resp_color = COLORS["responder"] if resp == "Responder" else COLORS["nonresponder"]
    st.markdown(f"""
    <div style='background:{COLORS["surface2"]}; border:1px solid {COLORS["border"]};
                border-radius:8px; padding:16px 20px; margin-bottom:20px;
                display:flex; align-items:center; gap:24px;'>
      <div>
        <div style='font-family:IBM Plex Mono; font-size:20px; font-weight:500;
                    color:{COLORS["text"]};'>{patient_id}</div>
        <div style='font-family:IBM Plex Mono; font-size:11px; margin-top:4px;
                    color:{resp_color};'>{resp}</div>
      </div>
      <div style='color:{COLORS["text_muted"]}; font-size:13px; font-family:IBM Plex Mono;'>
        {patient["n_cells"]:,} cells &nbsp;·&nbsp;
        pre={patient["n_pre"]} &nbsp;·&nbsp;
        post={patient["n_post"]}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Four metric cards ──
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Mean Escape",    f"{patient['mean_escape']:.4f}")
    with m2: st.metric("% High Escape",  f"{patient['pct_high_escape']*100:.1f}%")
    with m3: st.metric("Mean Uncertainty", f"{patient['mean_uncertainty']:.4f}")
    with m4:
        delta_pre_post = None
        if patient["mean_escape_pre"] and patient["mean_escape_post"]:
            delta_pre_post = patient["mean_escape_post"] - patient["mean_escape_pre"]
        st.metric("Δ Escape (post−pre)",
                  f"{delta_pre_post:+.4f}" if delta_pre_post is not None else "N/A")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # ── Cell type composition pie ──
    with col_left:
        st.markdown("## Cell Composition")
        comp = patient["cell_type_composition"]
        fig_pie = go.Figure(go.Pie(
            labels=list(comp.keys()),
            values=list(comp.values()),
            marker_colors=[CELL_TYPE_COLORS.get(ct, "#888") for ct in comp.keys()],
            hole=0.55,
            textinfo="percent",
            textfont=dict(size=10, family="IBM Plex Mono"),
        ))
        fig_pie.update_layout(
            **PLOTLY_THEME,
            height=280,
            showlegend=True,
            legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Escape by cell type bar ──
    with col_right:
        st.markdown("## Escape by Cell Type")
        esc = patient["escape_by_celltype"]
        sorted_esc = dict(sorted(
                        {k: v for k, v in esc.items() if v is not None}.items(),
                        key=lambda x: -x[1]))
        fig_bar = go.Figure(go.Bar(
            x=list(sorted_esc.values()),
            y=list(sorted_esc.keys()),
            orientation="h",
            marker_color=[CELL_TYPE_COLORS.get(ct, "#888") for ct in sorted_esc.keys()],
            marker_line_width=0,
        ))
        fig_bar.update_layout(
            **PLOTLY_THEME,
            height=280,
            xaxis_title="Mean escape probability",
            xaxis_range=[0, 1],
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Attention weights ──
    if attn and attn.get("pair_summary"):
        st.markdown("## GAT Attention · Top Cell-Type Interactions")
        pairs_df = pd.DataFrame(attn["pair_summary"])
        if not pairs_df.empty:
            pairs_df["mean_weight"] = pairs_df["mean_weight"].round(4)
            pairs_df.columns = ["Source Cell Type", "Target Cell Type",
                                 "Mean Attention", "Edge Count"]
            st.dataframe(pairs_df, use_container_width=True, hide_index=True)
    else:
        st.markdown(f"""
        <div style='font-family:IBM Plex Mono; font-size:11px; color:{COLORS["text_muted"]};
                    padding:12px; background:{COLORS["surface2"]}; border-radius:6px;
                    border:1px solid {COLORS["border"]};'>
          No intra-patient attention edges in top 5% for this patient.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── LLM Report ──
    st.markdown(f"## {mode} Report")

    cache_key = f"report_{patient_id}_{mode_key}"

    if cache_key not in st.session_state:
        st.session_state[cache_key] = None

    if st.button(f"Generate {mode} Report", key=f"btn_{mode_key}"):
        with st.spinner("Querying Claude API..."):
            result = api_post(f"/report/{patient_id}", {"mode": mode_key})
            if result:
                st.session_state[cache_key] = result["report"]

    if st.session_state[cache_key]:
        st.markdown(
            f'<div class="report-box">{st.session_state[cache_key].replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )
        st.download_button(
            label="Download report (.txt)",
            data=st.session_state[cache_key],
            file_name=f"TME-VI_{patient_id}_{mode_key}_report.txt",
            mime="text/plain",
        )
