"""
backend/api.py

FastAPI backend for TME-VI app.
Runs as a background thread inside the HF Space alongside Streamlit.

Endpoints:
  GET  /api/patients              → 32-patient list
  GET  /api/patient/{id}          → single patient detail
  GET  /api/patient/{id}/attention→ attention edge summary
  GET  /api/umap                  → UMAP point cloud
  GET  /api/metrics               → model performance
  GET  /api/celltypes             → cell type list + stats
  POST /api/report/{id}           → generate LLM report
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv
load_dotenv()

from .data_loader import (
    get_patients,
    get_patient,
    get_patient_attention,
    get_umap,
    get_metrics,
    get_cell_types,
    get_celltype_stats,
    get_attention_matrix,
)
from .llm_report import generate_report

app = FastAPI(title="TME-VI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Patients ─────────────────────────────────────────────────────────────────

@app.get("/api/patients")
def list_patients():
    """Return summary table for all 32 patients."""
    df = get_patients().reset_index()
    return {
        "patients": df[[
            "patient_id", "response", "n_cells",
            "mean_escape", "pct_high_escape", "mean_uncertainty",
        ]].to_dict(orient="records")
    }


@app.get("/api/patient/{patient_id}")
def patient_detail(patient_id: str):
    """Return full detail for a single patient."""
    patient = get_patient(patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return patient


@app.get("/api/patient/{patient_id}/attention")
def patient_attention(patient_id: str):
    """Return attention edge summary for a patient's TME."""
    patient = get_patient(patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return get_patient_attention(patient_id)


# ─── UMAP ─────────────────────────────────────────────────────────────────────

@app.get("/api/umap")
def umap_data(
    color_by: Literal["cell_type", "escape_prob", "uncertainty", "response"] = "cell_type",
    downsample: int = 5000,
):
    """
    Return UMAP coordinates with coloring variable.
    Downsamples to `downsample` points for fast frontend rendering.
    """
    df = get_umap()

    if downsample and downsample < len(df):
        df = df.sample(n=downsample, random_state=42)

    cols = ["umap_x", "umap_y", "patient_id", "cell_type", "response",
            "escape_prob", "uncertainty", "pred_label"]
    return {
        "points":    df[cols].to_dict(orient="records"),
        "color_by":  color_by,
        "n_total":   int(get_umap().shape[0]),
        "n_shown":   int(len(df)),
    }


# ─── Metrics ──────────────────────────────────────────────────────────────────

@app.get("/api/metrics")
def model_metrics():
    """Return all model performance metrics."""
    return get_metrics()


# ─── Cell types ───────────────────────────────────────────────────────────────

@app.get("/api/celltypes")
def cell_types():
    """Return cell type list with global stats."""
    return {
        "cell_types": get_cell_types(),
        "stats":      get_celltype_stats(),
    }


@app.get("/api/attention/matrix")
def attention_matrix():
    """Return cell-type x cell-type mean attention weight matrix."""
    matrix = get_attention_matrix()
    return {
        "cell_types": list(matrix.index),
        "matrix":     matrix.values.tolist(),
    }


# ─── LLM Report ───────────────────────────────────────────────────────────────

class ReportRequest(BaseModel):
    mode: Literal["clinical", "researcher"] = "clinical"


@app.post("/api/report/{patient_id}")
def patient_report(patient_id: str, req: ReportRequest):
    """Generate a LLM report for a patient. Returns full text (non-streaming)."""
    patient = get_patient(patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    report = generate_report(patient_id, mode=req.mode)
    return {
        "patient_id": patient_id,
        "mode":       req.mode,
        "report":     report,
    }


# ─── Health check ─────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    metrics = get_metrics()
    return {
        "status":      "ok",
        "n_patients":  metrics["n_patients"],
        "cell_auroc":  metrics["cell_auroc"],
    }
