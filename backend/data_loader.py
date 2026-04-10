"""
backend/data_loader.py

Loads app_data.pkl once at startup and exposes typed accessors.
All app code should go through this module — never load pkl directly.
"""

import pickle
from pathlib import Path
from functools import lru_cache
import pandas as pd
import numpy as np

DATA_PATH = Path(__file__).parent.parent / "data" / "app_data.pkl"


@lru_cache(maxsize=1)
def load_data() -> dict:
    """Load app_data.pkl exactly once, cache in memory."""
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    print(f"[data_loader] Loaded app_data.pkl — {len(data['umap']):,} cells, {len(data['patients'])} patients")
    return data


# ─── NaN/Inf safety ───────────────────────────────────────────────────────────

def _safe_float(v) -> float | None:
    """Convert to float, return None if NaN or Inf."""
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _clean_dict(d: dict) -> dict:
    """Recursively sanitize a dict for JSON serialization."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _clean_dict(v)
        elif isinstance(v, float):
            out[k] = _safe_float(v)
        elif isinstance(v, np.floating):
            out[k] = _safe_float(float(v))
        elif isinstance(v, np.integer):
            out[k] = int(v)
        else:
            out[k] = v
    return out


# ─── Typed accessors ──────────────────────────────────────────────────────────

def get_umap() -> pd.DataFrame:
    return load_data()["umap"]


def get_patients() -> pd.DataFrame:
    return load_data()["patients"]


def get_metrics() -> dict:
    return load_data()["metrics"]


def get_attention_edges() -> pd.DataFrame:
    return load_data()["attention_edges"]


def get_attention_matrix() -> pd.DataFrame:
    return load_data()["attention_matrix"]


def get_cell_types() -> list[str]:
    return load_data()["cell_types"]


def get_celltype_stats() -> dict:
    return load_data()["celltype_stats"]


def get_patient(patient_id: str) -> dict | None:
    """Return a single patient row as a plain dict, or None if not found."""
    patients = get_patients()
    if patient_id not in patients.index:
        return None
    row = patients.loc[patient_id]

    # Per-patient UMAP cells
    umap = get_umap()
    patient_cells = umap[umap["patient_id"] == patient_id]

    result = {
        "patient_id":            patient_id,
        "response":              str(row["response"]),
        "n_cells":               int(row["n_cells"]),
        "n_pre":                 int(row["n_pre"]),
        "n_post":                int(row["n_post"]),
        "mean_escape":           _safe_float(row["mean_escape"]),
        "std_escape":            _safe_float(row["std_escape"]),
        "pct_high_escape":       _safe_float(row["pct_high_escape"]),
        "mean_uncertainty":      _safe_float(row["mean_uncertainty"]),
        "mean_escape_pre":       _safe_float(row["mean_escape_pre"]),
        "mean_escape_post":      _safe_float(row["mean_escape_post"]),
        "mean_attn_received":    _safe_float(row["mean_attn_received"]),
        # Dicts — recursively cleaned
        "cell_type_composition": _clean_dict(row["cell_type_composition"]),
        "escape_by_celltype":    _clean_dict(row["escape_by_celltype"]),
        "top_escape_celltypes":  _clean_dict(row["top_escape_celltypes"]),
        # Histograms
        "escape_histogram": {
            "counts":    np.histogram(patient_cells["escape_prob"], bins=20, range=(0, 1))[0].tolist(),
            "bin_edges": np.histogram(patient_cells["escape_prob"], bins=20, range=(0, 1))[1].tolist(),
        },
        "uncertainty_histogram": {
            "counts":    np.histogram(patient_cells["uncertainty"], bins=20)[0].tolist(),
            "bin_edges": np.histogram(patient_cells["uncertainty"], bins=20)[1].tolist(),
        },
    }
    return result


def get_patient_attention(patient_id: str) -> dict:
    """Return attention edges where both src and dst belong to this patient."""
    edges = get_attention_edges()
    mask  = (edges["src_patient"] == patient_id) & (edges["dst_patient"] == patient_id)
    patient_edges = edges[mask]

    pair_summary = (
        patient_edges.groupby(["src_celltype", "dst_celltype"])["weight"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_weight", "count": "n_edges"})
        .sort_values("mean_weight", ascending=False)
        .head(10)
        .to_dict(orient="records")
    )

    # Clean floats in pair_summary
    pair_summary = [
        {k: (_safe_float(v) if isinstance(v, (float, np.floating)) else
             int(v) if isinstance(v, (int, np.integer)) else v)
         for k, v in row.items()}
        for row in pair_summary
    ]

    top_edges = (
        patient_edges.nlargest(20, "weight")[
            ["src_celltype", "dst_celltype", "weight", "src_escape", "dst_escape"]
        ].to_dict(orient="records")
    )
    top_edges = [
        {k: (_safe_float(v) if isinstance(v, (float, np.floating)) else v)
         for k, v in row.items()}
        for row in top_edges
    ]

    return {
        "n_edges":      int(len(patient_edges)),
        "pair_summary": pair_summary,
        "top_edges":    top_edges,
    }
