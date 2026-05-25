# TME-VI — Tumor Microenvironment Variational Inference

> An uncertainty-aware **Decoder Virtual Instrument** for predicting anti-PD-1 immunotherapy response in melanoma, built on the [AI Virtual Cell (AIVC)](https://www.cell.com/cell/fulltext/S0092-8674(24)01271-4) framework (Bunne et al., Cell 2024).

**[Live Demo →](https://huggingface.co/spaces/Jiang-Jun/tme-vi)**

---

## Motivation

Immune checkpoint blockade (anti-PD-1 therapy) achieves durable responses in only ~40% of metastatic melanoma patients. The composition and spatial organization of the tumor microenvironment (TME) — particularly the balance between cytotoxic effector T cells and exhausted or regulatory populations — is a key determinant of response. TME-VI asks: *can we learn a cell-level representation of the immune TME that predicts which patients will respond, and quantify our uncertainty in that prediction?*

---

## Architecture

TME-VI implements the **Decoder Virtual Instrument** pattern from the AIVC framework: a learned cellular encoder feeds a task-specific decoder that maps from latent cell state to a clinically meaningful readout.

```
Raw scRNA-seq (16,290 cells × 18,000 genes)
        │
        ▼
  [ Quality Control ]  ── qc.py
  (doublet removal, mt% filter, min counts)
        │
        ▼
  [ Normalization ]  ── normalize.py
  (total-count normalization → log1p, HVG selection: 2,000 genes)
        │
        ▼
  [ Graph Construction ]  ── build_graph.py
  (PCA-50 → sklearn KNN k=15 → Leiden clustering)
        │
        ▼
  [ scVI Encoder ]  ── 01_train_scvi.ipynb
  (Variational Autoencoder: 2,000 HVGs → 10-dim latent space)
  (batch correction across 32 patients built-in)
        │
        ▼
  [ GATDecoderVI ]  ── 02_train_gnn.ipynb
  (Graph Attention Network over cell neighborhood graph)
  (3 GAT layers, 8 attention heads, hidden dim 256)
  (input: 10-dim scVI latent + 8-dim cell type embedding)
        │
        ▼
  [ MC Dropout UQ ]  ── 05_uncertainty.ipynb
  (T=50 stochastic forward passes at inference)
  (mean prediction + epistemic uncertainty estimate)
        │
        ▼
  Cell-level immunotherapy response score + confidence
        │
        ▼
  [ Patient Aggregation ]
  (weighted mean over high-confidence cells per patient)
        │
        ▼
  Patient-level response prediction (responder / non-responder)
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source | GSE120575 — Sade-Feldman et al., *Cell* 2018 |
| Cells | 16,290 CD45+ tumor-infiltrating immune cells |
| Patients | 32 (16 responders / 16 non-responders) |
| Cancer type | Metastatic melanoma |
| Treatment | Anti-PD-1 (pembrolizumab / nivolumab) |

**Cell types annotated (8 classes):**
Cytotoxic CD8 T · CD4 T · Naive/Memory CD8 T · B cell · Macrophage · γδ T · Plasma cell · pDC

> Cluster annotations were grounded in marker gene evidence (e.g. Leiden cluster 4 corrected from Treg → Macrophage based on *CD68*, *LYZ*, *CSF1R* expression).

---

## Results

| Metric | Value |
|--------|-------|
| Cell-level AUROC | **0.885** |
| Confidence-filtered AUROC (top 20%) | **0.970** |
| Patient-level AUROC | **0.780** |

Uncertainty filtering retains the top 20% of cells by prediction confidence (lowest MC Dropout variance), yielding a high-precision subset suitable for clinical decision support framing.

---

## Interpretability

GAT attention weights from the final layer are extracted and visualized as a cell–cell interaction graph (top 5% edges; 17,861 edges retained). This surfaces biologically meaningful co-attention patterns — e.g. Cytotoxic CD8 T cells receiving high attention from adjacent exhausted populations in non-responders.

See `03_interpretability.ipynb` for attention analysis and UMAP visualization.

---

## Deployment Stack

```
HuggingFace Spaces (Docker SDK)
├── backend/
│   ├── api.py              # FastAPI endpoints
│   ├── data_loader.py      # app_data.pkl ingestion
│   └── llm_report.py       # Claude API streaming report generation
└── frontend/
    └── app.py              # Streamlit — 4 pages:
                            #   Overview (ROC, UQ analysis)
                            #   Cell Atlas (interactive UMAP)
                            #   Patients (32-patient table)
                            #   Report (LLM-generated clinical summary)
```

---

## AIVC Framework Positioning

TME-VI sits at the **cellular scale** of the multi-scale AIVC stack:

| Scale | Instrument | Project |
|-------|-----------|---------|
| 🔬 Molecular | Encoder VI | MutaGraph |
| 🧬 **Cellular** | **Decoder VI** | **TME-VI** |
| 📚 Knowledge | RAG Agent | CellScout |

The LLM layer (Claude API) functions as an **interpretability translator** — converting GNN numerical outputs into human-readable biological and clinical language, consistent with the AIVC vision of AI instruments that communicate findings to domain experts.

---

## Next Steps

Pan-cancer extension targeting NSCLC (GSE176021, Caushi et al. 2021) via CELLxGENE Census, enabling zero-shot transfer evaluation across cancer types and a universal immune TME representation.

---

## Stack

`scVI-tools` `PyTorch Geometric` `Scanpy/AnnData` `FastAPI` `Streamlit` `Anthropic Claude API` `Docker` `HuggingFace Spaces`

---

## Citation

If you build on this work, please also cite the AIVC framework:

```
Bunne et al. (2024). How to Build the Virtual Cell with Artificial Intelligence.
Cell, 189(7), 1151–1166. https://doi.org/10.1016/j.cell.2024.05.031
```
