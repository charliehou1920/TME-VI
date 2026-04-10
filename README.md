---
title: TME-VI · Tumor Immune Escape Decoder
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# TME-VI — Tumor Microenvironment Variational Inference

A Graph Attention Network Decoder Virtual Instrument for predicting immune escape in the tumor microenvironment, built within the AI Virtual Cell (AIVC) framework.

## Dataset
GSE120575 (Sade-Feldman et al. 2018) — 16,290 CD45⁺ immune cells from 32 melanoma patients undergoing anti-PD-1 immunotherapy.

## Pipeline
1. **scVI** → 50-dim Cellular Universal Representations
2. **GAT Decoder VI** → cell-level immune escape prediction
3. **MC Dropout** (T=50) → epistemic uncertainty quantification
4. **LLM layer** → Claude-generated clinical & researcher reports

## Performance
- Cell-level AUROC: 0.8852
- Confidence-filtered AUROC (top 20%): ~0.97
- Patient-level AUROC: ~0.78
