
---
title: PulmoLens — Pulmonary Nodule Spatial Grounding & Corrective Multilingual RAG
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🫁 PulmoLens

## Pulmonary Nodule Spatial Grounding & Corrective Multilingual RAG

PulmoLens is a research prototype combining frozen pulmonary nodule
spatial grounding with evidence-grounded multilingual retrieval and
citation-constrained generation.

### Pipeline

CT Slice
→ Frozen YOLO11s
→ LIDC Structured Evidence

Research Query
→ FAISS Retrieval
→ BGE Reranking
→ Relevance Gate
→ Evidence Bundle
→ Gemini
→ Claim Validation
→ Abstention

### Evidence Stores

- LIDC structured evidence: 240 records
- Global literature: 122 vectors
- Pakistan literature: 294 vectors
- Embedding dimension: 768

### Languages

- English
- Urdu
- Roman Urdu

### Safety Scope

PulmoLens is a research prototype.

It is not clinically validated and is not intended to provide
patient-specific diagnosis, malignancy determination, treatment
recommendations, or clinical decision-making.

YOLO output represents computational spatial-grounding results.

When sufficient evidence is unavailable, PulmoLens is designed to
return an insufficient-evidence state rather than inventing evidence.
