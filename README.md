---
title: PulmoLens — Pulmonary Nodule Spatial Grounding & Corrective Multilingual RAG
emoji: 🫁
colorFrom: red
colorTo: white
sdk: docker
app_port: 8501
pinned: false
---

# 🫁 PulmoLens

## Pulmonary Nodule Spatial Grounding & Corrective Multilingual RAG

PulmoLens is an AI research prototype combining frozen pulmonary nodule spatial grounding with evidence-grounded multilingual retrieval and citation-constrained generative synthesis.

---

### Key Workspaces & Capabilities

1. **🏠 Overview**: System architecture summary, exact dataset metrics, problem formulation, and pipeline diagnostics.
2. **🔬 CT Scan Analysis**: 2D axial pulmonary CT slice localization using a frozen YOLO11s model trained on LIDC-IDRI annotations, extracting spatial bounding box coordinates and confidence levels.
3. **💬 Research Assistant**: Multilingual RAG engine querying structured LIDC records and peer-reviewed literature with dense FAISS search, neural BGE reranking, relevance strictness gating, and Gemini-powered citation-bounded synthesis with downloadable PDF reports.
4. **📚 Evidence Explorer**: Interactive data browser across all 3 indexed collections with deep record inspection.
5. **🏛️ System Architecture**: Visual interactive topology diagram of the dual computer vision and retrieval pipelines.
6. **⚡ System Status**: Real-time diagnostic meters verifying model readiness and index integrity.

---

### Pipeline Architecture

```
[Computer Vision Workflow]
CT Slice ──► Frozen YOLO11s ──► Spatial Bounding Boxes ──► LIDC Structured Verification

[Multilingual Retrieval Workflow]
Research Query (EN / UR / Roman UR)
  │
  ├──► Multilingual Embedder (MPNet Base v2 - 768 dim)
  ├──► FAISS Vector Search (LIDC + Global + Pakistan Stores)
  ├──► Neural Reranking (BGE Reranker v2 M3)
  ├──► Relevance Strictness Gating
  └──► Gemini 2.5 Flash Grounded Generation ──► Lexical Claim Verification ──► PDF Report
```

---

### Evidence Stores

* **LIDC Structured Evidence**: 240 structured clinical records
* **Global Literature Store**: 122 peer-reviewed articles
* **Pakistan Literature Store**: 294 regional oncology & pulmonary papers
* **Dense Embedding Dimension**: 768 (`paraphrase-multilingual-mpnet-base-v2`)
* **Cross-Encoder Reranker**: `BAAI/bge-reranker-v2-m3`

---

### Supported Languages

* English
* Urdu (اردو)
* Roman Urdu

---

### Quick Start & Setup

#### 1. Prerequisites
* Python 3.10+ (or [`uv`](https://github.com/astral-sh/uv) package manager)

#### 2. Installation
```bash
# Clone repository
git clone <YOUR_REPO_URL>
cd PulmoLens-main

# Install dependencies
uv sync
# OR using pip:
pip install -r requirements.txt
```

#### 3. API Key Configuration
To enable the Gemini research synthesis assistant, obtain a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey) and provide it in either:

* **`.streamlit/secrets.toml`**:
  ```toml
  GEMINI_API_KEY = "your_actual_gemini_api_key_here"
  ```
* **`.env` file**:
  ```bash
  GEMINI_API_KEY="your_actual_gemini_api_key_here"
  ```

#### 4. Launch Application
```bash
uv run streamlit run app.py
# OR
streamlit run app.py
```
The application will launch and open your default browser at `http://localhost:8501`.

---

### Safety & Scope Notice

PulmoLens is a **research prototype**. It is not clinically validated and is not intended to provide patient-specific diagnosis, malignancy determination, treatment recommendations, or clinical decision-making. 

YOLO detections represent computational spatial-grounding candidate regions only. When sufficient evidence is unavailable, PulmoLens is designed to return an insufficient-evidence state rather than hallucinating or inventing citations.
