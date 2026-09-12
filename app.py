
import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import faiss

from PIL import Image
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer, CrossEncoder

from google import genai
from google.genai import types

from pydantic import BaseModel, Field


# ============================================================
# PULMOLENS CONFIGURATION
# ============================================================

APP_NAME = "PulmoLens"

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "yolo11s_lidc_best.pt"

INDEX_DIR = BASE_DIR / "index"

DATA_PATH = BASE_DIR / "data" / "lidc_structured_evidence.csv"

EMBEDDER_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

RERANKER_NAME = "BAAI/bge-reranker-v2-m3"

GEMINI_MODEL = "gemini-2.5-flash"

DENSE_TOP_K = 20
FINAL_TOP_K = 8

ABS_MIN_DENSE = 0.20
DEFAULT_RELATIVE_GATE = 0.70


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="PulmoLens",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .pulmo-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0;
    }

    .pulmo-subtitle {
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 1.4rem;
    }

    .evidence-card {
        padding: 14px;
        margin: 8px 0;
        border-radius: 10px;
        border: 1px solid #dbeafe;
        background: #f8fbff;
    }

    .safe-box {
        padding: 12px;
        border-radius: 8px;
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
    }

    .warning-box {
        padding: 12px;
        border-radius: 8px;
        background: #fffbeb;
        border: 1px solid #fde68a;
    }

    .danger-box {
        padding: 12px;
        border-radius: 8px;
        background: #fef2f2;
        border: 1px solid #fecaca;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="pulmo-title">🫁 PulmoLens</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="pulmo-subtitle">'
    "Pulmonary Nodule Spatial Grounding & Corrective Multilingual RAG"
    "</div>",
    unsafe_allow_html=True,
)

st.info(
    "Research prototype. PulmoLens is not a clinically validated "
    "diagnostic system and must not be used as a substitute for "
    "professional medical interpretation or treatment decisions."
)


# ============================================================
# PYDANTIC OUTPUT SCHEMA
# ============================================================

class Claim(BaseModel):
    text: str = Field(min_length=1)
    citations: List[str] = Field(default_factory=list)


class GroundedResponse(BaseModel):
    answer: str = ""
    claims: List[Claim] = Field(default_factory=list)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_yolo():

    if not MODEL_PATH.exists():
        return None

    return YOLO(str(MODEL_PATH))


@st.cache_resource
def load_retrieval_system():

    embedder = SentenceTransformer(EMBEDDER_NAME)

    reranker = CrossEncoder(
        RERANKER_NAME,
        max_length=512,
    )

    stores = {}

    mappings = {
        "lidc": (
            INDEX_DIR / "lidc.index",
            INDEX_DIR / "lidc_metadata.json",
        ),
        "literature": (
            INDEX_DIR / "literature.index",
            INDEX_DIR / "literature_metadata.json",
        ),
        "pakistan": (
            INDEX_DIR / "pakistan.index",
            INDEX_DIR / "pakistan_metadata.json",
        ),
    }

    for store_name, (index_path, metadata_path) in mappings.items():

        if not index_path.exists():
            raise FileNotFoundError(index_path)

        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)

        index = faiss.read_index(str(index_path))

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if index.ntotal != len(metadata):
            raise RuntimeError(
                f"{store_name}: FAISS/metadata mismatch "
                f"{index.ntotal} != {len(metadata)}"
            )

        stores[store_name] = {
            "index": index,
            "metadata": metadata,
        }

    return embedder, reranker, stores


yolo_model = load_yolo()
embedder, reranker, stores = load_retrieval_system()


# ============================================================
# DATA
# ============================================================

@st.cache_data
def load_structured_evidence():

    if not DATA_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(DATA_PATH)


structured_df = load_structured_evidence()


# ============================================================
# HELPERS
# ============================================================

def detect_language(query: str) -> str:

    if re.search(r"[\u0600-\u06FF]", query):
        return "urdu_script"

    q = query.lower()

    roman_markers = [
        "kya",
        "hai",
        "hain",
        "mein",
        "ke",
        "ki",
        "ka",
        "par",
        "ko",
    ]

    if any(word in q.split() for word in roman_markers):
        return "roman_urdu"

    return "english"


def extract_patient_id(query: str):

    match = re.search(
        r"(LIDC-IDRI-\d{4}|\b\d{4}\b)",
        query,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    raw = match.group(1).upper()

    if raw.isdigit():
        return f"LIDC-IDRI-{raw.zfill(4)}"

    return raw


def extract_nodule_id(query: str):

    patterns = [
        r"nodule\s*(?:id)?\s*[:#]?\s*(\d+)",
        r"nodule[- ]?(\d+)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            query,
            flags=re.IGNORECASE,
        )

        if match:
            return str(int(match.group(1)))

    return None


def exact_lidc_lookup(patient_id: str, nodule_id=None):

    results = []

    for meta in stores["lidc"]["metadata"]:

        meta_patient = str(
            meta.get("patient_id", "")
        ).upper()

        if meta_patient != patient_id.upper():
            continue

        if nodule_id is not None:

            meta_nodule = str(
                meta.get("nodule_id", "")
            ).strip()

            if meta_nodule != str(nodule_id):
                continue

        results.append(
            {
                "id": "D001",
                "store": "LIDC Dataset Evidence",
                "score": None,
                "rerank_score": None,
                "text": meta.get("text", ""),
                "title": meta.get("title", ""),
                "source_id": meta.get("source_id", ""),
                "patient_id": meta.get("patient_id", ""),
                "nodule_id": meta.get("nodule_id", ""),
                "exact": True,
            }
        )

    return results


def dense_retrieve(
    query: str,
    language: str,
    relative_ratio: float,
):

    q_vec = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)

    faiss.normalize_L2(q_vec)

    candidates = []

    # General research routing:
    # global + Pakistan
    for store_name in ["literature", "pakistan"]:

        if store_name not in stores:
            continue

        scores, indices = stores[store_name]["index"].search(
            q_vec,
            DENSE_TOP_K,
        )

        metadata = stores[store_name]["metadata"]

        for score, idx in zip(scores[0], indices[0]):

            if idx < 0:
                continue

            item = metadata[idx]

            candidates.append(
                {
                    "store": store_name,
                    "dense_score": float(score),
                    "metadata": item,
                }
            )

    if not candidates:
        return []

    top_score = max(
        x["dense_score"]
        for x in candidates
    )

    language_floor = (
        ABS_MIN_DENSE
        if language == "english"
        else 0.22
    )

    effective_threshold = max(
        language_floor,
        relative_ratio * top_score,
    )

    candidates = [
        x
        for x in candidates
        if x["dense_score"] >= effective_threshold
    ]

    if not candidates:
        return []

    # --------------------------------------------------------
    # BGE reranking
    # --------------------------------------------------------

    rerank_pairs = []

    for candidate in candidates:

        text = candidate["metadata"].get(
            "text",
            "",
        )

        rerank_pairs.append(
            [query, text]
        )

    rerank_scores = reranker.predict(
        rerank_pairs,
        show_progress_bar=False,
    )

    for candidate, rerank_score in zip(
        candidates,
        rerank_scores,
    ):
        candidate["rerank_score"] = float(
            rerank_score
        )

    candidates.sort(
        key=lambda x: x["rerank_score"],
        reverse=True,
    )

    # --------------------------------------------------------
    # Final relevance gate
    # --------------------------------------------------------

    if not candidates:
        return []

    best_rerank = candidates[0]["rerank_score"]

    rerank_floor = max(
        0.20,
        0.70 * best_rerank,
    )

    selected = [
        c
        for c in candidates
        if c["rerank_score"] >= rerank_floor
    ]

    selected = selected[:FINAL_TOP_K]

    output = []

    for i, candidate in enumerate(selected, start=1):

        meta = candidate["metadata"]

        output.append(
            {
                "id": f"E{i:03d}",
                "store": (
                    "Global Literature"
                    if candidate["store"] == "literature"
                    else "Pakistan Literature"
                ),
                "score": candidate["dense_score"],
                "rerank_score": candidate["rerank_score"],
                "text": meta.get("text", ""),
                "title": meta.get("title", ""),
                "source_id": meta.get("source_id", ""),
                "patient_id": meta.get("patient_id", ""),
                "nodule_id": meta.get("nodule_id", ""),
                "citation": meta.get("citation", ""),
                "pmid": meta.get("pmid", ""),
                "doi": meta.get("doi", ""),
                "source_url": meta.get("source_url", ""),
                "exact": False,
            }
        )

    return output


def retrieve(
    query: str,
    relative_ratio: float,
):

    language = detect_language(query)

    patient_id = extract_patient_id(query)

    nodule_id = extract_nodule_id(query)

    # --------------------------------------------------------
    # Exact LIDC entity route
    # --------------------------------------------------------

    if patient_id:

        exact = exact_lidc_lookup(
            patient_id,
            nodule_id,
        )

        return {
            "language": language,
            "patient_id": patient_id,
            "nodule_id": nodule_id,
            "evidence": exact,
            "grounded": bool(exact),
            "route": "exact_lidc_entity",
        }

    # --------------------------------------------------------
    # General research route
    # --------------------------------------------------------

    evidence = dense_retrieve(
        query,
        language,
        relative_ratio,
    )

    return {
        "language": language,
        "patient_id": None,
        "nodule_id": None,
        "evidence": evidence,
        "grounded": bool(evidence),
        "route": "hybrid_literature_retrieval",
    }


# ============================================================
# SAFETY FILTER
# ============================================================

UNSAFE_PATTERNS = [
    r"\bdefinitely\s+(has|have)\s+cancer\b",
    r"\bpatient\s+has\s+cancer\b",
    r"\bdiagnosed\s+with\s+cancer\b",
    r"\bmust\s+have\s+cancer\b",
    r"\bwhat\s+treatment\s+should\b",
    r"\bshould\s+(start|receive)\s+treatment\b",
    r"\bstart\s+treatment\b",
    r"\btreatment\s+should\b",
]


def unsafe_request(query: str) -> bool:

    return any(
        re.search(
            pattern,
            query,
            flags=re.IGNORECASE,
        )
        for pattern in UNSAFE_PATTERNS
    )


# ============================================================
# CLAIM SUPPORT
# ============================================================

def lexical_support(
    claim: str,
    evidence_text: str,
) -> bool:

    claim_tokens = set(
        re.findall(
            r"\b[a-zA-Z0-9]{4,}\b",
            claim.lower(),
        )
    )

    evidence_tokens = set(
        re.findall(
            r"\b[a-zA-Z0-9]{4,}\b",
            evidence_text.lower(),
        )
    )

    if not claim_tokens:
        return True

    overlap = len(
        claim_tokens & evidence_tokens
    )

    ratio = overlap / len(claim_tokens)

    return ratio >= 0.18


# ============================================================
# GEMINI GENERATION
# ============================================================

def generate_grounded_answer(
    query: str,
    evidence: List[Dict[str, Any]],
    language: str,
    api_key: str,
):

    if not evidence:
        return None, "INSUFFICIENT_EVIDENCE"

    client = genai.Client(api_key=api_key)

    evidence_bundle = []

    for item in evidence:

        citation = item["id"]

        evidence_bundle.append(
            f"[{citation}] "
            f"{item.get('text', '')}"
        )

    joined_evidence = "\n\n".join(
        evidence_bundle
    )

    language_name = {
        "english": "English",
        "urdu_script": "Urdu",
        "roman_urdu": "Roman Urdu",
    }.get(
        language,
        "English",
    )

    system_instruction = f"""
You are the evidence-bounded research assistant inside PulmoLens.

PulmoLens is a research prototype.

Answer strictly from the supplied Evidence Bundle.

Requested language:
{language_name}

Rules:

1. Do not invent facts.
2. Do not invent citation IDs.
3. Every factual claim must contain one or more citation IDs.
4. Only use citation IDs that exist in the Evidence Bundle.
5. Do not diagnose cancer or malignancy.
6. Do not recommend patient-specific treatment.
7. Do not invent measurements.
8. Do not infer information that is absent from the evidence.
9. When the evidence does not support a claim, omit that claim.
10. When evidence is insufficient, return no claims.
11. Preserve LIDC IDs, PMID, DOI and other identifiers exactly.
"""

    user_prompt = f"""
QUERY:
{query}

EVIDENCE BUNDLE:
{joined_evidence}
"""

    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=GroundedResponse,
                temperature=0.0,
            ),
        )

        data = GroundedResponse.model_validate_json(
            response.text
        )

        return data, None

    except Exception as exc:

        return None, str(exc)


# ============================================================
# VALIDATE GENERATED CLAIMS
# ============================================================

def validate_response(
    response: GroundedResponse,
    evidence: List[Dict[str, Any]],
):

    valid_ids = {
        item["id"]
        for item in evidence
    }

    surviving = []

    for claim in response.claims:

        if not claim.citations:
            continue

        if not set(claim.citations).issubset(valid_ids):
            continue

        cited_text = " ".join(
            item["text"]
            for item in evidence
            if item["id"] in claim.citations
        )

        if not lexical_support(
            claim.text,
            cited_text,
        ):
            continue

        surviving.append(
            claim
        )

    return surviving


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("PulmoLens Control")

output_language = st.sidebar.selectbox(
    "Output Language",
    [
        "English",
        "Urdu",
        "Roman Urdu",
    ],
)

relative_ratio = st.sidebar.slider(
    "Relative Retrieval Gate",
    min_value=0.50,
    max_value=0.90,
    value=DEFAULT_RELATIVE_GATE,
    step=0.05,
)

gemini_key = os.getenv(
    "GEMINI_API_KEY",
    "",
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🫁 CT Slice Grounding",
        "🔎 Evidence Retrieval",
        "📝 Grounded Research Answer",
    ]
)


# ============================================================
# TAB 1 — YOLO
# ============================================================

with tab1:

    st.subheader(
        "CT Slice Nodule Spatial Grounding"
    )

    st.caption(
        "Frozen YOLO11s research model. "
        "This interface performs 2D CT-slice inference; "
        "it is not a complete 3D CT diagnostic engine."
    )

    uploaded = st.file_uploader(
        "Upload CT slice",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded:

        image = Image.open(uploaded).convert(
            "RGB"
        )

        left, right = st.columns(2)

        with left:

            st.image(
                image,
                caption="Input CT Slice",
                use_container_width=True,
            )

        with right:

            if yolo_model is None:

                st.error(
                    "YOLO checkpoint is unavailable."
                )

            else:

                results = yolo_model(
                    image,
                    verbose=False,
                )

                plotted = results[0].plot()

                st.image(
                    plotted,
                    caption="PulmoLens YOLO11s Spatial Grounding",
                    use_container_width=True,
                )

                boxes = results[0].boxes

                if boxes is None or len(boxes) == 0:

                    st.warning(
                        "No nodule candidates detected "
                        "above the model threshold."
                    )

                else:

                    st.success(
                        f"{len(boxes)} nodule candidate(s) detected."
                    )

                    rows = []

                    for box in boxes:

                        conf = float(
                            box.conf[0]
                        )

                        xyxy = box.xyxy[0].cpu().numpy()

                        rows.append(
                            {
                                "confidence": round(
                                    conf,
                                    4,
                                ),
                                "x1": round(
                                    float(xyxy[0]),
                                    2,
                                ),
                                "y1": round(
                                    float(xyxy[1]),
                                    2,
                                ),
                                "x2": round(
                                    float(xyxy[2]),
                                    2,
                                ),
                                "y2": round(
                                    float(xyxy[3]),
                                    2,
                                ),
                            }
                        )

                    st.dataframe(
                        pd.DataFrame(rows),
                        use_container_width=True,
                    )


# ============================================================
# TAB 2 — RETRIEVAL
# ============================================================

with tab2:

    st.subheader(
        "Evidence Retrieval & Corrective Gate"
    )

    presets = [
        "Custom Query",
        "What imaging features are discussed for pulmonary nodules?",
        "What structured evidence is available for LIDC-IDRI-0001?",
        "Tell me about LIDC-IDRI-0008 nodule 1.",
        "پلمونری نوڈول کی سی ٹی خصوصیات کیا ہیں؟",
        "lung nodule ki CT features kya hain?",
        "What is the capital of France?",
        "Does LIDC-IDRI-9999 have a measured nodule?",
    ]

    preset = st.selectbox(
        "Preset query",
        presets,
    )

    query = st.text_area(
        "Research query",
        value="" if preset == "Custom Query" else preset,
        height=100,
    )

    run_retrieval = st.button(
        "🔎 Run PulmoLens Retrieval",
        type="primary",
    )

    if run_retrieval:

        if not query.strip():

            st.error(
                "Please enter a query."
            )

        elif unsafe_request(query):

            st.error(
                "This request is outside PulmoLens' "
                "safe research scope."
            )

            st.session_state.pop(
                "pulmo_retrieval",
                None,
            )

        else:

            result = retrieve(
                query=query,
                relative_ratio=relative_ratio,
            )

            st.session_state[
                "pulmo_retrieval"
            ] = result

            st.session_state[
                "pulmo_query"
            ] = query

    if "pulmo_retrieval" in st.session_state:

        result = st.session_state[
            "pulmo_retrieval"
        ]

        evidence = result["evidence"]

        st.markdown("---")

        if not result["grounded"]:

            st.error(
                "⛔ INSUFFICIENT_EVIDENCE"
            )

            if result["patient_id"]:

                st.caption(
                    f"No exact structured evidence was found for "
                    f"{result['patient_id']}."
                )

        else:

            st.success(
                f"✅ Evidence grounding passed — "
                f"{len(evidence)} evidence item(s)"
            )

            st.write(
                f"Route: `{result['route']}`"
            )

            if result["patient_id"]:
                st.write(
                    f"Patient: `{result['patient_id']}`"
                )

            if result["nodule_id"] is not None:
                st.write(
                    f"Nodule: `{result['nodule_id']}`"
                )

            for item in evidence:

                score_text = ""

                if item.get("score") is not None:
                    score_text += (
                        f"Dense: {item['score']:.4f}"
                    )

                if item.get("rerank_score") is not None:
                    score_text += (
                        f" | BGE: "
                        f"{item['rerank_score']:.4f}"
                    )

                st.markdown(
                    f"""
                    <div class="evidence-card">
                    <b>[{item['id']}] {item['store']}</b><br>
                    {score_text}<br><br>
                    {item.get('text', '')}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ============================================================
# TAB 3 — GENERATION
# ============================================================

with tab3:

    st.subheader(
        "Evidence-Constrained Research Answer"
    )

    retrieval = st.session_state.get(
        "pulmo_retrieval"
    )

    query = st.session_state.get(
        "pulmo_query",
        "",
    )

    if not retrieval or not retrieval["grounded"]:

        st.warning(
            "No validated evidence bundle is available."
        )

    else:

        st.write(
            f"**Query:** {query}"
        )

        if st.button(
            "⚡ Generate Grounded Answer",
            type="primary",
        ):

            if not gemini_key:

                st.error(
                    "GEMINI_API_KEY is not configured."
                )

            else:

                with st.spinner(
                    "Generating evidence-constrained answer..."
                ):

                    generated, error = (
                        generate_grounded_answer(
                            query=query,
                            evidence=retrieval["evidence"],
                            language=(
                                detect_language(query)
                            ),
                            api_key=gemini_key,
                        )
                    )

                if error:

                    st.error(
                        f"Generation failed: {error}"
                    )

                elif generated is None:

                    st.error(
                        "INSUFFICIENT_EVIDENCE"
                    )

                else:

                    valid_claims = validate_response(
                        generated,
                        retrieval["evidence"],
                    )

                    if not valid_claims:

                        st.error(
                            "⛔ No claims survived the "
                            "grounding validation gate."
                        )

                    else:

                        st.success(
                            f"✅ {len(valid_claims)} "
                            "grounded claim(s) validated."
                        )

                        for claim in valid_claims:

                            citations = " ".join(
                                f"[{c}]"
                                for c in claim.citations
                            )

                            st.markdown(
                                f"**• {claim.text}** "
                                f"{citations}"
                            )

                        st.markdown("---")

                        st.caption(
                            "Citations shown here refer to the "
                            "retrieved PulmoLens evidence bundle."
                        )
