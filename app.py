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

# PDF reporting
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)


# ============================================================
# PULMOLENS CONFIGURATION
# ============================================================

APP_NAME = "PulmoLens"

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "yolo11s_lidc_best.pt"
INDEX_DIR = BASE_DIR / "index"
DATA_PATH = BASE_DIR / "data" / "lidc_structured_evidence.csv"

EMBEDDER_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-mpnet-base-v2"
)

RERANKER_NAME = "BAAI/bge-reranker-v2-m3"
GEMINI_MODEL = "gemini-2.5-flash"

DENSE_TOP_K = 20
FINAL_TOP_K = 8

ABS_MIN_DENSE = 0.20
DEFAULT_RELATIVE_GATE = 0.70


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PulmoLens — Research Assistant",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL UI
# ============================================================

st.markdown(
    """
    <style>

    :root {
        --red: #B42318;
        --red-dark: #8A1C14;
        --red-soft: #FDECEC;
        --navy: #17324D;
        --text: #172033;
        --muted: #667085;
        --border: #E5E7EB;
        --surface: #FFFFFF;
        --surface-soft: #F8FAFC;
        --green: #15803D;
        --green-soft: #ECFDF3;
        --amber: #B45309;
        --amber-soft: #FFF7ED;
    }

    .stApp {
        background: #F6F8FB;
    }

    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E5E7EB;
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    .brand-wrap {
        background: linear-gradient(
            135deg,
            #B42318 0%,
            #8A1C14 100%
        );
        color: white;
        padding: 18px;
        border-radius: 18px;
        margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(180, 35, 24, 0.18);
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 13px;
    }

    .logo {
        width: 54px;
        height: 54px;
        min-width: 54px;
        border-radius: 14px;
        background: white;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #B42318;
        font-size: 30px;
        font-weight: 800;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }

    .brand-name {
        font-size: 27px;
        font-weight: 800;
        line-height: 1;
    }

    .brand-sub {
        font-size: 12px;
        margin-top: 6px;
        opacity: 0.92;
    }

    .hero {
        background:
            linear-gradient(
                135deg,
                rgba(180,35,24,0.08),
                rgba(255,255,255,0.96)
            );
        border: 1px solid #F0D4D1;
        border-radius: 22px;
        padding: 30px;
        margin-bottom: 24px;
    }

    .hero-kicker {
        color: var(--red);
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .hero-title {
        font-size: 38px;
        font-weight: 850;
        color: var(--navy);
        margin-top: 8px;
        margin-bottom: 8px;
    }

    .hero-text {
        font-size: 16px;
        color: #475467;
        line-height: 1.7;
        max-width: 900px;
    }

    .card {
        background: white;
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 21px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(16,24,40,0.03);
    }

    .card-title {
        color: var(--navy);
        font-size: 18px;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .card-text {
        color: var(--muted);
        line-height: 1.55;
        font-size: 14px;
    }

    .metric-card {
        background: white;
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 19px;
        text-align: left;
        min-height: 115px;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 800;
        color: var(--red);
        line-height: 1;
    }

    .metric-label {
        margin-top: 8px;
        font-weight: 700;
        color: var(--navy);
        font-size: 14px;
    }

    .metric-help {
        margin-top: 3px;
        color: var(--muted);
        font-size: 12px;
    }

    .status-good {
        background: var(--green-soft);
        border: 1px solid #A7F3D0;
        color: #166534;
        padding: 11px 14px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 13px;
        margin-bottom: 8px;
    }

    .status-warn {
        background: var(--amber-soft);
        border: 1px solid #FED7AA;
        color: #9A3412;
        padding: 11px 14px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 13px;
        margin-bottom: 8px;
    }

    .status-bad {
        background: var(--red-soft);
        border: 1px solid #FECACA;
        color: #991B1B;
        padding: 11px 14px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 13px;
        margin-bottom: 8px;
    }

    .step {
        background: white;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 17px;
        height: 100%;
    }

    .step-number {
        width: 31px;
        height: 31px;
        border-radius: 50%;
        background: var(--red-soft);
        color: var(--red);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .step-title {
        font-weight: 750;
        color: var(--navy);
        margin-bottom: 4px;
    }

    .step-text {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.5;
    }

    .evidence-card {
        background: white;
        border: 1px solid #E6E9EF;
        border-left: 4px solid var(--red);
        border-radius: 13px;
        padding: 15px;
        margin: 10px 0;
    }

    .evidence-id {
        color: var(--red);
        font-weight: 800;
        font-size: 13px;
    }

    .evidence-source {
        color: var(--navy);
        font-weight: 750;
        font-size: 13px;
        margin-left: 7px;
    }

    .evidence-text {
        color: #475467;
        line-height: 1.6;
        font-size: 13px;
        margin-top: 8px;
    }

    .answer-card {
        background: #FFFFFF;
        border: 1px solid #E4E7EC;
        border-radius: 18px;
        padding: 23px;
        margin-top: 10px;
    }

    .answer-title {
        color: var(--navy);
        font-size: 19px;
        font-weight: 800;
        margin-bottom: 15px;
    }

    .claim {
        background: #FAFAFA;
        border: 1px solid #EAECF0;
        border-radius: 12px;
        padding: 13px 15px;
        margin-bottom: 10px;
        color: #344054;
        line-height: 1.6;
    }

    .architecture-box {
        background: #101828;
        color: white;
        border-radius: 18px;
        padding: 24px;
        line-height: 1.8;
        font-family: monospace;
        white-space: pre-wrap;
        overflow-x: auto;
    }

    .small-note {
        color: var(--muted);
        font-size: 12px;
    }

    .research-badge {
        display: inline-block;
        background: #F2F4F7;
        color: #344054;
        border-radius: 999px;
        padding: 5px 10px;
        font-size: 11px;
        font-weight: 700;
        margin-right: 5px;
    }

    div.stButton > button {
        border-radius: 11px;
        font-weight: 750;
        min-height: 44px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PYDANTIC OUTPUT
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
            raise FileNotFoundError(str(index_path))

        if not metadata_path.exists():
            raise FileNotFoundError(str(metadata_path))

        index = faiss.read_index(str(index_path))

        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as f:
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


# Graceful loading
yolo_model = None
embedder = None
reranker = None
stores = {}

try:
    yolo_model = load_yolo()
except Exception as exc:
    st.session_state["yolo_error"] = str(exc)

try:
    retrieval_result = load_retrieval_system()
    embedder, reranker, stores = retrieval_result
except Exception as exc:
    st.session_state["retrieval_error"] = str(exc)


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

    if any(
        word in q.split()
        for word in roman_markers
    ):
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


def exact_lidc_lookup(
    patient_id: str,
    nodule_id=None,
):

    results = []

    if "lidc" not in stores:
        return results

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
                "source_id": meta.get(
                    "source_id",
                    "",
                ),
                "patient_id": meta.get(
                    "patient_id",
                    "",
                ),
                "nodule_id": meta.get(
                    "nodule_id",
                    "",
                ),
                "citation": meta.get(
                    "citation",
                    "",
                ),
                "pmid": meta.get(
                    "pmid",
                    "",
                ),
                "doi": meta.get(
                    "doi",
                    "",
                ),
                "source_url": meta.get(
                    "source_url",
                    "",
                ),
                "exact": True,
            }
        )

    return results


def dense_retrieve(
    query: str,
    language: str,
    relative_ratio: float,
):

    if embedder is None or reranker is None:
        return []

    q_vec = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)

    faiss.normalize_L2(q_vec)

    candidates = []

    for store_name in [
        "literature",
        "pakistan",
    ]:

        if store_name not in stores:
            continue

        scores, indices = stores[
            store_name
        ]["index"].search(
            q_vec,
            DENSE_TOP_K,
        )

        metadata = stores[
            store_name
        ]["metadata"]

        for score, idx in zip(
            scores[0],
            indices[0],
        ):

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

    rerank_pairs = []

    for candidate in candidates:

        text = candidate[
            "metadata"
        ].get(
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

        candidate[
            "rerank_score"
        ] = float(rerank_score)

    candidates.sort(
        key=lambda x: x["rerank_score"],
        reverse=True,
    )

    if not candidates:
        return []

    best_rerank = candidates[
        0
    ]["rerank_score"]

    rerank_floor = max(
        0.20,
        0.70 * best_rerank,
    )

    selected = [
        c
        for c in candidates
        if c["rerank_score"]
        >= rerank_floor
    ]

    selected = selected[:FINAL_TOP_K]

    output = []

    for i, candidate in enumerate(
        selected,
        start=1,
    ):

        meta = candidate[
            "metadata"
        ]

        output.append(
            {
                "id": f"E{i:03d}",
                "store": (
                    "Global Literature"
                    if candidate[
                        "store"
                    ] == "literature"
                    else "Pakistan Literature"
                ),
                "score": candidate[
                    "dense_score"
                ],
                "rerank_score": candidate[
                    "rerank_score"
                ],
                "text": meta.get(
                    "text",
                    "",
                ),
                "title": meta.get(
                    "title",
                    "",
                ),
                "source_id": meta.get(
                    "source_id",
                    "",
                ),
                "patient_id": meta.get(
                    "patient_id",
                    "",
                ),
                "nodule_id": meta.get(
                    "nodule_id",
                    "",
                ),
                "citation": meta.get(
                    "citation",
                    "",
                ),
                "pmid": meta.get(
                    "pmid",
                    "",
                ),
                "doi": meta.get(
                    "doi",
                    "",
                ),
                "source_url": meta.get(
                    "source_url",
                    "",
                ),
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
            "route": "Exact LIDC case lookup",
        }

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
        "route": "Medical literature search",
    }


# ============================================================
# SAFETY
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
        claim_tokens
        & evidence_tokens
    )

    ratio = overlap / len(
        claim_tokens
    )

    return ratio >= 0.18


# ============================================================
# GEMINI
# ============================================================

def generate_grounded_answer(
    query: str,
    evidence: List[Dict[str, Any]],
    output_language: str,
    api_key: str,
):

    if not evidence:
        return None, "INSUFFICIENT_EVIDENCE"

    client = genai.Client(
        api_key=api_key
    )

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
        "English": "English",
        "Urdu": "Urdu",
        "Roman Urdu": "Roman Urdu",
    }.get(
        output_language,
        "English",
    )

    system_instruction = f"""
You are the evidence-bounded research assistant
inside PulmoLens.

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
8. Do not infer information absent from the evidence.
9. If evidence does not support a claim, leave it out.
10. If evidence is insufficient, return no claims.
11. Preserve LIDC IDs, PMID and DOI exactly.
12. Keep explanations understandable for non-technical users.
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
# VALIDATE RESPONSE
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

        if not set(
            claim.citations
        ).issubset(valid_ids):
            continue

        cited_text = " ".join(
            item["text"]
            for item in evidence
            if item["id"]
            in claim.citations
        )

        if not lexical_support(
            claim.text,
            cited_text,
        ):
            continue

        surviving.append(claim)

    return surviving


# ============================================================
# PDF REPORT
# ============================================================

def build_pdf_report(
    query: str,
    claims: List[Claim],
    evidence: List[Dict[str, Any]],
    language: str,
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=42,
        leftMargin=42,
        topMargin=42,
        bottomMargin=42,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#B42318"),
        fontSize=22,
        alignment=TA_CENTER,
        spaceAfter=18,
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#17324D"),
        fontSize=14,
        spaceBefore=10,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#344054"),
        spaceAfter=8,
    )

    story = []

    story.append(
        Paragraph(
            "PulmoLens Research Report",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "Pulmonary Nodule Research Assistant",
            body_style,
        )
    )

    story.append(
        Paragraph(
            f"<b>Research question:</b> "
            f"{query}",
            body_style,
        )
    )

    story.append(
        Paragraph(
            f"<b>Output language:</b> "
            f"{language}",
            body_style,
        )
    )

    story.append(
        Spacer(1, 8)
    )

    story.append(
        Paragraph(
            "Grounded Findings",
            heading_style,
        )
    )

    for index, claim in enumerate(
        claims,
        start=1,
    ):

        citation_text = ", ".join(
            claim.citations
        )

        story.append(
            Paragraph(
                f"<b>{index}. </b>"
                f"{claim.text}"
                f" <b>[{citation_text}]</b>",
                body_style,
            )
        )

    story.append(
        Spacer(1, 8)
    )

    story.append(
        Paragraph(
            "Evidence Used",
            heading_style,
        )
    )

    table_data = [
        [
            "ID",
            "Source",
            "Title",
        ]
    ]

    for item in evidence:

        title = (
            item.get(
                "title",
                "",
            )
            or "Evidence record"
        )

        table_data.append(
            [
                item.get("id", ""),
                item.get("store", ""),
                title[:90],
            ]
        )

    table = Table(
        table_data,
        colWidths=[
            42,
            125,
            315,
        ],
        repeatRows=1,
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#B42318"
                    ),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.35,
                    colors.HexColor(
                        "#D0D5DD"
                    ),
                ),
                (
                    "FONTNAME",
                    (0, 1),
                    (-1, -1),
                    "Helvetica",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        colors.HexColor(
                            "#F9FAFB"
                        ),
                    ],
                ),
            ]
        )
    )

    story.append(table)

    story.append(
        Spacer(1, 18)
    )

    story.append(
        Paragraph(
            "Research Notice",
            heading_style,
        )
    )

    story.append(
        Paragraph(
            "PulmoLens is a research prototype. "
            "It is not a clinically validated diagnostic "
            "system and should not be used for diagnosis, "
            "treatment selection, or patient-specific "
            "medical decisions.",
            body_style,
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# SESSION STATE
# ============================================================

if "pulmo_retrieval" not in st.session_state:
    st.session_state["pulmo_retrieval"] = None

if "pulmo_query" not in st.session_state:
    st.session_state["pulmo_query"] = ""

if "pulmo_claims" not in st.session_state:
    st.session_state["pulmo_claims"] = []

if "grounding_result" not in st.session_state:
    st.session_state["grounding_result"] = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div class="brand-wrap">
        <div class="brand-row">
            <div class="logo">🫁</div>
            <div>
                <div class="brand-name">PulmoLens</div>
                <div class="brand-sub">
                    Research intelligence workspace
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Workspace",
    [
        "Overview",
        "CT Scan Analysis",
        "Research Assistant",
        "Evidence Explorer",
        "System Architecture",
        "System Status",
    ],
)

st.sidebar.markdown("---")

output_language = st.sidebar.selectbox(
    "Answer language",
    [
        "English",
        "Urdu",
        "Roman Urdu",
    ],
)

relative_ratio = st.sidebar.slider(
    "Evidence strictness",
    min_value=0.50,
    max_value=0.90,
    value=DEFAULT_RELATIVE_GATE,
    step=0.05,
    help=(
        "Higher values keep only evidence that is "
        "closely related to the best available result."
    ),
)

gemini_key = st.secrets.get(
    "GEMINI_API_KEY",
    os.getenv(
        "GEMINI_API_KEY",
        "",
    ),
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div style="
        display:flex;
        align-items:center;
        gap:12px;
        margin-bottom:14px;
    ">
        <div style="
            width:44px;
            height:44px;
            border-radius:12px;
            background:#B42318;
            color:white;
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:23px;
        ">
            🫁
        </div>

        <div>
            <div style="
                font-weight:850;
                font-size:24px;
                color:#17324D;
                line-height:1;
            ">
                PulmoLens
            </div>
            <div style="
                color:#667085;
                font-size:12px;
                margin-top:4px;
            ">
                Pulmonary Nodule Research Assistant
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PAGE — OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">
                Evidence-grounded medical research
            </div>

            <div class="hero-title">
                Understand the evidence around a CT finding.
            </div>

            <div class="hero-text">
                PulmoLens helps you explore pulmonary nodule research
                by combining CT image analysis, structured LIDC evidence,
                medical literature, and an evidence-controlled AI assistant.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">240</div>
                <div class="metric-label">LIDC records</div>
                <div class="metric-help">
                    Structured research evidence
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">122</div>
                <div class="metric-label">Global sources</div>
                <div class="metric-help">
                    Medical research literature
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">294</div>
                <div class="metric-label">Pakistan sources</div>
                <div class="metric-help">
                    Local research evidence
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">768-D</div>
                <div class="metric-label">Search space</div>
                <div class="metric-help">
                    Multilingual evidence retrieval
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div style='height:10px'></div>",
        unsafe_allow_html=True,
    )

    st.subheader("What problem does PulmoLens solve?")

    st.markdown(
        """
        <div class="card">
            <div class="card-title">
                Medical research can be difficult to connect.
            </div>

            <div class="card-text">
                A CT image, a patient identifier, a research paper,
                and a question may all contain useful information,
                but they usually live in different places.
                PulmoLens brings these pieces together into one
                research workflow and keeps the generated answer
                connected to supporting evidence.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("How PulmoLens works")

    cols = st.columns(4)

    steps = [
        (
            "1",
            "Find",
            "Looks for possible nodule regions in an uploaded CT slice."
        ),
        (
            "2",
            "Search",
            "Finds relevant structured and published research evidence."
        ),
        (
            "3",
            "Check",
            "Filters evidence before it is given to the language model."
        ),
        (
            "4",
            "Explain",
            "Creates a readable answer with supporting evidence."
        ),
    ]

    for col, step in zip(cols, steps):

        with col:
            number, title, text = step

            st.markdown(
                f"""
                <div class="step">
                    <div class="step-number">
                        {number}
                    </div>

                    <div class="step-title">
                        {title}
                    </div>

                    <div class="step-text">
                        {text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("Connected services")

    api1, api2, api3 = st.columns(3)

    with api1:

        if gemini_key:
            st.markdown(
                """
                <div class="status-good">
                    ● LLM API connected
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="status-warn">
                    ● LLM API key not configured
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption(
            "Gemini is used only after evidence retrieval."
        )

    with api2:

        backend_ready = (
            yolo_model is not None
            and bool(stores)
        )

        if backend_ready:
            st.markdown(
                """
                <div class="status-good">
                    ● AI backend connected
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="status-bad">
                    ● AI backend needs attention
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption(
            "YOLO, FAISS and retrieval components."
        )

    with api3:

        st.markdown(
            """
            <div class="status-good">
                ● Evidence library connected
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "LIDC, global and Pakistan research sources."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.info(
        "Research notice: PulmoLens is a prototype for research "
        "and education. It is not a diagnostic or treatment system."
    )


# ============================================================
# PAGE — CT ANALYSIS
# ============================================================

elif page == "CT Scan Analysis":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">
                CT image analysis
            </div>

            <div class="hero-title">
                Explore a CT slice
            </div>

            <div class="hero-text">
                Upload one CT image and PulmoLens will highlight
                possible pulmonary nodule candidates found by its
                frozen YOLO11s research model.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload a CT slice",
        type=[
            "png",
            "jpg",
            "jpeg",
        ],
        help=(
            "Upload one 2D CT image for research demonstration."
        ),
    )

    if uploaded:

        image = Image.open(
            uploaded
        ).convert("RGB")

        left, right = st.columns(
            [1, 1]
        )

        with left:

            st.markdown(
                """
                <div class="card-title">
                    Uploaded image
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.image(
                image,
                use_container_width=True,
            )

        with right:

            st.markdown(
                """
                <div class="card-title">
                    PulmoLens analysis
                </div>
                """,
                unsafe_allow_html=True,
            )

            if yolo_model is None:

                st.error(
                    "The CT analysis model is not available."
                )

            else:

                with st.spinner(
                    "Reviewing the CT slice..."
                ):

                    results = yolo_model(
                        image,
                        verbose=False,
                    )

                plotted = results[
                    0
                ].plot()

                st.image(
                    plotted,
                    caption=(
                        "Highlighted candidate regions"
                    ),
                    use_container_width=True,
                )

                boxes = results[
                    0
                ].boxes

                if boxes is None or len(boxes) == 0:

                    st.warning(
                        "No nodule candidates were detected "
                        "above the current model threshold."
                    )

                else:

                    st.success(
                        f"{len(boxes)} candidate region(s) detected."
                    )

                    rows = []

                    for number, box in enumerate(
                        boxes,
                        start=1,
                    ):

                        conf = float(
                            box.conf[0]
                        )

                        xyxy = (
                            box.xyxy[0]
                            .cpu()
                            .numpy()
                        )

                        rows.append(
                            {
                                "Candidate": number,
                                "Confidence": round(
                                    conf,
                                    3,
                                ),
                                "Left": round(
                                    float(
                                        xyxy[0]
                                    ),
                                    1,
                                ),
                                "Top": round(
                                    float(
                                        xyxy[1]
                                    ),
                                    1,
                                ),
                                "Right": round(
                                    float(
                                        xyxy[2]
                                    ),
                                    1,
                                ),
                                "Bottom": round(
                                    float(
                                        xyxy[3]
                                    ),
                                    1,
                                ),
                            }
                        )

                    st.dataframe(
                        pd.DataFrame(
                            rows
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

    else:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">
                    What happens here?
                </div>

                <div class="card-text">
                    1. Upload a CT slice.<br>
                    2. The research model checks the image.<br>
                    3. Possible candidate regions are highlighted.<br>
                    4. You can then use the Research Assistant
                       to explore related evidence.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "Important: this is 2D slice-level research inference, "
        "not complete 3D CT diagnosis."
    )


# ============================================================
# PAGE — RESEARCH ASSISTANT
# ============================================================

elif page == "Research Assistant":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">
                Evidence-grounded research assistant
            </div>

            <div class="hero-title">
                Ask a pulmonary nodule research question
            </div>

            <div class="hero-text">
                Ask in English, Urdu or Roman Urdu. PulmoLens searches
                its evidence library first, then asks the language model
                to explain only what the retrieved evidence supports.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    examples = [
        "What imaging features are discussed for pulmonary nodules?",
        "What structured evidence is available for LIDC-IDRI-0001?",
        "Tell me about LIDC-IDRI-0008 nodule 1.",
        "پلمونری نوڈول کی سی ٹی خصوصیات کیا ہیں؟",
        "lung nodule ki CT features kya hain?",
    ]

    selected_example = st.selectbox(
        "Example question",
        [
            "Write my own question"
        ]
        + examples,
    )

    default_query = (
        ""
        if selected_example
        == "Write my own question"
        else selected_example
    )

    query = st.text_area(
        "Your research question",
        value=default_query,
        height=125,
        placeholder=(
            "Example: What imaging features are discussed "
            "for pulmonary nodules?"
        ),
    )

    col1, col2 = st.columns(
        [1, 3]
    )

    with col1:

        search_clicked = st.button(
            "Search evidence",
            type="primary",
            use_container_width=True,
        )

    if search_clicked:

        st.session_state[
            "pulmo_claims"
        ] = []

        if not query.strip():

            st.error(
                "Please enter a research question."
            )

        elif unsafe_request(query):

            st.error(
                "This question is outside the safe research scope "
                "of PulmoLens. The system does not provide "
                "patient-specific diagnosis or treatment advice."
            )

            st.session_state[
                "pulmo_retrieval"
            ] = None

        elif not stores:

            st.error(
                "The evidence library could not be loaded."
            )

        else:

            with st.spinner(
                "Searching the evidence library..."
            ):

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

    retrieval = st.session_state.get(
        "pulmo_retrieval"
    )

    if retrieval:

        st.markdown("<br>", unsafe_allow_html=True)

        if not retrieval["grounded"]:

            st.markdown(
                """
                <div class="status-warn">
                    No sufficiently relevant evidence was found.
                    PulmoLens will not generate an unsupported answer.
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.success(
                f"Evidence search complete — "
                f"{len(retrieval['evidence'])} source(s) found."
            )

            info1, info2, info3 = st.columns(3)

            with info1:
                st.markdown(
                    f"""
                    **Research route**  
                    {retrieval["route"]}
                    """
                )

            with info2:
                st.markdown(
                    f"""
                    **Detected language**  
                    {retrieval["language"]}
                    """
                )

            with info3:
                st.markdown(
                    f"""
                    **Evidence items**  
                    {len(retrieval["evidence"])}
                    """
                )

            st.markdown(
                "### Supporting evidence"
            )

            for item in retrieval[
                "evidence"
            ]:

                score_text = ""

                if item.get(
                    "score"
                ) is not None:

                    score_text = (
                        f"Dense relevance: "
                        f"{item['score']:.3f}"
                    )

                if item.get(
                    "rerank_score"
                ) is not None:

                    if score_text:
                        score_text += " · "

                    score_text += (
                        f"Final relevance: "
                        f"{item['rerank_score']:.3f}"
                    )

                st.markdown(
                    f"""
                    <div class="evidence-card">

                        <span class="evidence-id">
                            [{item['id']}]
                        </span>

                        <span class="evidence-source">
                            {item['store']}
                        </span>

                        <div class="small-note">
                            {score_text}
                        </div>

                        <div class="evidence-text">
                            {item.get("text", "")}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(
                "### Generate the research explanation"
            )

            st.caption(
                "The language model will only receive the "
                "retrieved evidence shown above."
            )

            generate_clicked = st.button(
                "Generate grounded answer",
                type="primary",
                use_container_width=True,
            )

            if generate_clicked:

                if not gemini_key:

                    st.error(
                        "LLM API is not connected. "
                        "Add GEMINI_API_KEY in Streamlit secrets."
                    )

                else:

                    with st.spinner(
                        "Preparing a grounded explanation..."
                    ):

                        generated, error = (
                            generate_grounded_answer(
                                query=st.session_state[
                                    "pulmo_query"
                                ],
                                evidence=retrieval[
                                    "evidence"
                                ],
                                output_language=output_language,
                                api_key=gemini_key,
                            )
                        )

                    if error:

                        st.error(
                            f"Answer generation failed: {error}"
                        )

                    elif generated is None:

                        st.warning(
                            "There was not enough evidence to "
                            "generate a safe answer."
                        )

                    else:

                        valid_claims = (
                            validate_response(
                                generated,
                                retrieval[
                                    "evidence"
                                ],
                            )
                        )

                        st.session_state[
                            "pulmo_claims"
                        ] = valid_claims

                        if not valid_claims:

                            st.warning(
                                "No generated claims passed "
                                "the final evidence check."
                            )

                        else:

                            st.markdown(
                                """
                                <div class="answer-card">

                                    <div class="answer-title">
                                        PulmoLens Research Answer
                                    </div>

                                """,
                                unsafe_allow_html=True,
                            )

                            for claim in valid_claims:

                                citations = " ".join(
                                    f"[{c}]"
                                    for c in claim.citations
                                )

                                st.markdown(
                                    f"""
                                    <div class="claim">
                                        {claim.text}
                                        <br>
                                        <b>
                                            {citations}
                                        </b>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            st.markdown(
                                "</div>",
                                unsafe_allow_html=True,
                            )

                            st.success(
                                f"{len(valid_claims)} "
                                "claim(s) passed evidence validation."
                            )

                            # --------------------------------
                            # PDF REPORT
                            # --------------------------------

                            try:

                                pdf_bytes = (
                                    build_pdf_report(
                                        query=st.session_state[
                                            "pulmo_query"
                                        ],
                                        claims=valid_claims,
                                        evidence=retrieval[
                                            "evidence"
                                        ],
                                        language=output_language,
                                    )
                                )

                                st.download_button(
                                    label=(
                                        "Download research report (PDF)"
                                    ),
                                    data=pdf_bytes,
                                    file_name=(
                                        "PulmoLens_Research_Report.pdf"
                                    ),
                                    mime="application/pdf",
                                    use_container_width=True,
                                )

                            except Exception as exc:

                                st.warning(
                                    f"PDF report could not be created: "
                                    f"{exc}"
                                )


# ============================================================
# PAGE — EVIDENCE EXPLORER
# ============================================================

elif page == "Evidence Explorer":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">
                Evidence library
            </div>

            <div class="hero-title">
                Explore the sources behind PulmoLens
            </div>

            <div class="hero-text">
                This view lets you understand where PulmoLens gets
                its research evidence before an answer is generated.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b, c = st.columns(3)

    with a:
        st.metric(
            "LIDC evidence",
            "240",
        )

    with b:
        st.metric(
            "Global literature",
            "122",
        )

    with c:
        st.metric(
            "Pakistan literature",
            "294",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    source_choice = st.selectbox(
        "Evidence collection",
        [
            "LIDC Dataset Evidence",
            "Global Literature",
            "Pakistan Literature",
        ],
    )

    store_key = {
        "LIDC Dataset Evidence": "lidc",
        "Global Literature": "literature",
        "Pakistan Literature": "pakistan",
    }[source_choice]

    if store_key in stores:

        metadata = stores[
            store_key
        ]["metadata"]

        display_rows = []

        for item in metadata:

            display_rows.append(
                {
                    "Title": item.get(
                        "title",
                        "",
                    ),
                    "Source ID": item.get(
                        "source_id",
                        "",
                    ),
                    "Patient ID": item.get(
                        "patient_id",
                        "",
                    ),
                    "Nodule ID": item.get(
                        "nodule_id",
                        "",
                    ),
                }
            )

        df = pd.DataFrame(
            display_rows
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.warning(
            "This evidence store is not available."
        )


# ============================================================
# PAGE — ARCHITECTURE
# ============================================================

elif page == "System Architecture":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">
                Behind the interface
            </div>

            <div class="hero-title">
                How PulmoLens is built
            </div>

            <div class="hero-text">
                The technical pipeline is hidden from normal users,
                but researchers can inspect the architecture here.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader(
        "End-to-end architecture"
    )

    st.markdown(
        """
        <div class="architecture-box">
USER
  │
  ▼
PulmoLens Interface
  │
  ├───────────────┐
  │               │
CT Slice       Research Question
  │               │
  ▼               ▼
YOLO11s       Query Understanding
  │               │
  ▼               ▼
Candidate      Exact LIDC
Regions        Case Lookup
                    │
                    ▼
              FAISS Retrieval
                    │
                    ▼
               BGE Reranking
                    │
                    ▼
              Relevance Gate
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
   LIDC Evidence          Literature Evidence
        │                       │
        └───────────┬───────────┘
                    ▼
             Evidence Bundle
                    │
                    ▼
              Gemini 2.5 Flash
                    │
                    ▼
             Claim Validation
                    │
                    ▼
              Grounded Answer
                    │
                    ▼
              PDF Report
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    AI backend
                </div>

                <div class="card-text">
                    <b>YOLO11s</b><br>
                    Finds possible nodule candidate regions in
                    uploaded 2D CT slices.
                    <br><br>

                    <b>FAISS</b><br>
                    Finds relevant evidence from indexed sources.
                    <br><br>

                    <b>BGE reranker</b><br>
                    Reorders the retrieved evidence by relevance.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    AI reasoning layer
                </div>

                <div class="card-text">
                    <b>Gemini</b><br>
                    Converts retrieved evidence into a readable
                    research explanation.
                    <br><br>

                    <b>Validation</b><br>
                    Removes claims that do not have valid supporting
                    evidence.
                    <br><br>

                    <b>Abstention</b><br>
                    The system can refuse to answer when evidence
                    is insufficient.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader(
        "External connections"
    )

    p1, p2 = st.columns(2)

    with p1:

        if gemini_key:

            st.markdown(
                """
                <div class="status-good">
                    ● LLM API — Connected
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                """
                <div class="status-bad">
                    ● LLM API — Not connected
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption(
            "Gemini API provides the final language generation."
        )

    with p2:

        if yolo_model is not None and stores:

            st.markdown(
                """
                <div class="status-good">
                    ● AI Backend — Connected
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                """
                <div class="status-bad">
                    ● AI Backend — Needs attention
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption(
            "Local model + retrieval + evidence services."
        )


# ============================================================
# PAGE — SYSTEM STATUS
# ============================================================

elif page == "System Status":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">
                Platform health
            </div>

            <div class="hero-title">
                PulmoLens system status
            </div>

            <div class="hero-text">
                A simple view of the components currently available
                to the application.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    checks = []

    checks.append(
        (
            "YOLO11s model",
            yolo_model is not None,
            "CT candidate detection",
        )
    )

    checks.append(
        (
            "LIDC evidence store",
            "lidc" in stores,
            "240 structured evidence records",
        )
    )

    checks.append(
        (
            "Global literature store",
            "literature" in stores,
            "122 indexed sources",
        )
    )

    checks.append(
        (
            "Pakistan literature store",
            "pakistan" in stores,
            "294 indexed sources",
        )
    )

    checks.append(
        (
            "LLM API",
            bool(gemini_key),
            "Gemini 2.5 Flash",
        )
    )

    for name, ready, detail in checks:

        col1, col2 = st.columns(
            [2, 4]
        )

        with col1:

            if ready:

                st.markdown(
                    """
                    <div class="status-good">
                        ● Connected
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:

                st.markdown(
                    """
                    <div class="status-bad">
                        ● Not ready
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col2:

            st.markdown(
                f"**{name}**  \n{detail}"
            )

    st.markdown("<br>", unsafe_allow_html=True)

    if (
        yolo_model is not None
        and len(stores) == 3
        and gemini_key
    ):

        st.success(
            "PulmoLens is ready for the complete research workflow."
        )

    else:

        st.warning(
            "PulmoLens is partially available. "
            "Check the components above before deployment."
        )

    if st.session_state.get(
        "yolo_error"
    ):

        st.error(
            "YOLO loading detail: "
            + st.session_state[
                "yolo_error"
            ]
        )

    if st.session_state.get(
        "retrieval_error"
    ):

        st.error(
            "Retrieval loading detail: "
            + st.session_state[
                "retrieval_error"
            ]
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                Research safety
            </div>

            <div class="card-text">
                PulmoLens is a research prototype. It does not provide
                a clinical diagnosis, cancer diagnosis, treatment plan,
                or patient-specific medical recommendation.
                Its generated answers are limited to retrieved evidence.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        margin-top:35px;
        padding:18px 0 8px 0;
        border-top:1px solid #E5E7EB;
        color:#667085;
        font-size:12px;
        text-align:center;
    ">
        PulmoLens · Evidence-grounded pulmonary nodule research prototype
        · Research and education use only
    </div>
    """,
    unsafe_allow_html=True,
)
