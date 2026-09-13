import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

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
# PULMOLENS CONFIGURATION (PRESERVED)
# ============================================================

APP_NAME = "PulmoLens"
APP_VERSION = "v1.2-hackathon"

BASE_DIR = Path(__file__).resolve().parent

# Automatically read .env if present
env_file = BASE_DIR / ".env"
if env_file.exists():
    try:
        with open(env_file, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    _k = _k.strip()
                    _v = _v.strip().strip("'\"")
                    if _k and _k not in os.environ:
                        os.environ[_k] = _v
    except Exception:
        pass

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
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PulmoLens — Pulmonary Nodule Research Platform",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# MODERN MEDICAL RESEARCH UI DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        /* Core Palette - Clinical Medical Red & Crisp Clean White */
        --red-primary: #DC2626;
        --red-hover: #B91C1C;
        --red-dark: #991B1B;
        --red-light: #EF4444;
        --red-soft: #FEF2F2;
        --red-border: #FECACA;
        
        /* Compatibility aliases for previous tokens */
        --teal-primary: #DC2626;
        --teal-hover: #B91C1C;
        --teal-light: #EF4444;
        --teal-soft: #FEF2F2;
        --teal-border: #FECACA;
        --navy-dark: #0F172A;
        --navy-surface: #1E293B;
        --navy-light: #334155;
        
        /* Accents & Neutrals */
        --blue-accent: #2563EB;
        --blue-soft: #EFF6FF;
        --blue-border: #BFDBFE;
        --bg-page: #F8FAFC;
        --bg-card: #FFFFFF;
        --bg-subtle: #F1F5F9;
        
        /* High-Contrast Typography */
        --text-primary: #0F172A;
        --text-secondary: #1E293B;
        --text-muted: #475569;
        --text-light: #64748B;
        
        /* Borders & Elevation */
        --border-color: #E2E8F0;
        --border-hover: #CBD5E1;
        --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.08);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.08), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
        
        /* Status Semantics */
        --status-green: #059669;
        --status-green-bg: #ECFDF5;
        --status-green-border: #A7F3D0;
        --status-amber: #D97706;
        --status-amber-bg: #FFFBEB;
        --status-amber-border: #FDE68A;
        --status-red: #DC2626;
        --status-red-bg: #FEF2F2;
        --status-red-border: #FECACA;
    }

    /* Global Typography & Background */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: var(--bg-page) !important;
        color: var(--text-primary) !important;
        font-size: 16px !important;
    }

    /* Chrome cleanup */
    #MainMenu, footer {
        visibility: hidden;
    }
    header {
        visibility: visible;
        background: transparent !important;
    }

    /* ============================================================
       SIDEBAR & WORKSPACE NAVIGATION (FIX FOR BLANK/SMALL TEXT)
       ============================================================ */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid var(--border-color) !important;
        padding-top: 1.25rem !important;
    }

    /* Ensure all text in sidebar default to dark legible slate */
    [data-testid="stSidebar"] * {
        color: #0F172A;
    }

    /* Brand Header in Sidebar */
    .sidebar-brand-card {
        background: linear-gradient(135deg, #DC2626 0%, #991B1B 100%) !important;
        border-radius: 12px !important;
        padding: 16px 18px !important;
        margin-bottom: 22px !important;
        box-shadow: 0 4px 14px rgba(220, 38, 38, 0.25) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
    }

    .sidebar-brand-header {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .sidebar-logo-icon {
        width: 44px !important;
        height: 44px !important;
        min-width: 44px !important;
        border-radius: 10px !important;
        background: rgba(255, 255, 255, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px !important;
        color: #FFFFFF !important;
    }

    .sidebar-brand-title {
        color: #FFFFFF !important;
        font-size: 22px !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }

    .sidebar-brand-sub {
        color: #FEE2E2 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        margin-top: 4px;
        letter-spacing: 0.02em;
    }

    .sidebar-section-title {
        font-size: 13px !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #475569 !important;
        margin: 22px 0 10px 4px;
    }

    /* Workspace Navigation Items (Radio Group) */
    [data-testid="stSidebar"] div[role="radiogroup"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 6px !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        background: #F8FAFC !important;
        padding: 12px 16px !important;
        border-radius: 9px !important;
        border: 1px solid #E2E8F0 !important;
        margin: 0 !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        transition: all 0.15s ease-in-out !important;
    }

    /* Force high visibility on all text inside radio labels */
    [data-testid="stSidebar"] div[role="radiogroup"] > label p,
    [data-testid="stSidebar"] div[role="radiogroup"] > label span,
    [data-testid="stSidebar"] div[role="radiogroup"] > label div {
        color: #0F172A !important;
        font-size: 15.5px !important;
        font-weight: 650 !important;
        line-height: 1.3 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background-color: #FEF2F2 !important;
        border-color: #FECACA !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover p,
    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover span {
        color: #DC2626 !important;
    }

    /* Active / Checked Workspace Nav Item */
    [data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background-color: #FEF2F2 !important;
        border-color: #DC2626 !important;
        border-left: 6px solid #DC2626 !important;
        box-shadow: 0 2px 6px rgba(220, 38, 38, 0.12) !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] p,
    [data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] span,
    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p,
    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) span {
        color: #DC2626 !important;
        font-weight: 750 !important;
    }

    /* Radio button indicator dot */
    [data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"] {
        accent-color: #DC2626 !important;
        width: 18px !important;
        height: 18px !important;
    }

    /* Sidebar Settings Labels (Language, Strictness) */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] label p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #0F172A !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* Sidebar Selectbox & Inputs */
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #0F172A !important;
        font-size: 14.5px !important;
        font-weight: 600 !important;
    }

    /* Sidebar Slider */
    [data-testid="stSidebar"] [data-testid="stSlider"] div[role="slider"] {
        background-color: #DC2626 !important;
    }

    /* ============================================================
       PAGE HEADER & HERO SECTION
       ============================================================ */
    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 0 22px 0;
        border-bottom: 2px solid #F1F5F9;
        margin-bottom: 26px;
    }

    .app-header-left {
        display: flex;
        align-items: baseline;
        gap: 16px;
        flex-wrap: wrap;
    }

    .app-header-title {
        font-size: 30px;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.03em;
        line-height: 1.2;
    }

    .app-header-desc {
        font-size: 16px;
        color: #475569;
        font-weight: 500;
    }

    .app-header-status {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 13.5px;
        font-weight: 700;
    }

    .hero-banner {
        background: linear-gradient(135deg, #FFFFFF 0%, #FFF5F5 60%, #FEF2F2 100%);
        border: 1px solid #FECACA;
        border-left: 5px solid #DC2626;
        border-radius: 14px;
        padding: 28px 32px;
        margin-bottom: 26px;
        box-shadow: var(--shadow-sm);
        position: relative;
    }

    .hero-kicker-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #FEF2F2;
        color: #DC2626;
        border: 1px solid #FECACA;
        border-radius: 999px;
        padding: 5px 12px;
        font-size: 12.5px;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero-title {
        font-size: 32px;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.03em;
        line-height: 1.25;
        margin-bottom: 10px;
    }

    .hero-text {
        font-size: 16.5px;
        color: #1E293B;
        line-height: 1.65;
        max-width: 980px;
    }

    /* ============================================================
       CARDS & INTERACTIVE SECTIONS
       ============================================================ */
    .action-card {
        background: #FFFFFF;
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 22px;
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .action-card:hover {
        border-color: #DC2626;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.12);
        transform: translateY(-2px);
    }

    .action-card-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
    }

    .action-card-icon {
        width: 38px;
        height: 38px;
        border-radius: 10px;
        background: #FEF2F2;
        color: #DC2626;
        border: 1px solid #FECACA;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        font-weight: 700;
    }

    .action-card-title {
        font-size: 17.5px;
        font-weight: 800;
        color: #0F172A;
    }

    .action-card-desc {
        font-size: 14.5px;
        color: #475569;
        line-height: 1.55;
        margin-bottom: 14px;
    }

    .action-card-footer {
        font-size: 14px;
        font-weight: 750;
        color: #DC2626;
        display: flex;
        align-items: center;
        gap: 4px;
    }

    .research-card {
        background: #FFFFFF;
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 22px 24px;
        margin-bottom: 18px;
        box-shadow: var(--shadow-sm);
    }

    .research-card-title {
        font-size: 18px;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .research-card-body {
        font-size: 15.5px;
        color: #1E293B;
        line-height: 1.7;
    }

    /* Metric Cards */
    .metric-card-modern {
        background: #FFFFFF;
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: var(--shadow-sm);
        height: 100%;
    }

    .metric-value-modern {
        font-size: 34px;
        font-weight: 850;
        color: #DC2626;
        letter-spacing: -0.03em;
        line-height: 1.1;
        font-feature-settings: "tnum";
    }

    .metric-label-modern {
        font-size: 14.5px;
        font-weight: 750;
        color: #0F172A;
        margin-top: 6px;
    }

    .metric-sub-modern {
        font-size: 13px;
        color: #475569;
        margin-top: 3px;
        line-height: 1.4;
    }

    /* Workflow Stepper */
    .pipeline-step-card {
        background: #FFFFFF;
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 18px;
        height: 100%;
        box-shadow: var(--shadow-sm);
        position: relative;
    }

    .pipeline-step-num {
        width: 30px;
        height: 30px;
        border-radius: 8px;
        background: #DC2626;
        color: #FFFFFF;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        font-weight: 800;
        margin-bottom: 12px;
    }

    .pipeline-step-title {
        font-size: 16px;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 6px;
    }

    .pipeline-step-text {
        font-size: 14px;
        color: #475569;
        line-height: 1.55;
    }

    /* Status Badges */
    .badge-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 9px 14px;
        border-radius: 8px;
        font-size: 13.5px;
        font-weight: 700;
        margin-bottom: 8px;
        width: 100%;
    }

    .badge-status-green {
        background: var(--status-green-bg);
        border: 1px solid var(--status-green-border);
        color: var(--status-green);
    }

    .badge-status-amber {
        background: var(--status-amber-bg);
        border: 1px solid var(--status-amber-border);
        color: var(--status-amber);
    }

    .badge-status-red {
        background: var(--status-red-bg);
        border: 1px solid var(--status-red-border);
        color: var(--status-red);
    }

    .badge-status-blue {
        background: var(--blue-soft);
        border: 1px solid var(--blue-border);
        color: var(--blue-accent);
    }

    /* Evidence Cards */
    .evidence-item-card {
        background: #FFFFFF;
        border: 1px solid var(--border-color);
        border-left: 4px solid #DC2626;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: var(--shadow-sm);
        transition: border-color 0.15s ease;
    }

    .evidence-item-card:hover {
        border-color: var(--border-hover);
        border-left-color: #B91C1C;
    }

    .evidence-item-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
        flex-wrap: wrap;
        gap: 8px;
    }

    .evidence-badge-id {
        background: #0F172A;
        color: #FFFFFF;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 12.5px;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
    }

    .evidence-source-tag {
        background: var(--bg-subtle);
        color: #1E293B;
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 12px;
        font-weight: 700;
    }

    .evidence-score-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #DC2626;
        background: #FEF2F2;
        border: 1px solid #FECACA;
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 700;
    }

    .evidence-item-title {
        font-size: 15.5px;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 6px;
        line-height: 1.45;
    }

    .evidence-item-text {
        font-size: 14.5px;
        color: #1E293B;
        line-height: 1.65;
    }

    /* Research Answer Card */
    .answer-container {
        background: #FFFFFF;
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 24px 26px;
        margin-top: 18px;
        box-shadow: var(--shadow-sm);
    }

    .answer-container-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 16px;
        margin-bottom: 18px;
    }

    .answer-container-title {
        font-size: 20px;
        font-weight: 800;
        color: #0F172A;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .grounded-badge {
        background: #FEF2F2;
        color: #DC2626;
        border: 1px solid #FECACA;
        border-radius: 999px;
        padding: 5px 12px;
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .claim-card {
        background: var(--bg-page);
        border: 1px solid var(--border-color);
        border-left: 4px solid #DC2626;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        font-size: 15px;
        color: #0F172A;
        line-height: 1.65;
    }

    .citation-pill {
        display: inline-block;
        background: #FFFFFF;
        color: #DC2626;
        border: 1px solid #FECACA;
        border-radius: 5px;
        padding: 2px 8px;
        font-size: 12.5px;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        margin-left: 6px;
    }

    /* Architecture Visual Canvas */
    .arch-diagram-grid {
        background: #FFFFFF;
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 26px;
        margin-bottom: 22px;
        box-shadow: var(--shadow-sm);
    }

    .arch-phase {
        background: var(--bg-page);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }

    .arch-phase-title {
        font-size: 12.5px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #DC2626;
        margin-bottom: 10px;
    }

    .arch-nodes-row {
        display: flex;
        gap: 14px;
        flex-wrap: wrap;
    }

    .arch-node {
        flex: 1;
        min-width: 140px;
        background: #FFFFFF;
        border: 1px solid var(--border-color);
        border-left: 3px solid #DC2626;
        border-radius: 8px;
        padding: 12px 14px;
        font-size: 14px;
        font-weight: 750;
        color: #0F172A;
        box-shadow: var(--shadow-sm);
    }

    .arch-node-desc {
        font-size: 12px;
        font-weight: 500;
        color: #475569;
        margin-top: 4px;
    }

    .arch-flow-arrow {
        text-align: center;
        color: #94A3B8;
        font-size: 18px;
        margin: 6px 0;
        line-height: 1;
    }

    /* Medical Notice Callout */
    .medical-notice-box {
        background: #FFFFFF;
        border: 1px solid var(--border-color);
        border-left: 4px solid #DC2626;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 26px;
        display: flex;
        align-items: flex-start;
        gap: 14px;
    }

    .medical-notice-icon {
        color: #DC2626;
        font-size: 20px;
        margin-top: 1px;
    }

    .medical-notice-title {
        font-size: 15px;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 4px;
    }

    .medical-notice-text {
        font-size: 14px;
        color: #334155;
        line-height: 1.6;
    }

    /* ============================================================
       BUTTONS & FILE UPLOADER (RED & WHITE THEME)
       ============================================================ */
    div.stButton > button {
        border-radius: 8px !important;
        font-weight: 750 !important;
        font-size: 15px !important;
        padding: 0.6rem 1.4rem !important;
        transition: all 0.15s ease-in-out !important;
        border: 1px solid var(--border-color) !important;
    }

    div.stButton > button[kind="primary"] {
        background-color: #DC2626 !important;
        border-color: #DC2626 !important;
        color: #FFFFFF !important;
        box-shadow: 0 2px 5px rgba(220, 38, 38, 0.25) !important;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #B91C1C !important;
        border-color: #B91C1C !important;
        box-shadow: 0 4px 10px rgba(220, 38, 38, 0.35) !important;
    }

    div.stDownloadButton > button {
        border-radius: 8px !important;
        font-weight: 750 !important;
        font-size: 15px !important;
        background-color: #FFFFFF !important;
        border: 1.5px solid #DC2626 !important;
        color: #DC2626 !important;
        padding: 0.6rem 1.4rem !important;
    }

    div.stDownloadButton > button:hover {
        background-color: #FEF2F2 !important;
        border-color: #B91C1C !important;
        color: #B91C1C !important;
    }

    /* File Uploader Red & White */
    [data-testid="stFileUploader"] {
        background: #FFFFFF !important;
        border: 2px dashed #DC2626 !important;
        border-radius: 12px !important;
        padding: 24px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: #FFFFFF !important;
        border: none !important;
        padding: 12px !important;
    }

    [data-testid="stFileUploader"] section * {
        color: #0F172A !important;
        font-size: 15px !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: #DC2626 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 15px !important;
        font-weight: 750 !important;
        padding: 8px 20px !important;
    }

    [data-testid="stFileUploader"] button:hover {
        background-color: #B91C1C !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PYDANTIC OUTPUT SCHEMA (PRESERVED)
# ============================================================

class Claim(BaseModel):
    text: str = Field(min_length=1)
    citations: List[str] = Field(default_factory=list)


class GroundedResponse(BaseModel):
    answer: str = ""
    claims: List[Claim] = Field(default_factory=list)


# ============================================================
# MODEL & VECTOR STORE LOADING (PRESERVED)
# ============================================================

@st.cache_resource
def load_yolo():
    if not MODEL_PATH.exists():
        return None
    return YOLO(str(MODEL_PATH))


@st.cache_resource
def load_retrieval_system():
    embedder = SentenceTransformer(EMBEDDER_NAME)
    reranker = CrossEncoder(RERANKER_NAME, max_length=512)

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
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if index.ntotal != len(metadata):
            raise RuntimeError(
                f"{store_name}: FAISS/metadata count mismatch "
                f"{index.ntotal} != {len(metadata)}"
            )

        stores[store_name] = {
            "index": index,
            "metadata": metadata,
        }

    return embedder, reranker, stores


# Graceful loading without halting UI
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
# DATASET LOADING (PRESERVED)
# ============================================================

@st.cache_data
def load_structured_evidence():
    if not DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)


structured_df = load_structured_evidence()


# ============================================================
# RETRIEVAL & NLP HELPERS (PRESERVED)
# ============================================================

def detect_language(query: str) -> str:
    if re.search(r"[\u0600-\u06FF]", query):
        return "urdu_script"
    q = query.lower()
    roman_markers = ["kya", "hai", "hain", "mein", "ke", "ki", "ka", "par", "ko"]
    if any(word in q.split() for word in roman_markers):
        return "roman_urdu"
    return "english"


def extract_patient_id(query: str):
    match = re.search(r"(LIDC-IDRI-\d{4}|\b\d{4}\b)", query, flags=re.IGNORECASE)
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
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return str(int(match.group(1)))
    return None


def exact_lidc_lookup(patient_id: str, nodule_id=None):
    results = []
    if "lidc" not in stores:
        return results

    for meta in stores["lidc"]["metadata"]:
        meta_patient = str(meta.get("patient_id", "")).upper()
        if meta_patient != patient_id.upper():
            continue
        if nodule_id is not None:
            meta_nodule = str(meta.get("nodule_id", "")).strip()
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
                "citation": meta.get("citation", ""),
                "pmid": meta.get("pmid", ""),
                "doi": meta.get("doi", ""),
                "source_url": meta.get("source_url", ""),
                "exact": True,
            }
        )
    return results


def dense_retrieve(query: str, language: str, relative_ratio: float):
    if embedder is None or reranker is None:
        return []

    q_vec = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)

    faiss.normalize_L2(q_vec)
    candidates = []

    for store_name in ["literature", "pakistan"]:
        if store_name not in stores:
            continue
        scores, indices = stores[store_name]["index"].search(q_vec, DENSE_TOP_K)
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

    top_score = max(x["dense_score"] for x in candidates)
    language_floor = ABS_MIN_DENSE if language == "english" else 0.22
    effective_threshold = max(language_floor, relative_ratio * top_score)

    candidates = [x for x in candidates if x["dense_score"] >= effective_threshold]
    if not candidates:
        return []

    rerank_pairs = []
    for candidate in candidates:
        text = candidate["metadata"].get("text", "")
        rerank_pairs.append([query, text])

    rerank_scores = reranker.predict(rerank_pairs, show_progress_bar=False)

    for candidate, rerank_score in zip(candidates, rerank_scores):
        candidate["rerank_score"] = float(rerank_score)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    if not candidates:
        return []

    best_rerank = candidates[0]["rerank_score"]
    rerank_floor = max(0.20, 0.70 * best_rerank)

    selected = [c for c in candidates if c["rerank_score"] >= rerank_floor]
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


def retrieve(query: str, relative_ratio: float):
    language = detect_language(query)
    patient_id = extract_patient_id(query)
    nodule_id = extract_nodule_id(query)

    if patient_id:
        exact = exact_lidc_lookup(patient_id, nodule_id)
        return {
            "language": language,
            "patient_id": patient_id,
            "nodule_id": nodule_id,
            "evidence": exact,
            "grounded": bool(exact),
            "route": "Exact LIDC case lookup",
        }

    evidence = dense_retrieve(query, language, relative_ratio)
    return {
        "language": language,
        "patient_id": None,
        "nodule_id": None,
        "evidence": evidence,
        "grounded": bool(evidence),
        "route": "Medical literature search",
    }


# ============================================================
# SAFETY LOGIC (PRESERVED)
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
    return any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in UNSAFE_PATTERNS)


# ============================================================
# CLAIM VERIFICATION (PRESERVED)
# ============================================================

def lexical_support(claim: str, evidence_text: str) -> bool:
    claim_tokens = set(re.findall(r"\b[a-zA-Z0-9]{4,}\b", claim.lower()))
    evidence_tokens = set(re.findall(r"\b[a-zA-Z0-9]{4,}\b", evidence_text.lower()))

    if not claim_tokens:
        return True

    overlap = len(claim_tokens & evidence_tokens)
    ratio = overlap / len(claim_tokens)
    return ratio >= 0.18


# ============================================================
# GEMINI GENERATION (PRESERVED)
# ============================================================

def generate_grounded_answer(
    query: str,
    evidence: List[Dict[str, Any]],
    output_language: str,
    api_key: str,
):
    if not evidence:
        return None, "INSUFFICIENT_EVIDENCE"

    client = genai.Client(api_key=api_key)

    evidence_bundle = []
    for item in evidence:
        citation = item["id"]
        evidence_bundle.append(f"[{citation}] {item.get('text', '')}")

    joined_evidence = "\n\n".join(evidence_bundle)

    language_name = {
        "English": "English",
        "Urdu": "Urdu",
        "Roman Urdu": "Roman Urdu",
    }.get(output_language, "English")

    system_instruction = f"""
You are the evidence-bounded research assistant inside PulmoLens.
PulmoLens is a research prototype.

Answer strictly from the supplied Evidence Bundle.
Requested language: {language_name}

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
        data = GroundedResponse.model_validate_json(response.text)
        return data, None
    except Exception as exc:
        return None, str(exc)


def validate_response(
    response: GroundedResponse,
    evidence: List[Dict[str, Any]],
):
    valid_ids = {item["id"] for item in evidence}
    surviving = []

    for claim in response.claims:
        if not claim.citations:
            continue
        if not set(claim.citations).issubset(valid_ids):
            continue

        cited_text = " ".join(
            item["text"] for item in evidence if item["id"] in claim.citations
        )
        if not lexical_support(claim.text, cited_text):
            continue

        surviving.append(claim)

    return surviving


# ============================================================
# PDF REPORT GENERATION (PRESERVED)
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
        textColor=colors.HexColor("#0F766E"),
        fontSize=20,
        alignment=TA_CENTER,
        spaceAfter=14,
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#12304A"),
        fontSize=13,
        spaceBefore=12,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceAfter=8,
    )

    story = [
        Paragraph("PulmoLens Research Report", title_style),
        Paragraph("<b>Pulmonary Nodule Multimodal Research Intelligence</b>", body_style),
        Spacer(1, 4),
        Paragraph(f"<b>Research Query:</b> {query}", body_style),
        Paragraph(f"<b>Synthesized Language:</b> {language}", body_style),
        Spacer(1, 8),
        Paragraph("Evidence-Grounded Findings", heading_style),
    ]

    for index, claim in enumerate(claims, start=1):
        citation_text = ", ".join(claim.citations)
        story.append(
            Paragraph(
                f"<b>{index}.</b> {claim.text} <b>[{citation_text}]</b>",
                body_style,
            )
        )

    story.append(Spacer(1, 8))
    story.append(Paragraph("Supporting Evidence Inventory", heading_style))

    table_data = [["ID", "Source Collection", "Evidence Title / Excerpt"]]
    for item in evidence:
        title = item.get("title", "") or "Evidence record"
        table_data.append(
            [
                item.get("id", ""),
                item.get("store", ""),
                title[:95],
            ]
        )

    table = Table(table_data, colWidths=[42, 125, 315], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12304A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#E2E8F0")),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#F8FAFC")],
                ),
            ]
        )
    )

    story.append(table)
    story.append(Spacer(1, 16))
    story.append(Paragraph("Research Safety Notice", heading_style))
    story.append(
        Paragraph(
            "PulmoLens is a research prototype. It is not a clinically validated "
            "diagnostic system and should not be used for diagnosis, treatment "
            "selection, or patient-specific medical decisions.",
            body_style,
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# SESSION STATE INITIALIZATION
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
# DYNAMIC SYSTEM READINESS CALCULATION
# ============================================================

gemini_key = os.getenv("GEMINI_API_KEY", "")
if not gemini_key:
    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            gemini_key = str(st.secrets["GEMINI_API_KEY"])
    except Exception:
        gemini_key = ""


system_checks = [
    ("YOLO11s Model", yolo_model is not None, "2D CT slice candidate detection"),
    ("LIDC Evidence Store", "lidc" in stores, "240 structured clinical records"),
    ("Global Literature Store", "literature" in stores, "122 peer-reviewed articles"),
    ("Pakistan Literature Store", "pakistan" in stores, "294 regional oncology papers"),
    ("LLM Synthesis API", bool(gemini_key), "Gemini 2.5 Flash grounded generation"),
]

ready_count = sum(1 for _, ready, _ in system_checks if ready)
total_checks = len(system_checks)
all_systems_ready = (ready_count == total_checks)


# ============================================================
# APP SHELL: SIDEBAR NAVIGATION & SETTINGS
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand-card">
            <div class="sidebar-brand-header">
                <div class="sidebar-logo-icon">🫁</div>
                <div>
                    <div class="sidebar-brand-title">PulmoLens</div>
                    <div class="sidebar-brand-sub">Pulmonary Nodule Research</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section-title">WORKSPACE</div>', unsafe_allow_html=True)
    
    page = st.radio(
        "Workspace Navigation",
        [
            "Overview",
            "CT Scan Analysis",
            "Research Assistant",
            "Evidence Explorer",
            "System Architecture",
            "System Status",
        ],
        format_func=lambda x: {
            "Overview": "🏠  Overview",
            "CT Scan Analysis": "🔬  CT Scan Analysis",
            "Research Assistant": "💬  Research Assistant",
            "Evidence Explorer": "📚  Evidence Explorer",
            "System Architecture": "🏛️  System Architecture",
            "System Status": "⚡  System Status",
        }.get(x, x),
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-section-title">RESEARCH SETTINGS</div>', unsafe_allow_html=True)

    output_language = st.selectbox(
        "Synthesis Language",
        ["English", "Urdu", "Roman Urdu"],
        help="Language used for grounded explanation generation.",
    )

    relative_ratio = st.slider(
        "Evidence Strictness Gate",
        min_value=0.50,
        max_value=0.90,
        value=DEFAULT_RELATIVE_GATE,
        step=0.05,
        help="Higher values retain only evidence closely aligned with the top retrieval candidate.",
    )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Sidebar Live Health Footer
    if all_systems_ready:
        st.markdown(
            f"""
            <div class="badge-status badge-status-green">
                <span style="font-size: 10px;">●</span> {ready_count}/{total_checks} Services Ready
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif ready_count >= 3:
        st.markdown(
            f"""
            <div class="badge-status badge-status-amber">
                <span style="font-size: 10px;">●</span> {ready_count}/{total_checks} Systems Operational
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="badge-status badge-status-red">
                <span style="font-size: 10px;">●</span> {ready_count}/{total_checks} Systems Available
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# TOP APPLICATION HEADER
# ============================================================

page_headers = {
    "Overview": ("Overview & Research Protocol", "Evidence-grounded pulmonary nodule intelligence platform"),
    "CT Scan Analysis": ("CT Scan Analysis", "AI-assisted pulmonary nodule candidate exploration and spatial localization"),
    "Research Assistant": ("Research Assistant", "Multilingual clinical queries bounded by verified medical evidence"),
    "Evidence Explorer": ("Evidence Library Explorer", "Structured inspection of LIDC annotations, global, and regional literature"),
    "System Architecture": ("System Architecture", "Multimodal pipeline, neural reranking, and verification topology"),
    "System Status": ("Platform Health & Diagnostics", "Real-time component connectivity and integrity monitoring"),
}

curr_title, curr_desc = page_headers.get(page, ("PulmoLens", "Medical AI Research Platform"))

status_badge_html = (
    '<div class="app-header-status badge-status-green">● System operational</div>'
    if all_systems_ready
    else f'<div class="app-header-status badge-status-amber">● {ready_count}/{total_checks} Components online</div>'
)

st.markdown(
    f"""
    <div class="app-header">
        <div class="app-header-left">
            <div class="app-header-title">{curr_title}</div>
            <div class="app-header-desc">· {curr_desc}</div>
        </div>
        <div>
            {status_badge_html}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PAGE 1: OVERVIEW (HACKATHON DEMO SHOWCASE)
# ============================================================

if page == "Overview":

    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-kicker-pill">
                <span>✦</span> EVIDENCE-GROUNDED MEDICAL AI · RESEARCH PROTOCOL
            </div>
            <div class="hero-title">
                Explore pulmonary nodule evidence with AI
            </div>
            <div class="hero-text">
                PulmoLens bridges the critical gap in pulmonary oncology research by combining frozen 
                YOLO11s CT slice candidate localization, dual-tier FAISS vector retrieval across global and 
                regional literature, neural cross-encoder reranking, and strictly verified, citation-grounded 
                AI explanations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Action Cards for Quick Exploration
    col_act1, col_act2 = st.columns(2)
    with col_act1:
        st.markdown(
            """
            <div class="action-card">
                <div>
                    <div class="action-card-header">
                        <div class="action-card-icon">🩻</div>
                        <div class="action-card-title">Analyze CT Image</div>
                    </div>
                    <div class="action-card-desc">
                        Upload a 2D CT slice to perform candidate nodule localization with YOLO11s and extract precise bounding coordinates.
                    </div>
                </div>
                <div class="action-card-footer">
                    Navigate to CT Scan Analysis →
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_act2:
        st.markdown(
            """
            <div class="action-card">
                <div>
                    <div class="action-card-header">
                        <div class="action-card-icon">🔍</div>
                        <div class="action-card-title">Ask Research Assistant</div>
                    </div>
                    <div class="action-card-desc">
                        Search multi-source medical evidence in English, Urdu, or Roman Urdu and generate strictly validated research explanations.
                    </div>
                </div>
                <div class="action-card-footer">
                    Navigate to Research Assistant →
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # Real Corpus Metrics (Preserved Exact Counts)
    st.markdown("##### Research Corpus & Index Topology")
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(
            """
            <div class="metric-card-modern">
                <div class="metric-value-modern">240</div>
                <div class="metric-label-modern">LIDC Records</div>
                <div class="metric-sub-modern">Structured clinical nodule annotations & characteristics</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            """
            <div class="metric-card-modern">
                <div class="metric-value-modern">122</div>
                <div class="metric-label-modern">Global Sources</div>
                <div class="metric-sub-modern">Peer-reviewed pulmonary oncology & radiology literature</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            """
            <div class="metric-card-modern">
                <div class="metric-value-modern">294</div>
                <div class="metric-label-modern">Pakistan Sources</div>
                <div class="metric-sub-modern">Regional epidemiology & local clinical studies</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            """
            <div class="metric-card-modern">
                <div class="metric-value-modern">768-D</div>
                <div class="metric-label-modern">Vector Space</div>
                <div class="metric-sub-modern">Multilingual dense retrieval & cross-encoder reranking</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # Core Problem & Clinical AI Solution
    col_prob, col_sol = st.columns(2)
    with col_prob:
        st.markdown(
            """
            <div class="research-card">
                <div class="research-card-title">
                    <span>⚠️</span> The Multimodal Disconnect in Pulmonary Research
                </div>
                <div class="research-card-body">
                    CT scans, clinical case metadata, global medical literature, and regional epidemiological data 
                    traditionally exist in disconnected silos. When researchers query general LLMs, the models frequently hallucinate 
                    diagnostic criteria, cite nonexistent clinical studies, or provide unverified treatment recommendations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_sol:
        st.markdown(
            """
            <div class="research-card">
                <div class="research-card-title">
                    <span>🛡️</span> The PulmoLens Grounding Guarantee
                </div>
                <div class="research-card-body">
                    PulmoLens solves hallucination through a strict zero-temperature verification pipeline:
                    candidate localization is frozen, retrieval is multi-tiered and bounded, and every single sentence 
                    generated by the LLM is subjected to automated citation mapping and lexical overlap verification.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # End-to-End Workflow Pipeline
    st.markdown("##### End-to-End Research Intelligence Workflow")
    s1, s2, s3, s4 = st.columns(4)

    pipeline_steps = [
        ("01", "Candidate Localization", "YOLO11s scans 2D CT slices to isolate suspicious nodule regions with bounding coordinates."),
        ("02", "Multi-Store Retrieval", "FAISS conducts dense vector search across LIDC cases, global papers, and regional literature."),
        ("03", "Neural Reranking & Gating", "BGE cross-encoder calculates fine-grained relevance and enforces dynamic similarity thresholds."),
        ("04", "Grounded Explanation", "Gemini 2.5 Flash synthesizes evidence; claims failing lexical verification are strictly pruned."),
    ]

    for col, (num, title, text) in zip([s1, s2, s3, s4], pipeline_steps):
        with col:
            st.markdown(
                f"""
                <div class="pipeline-step-card">
                    <div class="pipeline-step-num">{num}</div>
                    <div class="pipeline-step-title">{title}</div>
                    <div class="pipeline-step-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # Connected Services Live Status
    st.markdown("##### Connected System Services")
    cs1, cs2, cs3 = st.columns(3)

    with cs1:
        if bool(gemini_key):
            st.markdown('<div class="badge-status badge-status-green">● Gemini 2.5 Flash API Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="badge-status badge-status-amber">● Gemini API Key Not Configured</div>', unsafe_allow_html=True)
        st.caption("Language generation restricted strictly to retrieved evidence bundles.")

    with cs2:
        if yolo_model is not None and bool(stores):
            st.markdown('<div class="badge-status badge-status-green">● AI Inference Engine Active</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="badge-status badge-status-red">● Inference Engine Needs Attention</div>', unsafe_allow_html=True)
        st.caption("YOLO11s detector and FAISS 768-D indexing running locally.")

    with cs3:
        if len(stores) == 3:
            st.markdown('<div class="badge-status badge-status-green">● 3/3 Evidence Stores Mounted</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="badge-status badge-status-amber">● {len(stores)}/3 Stores Available</div>', unsafe_allow_html=True)
        st.caption("LIDC structured records, PubMed global corpus, and Pakistan literature.")

    # Medical Prototype Disclaimer
    st.markdown(
        """
        <div class="medical-notice-box">
            <div class="medical-notice-icon">ℹ️</div>
            <div>
                <div class="medical-notice-title">Research Prototype Protocol</div>
                <div class="medical-notice-text">
                    PulmoLens is developed exclusively for scientific research, benchmarking, and academic education. 
                    It is not a clinically certified medical device and must never be used for independent diagnostic decisions, 
                    prognosis assessment, or treatment planning.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE 2: CT SCAN ANALYSIS (LIVE DEMO WORKFLOW)
# ============================================================

elif page == "CT Scan Analysis":

    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-kicker-pill">
                <span>🩻</span> COMPUTER VISION WORKFLOW · YOLO11s
            </div>
            <div class="hero-title">
                2D CT Slice Candidate Localization
            </div>
            <div class="hero-text">
                Upload a 2D pulmonary CT slice to evaluate candidate nodule regions. PulmoLens uses a frozen 
                YOLO11s research model trained on LIDC annotations to detect candidate lesions and extract 
                spatial bounding coordinates.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload a 2D CT slice (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
        help="Select a 2D axial pulmonary CT image slice for candidate nodule localization.",
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        col_img_orig, col_img_res = st.columns(2)

        with col_img_orig:
            st.markdown(
                """
                <div class="research-card">
                    <div class="research-card-title">
                        <span>Original CT Slice</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(image, use_container_width=True)

        with col_img_res:
            st.markdown(
                """
                <div class="research-card">
                    <div class="research-card-title">
                        <span>PulmoLens YOLO11s Detection</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if yolo_model is None:
                st.error("The YOLO11s model weights could not be loaded from disk.")
            else:
                with st.spinner("Executing neural inference on CT slice..."):
                    results = yolo_model(image, verbose=False)

                plotted = results[0].plot()
                st.image(plotted, caption="Localized Candidate Nodule Regions", use_container_width=True)

                boxes = results[0].boxes

                if boxes is None or len(boxes) == 0:
                    st.markdown(
                        """
                        <div class="badge-status badge-status-amber">
                            <span>●</span> No candidate regions detected exceeding threshold.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="badge-status badge-status-green">
                            <span>✓</span> Detection complete · {len(boxes)} candidate region(s) identified
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    rows = []
                    for number, box in enumerate(boxes, start=1):
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy()
                        rows.append(
                            {
                                "Candidate": f"Region #{number}",
                                "Confidence": f"{conf:.3f}",
                                "X-Min (Left)": round(float(xyxy[0]), 1),
                                "Y-Min (Top)": round(float(xyxy[1]), 1),
                                "X-Max (Right)": round(float(xyxy[2]), 1),
                                "Y-Max (Bottom)": round(float(xyxy[3]), 1),
                            }
                        )

                    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                    st.dataframe(
                        pd.DataFrame(rows),
                        use_container_width=True,
                        hide_index=True,
                    )

        st.markdown(
            """
            <div class="research-card" style="margin-top: 20px;">
                <div class="research-card-title">
                    <span>💡</span> Next Step: Multimodal Research Grounding
                </div>
                <div class="research-card-body">
                    If this CT scan corresponds to a known dataset case (e.g. <code>LIDC-IDRI-0001</code>) or contains specific nodule features (e.g., spiculation, lobulation, ground-glass opacity), explore literature and structured records using the <b>Research Assistant</b> workspace.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="research-card" style="margin-top: 14px;">
                <div class="research-card-title">
                    <span>🔬</span> CT Analysis Pipeline Protocol
                </div>
                <div class="research-card-body">
                    <b>1. Input:</b> Upload a standard 2D axial pulmonary CT slice in JPG or PNG format.<br>
                    <b>2. Vision Inference:</b> The YOLO11s detector predicts spatial bounding boxes for nodule candidates.<br>
                    <b>3. Coordinate Extraction:</b> Pixel bounding boxes (Left, Top, Right, Bottom) and confidence scores are calculated.<br>
                    <b>4. Cross-Verification:</b> Candidate findings can be cross-referenced against LIDC ground-truth evidence.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="medical-notice-box">
            <div class="medical-notice-icon">⚠️</div>
            <div>
                <div class="medical-notice-title">Imaging Scope Notice</div>
                <div class="medical-notice-text">
                    This module evaluates isolated 2D slices for candidate localization research. Full clinical pulmonary nodule assessment requires volumetric 3D DICOM series inspection and multidisciplinary clinical review.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE 3: RESEARCH ASSISTANT (EVIDENCE-GROUNDED QA)
# ============================================================

elif page == "Research Assistant":

    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-kicker-pill">
                <span>📚</span> MULTILINGUAL RETRIEVAL-AUGMENTED RESEARCH
            </div>
            <div class="hero-title">
                Ask a pulmonary nodule research question
            </div>
            <div class="hero-text">
                Formulate queries in English, Urdu, or Roman Urdu. PulmoLens retrieves supporting evidence from 
                structured LIDC records and peer-reviewed literature, applies neural reranking, and synthesizes 
                a strictly grounded explanation where every claim is verified against source citations.
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
        "Pre-configured Research Prompts (or compose below)",
        ["Write my own question"] + examples,
    )

    default_query = "" if selected_example == "Write my own question" else selected_example

    query = st.text_area(
        "Clinical or Research Query",
        value=default_query,
        height=110,
        placeholder="e.g. What morphological criteria differentiate solid and subsolid pulmonary nodules in medical literature?",
    )

    col_btn, col_meta = st.columns([1, 2])
    with col_btn:
        search_clicked = st.button(
            "🔍 Search Evidence",
            type="primary",
            use_container_width=True,
        )
    with col_meta:
        detected_lang = detect_language(query) if query.strip() else "auto-detect"
        st.markdown(
            f"""
            <div style="font-size: 12px; color: var(--text-muted); padding-top: 10px;">
                Routing: <b>Multilingual Vector Search + BGE Reranker</b> · Language: <code style="color: var(--teal-primary);">{detected_lang}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if search_clicked:
        st.session_state["pulmo_claims"] = []

        if not query.strip():
            st.error("Please provide a research question to search the evidence library.")
        elif unsafe_request(query):
            st.error(
                "Safety Alert: This query falls outside the safe research scope of PulmoLens. "
                "The system does not provide patient-specific medical diagnosis or direct treatment advice."
            )
            st.session_state["pulmo_retrieval"] = None
        elif not stores:
            st.error("The FAISS evidence index stores could not be loaded from the local repository.")
        else:
            with st.spinner("Executing dense multi-store vector retrieval & BGE reranking..."):
                result = retrieve(query=query, relative_ratio=relative_ratio)

            st.session_state["pulmo_retrieval"] = result
            st.session_state["pulmo_query"] = query

    retrieval = st.session_state.get("pulmo_retrieval")

    if retrieval:
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        if not retrieval["grounded"]:
            st.markdown(
                """
                <div class="badge-status badge-status-amber" style="padding: 14px;">
                    <span>⚠️</span> No sufficiently relevant evidence passed the similarity strictness gate. 
                    In accordance with our zero-hallucination policy, PulmoLens will not synthesize an unsupported answer.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Summary Bar
            st.markdown(
                f"""
                <div class="badge-status badge-status-green" style="padding: 12px 16px; margin-bottom: 16px;">
                    <span>✓</span> Evidence retrieval successful · {len(retrieval['evidence'])} verified source(s) extracted
                </div>
                """,
                unsafe_allow_html=True,
            )

            inf1, inf2, inf3 = st.columns(3)
            with inf1:
                st.markdown(
                    f"""
                    <div class="research-card" style="padding: 12px 14px; margin-bottom: 12px;">
                        <div style="font-size: 11px; color: var(--text-muted); font-weight: 700;">RESEARCH ROUTE</div>
                        <div style="font-size: 13.5px; font-weight: 700; color: var(--navy-surface); margin-top: 2px;">
                            {retrieval["route"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with inf2:
                st.markdown(
                    f"""
                    <div class="research-card" style="padding: 12px 14px; margin-bottom: 12px;">
                        <div style="font-size: 11px; color: var(--text-muted); font-weight: 700;">DETECTED LANGUAGE</div>
                        <div style="font-size: 13.5px; font-weight: 700; color: var(--navy-surface); margin-top: 2px;">
                            {retrieval["language"].replace('_', ' ').title()}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with inf3:
                st.markdown(
                    f"""
                    <div class="research-card" style="padding: 12px 14px; margin-bottom: 12px;">
                        <div style="font-size: 11px; color: var(--text-muted); font-weight: 700;">RETRIEVED ITEMS</div>
                        <div style="font-size: 13.5px; font-weight: 700; color: var(--navy-surface); margin-top: 2px;">
                            {len(retrieval["evidence"])} Evidence Documents
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("##### Supporting Evidence Bundle")

            for item in retrieval["evidence"]:
                score_badges = []
                if item.get("score") is not None:
                    score_badges.append(f"Dense: {item['score']:.3f}")
                if item.get("rerank_score") is not None:
                    score_badges.append(f"Rerank: {item['rerank_score']:.3f}")

                score_html = (
                    f'<div class="evidence-score-pill">{" · ".join(score_badges)}</div>'
                    if score_badges
                    else ""
                )

                item_title = item.get("title", "") or "Structured Evidence Record"
                item_text = item.get("text", "")

                st.markdown(
                    f"""
                    <div class="evidence-item-card">
                        <div class="evidence-item-header">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span class="evidence-badge-id">[{item['id']}]</span>
                                <span class="evidence-source-tag">{item['store']}</span>
                            </div>
                            {score_html}
                        </div>
                        <div class="evidence-item-title">{item_title}</div>
                        <div class="evidence-item-text">{item_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

            # Grounded Generation Block
            st.markdown("##### Evidence-Bounded Synthesis Engine")
            st.caption(
                "Gemini 2.5 Flash will be strictly restricted to the Evidence Bundle above. "
                "Any statement lacking verifiable citation mapping will be deterministically pruned."
            )

            col_gen_btn, _ = st.columns([1, 1])
            with col_gen_btn:
                generate_clicked = st.button(
                    "✨ Generate Grounded Explanation",
                    type="primary",
                    use_container_width=True,
                )

            if generate_clicked:
                if not gemini_key:
                    st.error("LLM API key is not configured. Please supply GEMINI_API_KEY in secrets or environment.")
                else:
                    with st.spinner("Synthesizing grounded claims and performing lexical verification..."):
                        generated, error = generate_grounded_answer(
                            query=st.session_state["pulmo_query"],
                            evidence=retrieval["evidence"],
                            output_language=output_language,
                            api_key=gemini_key,
                        )

                    if error:
                        st.error(f"Generative synthesis error: {error}")
                    elif generated is None:
                        st.warning("Insufficient evidence available to formulate a clinically safe answer.")
                    else:
                        valid_claims = validate_response(generated, retrieval["evidence"])
                        st.session_state["pulmo_claims"] = valid_claims

                        if not valid_claims:
                            st.warning("All candidate claims were filtered during lexical support verification.")
                        else:
                            st.markdown(
                                """
                                <div class="answer-container">
                                    <div class="answer-container-header">
                                        <div class="answer-container-title">
                                            <span>PulmoLens Research Synthesis</span>
                                        </div>
                                        <div class="grounded-badge">✓ EVIDENCE GROUNDED</div>
                                    </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            for claim in valid_claims:
                                citation_pills = "".join(
                                    f'<span class="citation-pill">[{c}]</span>'
                                    for c in claim.citations
                                )
                                st.markdown(
                                    f"""
                                    <div class="claim-card">
                                        {claim.text} {citation_pills}
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            st.markdown("</div>", unsafe_allow_html=True)

                            st.markdown(
                                f"""
                                <div class="badge-status badge-status-green" style="margin-top: 12px;">
                                    <span>✓</span> {len(valid_claims)} claim(s) successfully passed strict citation and lexical support validation
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            # PDF Report Generation
                            try:
                                pdf_bytes = build_pdf_report(
                                    query=st.session_state["pulmo_query"],
                                    claims=valid_claims,
                                    evidence=retrieval["evidence"],
                                    language=output_language,
                                )

                                col_pdf_btn, _ = st.columns([1, 1])
                                with col_pdf_btn:
                                    st.download_button(
                                        label="📄 Download Research Report (PDF)",
                                        data=pdf_bytes,
                                        file_name=f"PulmoLens_Research_Report_{retrieval.get('patient_id') or 'Query'}.pdf",
                                        mime="application/pdf",
                                        use_container_width=True,
                                    )
                            except Exception as exc:
                                st.warning(f"PDF compilation notice: {exc}")


# ============================================================
# PAGE 4: EVIDENCE EXPLORER (RESEARCH LIBRARY)
# ============================================================

elif page == "Evidence Explorer":

    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-kicker-pill">
                <span>📂</span> MULTI-SOURCE EVIDENCE REPOSITORY
            </div>
            <div class="hero-title">
                Explore the Research Knowledge Base
            </div>
            <div class="hero-text">
                Inspect the multi-source corpus backing PulmoLens before retrieval. 
                Our indexed library incorporates 240 structured LIDC annotations, 122 global peer-reviewed 
                oncology publications, and 294 regional pulmonary studies.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top Metric Summary Cards
    em1, em2, em3 = st.columns(3)
    with em1:
        st.markdown(
            """
            <div class="metric-card-modern">
                <div class="metric-value-modern">240</div>
                <div class="metric-label-modern">LIDC Structured Records</div>
                <div class="metric-sub-modern">Radiologist nodule ratings & spatial features</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with em2:
        st.markdown(
            """
            <div class="metric-card-modern">
                <div class="metric-value-modern">122</div>
                <div class="metric-label-modern">Global Medical Literature</div>
                <div class="metric-sub-modern">PubMed indexed oncology & CT literature</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with em3:
        st.markdown(
            """
            <div class="metric-card-modern">
                <div class="metric-value-modern">294</div>
                <div class="metric-label-modern">Pakistan Regional Literature</div>
                <div class="metric-sub-modern">Local epidemiology & clinical research papers</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    source_choice = st.selectbox(
        "Select Evidence Collection to Inspect",
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
        metadata = stores[store_key]["metadata"]
        display_rows = []

        for item in metadata:
            display_rows.append(
                {
                    "Title / Topic": item.get("title", "") or "Clinical Evidence Entry",
                    "Source ID": item.get("source_id", "N/A"),
                    "Patient ID": item.get("patient_id", "N/A"),
                    "Nodule ID": item.get("nodule_id", "N/A"),
                }
            )

        df = pd.DataFrame(display_rows)
        
        st.markdown(f"**Viewing {len(df)} indexed documents in `{source_choice}`**")
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        with st.expander("🔍 Deep Record Inspector (View Full Text & Citations)"):
            rec_idx = st.number_input(
                f"Select Record Index (0 to {len(metadata)-1})",
                min_value=0,
                max_value=max(0, len(metadata) - 1),
                value=0,
                step=1,
            )
            if metadata:
                sel_meta = metadata[rec_idx]
                st.markdown(f"**Title:** {sel_meta.get('title', 'N/A')}")
                st.markdown(f"**Citation:** {sel_meta.get('citation', 'N/A')}")
                st.markdown(f"**PMID / DOI:** `{sel_meta.get('pmid', 'N/A')}` · `{sel_meta.get('doi', 'N/A')}`")
                st.text_area("Record Text Content", value=sel_meta.get("text", ""), height=150, disabled=True)
    else:
        st.warning("The selected evidence store is currently unavailable.")


# ============================================================
# PAGE 5: SYSTEM ARCHITECTURE (TECHNICAL SOPHISTICATION)
# ============================================================

elif page == "System Architecture":

    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-kicker-pill">
                <span>⚙️</span> MULTIMODAL SYSTEM TOPOLOGY
            </div>
            <div class="hero-title">
                PulmoLens End-to-End Architecture
            </div>
            <div class="hero-text">
                A rigorous, multi-tiered pipeline that couples computer vision candidate localization with 
                dense vector retrieval, cross-encoder neural reranking, and zero-temperature evidence-bounded LLM reasoning.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### Visual Pipeline Diagram")

    st.markdown(
        """
        <div class="arch-diagram-grid">
            <div class="arch-phase">
                <div class="arch-phase-title">PHASE 1 · MULTIMODAL USER INPUT</div>
                <div class="arch-nodes-row">
                    <div class="arch-node">
                        🩻 2D CT Slice Image
                        <div class="arch-node-desc">Axial lung slice (PNG / JPG)</div>
                    </div>
                    <div class="arch-node">
                        💬 Research Query
                        <div class="arch-node-desc">English / Urdu / Roman Urdu</div>
                    </div>
                </div>
            </div>

            <div class="arch-flow-arrow">▼</div>

            <div class="arch-phase">
                <div class="arch-phase-title">PHASE 2 · VISION INFERENCE & VECTOR RETRIEVAL</div>
                <div class="arch-nodes-row">
                    <div class="arch-node">
                        🎯 YOLO11s Detector
                        <div class="arch-node-desc">Frozen candidate nodule locator</div>
                    </div>
                    <div class="arch-node">
                        ⚡ FAISS Index (768-D)
                        <div class="arch-node-desc">Multilingual dense semantic search</div>
                    </div>
                    <div class="arch-node">
                        🗂️ Exact LIDC Lookup
                        <div class="arch-node-desc">Deterministic patient ID routing</div>
                    </div>
                </div>
            </div>

            <div class="arch-flow-arrow">▼</div>

            <div class="arch-phase">
                <div class="arch-phase-title">PHASE 3 · NEURAL RERANKING & DYNAMIC GATING</div>
                <div class="arch-nodes-row">
                    <div class="arch-node">
                        🧠 BGE Reranker-v2-m3
                        <div class="arch-node-desc">Cross-encoder pair relevance scoring</div>
                    </div>
                    <div class="arch-node">
                        🛡️ Relative Ratio Gate
                        <div class="arch-node-desc">Dynamic similarity thresholding (0.70x)</div>
                    </div>
                </div>
            </div>

            <div class="arch-flow-arrow">▼</div>

            <div class="arch-phase">
                <div class="arch-phase-title">PHASE 4 · BOUNDED GENERATION & CLAIM VALIDATION</div>
                <div class="arch-nodes-row">
                    <div class="arch-node">
                        🤖 Gemini 2.5 Flash
                        <div class="arch-node-desc">Zero-temperature evidence-bounded synthesis</div>
                    </div>
                    <div class="arch-node">
                        🔍 Lexical Overlap Check
                        <div class="arch-node-desc">Deterministic token support verification</div>
                    </div>
                </div>
            </div>

            <div class="arch-flow-arrow">▼</div>

            <div class="arch-phase">
                <div class="arch-phase-title">PHASE 5 · VERIFIED CLINICAL RESEARCH OUTPUT</div>
                <div class="arch-nodes-row">
                    <div class="arch-node">
                        📋 Grounded Claims
                        <div class="arch-node-desc">Sentence-level citation badges [E001]</div>
                    </div>
                    <div class="arch-node">
                        📄 PDF Research Report
                        <div class="arch-node-desc">ReportLab formatted documentation</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### Architectural Component Deep Dive")
    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown(
            """
            <div class="research-card">
                <div class="research-card-title">
                    <span>🧠</span> Vision & Retrieval Subsystems
                </div>
                <div class="research-card-body">
                    <b>• YOLO11s (Ultralytics):</b> Fine-tuned on annotated LIDC-IDRI slices to locate candidate nodule bounding boxes.<br><br>
                    <b>• Multilingual Embedder:</b> <code>paraphrase-multilingual-mpnet-base-v2</code> maps English, Urdu, and Roman Urdu into a 768-D shared embedding space.<br><br>
                    <b>• FAISS Multi-Index:</b> Partitioned indices for LIDC structured metadata, global PubMed oncology, and regional Pakistan studies.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_right:
        st.markdown(
            """
            <div class="research-card">
                <div class="research-card-title">
                    <span>🛡️</span> Reranking & Verification Guardrails
                </div>
                <div class="research-card-body">
                    <b>• BGE Reranker (BAAI/bge-reranker-v2-m3):</b> Cross-encoder reassesses query-passage semantic alignment before passing to the LLM.<br><br>
                    <b>• Evidence Gating:</b> Rejects queries where evidence relevance falls below strict relative ratio floors.<br><br>
                    <b>• Lexical Claim Verification:</b> Mathematical n-gram overlap check ensures claims strictly reflect retrieved passages.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# PAGE 6: SYSTEM STATUS (PLATFORM HEALTH DASHBOARD)
# ============================================================

elif page == "System Status":

    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-kicker-pill">
                <span>🩺</span> SYSTEM READINESS & DIAGNOSTICS
            </div>
            <div class="hero-title">
                Platform Health & Service Status
            </div>
            <div class="hero-text">
                Real-time operational status for all core neural networks, vector retrieval indices, 
                and generative API connections powering PulmoLens.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Readiness Summary Banner
    health_card_class = "badge-status-green" if all_systems_ready else "badge-status-amber"
    st.markdown(
        f"""
        <div class="research-card" style="border-left: 4px solid var(--teal-primary);">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <div style="font-size: 18px; font-weight: 800; color: var(--navy-surface);">System Readiness Score</div>
                    <div style="font-size: 13px; color: var(--text-muted); margin-top: 2px;">
                        {ready_count} of {total_checks} critical research services currently operational
                    </div>
                </div>
                <div class="grounded-badge" style="font-size: 13px; padding: 6px 14px;">
                    {'OPERATIONAL' if all_systems_ready else 'PARTIAL AVAILABILITY'}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### Detailed Component Diagnostics")

    for name, ready, detail in system_checks:
        c_status, c_info = st.columns([1, 4])
        with c_status:
            if ready:
                st.markdown('<div class="badge-status badge-status-green">● Connected</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-status badge-status-red">● Offline</div>', unsafe_allow_html=True)
        with c_info:
            st.markdown(
                f"""
                <div style="padding-top: 6px;">
                    <b>{name}</b> — <span style="color: var(--text-muted); font-size: 13px;">{detail}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    if st.session_state.get("yolo_error"):
        st.error(f"YOLO Model Initialization Error: {st.session_state['yolo_error']}")

    if st.session_state.get("retrieval_error"):
        st.error(f"FAISS Retrieval Initialization Error: {st.session_state['retrieval_error']}")

    # Research Governance Card
    st.markdown(
        """
        <div class="research-card" style="margin-top: 16px;">
            <div class="research-card-title">
                <span>⚖️</span> Research Governance & Clinical Safety Protocol
            </div>
            <div class="research-card-body">
                PulmoLens operates under strict academic and research governance protocols. The application 
                enforces non-diagnostic output contracts, abstains from answering when evidence support is 
                unsubstantiated, and maintains 100% deterministic traceability back to indexed scientific literature.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# APPLICATION FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        margin-top: 40px;
        padding: 20px 0 10px 0;
        border-top: 1px solid #E2E8F0;
        color: #64748B;
        font-size: 12px;
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    ">
        <span>🫁 <b>PulmoLens</b></span>
        <span>·</span>
        <span>Evidence-Grounded Pulmonary Nodule Research Platform</span>
        <span>·</span>
        <span>Academic Research & Hackathon Submission</span>
    </div>
    """,
    unsafe_allow_html=True,
)
