import os
import tempfile
from datetime import datetime

import cv2
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from helmet_detection import detect_helmet
from plate_detection import detect_plate
from ocr import read_plate
from registry import get_registry, save_violation
from sms_alert import send_sms_alert


st.set_page_config(
    page_title="Helmet Watch Pro | AI for Safer Roads",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "nav" not in st.session_state:
    st.session_state.nav = "dashboard"
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "processed_id" not in st.session_state:
    st.session_state.processed_id = None
if "flash" not in st.session_state:
    st.session_state.flash = None


THEME_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background: #f3f6fb !important;
        color: #0f172a !important;
        font-family: 'Inter', sans-serif !important;
    }

    #MainMenu, footer, header [data-testid="stToolbar"] { display: none !important; }
    [data-testid="stHeader"] { background: transparent !important; }

    .block-container {
        padding: 1.1rem 1.6rem 2rem 1.6rem !important;
        max-width: 1440px !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1f3a 0%, #123056 48%, #0d2748 100%) !important;
        border-right: 0 !important;
        transform: none !important;
        visibility: visible !important;
        min-width: 260px !important;
        width: 260px !important;
        left: 0 !important;
        margin-left: 0 !important;
    }
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stExpandSidebarButton"] {
        display: none !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        background: transparent !important;
        padding: 1.1rem 0.85rem 1.4rem 0.85rem !important;
    }
    [data-testid="stSidebar"] * { color: #e8eef7 !important; }
    [data-testid="stSidebar"] .stButton button {
        background: transparent !important;
        color: #d5e2f3 !important;
        border: 0 !important;
        border-radius: 12px !important;
        justify-content: flex-start !important;
        font-weight: 600 !important;
        padding: 0.72rem 0.9rem !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: rgba(255,255,255,0.08) !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] button[kind="primary"] {
        background: #2563eb !important;
        color: #ffffff !important;
        box-shadow: 0 8px 18px rgba(37, 99, 235, 0.35) !important;
    }

    .brand-wrap {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        padding: 0.35rem 0.4rem 1.2rem 0.4rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1rem;
    }
    .brand-mark {
        width: 42px; height: 42px; border-radius: 12px;
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.35rem;
        box-shadow: 0 8px 20px rgba(37,99,235,0.4);
    }
    .brand-name { font-size: 1.05rem; font-weight: 800; line-height: 1.15; }
    .brand-sub { font-size: 0.72rem; color: #93c5fd !important; margin-top: 2px; }

    .sidebar-foot {
        margin-top: 2.2rem;
        padding: 1rem 0.6rem 0.2rem 0.6rem;
        border-top: 1px solid rgba(255,255,255,0.08);
        color: #93a4c0 !important;
        font-size: 0.75rem;
        line-height: 1.45;
    }

    .top-title { font-size: 1.55rem; font-weight: 800; color: #0f172a; margin: 0; }
    .top-sub { color: #64748b; font-size: 0.9rem; margin-top: 2px; }

    .user-chip {
        display: flex; align-items: center; gap: 0.55rem;
        background: #fff; border: 1px solid #e2e8f0; border-radius: 999px;
        padding: 0.35rem 0.75rem 0.35rem 0.4rem;
    }
    .user-avatar {
        width: 28px; height: 28px; border-radius: 50%;
        background: #dbeafe; display: flex; align-items: center; justify-content: center;
    }

    .kpi-card {
        border-radius: 16px;
        padding: 1rem 1.1rem;
        display: flex; align-items: center; gap: 0.9rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
        border: 1px solid transparent;
        min-height: 96px;
    }
    .kpi-blue { background: #eef6ff; border-color: #dbeafe; }
    .kpi-pink { background: #fdf2f8; border-color: #fce7f3; }
    .kpi-green { background: #ecfdf5; border-color: #dcfce7; }
    .kpi-purple { background: #f5f3ff; border-color: #ede9fe; }
    .kpi-icon {
        width: 46px; height: 46px; border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.25rem; flex-shrink: 0;
    }
    .kpi-label { font-size: 0.82rem; color: #64748b; font-weight: 600; }
    .kpi-value { font-size: 1.7rem; font-weight: 800; color: #0f172a; line-height: 1.15; }
    .kpi-sub { font-size: 0.75rem; color: #94a3b8; }
    .kpi-blue .kpi-icon { background: #dbeafe; }
    .kpi-pink .kpi-icon { background: #fce7f3; }
    .kpi-green .kpi-icon { background: #dcfce7; }
    .kpi-purple .kpi-icon { background: #ede9fe; }

    [data-testid="stFileUploaderDropzone"] {
        min-height: 220px !important;
        align-items: center !important;
        justify-content: center !important;
        border: 2px dashed #bfdbfe !important;
        background: #f8fbff !important;
        border-radius: 16px !important;
    }
    [data-testid="stFileUploader"] section {
        background: #fff !important;
        border: 1px dashed #bfdbfe !important;
        border-radius: 14px !important;
        padding: 0.4rem 0.6rem !important;
    }
    [data-testid="stFileUploader"] button {
        background: #2563eb !important;
        color: #fff !important;
        border: 0 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
    }

    .panel {
        background: #fff;
        border: 1px solid #e8eef5;
        border-radius: 18px;
        padding: 1.15rem 1.2rem 1.2rem 1.2rem;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
        margin-bottom: 1rem;
    }
    .panel-title { font-size: 1.05rem; font-weight: 700; color: #0f172a; }
    .panel-sub { color: #64748b; font-size: 0.84rem; margin: 0.15rem 0 0.9rem 0; }

    .dropzone {
        border: 2px dashed #bfdbfe;
        background: #f8fbff;
        border-radius: 16px;
        min-height: 280px;
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        text-align: center;
        color: #64748b;
        padding: 1.5rem;
    }
    .dropzone h4 { color: #1e3a8a; margin: 0.6rem 0 0.3rem 0; font-size: 1rem; }

    .result-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.9rem 1rem;
    }
    .result-ok { color: #16a34a; font-weight: 700; }
    .result-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 0.62rem 0;
        border-bottom: 1px solid #eef2f7;
        font-size: 0.9rem;
    }
    .result-row:last-child { border-bottom: 0; }
    .result-label { color: #64748b; }
    .result-val { font-weight: 700; color: #0f172a; }
    .status-bad { color: #dc2626; }
    .status-good { color: #16a34a; }

    [data-testid="stFileUploader"] { margin-top: -8px; }
    div[data-testid="stImage"] { border-radius: 12px; overflow: hidden; }
    [data-testid="stDataFrame"] {
        border: 1px solid #e8eef5; border-radius: 12px; overflow: hidden;
    }

    .login-card {
        background: #fff; border-radius: 22px; padding: 2rem 2.1rem;
        box-shadow: 0 20px 50px rgba(15,23,42,0.12);
        border: 1px solid #e8eef5;
    }
    .badge-soft {
        display: inline-block; background: #dbeafe; color: #1d4ed8;
        border-radius: 999px; padding: 0.25rem 0.7rem;
        font-size: 0.75rem; font-weight: 700;
    }
    .demo-note {
        background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 12px;
        padding: 0.7rem 0.85rem; font-size: 0.84rem; color: #1e3a8a; margin: 0.8rem 0 1rem 0;
    }

    .header-logout button {
        background: #fff !important;
        color: #ef4444 !important;
        border: 1px solid #fca5a5 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        box-shadow: none !important;
    }
</style>
"""

LOGIN_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700;800&family=Outfit:wght@300;400;500;600;700;800&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background-color: #06090e !important;
        background-image:
            radial-gradient(rgba(0, 242, 254, 0.08) 1.2px, transparent 1.2px),
            linear-gradient(145deg, #05080e 0%, #09101c 45%, #060a12 100%) !important;
        background-size: 28px 28px, 100% 100% !important;
        color: #e2e8f0 !important;
        font-family: 'Outfit', -apple-system, sans-serif !important;
    }

    #MainMenu, footer, header [data-testid="stToolbar"] { display: none !important; }
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stExpandSidebarButton"] { display: none !important; }

    .block-container {
        padding-top: 1.4rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1420px !important;
    }

    .text-cyan {
        color: #00f2fe !important;
        text-shadow: 0 0 20px rgba(0, 242, 254, 0.4);
    }
    .text-orange {
        color: #ff6a00 !important;
        text-shadow: 0 0 20px rgba(255, 106, 0, 0.4);
    }
    .accent-swoop {
        height: 3px;
        width: 120px;
        background: linear-gradient(90deg, #ff6a00 0%, #ee0979 100%);
        border-radius: 999px;
        margin: 0.6rem 0 1.2rem 0;
        box-shadow: 0 0 12px rgba(255, 106, 0, 0.6);
    }

    .auth-hero-shell {
        background: rgba(10, 16, 28, 0.88);
        border: 1px solid rgba(0, 242, 254, 0.25);
        border-radius: 26px;
        padding: 2.6rem;
        backdrop-filter: blur(20px);
        box-shadow: 0 25px 60px rgba(0, 0, 0, 0.6), 0 0 35px rgba(0, 242, 254, 0.12);
    }
    .auth-pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.6rem;
        background: rgba(0, 242, 254, 0.1);
        border: 1px solid rgba(0, 242, 254, 0.35);
        border-radius: 999px;
        padding: 0.4rem 1rem;
        color: #00f2fe;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-weight: 700;
    }
    .hero-main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1.1;
        color: #ffffff;
        margin: 1.2rem 0 0.4rem 0;
        letter-spacing: -0.02em;
    }
    .hero-sub-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #ff6a00;
        margin-bottom: 1.2rem;
    }
    .feature-grid-stack {
        display: flex;
        flex-direction: column;
        gap: 0.9rem;
        margin: 1.8rem 0;
    }
    .feature-grid-item {
        display: flex;
        align-items: center;
        gap: 1rem;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 14px;
        padding: 0.85rem 1.1rem;
    }
    .feature-grid-icon {
        width: 2.6rem;
        height: 2.6rem;
        border-radius: 10px;
        background: rgba(0, 242, 254, 0.14);
        border: 1px solid rgba(0, 242, 254, 0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        flex-shrink: 0;
    }
    .demo-hud-callout {
        background: rgba(255, 106, 0, 0.09);
        border: 1px solid rgba(255, 106, 0, 0.28);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.84rem;
    }
    .demo-hud-tag {
        background: #ff6a00;
        color: #ffffff;
        font-weight: 800;
        font-size: 0.68rem;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        letter-spacing: 0.08em;
    }

    [data-testid="stForm"] {
        background: rgba(10, 16, 28, 0.88) !important;
        border: 1px solid rgba(0, 242, 254, 0.25) !important;
        border-radius: 24px !important;
        padding: 2.2rem !important;
        backdrop-filter: blur(20px) !important;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5) !important;
    }
    div[data-baseweb="input"] {
        background-color: rgba(14, 22, 38, 0.9) !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        border-radius: 10px !important;
        color: #f8fafc !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #00f2fe !important;
        box-shadow: 0 0 0 3px rgba(0, 242, 254, 0.3) !important;
    }
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 50%, #ff6a00 100%) !important;
        color: #ffffff !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.04em !important;
        border: 1px solid rgba(0, 242, 254, 0.4) !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.4rem !important;
        box-shadow: 0 4px 18px rgba(0, 198, 255, 0.4) !important;
    }
    [data-testid="stWidgetLabel"] p, label {
        color: #94a3b8 !important;
    }
</style>
"""


def init_session():
    defaults = {
        "authenticated": False,
        "nav": "dashboard",
        "uploader_key": 0,
        "analysis": None,
        "processed_id": None,
        "flash": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def annotate_image(image_path, helmet_detections, plate_result=None, plate_number=""):
    img = cv2.imread(image_path)
    if img is None:
        return None, None

    for det in helmet_detections:
        cls_name = det.get("class", "").lower()
        conf = det.get("confidence", 0.0)
        box = det.get("box", [])
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        is_violation = ("no" in cls_name and "helmet" in cls_name)
        box_color = (56, 56, 230) if is_violation else (70, 190, 80)
        label = f"{'no-helmet' if is_violation else 'helmet'} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y1 = max(0, y1 - 22)
        cv2.rectangle(img, (x1, label_y1), (x1 + w + 8, y1), box_color, -1)
        cv2.putText(img, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    if plate_result and "box" in plate_result:
        pbox = plate_result["box"]
        if len(pbox) == 4:
            px1, py1, px2, py2 = [int(v) for v in pbox]
            pconf = plate_result.get("confidence", 0.0)
            plate_color = (210, 120, 40)
            plabel = f"plate {pconf:.2f}"
            cv2.rectangle(img, (px1, py1), (px2, py2), plate_color, 2)
            (pw, ph), _ = cv2.getTextSize(plabel, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            plabel_y1 = max(0, py1 - 20)
            cv2.rectangle(img, (px1, plabel_y1), (px1 + pw + 8, py1), plate_color, -1)
            cv2.putText(img, plabel, (px1 + 4, py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            if plate_number and plate_number != "UNKNOWN":
                cv2.putText(
                    img,
                    plate_number,
                    (px1 + 6, min(img.shape[0] - 8, py2 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

    annotated_path = os.path.join(tempfile.gettempdir(), "helmet_watch_annotated.jpg")
    cv2.imwrite(annotated_path, img)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), annotated_path


def has_no_helmet(detections):
    for detection in detections:
        class_name = detection.get("class", "").lower()
        if "no" in class_name and "helmet" in class_name:
            return True, float(detection.get("confidence", 0.0))
    helmet_conf = 0.0
    for detection in detections:
        class_name = detection.get("class", "").lower()
        if "helmet" in class_name:
            helmet_conf = max(helmet_conf, float(detection.get("confidence", 0.0)))
    return False, helmet_conf


INVALID_VEHICLE_NUMBERS = {"", "UNKNOWN", "NOT DETECTED"}
OCR_CONFIDENCE_THRESHOLD = 0.80


def is_valid_vehicle_number(vehicle_number):
    return str(vehicle_number or "").strip().upper() not in INVALID_VEHICLE_NUMBERS


def verified_confidence(value, vehicle_number):
    if not is_valid_vehicle_number(vehicle_number):
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None

    if confidence <= 1:
        confidence *= 100
    return confidence if confidence >= OCR_CONFIDENCE_THRESHOLD * 100 else None


def run_analysis(image_path, filename):
    helmet_detections = detect_helmet(image_path)
    no_helmet, helmet_confidence = has_no_helmet(helmet_detections)

    plate_result = None
    vehicle_number = "NOT DETECTED"
    plate_confidence = 0.0
    ocr_confidence = 0.0
    plate_valid = False

    if no_helmet:
        plate_result = detect_plate(image_path)
        if plate_result is not None:
            ocr_result = read_plate(plate_result["crop"])
            plate_confidence = plate_result.get("confidence", 0.0)
            vehicle_number = ocr_result.get("plate_number", "NOT DETECTED")
            ocr_confidence = float(ocr_result.get("confidence", 0.0))
            plate_valid = bool(ocr_result.get("valid", False))
            if not (
                is_valid_vehicle_number(vehicle_number)
                and plate_valid
                and ocr_confidence >= OCR_CONFIDENCE_THRESHOLD
            ):
                vehicle_number = "NOT DETECTED"
                ocr_confidence = 0.0
                plate_valid = False

    annotated_rgb, annotated_path = annotate_image(
        image_path,
        helmet_detections,
        plate_result,
        vehicle_number,
    )

    return {
        "filename": filename,
        "source_path": image_path,
        "annotated_rgb": annotated_rgb,
        "annotated_path": annotated_path,
        "helmet_detections": helmet_detections,
        "no_helmet": no_helmet,
        "helmet_confidence": helmet_confidence,
        "plate_result": plate_result,
        "plate_confidence": plate_confidence,
        "vehicle_number": vehicle_number,
        "ocr_confidence": ocr_confidence,
        "plate_valid": plate_valid,
        "saved": False,
        "sms_status": (
            "Not applicable: no violation"
            if not no_helmet
            else "Ready after verified save"
            if plate_valid and ocr_confidence >= OCR_CONFIDENCE_THRESHOLD
            else "Not sent: plate not verified"
        ),
    }


def goto(page):
    st.session_state.nav = page
    st.rerun()


def render_sidebar():
    components.html(
        """
        <script>
        const doc = window.parent.document;
        const btn = doc.querySelector('[data-testid="stExpandSidebarButton"]');
        if (btn) { btn.click(); }
        </script>
        """,
        height=0,
    )
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-wrap">
                <div class="brand-mark">🪖</div>
                <div>
                    <div class="brand-name">Helmet Watch Pro</div>
                    <div class="brand-sub">AI for Safer Roads</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        items = [
            ("dashboard", "🏠  Dashboard"),
            ("analyze", "🖼️  Analyze Image"),
            ("registry", "📋  Violation Registry"),
        ]
        for key, label in items:
            is_active = st.session_state.nav == key
            if st.button(
                label,
                key=f"nav_{key}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.nav = key
                st.rerun()

        st.markdown(
            """
            <div class="sidebar-foot">
                <b>Safer Riders</b><br/>Stronger Communities
                <div style="margin-top:10px;opacity:0.8;">© 2026 Helmet Watch Pro<br/>All rights reserved</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_topbar(title, subtitle):
    left, right = st.columns([3.4, 1.6], vertical_alignment="center")
    with left:
        st.markdown(f"<div class='top-title'>{title}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='top-sub'>{subtitle}</div>", unsafe_allow_html=True)
    with right:
        c1, c2, c3 = st.columns([0.4, 1.2, 0.8], vertical_alignment="center")
        with c1:
            st.markdown("<div style='font-size:1.25rem;text-align:center;'>🔔</div>", unsafe_allow_html=True)
        with c2:
            st.markdown(
                """
                <div class="user-chip">
                    <div class="user-avatar">👤</div>
                    <div style="font-weight:700;font-size:0.9rem;">Admin</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown('<div class="header-logout">', unsafe_allow_html=True)
            if st.button("Logout", key="logout_btn", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.analysis = None
                st.session_state.processed_id = None
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


def process_upload(uploaded_file):
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.processed_id == file_id and st.session_state.analysis:
        return

    if uploaded_file.size > 10 * 1024 * 1024:
        st.error("Image is larger than 10MB. Please upload a smaller file.")
        return

    temp_dir = tempfile.mkdtemp()
    image_path = os.path.join(temp_dir, uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Analyzing traffic image..."):
        st.session_state.analysis = run_analysis(image_path, uploaded_file.name)
        st.session_state.processed_id = file_id
        st.session_state.flash = None


def render_analyze_panel():
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">Analyze Traffic Image</div>
            <div class="panel-sub" style="margin-bottom:0;">Upload a traffic image to detect helmet violations and extract vehicle number plate.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    analysis = st.session_state.analysis
    left, mid, right = st.columns([1.05, 1.25, 1.05], gap="medium")

    with left:
        st.markdown('<div class="panel-title" style="font-size:0.95rem;margin-bottom:0.6rem;">Upload Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Browse Image",
            type=["jpg", "jpeg", "png"],
            key=f"traffic_upload_{st.session_state.uploader_key}",
        )
        if uploaded_file is not None:
            process_upload(uploaded_file)
            analysis = st.session_state.analysis
        elif analysis is None:
            st.caption("JPG, PNG, JPEG · Max 10MB")

    with mid:
        st.markdown('<div class="panel-title" style="font-size:0.95rem;margin-bottom:0.6rem;">Uploaded Image</div>', unsafe_allow_html=True)
        if analysis and analysis.get("annotated_rgb") is not None:
            st.image(analysis["annotated_rgb"], use_container_width=True)
        elif analysis:
            st.image(analysis["source_path"], use_container_width=True)
        else:
            st.markdown(
                """
                <div class="dropzone" style="min-height:240px;">
                    <div>Preview appears here after upload.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown('<div class="panel-title" style="font-size:0.95rem;margin-bottom:0.6rem;">Detection Result</div>', unsafe_allow_html=True)
        if analysis is None:
            st.markdown(
                """
                <div class="dropzone" style="min-height:240px;">
                    <div>Results will appear after the image is processed.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            no_helmet = analysis["no_helmet"]
            vehicle = analysis["vehicle_number"]
            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-ok">✅ Image processed successfully!</div>
                    <div class="result-row">
                        <span class="result-label">🪖 Helmet Status</span>
                        <span class="result-val {'status-bad' if no_helmet else 'status-good'}">
                            {'NO HELMET' if no_helmet else 'HELMET'}
                        </span>
                    </div>
                    <div class="result-row">
                        <span class="result-label">🚘 License Plate Detection</span>
                        <span class="result-val">{'Detected' if analysis.get('plate_result') else 'Not detected'}</span>
                    </div>
                    <div class="result-row">
                        <span class="result-label">🔢 OCR Result</span>
                        <span class="result-val">{vehicle if analysis.get('plate_valid') else 'Invalid / not detected'}</span>
                    </div>
                    <div class="result-row">
                        <span class="result-label">⚠️ Violation</span>
                        <span class="result-val {'status-bad' if no_helmet else 'status-good'}">
                            {'Yes (No Helmet)' if no_helmet else 'No'}
                        </span>
                    </div>
                    <div class="result-row">
                        <span class="result-label">📱 SMS Notification</span>
                        <span class="result-val">{analysis.get('sms_status', 'Not sent')}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            b1, b2 = st.columns(2)
            with b1:
                save_disabled = (
                    (not no_helmet)
                    or analysis.get("saved", False)
                    or not analysis.get("plate_valid", False)
                    or not is_valid_vehicle_number(vehicle)
                    or verified_confidence(analysis.get("ocr_confidence"), vehicle) is None
                )
                if st.button("💾 Save to Registry", use_container_width=True, disabled=save_disabled, type="primary"):
                    saved, message = save_violation({
                        "violation": no_helmet,
                        "plate_valid": analysis.get("plate_valid", False),
                        "vehicle_number": vehicle,
                        "confidence": analysis["ocr_confidence"],
                    })
                    if saved:
                        st.session_state.analysis["saved"] = True
                        sms_allowed = (
                            no_helmet
                            and analysis.get("plate_valid", False)
                            and is_valid_vehicle_number(vehicle)
                            and verified_confidence(
                                analysis.get("ocr_confidence"), vehicle
                            ) is not None
                        )
                        if sms_allowed:
                            sms_sent, sms_status, sms_result = send_sms_alert(
                                vehicle,
                                analysis["ocr_confidence"],
                            )
                            if sms_sent:
                                st.session_state.analysis["sms_status"] = "Accepted by Twilio"
                                st.session_state.flash = (
                                    "success",
                                    f"✅ {message}. Twilio accepted the SMS request; delivery is not confirmed. Message SID: {sms_result}",
                                )
                            elif sms_status == "not_configured":
                                st.session_state.analysis["sms_status"] = "Not configured"
                                st.session_state.flash = (
                                    "info",
                                    f"{message}. SMS alert not sent: {sms_result}",
                                )
                            else:
                                st.session_state.analysis["sms_status"] = "Failed"
                                st.session_state.flash = (
                                    "error",
                                    f"⚠️ Violation saved, but SMS alert could not be sent. {sms_result}",
                                )
                        else:
                            st.session_state.analysis["sms_status"] = "Not sent: verification failed"
                            st.session_state.flash = ("info", f"{message}. SMS not sent: verification failed.")
                    else:
                        st.session_state.analysis["sms_status"] = "Not sent: violation not saved"
                        st.session_state.flash = (
                            "info",
                            f"{message} SMS not sent.",
                        )
                    st.rerun()
            with b2:
                if st.button("🧹 Clear", use_container_width=True):
                    st.session_state.analysis = None
                    st.session_state.processed_id = None
                    st.session_state.flash = None
                    st.session_state.uploader_key += 1
                    st.rerun()

            if analysis.get("saved"):
                st.caption("This violation is already saved to the registry.")

    if st.session_state.flash:
        kind, text = st.session_state.flash
        if kind == "success":
            st.success(text)
        elif kind == "error":
            st.error(text)
        else:
            st.info(text)


def registry_table(rows, limit=None, show_search=False):
    if not rows:
        st.info("No violations recorded in the registry yet.")
        return

    ordered = sorted(rows, key=lambda r: (r.get("Date", ""), r.get("Time", "")), reverse=True)
    if limit:
        ordered = ordered[:limit]

    table = []
    for row in ordered:
        table.append({
            "Date": row.get("Date", ""),
            "Time": row.get("Time", ""),
            "Vehicle Number": row.get("Vehicle_Number", ""),
            "Violation": row.get("Violation", ""),
        })
    df = pd.DataFrame(table)

    if show_search:
        query = st.text_input("Search vehicle number", placeholder="Search by vehicle number...")
        if query:
            df = df[df["Vehicle Number"].str.contains(query.strip().upper(), na=False)]

    st.dataframe(df, use_container_width=True, hide_index=True)
    return df


def render_recent_violations():
    rows = get_registry()
    head_l, head_r = st.columns([4, 1], vertical_alignment="center")
    with head_l:
        st.markdown(
            """
            <div class="panel-title">Recent Violations</div>
            <div class="panel-sub" style="margin-bottom:0;">Latest detected helmet violations from the registry.</div>
            """,
            unsafe_allow_html=True,
        )
    with head_r:
        if st.button("View All →", use_container_width=True):
            goto("registry")
    registry_table(rows, limit=5)


def login_page():
    st.markdown(LOGIN_CSS, unsafe_allow_html=True)

    _, center_col, _ = st.columns([0.4, 11.2, 0.4])
    with center_col:
        brand_col, form_col = st.columns([1.2, 1], gap="large")

        with brand_col:
            st.markdown(
                """
                <div class="auth-hero-shell">
                    <div>
                        <div class="auth-pill-badge">
                            <span>● LIVE HUD</span> &nbsp;|&nbsp; <span>HIGHWAY SURVEILLANCE v2.4</span>
                        </div>
                        <div class="hero-main-title">
                            HELMET <span class="text-cyan">DETECTION</span>
                        </div>
                        <div class="hero-sub-title">
                            with Computer Vision: <span class="text-orange">AI Traffic Defense</span>
                        </div>
                        <div class="accent-swoop"></div>
                        <p style="color: #94a3b8; font-size: 0.98rem; line-height: 1.65; max-width: 540px;">
                            Detects no-helmet violations, validates Indian registration plates, records verified events,
                            and sends SMS alerts only after a violation is saved.
                        </p>
                        <div style="color:#cbd5e1;font-size:0.76rem;line-height:1.6;margin:0.8rem 0 1rem 0;">
                            Traffic Image → YOLOv8 Helmet Detection → No-Helmet Violation → License Plate Detection →
                            OpenCV Crop → PaddleOCR → Indian Plate Validation → CSV Violation Registry →
                            Streamlit Dashboard → SMS Alert
                        </div>
                        <div class="feature-grid-stack">
                            <div class="feature-grid-item">
                                <div class="feature-grid-icon">🪖</div>
                                <div>
                                    <div style="font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: #f1f5f9;">
                                        YOLOv8 Dual Rider Classification
                                    </div>
                                    <div style="color: #64748b; font-size: 0.78rem;">
                                        Sub-second helmet compliance verification at traffic junctions
                                    </div>
                                </div>
                            </div>
                            <div class="feature-grid-item">
                                <div class="feature-grid-icon">🔍</div>
                                <div>
                                    <div style="font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: #f1f5f9;">
                                        Neural License Plate Localization & OCR
                                    </div>
                                    <div style="color: #64748b; font-size: 0.78rem;">
                                        Multi-variant CLAHE preprocessing & Indian plate validation
                                    </div>
                                </div>
                            </div>
                            <div class="feature-grid-item">
                                <div class="feature-grid-icon">📲</div>
                                <div>
                                    <div style="font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: #f1f5f9;">
                                        Instant Citation Dispatch & Registry
                                    </div>
                                    <div style="color: #64748b; font-size: 0.78rem;">
                                        Twilio SMS citation gateway & tamper-proof audit trail
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; color: #64748b; font-size: 0.78rem; display: flex; align-items: center; gap: 0.5rem; border-top: 1px solid rgba(148, 163, 184, 0.12); padding-top: 1rem;">
                        <span style="color: #00f2fe;">●</span> Station Alpha-07 Active &nbsp;|&nbsp; 🔒 256-bit TLS Encrypted
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with form_col:
            with st.form("operator_login_form", clear_on_submit=False):
                st.markdown(
                    """
                    <div style="margin-bottom: 1.2rem;">
                        <div style="font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 800; color: #f8fafc;">
                            Command <span class="text-cyan">Console</span>
                        </div>
                        <div style="color: #94a3b8; font-size: 0.88rem; margin-top: 0.2rem;">
                            Surveillance node authentication terminal. Enter authorized operator credentials.
                        </div>
                    </div>
                    <div class="demo-hud-callout">
                        <span class="demo-hud-tag">DEMO ACCESS</span>
                        <span style="font-family: 'JetBrains Mono', monospace; color: #f8fafc; font-size: 0.82rem;">
                            User: <b style="color: #00f2fe;">admin</b> &nbsp;•&nbsp; Pass: <b style="color: #ff6a00;">helmet123</b>
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                username = st.text_input(
                    "Operator Identifier",
                    value="admin",
                    placeholder="Enter operator username (e.g. admin)",
                )
                password = st.text_input(
                    "Security Passcode",
                    value="helmet123",
                    type="password",
                    placeholder="Enter security passcode",
                )

                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                submit = st.form_submit_button("Authenticate Terminal ➔", use_container_width=True)

                if submit:
                    if username.strip() == "admin" and password == "helmet123":
                        st.session_state.authenticated = True
                        st.session_state.nav = "dashboard"
                        st.rerun()
                    else:
                        st.error("❌ Access Denied: Invalid Operator Identifier or Passcode.")

                st.markdown(
                    """
                    <div style="margin-top: 1.2rem; color: #64748b; font-size: 0.76rem; text-align: center; border-top: 1px solid rgba(148,163,184,0.1); padding-top: 0.8rem;">
                        🔒 Authorized Law Enforcement Interface. All surveillance scans & citation dispatches are audited.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def page_dashboard():
    render_topbar("Welcome, Admin 👋", "Monitor. Detect. Enforce. Build Safer Roads.")
    st.write("")
    render_analyze_panel()
    st.write("")
    render_recent_violations()


def page_analyze():
    render_topbar("Analyze Image", "Upload a traffic photo, inspect detections, then save a citation.")
    st.write("")
    render_analyze_panel()


def page_registry():
    render_topbar("Violation Registry", "Search, inspect and export recorded helmet violations.")
    st.write("")
    rows = get_registry()
    df = registry_table(rows, show_search=True)
    if df is not None and not df.empty:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export Registry (.CSV)",
            data=csv_bytes,
            file_name=f"violation_registry_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
        st.caption(f"Showing {len(df)} record(s).")


def app_dashboard():
    render_sidebar()
    page = st.session_state.nav
    if page == "analyze":
        page_analyze()
    elif page == "registry":
        page_registry()
    else:
        page_dashboard()


init_session()
if st.session_state.authenticated:
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    app_dashboard()
else:
    login_page()
