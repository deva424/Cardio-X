"""Shared presentation helpers for the Streamlit pages."""

from pathlib import Path

import joblib
import streamlit as st
import tensorflow as tf

MODEL_PATH = Path("models/best_1d_cnn_model.keras")
SCALER_PATH = Path("models/scaler.joblib")


def apply_theme() -> None:
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root{--ink:#10243e;--muted:#68788d;--line:#dce6f0;--mint:#28c7a6;--paper:#f6f9fc}.stApp{background:var(--paper);color:var(--ink);font-family:Manrope,sans-serif}#MainMenu,footer{visibility:hidden}.block-container{max-width:1240px;padding-top:2.1rem;padding-bottom:3rem}[data-testid="stHeader"]{background:rgba(246,249,252,.85)}h1,h2,h3{color:var(--ink);letter-spacing:-.04em;font-weight:800!important}.eyebrow,.section-label{font-family:'DM Mono',monospace;letter-spacing:.13em;font-size:.7rem;font-weight:500;text-transform:uppercase;color:#718198}.hero{background:radial-gradient(circle at 82% 18%,#366ae2 0,#173c78 34%,#10243e 78%);border-radius:24px;color:#fff;padding:38px 42px;position:relative;overflow:hidden;box-shadow:0 14px 32px rgba(23,60,120,.18)}.hero:after{content:'';position:absolute;width:340px;height:340px;border:1px solid rgba(255,255,255,.13);border-radius:50%;right:-105px;top:-160px;box-shadow:0 0 0 34px rgba(255,255,255,.045),0 0 0 70px rgba(255,255,255,.035)}.hero h1{color:#fff;font-size:2.55rem;line-height:1.1;margin:.45rem 0 .7rem}.hero p{color:#d4e1f7;max-width:620px;line-height:1.7;margin:0}.hero .eyebrow{color:#8ff0d9}.pill{display:inline-block;margin-top:1.35rem;padding:7px 11px;border-radius:99px;background:rgba(143,240,217,.14);border:1px solid rgba(143,240,217,.28);color:#b6fae9;font-size:.73rem;font-family:'DM Mono',monospace}.metric-card,.feature-card{background:#fff;border:1px solid var(--line);border-radius:16px;padding:19px;min-height:116px;box-sizing:border-box;box-shadow:0 5px 18px rgba(26,55,86,.035)}.feature-card{min-height:175px}.feature-card h3{margin:0 0 .5rem;font-size:1rem}.feature-card p{color:var(--muted);font-size:.88rem;line-height:1.65}.metric-name{color:var(--muted);font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em}.metric-value{color:var(--ink);font-size:1.7rem;font-weight:800;letter-spacing:-.06em;margin-top:8px}.metric-caption{color:#8190a2;font-size:.76rem;margin-top:2px}.result-card{background:linear-gradient(135deg,#10243e,#173c78);border-radius:16px;padding:19px;color:#fff;min-height:116px;box-sizing:border-box}.result-card .metric-name{color:#9db7dd}.result-card .metric-value{color:#fff;font-size:1.2rem}.result-card .metric-caption{color:#c4d6f1}.stDataFrame{border:1px solid var(--line);border-radius:14px;overflow:hidden}[data-testid="stFileUploader"]{background:#fff;border:1px dashed #b9c9dc;border-radius:16px;padding:9px 16px 2px}[data-testid="stAlert"]{border-radius:12px}.chart-shell{background:#fff;border:1px solid var(--line);border-radius:16px;padding:8px 14px 3px}.step-number{color:var(--mint);font-family:'DM Mono',monospace;font-size:.78rem;font-weight:700}
    </style>""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        return None, None
    return (tf.keras.models.load_model(MODEL_PATH, compile=False),
            joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None)


def metric_card(title: str, value: str, caption: str, featured: bool = False) -> None:
    kind = "result-card" if featured else "metric-card"
    st.markdown(f'<div class="{kind}"><div class="metric-name">{title}</div><div class="metric-value">{value}</div><div class="metric-caption">{caption}</div></div>', unsafe_allow_html=True)
