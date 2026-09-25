from pathlib import Path

import streamlit as st

from src.cardio_x.ui import MODEL_PATH, apply_theme

st.set_page_config(page_title="Cardio-X | ECG Intelligence", page_icon="+", layout="wide")
apply_theme()

st.markdown("""<style>
.landing-hero{min-height:455px;background:linear-gradient(125deg,#081b34 0%,#123a70 58%,#176c8a 100%);border-radius:28px;position:relative;overflow:hidden;padding:58px 58px;box-sizing:border-box;color:white;box-shadow:0 24px 55px rgba(15,49,92,.22)}
.landing-hero:before{content:'';position:absolute;inset:0;background-image:linear-gradient(rgba(183,243,241,.07) 1px,transparent 1px),linear-gradient(90deg,rgba(183,243,241,.07) 1px,transparent 1px);background-size:36px 36px;mask-image:linear-gradient(90deg,transparent 0%,#000 42%,#000 100%)}
.landing-copy{position:relative;z-index:2;max-width:620px}.landing-copy h1{color:white!important;font-size:3.35rem;line-height:1.03;margin:.55rem 0 1rem}.landing-copy p{color:#c8d8ee;line-height:1.7;font-size:1.02rem;max-width:535px}.landing-art{position:absolute;right:1%;bottom:12%;width:48%;opacity:.95}.pulse-path{fill:none;stroke:#7cf1d3;stroke-width:5;stroke-linecap:round;stroke-linejoin:round;filter:drop-shadow(0 0 9px rgba(124,241,211,.7));stroke-dasharray:850;stroke-dashoffset:850;animation:trace 4s ease-in-out infinite}.pulse-shadow{fill:none;stroke:#8ec3ff;stroke-width:18;opacity:.12;filter:blur(7px)}@keyframes trace{0%{stroke-dashoffset:850;opacity:.3}35%,75%{stroke-dashoffset:0;opacity:1}100%{stroke-dashoffset:-850;opacity:.25}}.orb{position:absolute;width:260px;height:260px;border:1px solid rgba(159,234,255,.22);border-radius:50%;right:9%;top:12%;box-shadow:0 0 0 30px rgba(159,234,255,.035),0 0 0 62px rgba(159,234,255,.025)}
.upload-panel{position:relative;z-index:4;max-width:570px;margin-top:2rem;background:rgba(255,255,255,.1);border:1px solid rgba(181,235,255,.28);backdrop-filter:blur(12px);border-radius:18px;padding:15px 18px 5px}.upload-panel p{font-size:.82rem;color:#bfdaea;margin:0 0 .55rem}.upload-panel [data-testid="stFileUploader"]{background:rgba(255,255,255,.9);border:0;border-radius:12px}.upload-panel [data-testid="stFileUploader"] *{color:#10243e!important}
</style>""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error(f"The model file is missing. Expected: {MODEL_PATH}")
    st.stop()

st.markdown("""<section class="landing-hero"><div class="orb"></div><svg class="landing-art" viewBox="0 0 850 240" aria-hidden="true"><path class="pulse-shadow" d="M5 130 H150 L190 128 L220 70 L252 192 L292 35 L337 151 L370 130 H520 L552 112 L580 147 L610 130 H845"/><path class="pulse-path" d="M5 130 H150 L190 128 L220 70 L252 192 L292 35 L337 151 L370 130 H520 L552 112 L580 147 L610 130 H845"/></svg><div class="landing-copy"><div class="eyebrow" style="color:#8ff0d9">CARDIO-X / ECG INTELLIGENCE</div><h1>Every heartbeat<br>tells a story.</h1><p>Enter the analysis studio to transform raw ECG samples into clear, visual heartbeat insights.</p><div class="pill">AI-ASSISTED REVIEW &nbsp; / &nbsp; 5 RHYTHM CLASSES</div><div class="upload-panel"><p>Upload your ECG CSV to begin. Your results will open in a dedicated analysis page.</p>""", unsafe_allow_html=True)

upload = st.file_uploader("Upload ECG CSV", type="csv", label_visibility="collapsed")
st.markdown("</div></div></section>", unsafe_allow_html=True)

if upload is not None:
    st.session_state["ecg_upload"] = {"name": upload.name, "data": upload.getvalue()}
    st.switch_page("pages/1_Analyze_ECG.py")

st.markdown('<div class="section-label">Inside the studio</div><h2>A simple path from signal to insight</h2>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
for column, number, title, text in [
    (col1, "01", "Upload", "Bring a CSV with 187 ECG signal values per heartbeat."),
    (col2, "02", "Analyze", "The model scores five rhythm patterns for every row."),
    (col3, "03", "Explore", "Review confidence, distribution, and individual waveforms."),
]:
    with column:
        st.markdown(f'<div class="feature-card"><div class="step-number">{number}</div><h3>{title}</h3><p>{text}</p></div>', unsafe_allow_html=True)
