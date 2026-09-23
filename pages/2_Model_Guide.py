import streamlit as st

from src.cardio_x.constants import CLASS_INFO
from src.cardio_x.ui import apply_theme

st.set_page_config(page_title="Model Guide | Cardio-X", page_icon="+", layout="wide")
apply_theme()
st.markdown('<div class="eyebrow">Cardio-X / Interpretation guide</div><h1>Model guide</h1><p style="color:#68788d">A practical reference for the model outputs shown in the ECG analysis workspace.</p>', unsafe_allow_html=True)

st.markdown('<div class="section-label">Classification system</div><h2>Five heartbeat categories</h2>', unsafe_allow_html=True)
for index, (name, description, action) in CLASS_INFO.items():
    with st.expander(f"Class {index} - {name}", expanded=index == 0):
        st.markdown(f"**What the model detected**  \n{description}\n\n**Suggested next step**  \n{action}")

st.markdown('<div class="section-label">Workflow</div><h2>How an analysis is produced</h2>', unsafe_allow_html=True)
columns = st.columns(3)
steps = [("01", "Upload", "The app reads the first 187 numeric signal values from each CSV row."), ("02", "Prepare", "Signals are standardized and reshaped to the model input format."), ("03", "Classify", "The CNN returns probabilities for five rhythm categories; the highest is displayed.")]
for column, (number, title, text) in zip(columns, steps):
    with column:
        st.markdown(f'<div class="feature-card"><div class="step-number">{number}</div><h3>{title}</h3><p>{text}</p></div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">Important limitation</div><h2>Interpret results responsibly</h2>', unsafe_allow_html=True)
st.warning("Confidence reflects the model's relative certainty, not clinical certainty. Cardio-X is an educational and research tool, not a diagnostic device. Clinical decisions must be made by qualified healthcare professionals.")
