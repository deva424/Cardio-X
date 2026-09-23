from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

from src.cardio_x.constants import CLASS_INFO
from src.cardio_x.preprocessing import extract_uploaded_signals, prepare_uploaded_features
from src.cardio_x.ui import apply_theme, load_artifacts, metric_card

st.set_page_config(page_title="Analyze ECG | Cardio-X", page_icon="+", layout="wide")
apply_theme()
st.markdown('<div class="eyebrow">Cardio-X / Analysis workspace</div><h1>Analyze ECG signals</h1><p style="color:#68788d">Upload a dataset to generate beat-level classifications and inspect the waveform behind each prediction.</p>', unsafe_allow_html=True)

model, scaler = load_artifacts()
if model is None:
    st.error("The model file is missing. Expected: models/best_1d_cnn_model.keras")
    st.stop()

stored_upload = st.session_state.get("ecg_upload")
if stored_upload:
    header_col, reset_col, _ = st.columns((3, 1, 2))
    with header_col:
        st.success(f"Loaded dataset: {stored_upload['name']}")
    with reset_col:
        if st.button("Use another CSV", use_container_width=True):
            del st.session_state["ecg_upload"]
            st.switch_page("app.py")
    upload = BytesIO(stored_upload["data"])
else:
    st.markdown('<div class="section-label">Data intake</div><h2>Upload a heartbeat dataset</h2><p style="color:#68788d;font-size:.88rem">The CSV needs at least 187 numeric ECG signal columns. A <code>class_label</code> column is optional.</p>', unsafe_allow_html=True)
    upload = st.file_uploader("Drop your ECG CSV here", type="csv", label_visibility="collapsed")
if scaler is None:
    st.info("This supplied model has no saved training scaler. Uploaded rows are standardized as a batch; use results as exploratory analysis.")

if upload is None:
    st.markdown('<div class="section-label">Ready when you are</div>', unsafe_allow_html=True)
    left, middle, right = st.columns(3)
    with left: metric_card("Signal shape", "187 points", "Per heartbeat row")
    with middle: metric_card("Model output", "5 classes", "Rhythm categories")
    with right: metric_card("Review mode", "Interactive", "Select individual beats")
    st.stop()

try:
    frame = pd.read_csv(upload)
    signals = extract_uploaded_signals(frame)
    with st.spinner("Reading waveform patterns..."):
        probabilities = model.predict(prepare_uploaded_features(signals, scaler), verbose=0)
    labels = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1) * 100
    names = [CLASS_INFO[int(label)][0] for label in labels]
    summary = pd.Series(names).value_counts().rename_axis("Prediction").reset_index(name="Beats")
    dominant = summary.iloc[0]

    st.markdown('<div class="section-label">Analysis overview</div><h2>Signal classification summary</h2>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns((1, 1, 1, 1.35))
    with col1: metric_card("Beats analyzed", str(len(frame)), "Rows in this upload")
    with col2: metric_card("Mean confidence", f"{confidence.mean():.1f}%", "Across all predictions")
    with col3: metric_card("Anomaly flagged", str(int(np.count_nonzero(labels))), "Non-normal patterns")
    with col4: metric_card("Most common result", str(dominant["Prediction"]), f'{int(dominant["Beats"])} beats classified', True)

    table_col, distribution_col = st.columns((1.65, 1))
    with table_col:
        st.markdown('<div class="section-label">Prediction ledger</div><h2>Beat-by-beat results</h2>', unsafe_allow_html=True)
        results = pd.DataFrame({"Prediction": names, "Confidence (%)": np.round(confidence, 1)})
        if "class_label" in frame.columns:
            results.insert(0, "Actual label", frame["class_label"].values)
        st.dataframe(results, hide_index=True, use_container_width=True, height=315)
    with distribution_col:
        st.markdown('<div class="section-label">Distribution</div><h2>Pattern mix</h2>', unsafe_allow_html=True)
        st.bar_chart(summary.set_index("Prediction"), color="#28c7a6", height=315)

    st.markdown('<div class="section-label">Waveform explorer</div><h2>Inspect an individual heartbeat</h2>', unsafe_allow_html=True)
    chosen = st.slider("Select heartbeat row", 0, len(frame) - 1, 0)
    chart_col, insight_col = st.columns((1.75, 1))
    with chart_col:
        st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
        st.line_chart(pd.DataFrame({"ECG amplitude": signals[chosen]}), color="#4f82f5", height=300)
        st.markdown('</div>', unsafe_allow_html=True)
    with insight_col:
        name, description, action = CLASS_INFO[int(labels[chosen])]
        metric_card("Selected beat", name, f"Confidence: {confidence[chosen]:.1f}%", True)
        st.markdown(f"<div style='padding:14px 4px 0;color:#526278;font-size:.9rem;line-height:1.6'><b style='color:#10243e'>Pattern note</b><br>{description}<br><br><b style='color:#10243e'>Suggested next step</b><br>{action}</div>", unsafe_allow_html=True)
except Exception as error:
    st.error(f"Could not analyze this CSV: {error}")
