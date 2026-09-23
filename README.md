# Cardio-X

### ECG arrhythmia exploration with a 1D convolutional neural network

Cardio-X is a Streamlit application for exploring ECG heartbeat data. Upload a CSV of ECG samples to receive model classifications, confidence scores, class distribution, and an interactive waveform view.

> **Important:** Cardio-X is an educational and research project. It is not a medical device and must not be used for diagnosis, treatment, or clinical decision-making.

## Highlights

| | Capability | What it provides |
| --- | --- | --- |
| 01 | Interactive dashboard | A dedicated landing screen and ECG analysis workspace. |
| 02 | CNN inference | Five-class heartbeat classification from 187-point ECG signals. |
| 03 | Visual review | Prediction table, confidence scores, class distribution, and per-beat waveform chart. |
| 04 | Training pipeline | Reproducible training, evaluation, and confusion-matrix generation. |

## App experience

1. Open the Cardio-X landing page.
2. Upload a CSV containing ECG signal values.
3. The app moves to the analysis workspace automatically.
4. Review aggregate results, individual predictions, and waveform detail.

The app accepts at least **187 numeric signal columns** per row. An optional `class_label` column is displayed for comparison when present.

## Quick start

### 1. Create and activate a virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

### 3. Start Cardio-X

```powershell
py -m streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Dataset format

For model training and evaluation, place these headerless MIT-BIH files in `data/raw/`:

```text
data/raw/
├── mitbih_train.csv
└── mitbih_test.csv
```

Each row must contain:

```text
187 ECG signal values, followed by 1 class label
```

The dashboard supports CSVs with column headers as long as they contain at least 187 numeric signal columns.

## Train the model

```powershell
py -m src.cardio_x.train `
  --train data/raw/mitbih_train.csv `
  --test data/raw/mitbih_test.csv
```

Training produces:

```text
models/
├── best_1d_cnn_model.keras
└── scaler.joblib
```

`scaler.joblib` stores the training preprocessing configuration and is recommended for consistent inference.

## Evaluate a trained model

```powershell
py -m src.cardio_x.evaluate --test data/raw/mitbih_test.csv
```

This prints a classification report and saves a confusion matrix to:

```text
reports/confusion_matrix.png
```

## Project structure

```text
ARR_detection/
├── app.py                         # Creative upload landing page
├── pages/
│   ├── 1_Analyze_ECG.py           # Results dashboard and waveform explorer
│   └── 2_Model_Guide.py           # Model classes and interpretation guide
├── src/cardio_x/
│   ├── constants.py               # Class names and guidance text
│   ├── model.py                   # 1D CNN architecture
│   ├── preprocessing.py           # Dataset and inference preprocessing
│   ├── train.py                   # Training entry point
│   ├── evaluate.py                # Evaluation entry point
│   └── ui.py                      # Shared UI theme and artifact loading
├── models/
│   └── best_1d_cnn_model.keras    # Supplied trained model
├── notebooks/
│   └── app_py.ipynb               # Original Colab reference notebook
├── requirements.txt
└── README.md
```

## Heartbeat classes

| Class | Label |
| ---: | --- |
| 0 | Normal Beat |
| 1 | Supraventricular Ectopic Beat |
| 2 | Ventricular Ectopic Beat |
| 3 | Fusion Beat |
| 4 | Unclassifiable Beat |

## Model artifact note

The supplied `models/best_1d_cnn_model.keras` is ready to use. Its matching training scaler was not supplied, so the dashboard standardizes uploaded rows as a batch. Retrain the model or add `models/scaler.joblib` to use the original training scaler during inference.

## Troubleshooting

| Issue | Resolution |
| --- | --- |
| `ModuleNotFoundError: tensorflow` | Run `py -m pip install -r requirements.txt`. |
| Model file not found | Confirm `models/best_1d_cnn_model.keras` exists. |
| CSV cannot be analyzed | Confirm the file contains at least 187 numeric ECG columns. |
| The UI does not update | Stop Streamlit with `Ctrl+C`, then start it again. |

---

Built for ECG data exploration with Streamlit, TensorFlow, and scikit-learn.
