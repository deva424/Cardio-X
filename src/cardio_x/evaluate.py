import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from .constants import CLASS_INFO
from .preprocessing import load_dataset, transform_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Cardio-X model.")
    parser.add_argument("--test", required=True)
    parser.add_argument("--model", default="models/best_1d_cnn_model.keras")
    parser.add_argument("--scaler", default="models/scaler.joblib")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    features, y_true = load_dataset(args.test)
    model = tf.keras.models.load_model(args.model)
    features = transform_features(features, joblib.load(args.scaler))
    y_pred = np.argmax(model.predict(features, verbose=0), axis=1)
    labels = sorted(CLASS_INFO)
    print(classification_report(y_true, y_pred, labels=labels, target_names=[CLASS_INFO[i][0] for i in labels]))

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(report_dir / "confusion_matrix.png", dpi=150)


if __name__ == "__main__":
    main()
