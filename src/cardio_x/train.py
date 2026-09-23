import argparse
from pathlib import Path

from imblearn.over_sampling import SMOTE
import tensorflow as tf

from .model import build_model
from .preprocessing import fit_transform_training_data, load_dataset, save_scaler, transform_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Cardio-X 1D CNN.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--model-output", default="models/best_1d_cnn_model.keras")
    parser.add_argument("--scaler-output", default="models/scaler.joblib")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    x_train, y_train = load_dataset(args.train)
    x_test, y_test = load_dataset(args.test)
    x_resampled, y_resampled = SMOTE(random_state=42).fit_resample(x_train, y_train)
    x_scaled, scaler = fit_transform_training_data(x_resampled)
    x_train_cnn = x_scaled.reshape(-1, x_scaled.shape[1], 1)
    x_test_cnn = transform_features(x_test, scaler)

    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model = build_model(x_train_cnn.shape[1:], len(set(y_resampled)))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True, mode="max"),
    ]
    model.fit(x_train_cnn, y_resampled, epochs=args.epochs, batch_size=args.batch_size,
              validation_split=0.2, callbacks=callbacks)
    save_scaler(scaler, args.scaler_output)
    loss, accuracy = model.evaluate(x_test_cnn, y_test, verbose=0)
    print(f"Test loss: {loss:.4f} | Test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
