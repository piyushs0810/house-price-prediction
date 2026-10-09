import joblib
import numpy as np
from pathlib import Path

from feature import add_features


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "house_price_model.pkl"


def load_model():
    """Load the saved house price prediction pipeline."""
    return joblib.load(MODEL_PATH)


def predict_price(model, input_data):
    """
    Predict a house price from its original input features.

    The function adds engineered features before prediction.
    The trained model predicts log-transformed prices, which
    are converted back to the original price scale.
    """

    engineered_data = add_features(input_data)

    prediction_log = model.predict(engineered_data)

    prediction = np.expm1(prediction_log[0])

    return float(prediction)