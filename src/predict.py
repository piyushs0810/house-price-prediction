import joblib
import numpy as np
import pandas as pd
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
    """

    input_df = pd.DataFrame([input_data])
    
    engineered_data = add_features(input_df)

    prediction_log = model.predict(engineered_data)

    prediction = np.expm1(prediction_log[0])

    return float(prediction)