import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path

from predict import load_model, predict_price


st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

@st.cache_resource
def get_model():
    """Load and cache the trained ML model."""
    return load_model()


BASE_DIR = Path(__file__).resolve().parent.parent
REFERENCE_PATH = BASE_DIR / "src" / "reference_values.json"

with open(REFERENCE_PATH, "r", encoding="utf-8") as file:
    reference_data = json.load(file)

default_values = reference_data["default_values"]
dropdown_options = reference_data["dropdown_options"]

model = get_model()

st.markdown(
    """
    <style>

    .main {
        padding-top: 2rem;
    }

    h1 {
        font-size: 3rem;
    }

    .prediction-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
    }

    .prediction-price {
        font-size: 2.5rem;
        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:

    st.header("About the Model")

    st.write(
        """
        This application uses a Gradient Boosting regression
        model trained on the Ames Housing dataset.
        """
    )

    st.divider()

    st.subheader("Model Performance")

    st.metric(
        "5-Fold CV Log RMSE",
        "0.1278"
    )

    st.write(
        "Lower RMSE indicates better prediction performance."
    )

    st.divider()

    st.subheader("How it works")

    st.write(
        """
        1. Enter house characteristics.
        2. Missing features receive typical dataset values.
        3. The saved ML pipeline preprocesses the data.
        4. Gradient Boosting predicts the log house price.
        5. The prediction is converted back to the original price.
        """
    )

    st.divider()

    st.caption(
        "Built as an end-to-end machine learning portfolio project."
    )


st.title("🏠 House Price Predictor")

st.markdown(
    "### Estimate a house's market value using machine learning."
)

st.write(
    """
    Enter the characteristics of a house below and let the
    trained Gradient Boosting model generate an estimated
    sale price.
    """
)

st.header("🏠 House Information")
st.subheader("Basic Information")

col1, col2 = st.columns(2)

with col1:

    overall_qual = st.slider(
        "Overall Quality",
        min_value=1,
        max_value=10,
        value=5,
        help="Overall material and finish quality of the house."
    )

    year_built = st.number_input(
        "Year Built",
        min_value=1872,
        max_value=2026,
        value=2000,
        step=1
    )

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=0,
        max_value=10,
        value=3,
        step=1
    )


with col2:

    gr_liv_area = st.number_input(
        "Above Ground Living Area (sq ft)",
        min_value=300,
        max_value=5000,
        value=1500,
        step=50
    )

    full_bath = st.number_input(
        "Full Bathrooms",
        min_value=0,
        max_value=5,
        value=2,
        step=1
    )

    tot_rms = st.number_input(
        "Total Rooms Above Ground",
        min_value=1,
        max_value=15,
        value=7,
        step=1
    )

st.subheader("Basement")

col1, col2 = st.columns(2)

with col1:

    total_bsmt_sf = st.number_input(
        "Basement Area (sq ft)",
        min_value=0,
        max_value=5000,
        value=1000,
        step=50
    )


with col2:

    bsmt_qual = st.selectbox(
        "Basement Quality",
        ["None", "Po", "Fa", "TA", "Gd", "Ex"],
        index=4
    )

st.subheader("Garage")

col1, col2 = st.columns(2)

with col1:

    garage_cars = st.number_input(
        "Garage Capacity (cars)",
        min_value=0,
        max_value=5,
        value=2,
        step=1
    )


with col2:

    garage_area = st.number_input(
        "Garage Area (sq ft)",
        min_value=0,
        max_value=1500,
        value=400,
        step=25
    )


st.subheader("House Quality")

col1, col2 = st.columns(2)

with col1:

    kitchen_qual = st.selectbox(
        "Kitchen Quality",
        ["Po", "Fa", "TA", "Gd", "Ex"],
        index=2
    )


with col2:

    garage_finish = st.selectbox(
        "Garage Finish",
        ["None", "Unf", "RFn", "Fin"],
        index=2
    )

st.subheader("Other Features")

col1, col2 = st.columns(2)

with col1:

    central_air = st.selectbox(
        "Central Air Conditioning",
        ["Y", "N"]
    )


with col2:

  neighborhood = st.selectbox(
    "Neighborhood",
    dropdown_options["Neighborhood"]
)


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🏠 Predict House Price",
    use_container_width=True
):

    # Start with a complete row containing default values.
    input_data = default_values.copy()

    # --------------------------------------------------------
    # Numerical features
    # --------------------------------------------------------

    input_data["OverallQual"] = overall_qual
    input_data["GrLivArea"] = gr_liv_area
    input_data["YearBuilt"] = year_built
    input_data["TotalBsmtSF"] = total_bsmt_sf
    input_data["FullBath"] = full_bath
    input_data["TotRmsAbvGrd"] = tot_rms
    input_data["BedroomAbvGr"] = bedrooms
    input_data["GarageCars"] = garage_cars
    input_data["GarageArea"] = garage_area

    input_data["BsmtQual"] = (
        np.nan
        if bsmt_qual == "None"
        else bsmt_qual
    )

    input_data["KitchenQual"] = kitchen_qual

    input_data["GarageFinish"] = (
        np.nan
        if garage_finish == "None"
        else garage_finish
    )

    input_data["CentralAir"] = central_air

    input_data["Neighborhood"] = neighborhood

    
    prediction = predict_price(
        model,
        input_data
    )

    st.divider()

    st.subheader("Estimated House Price")

    st.markdown(
    f"""
    <div style="
        padding: 28px;
        border-radius: 16px;
        text-align: center;
        border: 1px solid rgba(128,128,128,0.35);
        margin-top: 20px;
    ">
        <div style="font-size: 1.1rem;">
            Estimated Sale Price
        </div>
        <div style="
            font-size: 2.6rem;
            font-weight: 700;
            margin-top: 10px;
        ">
            ${prediction:,.0f}
        </div>
        <div style="
            font-size: 0.85rem;
            margin-top: 12px;
            opacity: 0.75;
        ">
            Generated by the trained Gradient Boosting model
        </div>
    </div>
    """,
    unsafe_allow_html=True
    )

    st.caption(
        "This is a machine-learning estimate, "
        "not a professional property valuation."
    )

st.divider()

st.subheader("About this Project")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Training Houses",
        "1,460"
    )

with col2:
    st.metric(
        "Input Features",
        "79"
    )

with col3:
    st.metric(
        "Best CV RMSE",
        "0.1302"
    )