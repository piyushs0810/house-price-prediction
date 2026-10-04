import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# Load trained model
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "house_price_model.pkl"
DATA_PATH = BASE_DIR / "data" / "train.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_training_data():
    return pd.read_csv(DATA_PATH)


model = load_model()
training_data = load_training_data()

#adding defalualt values for the input fields

X_reference = training_data.drop(
    columns=["SalePrice", "Id"]
)

default_values = X_reference.iloc[[0]].copy()

for column in X_reference.columns:

    if pd.api.types.is_numeric_dtype(X_reference[column]):

        default_values[column] = X_reference[column].median()

    else:

        default_values[column] = X_reference[column].mode()[0]

# Page configuration

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)
st.markdown("""
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

.small-text {
    font-size: 0.9rem;
    color: #666;
}

</style>
""", unsafe_allow_html=True)
with st.sidebar:

    st.header("About the Model")

    st.write(
        "This application uses a Gradient Boosting "
        "regression model trained on the Ames Housing dataset."
    )

    st.divider()

    st.subheader("Model Performance")

    st.metric(
        "5-Fold CV Log RMSE",
        "0.1302"
    )

    st.write(
        "Lower RMSE indicates better prediction performance."
    )

    st.divider()

    st.subheader("How it works")

    st.write(
        """
        1. User enters house characteristics.
        2. Missing features receive typical dataset values.
        3. Data is preprocessed automatically.
        4. Gradient Boosting predicts the log house price.
        5. The prediction is converted back to the original price scale.
        """
    )

    st.divider()

    st.caption(
        "Built as an end-to-end machine learning portfolio project."
    )


# User inputs
 
st.title("🏠 House Price Predictor")

st.markdown(
    "### Estimate a house's market value using machine learning."
)

st.write(
    "Enter the characteristics below and let the trained "
    "Gradient Boosting model generate an estimated sale price."
)

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
        sorted(X_reference["Neighborhood"].dropna().unique())
    )



# Prediction

if st.button("🏠 Predict House Price", use_container_width=True):

    input_data = default_values.copy()

    # Numerical inputs
    input_data["OverallQual"] = overall_qual
    input_data["GrLivArea"] = gr_liv_area
    input_data["YearBuilt"] = year_built
    input_data["TotalBsmtSF"] = total_bsmt_sf
    input_data["FullBath"] = full_bath
    input_data["TotRmsAbvGrd"] = tot_rms
    input_data["BedroomAbvGr"] = bedrooms
    input_data["GarageCars"] = garage_cars
    input_data["GarageArea"] = garage_area

    # Categorical inputs
    input_data["BsmtQual"] = (
        np.nan if bsmt_qual == "None" else bsmt_qual
    )

    input_data["KitchenQual"] = kitchen_qual

    input_data["GarageFinish"] = (
        np.nan if garage_finish == "None" else garage_finish
    )

    input_data["CentralAir"] = central_air

    input_data["Neighborhood"] = neighborhood

    # Make prediction
    prediction_log = model.predict(input_data)

    prediction = np.expm1(prediction_log[0])

    st.divider()

    st.subheader("Estimated House Price")

    st.markdown(
        f"""
        <div class="prediction-box">
            <div class="prediction-price">${prediction:,.0f}</div>
            <div class="small-text">Prediction generated by the trained Gradient Boosting model.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

st.subheader("About this project")

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

