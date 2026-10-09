# House Price Prediction

A machine learning project that predicts residential property prices using housing characteristics. Built with Python, pandas, scikit-learn, and Streamlit, this project demonstrates the end-to-end machine learning workflow, from exploratory data analysis and preprocessing to model evaluation and deployment.

## Live Demo

**Web App:** https://house-price-prediction-ixrynqyx5fdf6eu9hvrryx.streamlit.app/

**GitHub Repository:** https://github.com/piyushs0810/house-price-prediction

## Overview

The goal of this project is to estimate house prices based on features such as:

- Overall material and finish quality
- Above-ground living area
- Construction year
- Basement area
- Garage capacity and area
- Number of bathrooms and rooms
- Neighborhood and other property characteristics

The application accepts user inputs and returns an estimated house price using a trained machine learning model.

## Dataset

The project uses the Ames Housing dataset associated with the Kaggle House Prices competition.

The training dataset contains 1,460 observations and the original dataset includes 79 input features after excluding the `Id` column.

The target variable is 'SalePrice'.

## Machine Learning Workflow

### 1. Exploratory Data Analysis

- Examined dataset dimensions, data types, and missing values.
- Investigated numerical relationships with the target variable.
- Identified influential features, including overall quality, living area, and garage capacity.
- Examined the distribution of house prices.

### 2. Data Preprocessing

The preprocessing pipeline handles numerical and categorical variables separately.

**Numerical features**
- Missing-value imputation using the median.
- Standardization using `StandardScaler`.

**Categorical features**
- Missing-value imputation using the most frequent value.
- One-hot encoding using `OneHotEncoder(handle_unknown="ignore")`.

Preprocessing is fitted on training data within the machine learning pipeline to avoid leaking validation information into the training process.

### 3. Feature Engineering

Six additional features were created to represent useful combinations and age-related properties:

| Feature | Description |
|---|---|
| `TotalSF` | Combined basement, first-floor, and second-floor area |
| `TotalBathrooms` | Total bathrooms, counting half-bathrooms as 0.5 |
| `TotalPorchSF` | Combined porch and deck area |
| `HouseAge` | Year sold minus year built |
| `RemodAge` | Year sold minus year remodeled |
| `TotalBsmtFinSF` | Combined finished basement areas |

These features provide the model with additional information about property size, layout, and age.

### 4. Model Training and Selection

Several regression models were evaluated:

- Linear Regression
- Random Forest Regressor
- Tuned Random Forest Regressor
- Gradient Boosting Regressor
- Gradient Boosting with engineered features

Five-fold cross-validation was used to compare model performance.

The target was transformed using `log1p(SalePrice)` during training. Predictions are transformed back into the original price scale using `expm1()`.

### 5. Evaluation

The primary comparison metric is Root Mean Squared Error on the log-transformed target (Log RMSE).

| Model | Mean 5-Fold CV Log RMSE |
|---|---:|
| Linear Regression | 0.1693 |
| Random Forest | 0.1443 |
| Tuned Random Forest | 0.1404 |
| Gradient Boosting | 0.1302 |
| Gradient Boosting with Feature Engineering | **0.1279** |

Lower Log RMSE indicates better performance.

The feature-engineered Gradient Boosting model achieved a mean cross-validation Log RMSE of approximately **0.1279**, compared with approximately 0.1302 for the earlier Gradient Boosting model.

These results are cross-validation results, not a guarantee of prediction accuracy for every individual house.

### 6. Web Application

The Streamlit application allows users to enter selected house characteristics and receive a predicted sale price.

The application uses the same feature-engineering logic as the training workflow to ensure that the model receives the expected input features.

## Project Structure

```text
house_price_prediction/
├── data/
├── notebooks/
│   └── 01_exploration.ipynb
├── model/
│   └── house_price_model.pkl
├── src/
│   ├── app.py
│   ├── predict.py
│   └── features.py
├── requirements.txt
└── README.md
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/piyushs0810/house-price-prediction.git
cd house-price-prediction
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run src/app.py
```

The application will open in your browser.

Ensure that the trained model and any required application configuration files are available in the expected locations.

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- Streamlit
- Joblib
- Matplotlib and Seaborn for exploratory analysis

## Limitations

- The model is trained on the Ames Housing dataset and may not generalize to housing markets in other regions.
- Predictions are estimates, not professional property valuations.
- Model performance depends on the quality and representativeness of the training data.
- Cross-validation results do not guarantee the same performance on unseen real-world properties.

## Future Improvements

- Add more detailed model diagnostics.
- Improve input validation and user experience.
- Evaluate additional regression algorithms.
- Investigate prediction intervals and uncertainty estimates.
- Expand the application with clearer explanations of the factors influencing predictions.
  
## Screenshots

### Prediction Interface
![Prediction Interface](screenshots/app.png)

### Example Prediction
![Example Prediction](screenshots/prediction.png)

## Author

**Piyush Shrivastava**

GitHub: [piyushs0810](https://github.com/piyushs0810)
