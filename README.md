# 🏠 House Price Prediction

An end-to-end machine learning project that predicts house sale prices using the Ames Housing dataset.

The project covers the complete machine learning workflow:

**Data Analysis → Preprocessing → Model Training → Cross-Validation → Model Comparison → Model Saving → Streamlit Deployment**

---

## 🎯 Project Objective

The goal of this project is to build a machine learning model capable of estimating house sale prices from features such as:

- Overall house quality
- Living area
- Year built
- Basement area
- Garage capacity
- Number of bathrooms
- Neighborhood
- Kitchen quality
- And other property characteristics

The target variable is `SalePrice`.

---

## 📊 Dataset

The project uses the Ames Housing dataset from the Kaggle House Prices competition.

The training dataset contains:

- 1,460 houses
- 79 explanatory features
- Numerical and categorical variables

The target variable is:

```text
SalePrice
```

The target was transformed using:

```python
np.log1p(SalePrice)
```

because the competition evaluates predictions using logarithmic RMSE.

---

## 🔎 Machine Learning Workflow

### 1. Exploratory Data Analysis

The dataset was analyzed to understand:

- Feature distributions
- Missing values
- Correlations with house prices
- Numerical and categorical features
- Relationships between important house characteristics and `SalePrice`

`OverallQual` showed one of the strongest relationships with `SalePrice`.

### 2. Missing Value Handling

Missing values were handled differently depending on the feature type.

Numerical features:

```text
Median imputation
```

Categorical features:

```text
Most-frequent imputation
```

Some missing values in the original dataset represent the absence of a feature, such as no garage, basement, fence, or swimming pool.

### 3. Feature Preprocessing

A Scikit-learn `ColumnTransformer` was used to process numerical and categorical features separately.

Numerical pipeline:

```text
Median Imputation
        ↓
Standard Scaling
```

Categorical pipeline:

```text
Most-Frequent Imputation
        ↓
One-Hot Encoding
```

The complete preprocessing pipeline was combined with the machine learning model.

---

## 🤖 Models Tested

Several regression approaches were compared:

| Model | 5-Fold CV Log RMSE |
|---|---:|
| Linear Regression | 0.1693 |
| Random Forest | 0.1443 |
| Tuned Random Forest | 0.1404 |
| Gradient Boosting | **0.1302** |
| Tuned Gradient Boosting | 0.1317 |

Lower RMSE is better.

The original Gradient Boosting model performed best during cross-validation.

An important observation was that hyperparameter tuning did not automatically improve every model. The tuned Gradient Boosting model performed slightly worse than the original configuration.

---

## 🏆 Final Model

The final model is:

```text
GradientBoostingRegressor
```

with:

```text
n_estimators = 300
learning_rate = 0.05
max_depth = 3
random_state = 42
```

The model achieved approximately:

```text
5-Fold CV Log RMSE: 0.1302
```

The complete Scikit-learn pipeline, including preprocessing and the trained model, is saved using Joblib.

---

## 🌐 Streamlit Application

The project includes an interactive Streamlit application where users can enter house characteristics and receive an estimated sale price.

The application:

1. Collects user input
2. Creates a complete feature row
3. Applies the saved preprocessing pipeline
4. Predicts the log-transformed price
5. Converts the prediction back to the original price scale
6. Displays the estimated sale price

The application is intended as a demonstration of the machine learning model and is **not a professional property valuation tool**.

---

## 📁 Project Structure

```text
house_price_prediction/
│
├── data/
│   └── data_description.txt
│
├── model/
│   └── house_price_model.pkl
│
├── notebooks/
│   └── 01_exploration.ipynb
│
├── src/
│   └── app.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Running the Project Locally

Clone the repository and enter the project directory:

```bash
git clone <repository-url>
cd house_price_prediction
```

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the Kaggle dataset and place the required files inside:

```text
data/
```

Then launch the application:

```bash
streamlit run src/app.py
```

---

## 🚀 Future Improvements

Possible improvements include:

- XGBoost / LightGBM comparison
- More advanced feature engineering
- Hyperparameter optimization
- Prediction confidence/uncertainty estimates
- Better handling of structural missing values
- More detailed model explainability
- Improved user input coverage
- Cloud deployment

---

## 🧠 What I Learned

This project helped me understand the complete machine learning workflow rather than only training a model:

- Exploratory Data Analysis
- Missing-value handling
- Feature preprocessing
- Numerical and categorical pipelines
- One-hot encoding
- Data leakage
- Train/test splitting
- Cross-validation
- Hyperparameter tuning
- Model comparison
- Log-transformed targets
- Model serialization with Joblib
- Building an ML application with Streamlit
- Structuring an ML project for deployment


