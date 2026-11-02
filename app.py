import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

base_dir = Path(__file__).resolve().parent
model = joblib.load(base_dir / "model.joblib")
feature_names = joblib.load(base_dir / "feature_names.joblib")
cities = list(model.named_steps["prep"].named_transformers_["cat"].categories_[0])

st.title("🏠 Welcome to House Price Prediction Application")

st.sidebar.header("House Information")

city = st.sidebar.selectbox("enter city", cities, index=cities.index("Seattle") if "Seattle" in cities else 0)

bedrooms = st.sidebar.number_input("enter bedrooms", min_value=1, max_value=9, value=3)

bathrooms = st.sidebar.number_input("enter bathrooms", min_value=0.5, max_value=8.0, value=2.0, step=0.25)

sqft_living = st.sidebar.number_input("enter living area (sqft)", min_value=300, max_value=14000, value=2000)

sqft_lot = st.sidebar.number_input("enter lot size (sqft)", min_value=500, max_value=1100000, value=7500)

floors = st.sidebar.selectbox("enter floors", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5], index=2)

sqft_basement = st.sidebar.number_input("enter basement area (sqft)", min_value=0, max_value=5000, value=0)

waterfront = st.sidebar.selectbox("enter waterfront", [0, 1])

view = st.sidebar.selectbox("enter view", [0, 1, 2, 3, 4])

condition = st.sidebar.selectbox("enter condition", [1, 2, 3, 4, 5], index=2)

yr_built = st.sidebar.number_input("enter year built", min_value=1900, max_value=2026, value=1990)

yr_renovated = st.sidebar.number_input("enter year renovated (0 if never)", min_value=0, max_value=2026, value=0)

input_values = pd.DataFrame([{
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "sqft_living": sqft_living,
    "sqft_lot": sqft_lot,
    "floors": floors,
    "waterfront": waterfront,
    "view": view,
    "condition": condition,
    "sqft_basement": sqft_basement,
    "yr_built": yr_built,
    "yr_renovated": yr_renovated,
    "city": city
}])

input_df = input_values[feature_names]

st.subheader("Details of the House")
st.dataframe(input_df, use_container_width=True)

if st.button("predict", type="primary"):
    prediction = model.predict(input_df)[0]
    st.subheader("Prediction result")
    st.success(f"Estimated house price: {prediction:,.0f}")
