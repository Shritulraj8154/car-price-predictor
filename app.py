import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# ---------------- File Check ----------------
if not os.path.exists("model.pkl"):
    st.error("model.pkl not found. Run train_model.py first.")
    st.stop()

if not os.path.exists("cleaned car.csv"):
    st.error("cleaned car.csv not found.")
    st.stop()

# ---------------- Load ----------------
model = pickle.load(open("model.pkl", "rb"))
car = pd.read_csv("cleaned car.csv")

X = car[['name','company','year','kms_driven','fuel_type']]
y = car['Price']

# Accuracy
y_pred = model.predict(X)
accuracy = r2_score(y, y_pred)

# ---------------- Title ----------------
st.markdown("<h1 style='text-align:center; color:#1f77b4;'>🚗 Car Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Machine Learning Based Used Car Price Prediction</p>", unsafe_allow_html=True)

st.write("---")

# ---------------- Inputs ----------------
col1, col2 = st.columns(2)

with col1:
    name = st.selectbox("Car Name", sorted(car["name"].unique()))
    year = st.number_input("Year of Purchase", min_value=1990, max_value=2025)

with col2:
    company = st.selectbox("Company", sorted(car["company"].unique()))
    fuel = st.selectbox("Fuel Type", sorted(car["fuel_type"].unique()))

kms = st.number_input("Kilometers Driven", min_value=0, step=1000)

# ---------------- Prediction ----------------
if st.button("Predict Price 💰"):

    input_data = pd.DataFrame(
        [[name, company, year, kms, fuel]],
        columns=["name","company","year","kms_driven","fuel_type"]
    )

    prediction = model.predict(input_data)
    final_price = round(prediction[0], 2)

    st.success(f"Estimated Car Price: ₹ {final_price}")

st.write("---")

# ---------------- Model Performance ----------------
st.subheader("📊 Model Performance")

st.write(f"R² Score (Accuracy): {round(accuracy*100,2)} %")

fig = plt.figure()
plt.scatter(y, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")

st.pyplot(fig)

st.write("---")
st.markdown("<p style='text-align:center;'>Made with ❤️ using Streamlit & Scikit-Learn</p>", unsafe_allow_html=True)