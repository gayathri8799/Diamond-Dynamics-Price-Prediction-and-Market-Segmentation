import streamlit as st
import numpy as np
import pickle

# Load saved models
price_model = pickle.load(open("price_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
cluster_model = pickle.load(open("cluster_model.pkl", "rb"))

# Ordinal categories (must match training)
cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

# Cluster name mapping
cluster_names = {
    0: "Affordable Small Diamonds",
    1: "Mid-range Balanced Diamonds",
    2: "Premium Heavy Diamonds"
}

# Streamlit UI
st.set_page_config(page_title="Diamond Predictor", layout="centered")

st.title("💎 Diamond Price & Market Segment Predictor")
st.write("Predict **diamond price (INR)** and **market segment** based on diamond attributes.")

st.divider()

# User Inputs
st.subheader("🔹 Enter Diamond Details")

carat = st.number_input("Carat", min_value=0.1, step=0.01)
x = st.number_input("Length (x in mm)", min_value=0.1, step=0.01)
y = st.number_input("Width (y in mm)", min_value=0.1, step=0.01)
z = st.number_input("Depth (z in mm)", min_value=0.1, step=0.01)

cut = st.selectbox("Cut", cut_order)
color = st.selectbox("Color", color_order)
clarity = st.selectbox("Clarity", clarity_order)

# Feature Engineering
volume = x * y * z

# Convert categorical to ordinal values
cut_val = cut_order.index(cut)
color_val = color_order.index(color)
clarity_val = clarity_order.index(clarity)

# Regression input (8 features)
reg_input = np.array([[carat, cut_val, color_val, clarity_val, x, y, z, volume]])
reg_scaled = scaler.transform(reg_input)

# Predict price first (needed for clustering)
log_price_pred = price_model.predict(reg_scaled)
predicted_price_inr = np.expm1(log_price_pred)[0]

# Clustering input (6 features ONLY)
cluster_input = np.array([[carat, cut_val, color_val, clarity_val,
                           predicted_price_inr, volume]])

st.divider()

#  Price Prediction Module
st.subheader("📈 Price Prediction")

if st.button("Predict Price 💰"):
    log_price_pred = price_model.predict(reg_scaled)
    predicted_price_inr = np.expm1(log_price_pred)[0]
    st.success(f"💰 Predicted Diamond Price: ₹ {predicted_price_inr:,.2f}")

#  Market Segment Prediction
st.subheader("📊 Market Segment Prediction")
if st.button("Predict Market Segment 🏷️"):
    log_price_pred = price_model.predict(reg_scaled)
    predicted_price_inr = np.expm1(log_price_pred)[0]

    cluster_input = np.array([[carat, cut_val, color_val, clarity_val,
                               predicted_price_inr, volume]])

    cluster = cluster_model.predict(cluster_input)[0]
    cluster_name = cluster_names.get(cluster, "Unknown Segment")

    st.info(f"🔢 Cluster Number: {cluster}")
    st.success(f"🏷️ Market Segment: {cluster_name}")