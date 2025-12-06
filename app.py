import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.set_page_config(page_title="Diamond Analysis App", layout="wide")


# LOAD MODELS & PREPROCESSORS

price_model = joblib.load("best_price_model.pkl")           # Regression pipeline
cluster_model = joblib.load("best_clustering_model.pkl")   # Clustering model
scaler = joblib.load("cluster_scaler.pkl")                 # Scaler for clustering
cluster_columns = joblib.load("cluster_columns.pkl")       # Columns used during clustering training


# APP HEADER

st.title("💎 Diamond Price Prediction & Market Segmentation")


# SIDEBAR NAVIGATION

menu = st.sidebar.radio(
    "Navigation",
    ["📤 Upload Data", "💰 Price Prediction", "📊 Clustering"]
)


# UPLOAD DATA

if menu == "📤 Upload Data":
    st.header("📤 Upload Diamond Dataset")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success("File uploaded successfully!")
        st.subheader("🔍 Preview of Data")
        st.dataframe(df.head())
        st.subheader("📌 Dataset Summary")
        st.write(df.describe())

#price prediction

elif menu == "💰 Price Prediction":

    st.header("💰 Diamond Price Prediction (Regression Model)")
    st.write("Enter diamond properties below:")

    import pandas as pd
    import joblib

    # Load model ONCE
    price_model = joblib.load("best_price_model.pkl")

    # --- Inputs ---
    carat = st.number_input("Carat", 0.1, 5.0, 1.0, 0.01)
    depth = st.number_input("Depth", 50.0, 80.0, 61.0)
    table = st.number_input("Table", 40.0, 100.0, 57.0)
    x = st.number_input("Length (x)", 0.1, 20.0, 5.0)
    y = st.number_input("Width (y)", 0.1, 20.0, 5.0)
    z = st.number_input("Height (z)", 0.1, 20.0, 3.0)

    cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
    color = st.selectbox("Color", list("DEFGHIJ"))
    clarity = st.selectbox("Clarity", ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"])

    if st.button("Predict Price"):
        # Correct input DataFrame
        input_data = pd.DataFrame([{
            "carat": carat,
            "depth": depth,
            "table": table,
            "x": x,
            "y": y,
            "z": z,
            "cut": cut,
            "color": color,
            "clarity": clarity
        }])

        st.write("INPUT DATA:", input_data)

        try:
            predicted_price = price_model.predict(input_data)[0]
            st.success(f"💎 Predicted Diamond Price: ₹ {predicted_price:,.2f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# CLUSTERING

elif menu == "📊 Clustering":
    st.header("📊 Diamond Market Segmentation (Clustering)")

    # Upload CSV
    uploaded = st.file_uploader("Upload a CSV file for clustering", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success("Data uploaded successfully!")
        
        st.subheader("Preview:")
        st.dataframe(df.head())  # Show first 5 rows

        
        #  Prepare Data for Clustering
        
        # Drop price column if present
        cluster_df = df.drop(["price"], axis=1, errors="ignore")

        # One-hot encode categorical columns
        cluster_encoded = pd.get_dummies(cluster_df)

        # Align columns with saved clustering model columns
        # cluster_columns.pkl should be saved when training
        cluster_columns = joblib.load("cluster_columns.pkl")  # columns used during training
        cluster_encoded = cluster_encoded.reindex(columns=cluster_columns, fill_value=0)

        # Scale features
        scaled_data = scaler.transform(cluster_encoded)

        # Predict cluster labels
        labels = cluster_model.predict(scaled_data)
        df["cluster"] = labels

        
        #  Cluster Summary
        
        st.subheader("Cluster Summary")
        cluster_summary = df.groupby("cluster").agg({
            "carat": "mean",
            "price": ["mean", "count"]
        }).round(2)
        st.dataframe(cluster_summary)

        
        #  PCA 2D Visualization
        
        st.subheader("🎨 PCA 2D Cluster Visualization")
        pca = PCA(n_components=2)
        pca_2D = pca.fit_transform(scaled_data)

        fig, ax = plt.subplots(figsize=(7,5))
        scatter = ax.scatter(pca_2D[:,0], pca_2D[:,1], c=labels, cmap="viridis")
        plt.title("Clusters (PCA 2D)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.colorbar(scatter, ax=ax, label="Cluster")
        st.pyplot(fig)

        
        #  Cluster Naming
        
        st.subheader("📌 Cluster Naming Based on Rules")
        cluster_names = {}
        for c in df["cluster"].unique():
            cluster_data = df[df["cluster"] == c]
        
            # Convert label to native int
            c_int = int(c)
        
            avg_price = cluster_data["price"].mean() if "price" in cluster_data.columns else 0
            avg_carat = cluster_data["carat"].mean()
        
            if avg_carat > 1.5 and avg_price > df["price"].mean() if "price" in df.columns else 0:
                name = "💎 Premium Heavy Diamonds"
            elif avg_carat < 0.5:
                name = "💍 Affordable Small Diamonds"
            else:
                name = "🔷 Mid-range Balanced Diamonds"
        
            cluster_names[c_int] = name
        
        df["cluster_name"] = df["cluster"].map(cluster_names)
        st.write(cluster_names)
        # Show labeled clusters
        st.subheader("📝 Labeled Clusters Preview")
        display_cols = ["carat", "price", "cut", "color", "clarity", "cluster", "cluster_name"]
        display_cols = [col for col in display_cols if col in df.columns]
        st.dataframe(df[display_cols].head(10))  # show first 10 rows

        # Save clustered output
        df.to_csv("clustered_output.csv", index=False)
        st.success("Clustered file saved: clustered_output.csv")