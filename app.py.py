import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# ✅ Set this before any other Streamlit command
st.set_page_config(page_title="Crop Recommendation", layout="centered")

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("Crop_recommendation.csv")

df = load_data()

# Prepare features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# --- Streamlit UI ---
st.title("🌾 Crop Recommendation System")
st.markdown("Enter your soil and climate values to get the most suitable crop.")

# Input sliders
n = st.slider("Nitrogen (N)", 0, 140, 90)
p = st.slider("Phosphorus (P)", 5, 145, 40)
k = st.slider("Potassium (K)", 5, 205, 40)
temperature = st.slider("Temperature (°C)", 10.0, 45.0, 25.0)
humidity = st.slider("Humidity (%)", 10.0, 100.0, 80.0)
ph = st.slider("pH", 3.5, 9.5, 6.5)
rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0)

# Prediction button
if st.button("Recommend Crop"):
    input_data = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)[0]
    st.success(f"✅ We recommend growing: **{prediction}**")

# --- Feature Importance ---
st.subheader("📊 Feature Importance")

importances = model.feature_importances_
features = X.columns

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x=importances, y=features, ax=ax, palette="viridis")
ax.set_title("Feature Importance of Input Parameters")
st.pyplot(fig)

