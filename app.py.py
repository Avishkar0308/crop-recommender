import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Crop_recommendation.csv")
    return df

df = load_data()

# Split data
X = df.drop("label", axis=1)
y = df["label"]

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit App
st.set_page_config(page_title="Crop Recommendation", layout="centered")
st.title("🌾 Crop Recommendation System")
st.markdown("Enter soil and climate data to get the best crop suggestion!")

# Input sliders
n = st.slider("Nitrogen (N)", 0, 140, 90)
p = st.slider("Phosphorus (P)", 5, 145, 40)
k = st.slider("Potassium (K)", 5, 205, 40)
temperature = st.slider("Temperature (°C)", 10.0, 45.0, 25.0)
humidity = st.slider("Humidity (%)", 10.0, 100.0, 80.0)
ph = st.slider("pH", 3.5, 9.5, 6.5)
rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0)

# Predict button
if st.button("Recommend Crop"):
    input_data = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)[0]
    st.success(f"✅ Recommended Crop: **{prediction}**")

# Feature Importance Visualization
st.subheader("📊 Feature Importance")
importances = model.feature_importances_
feature_names = X.columns

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x=importances, y=feature_names, ax=ax, palette="viridis")
ax.set_title("Feature Importance")
st.pyplot(fig)
