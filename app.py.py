import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Set page config
st.set_page_config(page_title="Smart Crop Recommendation", layout="centered")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Crop_recommendation.csv")

df = load_data()

# Prepare features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Climate zone mapping
CLIMATE_ZONES = {
    "Tropical": {
        "lat_range": (-23.5, 23.5),
        "lon_range": (-180, 180),
        "defaults": [28.0, 80.0, 200.0]
    },
    "Temperate": {
        "lat_range": (23.5, 66.5),
        "lon_range": (-180, 180),
        "defaults": [20.0, 60.0, 100.0]
    },
    "Arid": {
        "lat_range": (15, 35),
        "lon_range": (-20, 60),
        "defaults": [35.0, 30.0, 20.0]
    },
    "Default": {
        "lat_range": (-90, 90),
        "lon_range": (-180, 180),
        "defaults": [25.0, 60.0, 100.0]
    }
}

# Inject JS to get GPS
def get_gps_location():
    gps_js = """
    <script>
    function sendCoords() {
        navigator.geolocation.getCurrentPosition(
            position => {
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;
                window.parent.postMessage({type:"coords", lat, lon}, "*");
            },
            error => {
                window.parent.postMessage({type:"coords_error", error: error.message}, "*");
            }
        );
    }
    sendCoords();
    </script>
    """
    st.components.v1.html(gps_js, height=0)

# Check for query params
def check_for_coords():
    params = st.experimental_get_query_params()
    if params.get("lat") and params.get("lon"):
        try:
            lat = float(params["lat"][0])
            lon = float(params["lon"][0])
            return lat, lon
        except:
            return None
    return None

# Get climate defaults
def get_climate_defaults(lat, lon):
    for zone, data in CLIMATE_ZONES.items():
        if (data["lat_range"][0] <= lat <= data["lat_range"][1] and 
            data["lon_range"][0] <= lon <= data["lon_range"][1]):
            return data["defaults"], zone
    return CLIMATE_ZONES["Default"]["defaults"], "Unknown"

# App UI
st.title("🌍 GPS-Enabled Crop Recommendation")
st.markdown("""This system automatically detects your location to suggest optimal crops 
            based on local climate conditions.""")

# Init session
if "coords" not in st.session_state:
    st.session_state.coords = check_for_coords()
if "climate_zone" not in st.session_state:
    st.session_state.climate_zone = None

# Get GPS
get_gps_location()

# Show location if available
if st.session_state.coords:
    lat, lon = st.session_state.coords
    defaults, zone = get_climate_defaults(lat, lon)
    st.session_state.climate_zone = zone
    
    st.success(f"📍 Detected location: {lat:.4f}, {lon:.4f}")
    st.info(f"🌦️ Climate zone: {zone} (Temp: {defaults[0]}°C, Humidity: {defaults[1]}%, Rainfall: {defaults[2]}mm)")

# Input form
with st.form("crop_form"):
    st.subheader("🌱 Soil Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        n = st.slider("Nitrogen (N)", 0, 140, 90)
        p = st.slider("Phosphorus (P)", 5, 145, 40)
        k = st.slider("Potassium (K)", 5, 205, 40)
    
    with col2:
        ph = st.slider("pH", 3.5, 9.5, 6.5)
        if st.session_state.coords:
            defaults, _ = get_climate_defaults(*st.session_state.coords)
            temperature = st.slider("Temperature (°C)", 10.0, 45.0, float(defaults[0]))
            humidity = st.slider("Humidity (%)", 10.0, 100.0, float(defaults[1]))
            rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, float(defaults[2]))
        else:
            temperature = st.slider("Temperature (°C)", 10.0, 45.0, 25.0)
            humidity = st.slider("Humidity (%)", 10.0, 100.0, 60.0)
            rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0)
    
    submitted = st.form_submit_button("Recommend Crop")

# Prediction
if submitted:
    input_data = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)[0]
    
    st.success(f"✅ Recommended Crop: **{prediction}**")
    
    if st.session_state.coords:
        st.info(f"Recommendation based on {st.session_state.climate_zone} climate conditions")

# Feature Importance
st.subheader("📊 How Different Factors Affect Recommendations")
fig, ax = plt.subplots(figsize=(10, 5))
importances = model.feature_importances_
features = X.columns
sns.barplot(x=importances, y=features, hue=features, palette="viridis", legend=False)
ax.set_title("Feature Importance")
ax.set_xlabel("Importance Score")
st.pyplot(fig)

# Simulate Location Help
st.markdown("""
### 🧭 Testing Without GPS
You can simulate locations by adding URL parameters in the browser:
- Example: `http://localhost:8501/?lat=19.5&lon=73.8`
""")
