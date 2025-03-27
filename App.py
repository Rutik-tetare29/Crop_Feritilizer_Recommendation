import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load trained models
with open("crop_recommendation_2.pkl", "rb") as f:
    crop_model = pickle.load(f)

with open("crop_scalor.pkl", "rb") as scaler_file:
    sc = pickle.load(scaler_file)

with open("fertilizer_recommendation.pkl", "rb") as f:
    fertilizer_model = pickle.load(f)

with open("fertilizer_scalor.pkl", "rb") as f:
    fc = pickle.load(f)


# Crop Mapping
crop_dict = {
    1: 'Rice', 2: 'Maize', 3: 'Jute', 4: 'Cotton', 5: 'Coconut', 6: 'Papaya',
    7: 'Orange', 8: 'Apple', 9: 'Muskmelon', 10: 'Watermelon', 11: 'Grapes',
    12: 'Mango', 13: 'Banana', 14: 'Pomegranate', 15: 'Lentil', 16: 'Blackgram',
    17: 'Mungbean', 18: 'Mothbeans', 19: 'Pigeonpeas', 20: 'Kidneybeans',
    21: 'Chickpea', 22: 'Coffee'
}

# Fertilizer Mapping
fert_dict = {
    1: 'Urea', 2: 'DAP', 3: '14-35-14', 4: '28-28',
    5: '17-17-17', 6: '20-20', 7: '10-26-26'
}

# Streamlit UI
st.title("🌱 Crop & Fertilizer Recommendation System")
st.markdown("Enter soil & climate details to get the **best crop & fertilizer recommendations.**")

# User Inputs
nitrogen = st.number_input("Nitrogen (N)", min_value=0)
phosphorus = st.number_input("Phosphorus (P)", min_value=0)
potassium = st.number_input("Potassium (K)", min_value=0)
temperature = st.number_input("Temperature (°C)", min_value=0.0, step=0.1)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
pH = st.number_input("Soil pH Level", min_value=0.0, max_value=14.0, step=0.1)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=0.1)

# Predict Crop & Fertilizer
if st.button("🔍 Recommend Crop & Fertilizer"):
    # Crop Prediction
    crop_input = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall]])
    crop_input = sc.transform(crop_input)
    crop_prediction = crop_model.predict(crop_input).reshape(1,-1)[0]
    predicted_crop = crop_dict.get(crop_prediction[0], "Unknown Crop")

    # Fertilizer Prediction (Now using 7 features instead of 8)
   
    fert_input = np.array([[temperature,humidity,0.6,2,crop_prediction[0],nitrogen, potassium, phosphorus]])  # Add a placeholder value
    fert_input = fc.transform(fert_input)
    fert_prediction = fertilizer_model.predict(fert_input).reshape(1,-1)[0]
    predicted_fertilizer = fert_dict.get(fert_prediction[0], "Unknown Fertilizer")

    # Display Results
    st.success(f"✅ Recommended Crop: **{predicted_crop}**")
    st.success(f"✅ Recommended Fertilizer: **{predicted_fertilizer}**")
