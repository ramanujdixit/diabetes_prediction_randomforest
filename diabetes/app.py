import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model
with open("model.pkl", "rb") as f:  # note 'rb' for reading
    model = pickle.load(f)


# Since scaler was fit on training data, recreate StandardScaler and manually set mean_ and scale_ if saved
# For simplicity, here we fit on sample data or you can save and load scaler similarly

scaler = StandardScaler()

# User input sliders for each feature (replace ranges with your actual feature value ranges)
pregnancies = st.slider('Pregnancies', 0, 20, 1)
glucose = st.slider('Glucose', 0, 200, 120)
insulin = st.slider('Insulin', 0.0, 900.0, 79.0)
bmi = st.slider('BMI', 10.0, 70.0, 32.0)
diabetes_pedigree = st.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5)
age = st.slider('Age', 10, 100, 33)

# Create input array for prediction
input_data = np.array([[pregnancies, glucose, insulin, bmi, diabetes_pedigree, age]])

# Apply same outlier capping and scaling as training (for demo, just scaling)
input_scaled = scaler.fit_transform(input_data)  # In deployment, use saved scaler's transform only

if st.button('Predict'):
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.write("The patient is likely to have diabetes.")
    else:
        st.write("The patient is unlikely to have diabetes.")
