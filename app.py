import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- Load artifacts ----------------
model = joblib.load(os.path.join(BASE_DIR, "heart_svm_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "heart_scaler.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "heart_columns.pkl"))


st.title("❤️ Heart Disease Prediction System")
st.write("Enter patient details to predict the risk of heart disease.")

# ---------------- Input Fields ----------------
age = st.slider("Age", 1, 120, 45)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Serum Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0)
slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])


NUMERIC_COLS = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']


# ---------------- Create empty row ----------------
input_data = pd.DataFrame(0, index=[0], columns=columns)

# ---------------- Numerical features ----------------
input_data["Age"] = age
input_data["RestingBP"] = resting_bp
input_data["Cholesterol"] = cholesterol
input_data["FastingBS"] = fasting_bs
input_data["MaxHR"] = max_hr
input_data["Oldpeak"] = oldpeak

# ---------------- Categorical: Sex ----------------
if sex == "Male":
    input_data["Sex_M"] = 1

# ---------------- Categorical: Exercise Angina ----------------
if exercise_angina == "Yes":
    input_data["ExerciseAngina"] = 1

# ---------------- One-Hot: Chest Pain ----------------
input_data[f"ChestPainType_{chest_pain}"] = 1

# ---------------- One-Hot: Resting ECG ----------------
if resting_ecg == "Normal":
    input_data["RestingECG_Normal"] = 1
elif resting_ecg == "ST":
    input_data["RestingECG_ST"] = 1
else:
    input_data["RestingECG_LVH"] = 1

# ---------------- One-Hot: ST Slope ----------------
input_data[f"ST_Slope_{slope}"] = 1

# ---------------- Prediction ----------------
if st.button("Predict"):
    
    input_data[NUMERIC_COLS] = scaler.transform(input_data[NUMERIC_COLS])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
