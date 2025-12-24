# ❤️ Heart Disease Prediction System

![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.1-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

An end-to-end **Machine Learning + Deployment** project that predicts the **risk of heart disease** based on patient health parameters.  
This project demonstrates real-world ML engineering, including preprocessing, model training, artifact management, and deployment using **Streamlit**.

---

## 🚀 Project Overview

Heart disease is one of the leading causes of death worldwide.  
This system provides a deployable ML solution that:

- Accepts **human-readable medical inputs**
- Reconstructs the **exact training feature pipeline**
- Predicts whether a patient is at **high or low risk** of heart disease
- Is usable by **non-technical users**

---

## 🧠 Machine Learning Pipeline (`heart_disease.ipynb`)

### 📊 Dataset
- Public heart disease dataset
- Feature types:
  - **Numerical:** Age, RestingBP, Cholesterol, MaxHR, Oldpeak
  - **Categorical:** ChestPainType, RestingECG, ST_Slope, Sex, ExerciseAngina
- Target variable:
  - `0` → No heart disease
  - `1` → Heart disease present

---

### 🔄 Data Preprocessing

#### One-Hot Encoding
Categorical features were converted using **one-hot encoding** to avoid ordinal assumptions and ensure compatibility with ML models.

#### Feature Scaling
Only numerical features were scaled using `StandardScaler`:

```python
NUMERIC_COLS = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
```

---
### 🧪 Train–Test Split

```python
train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
```

---

### 💾 Saved Artifacts

```python
joblib.dump(SVMModel, "heart_svm_model.pkl")
joblib.dump(scaler, "heart_scaler.pkl")
joblib.dump(X_scaled.columns.tolist(), "heart_columns.pkl")
```



---

## 🔁 How Prediction Works

User Input
   ↓
Feature Reconstruction (One-Hot Encoding)
   ↓
Numerical Feature Scaling
   ↓
Trained SVM Model
   ↓
Heart Disease Risk Prediction



## 🖥️ Streamlit Application (`app.py`)

The Streamlit application provides a clean and intuitive interface for real-time predictions.

### Key Features
- User-friendly medical inputs
- Real-time inference
- Training–deployment consistency
- Robust preprocessing logic


<details>
<summary><strong>📦 Inference Logic</strong></summary>

1. Create an empty DataFrame using the saved column schema  
2. Populate numerical and one-hot encoded categorical features  
3. Apply scaling **only to numerical columns**  
4. Predict using the trained SVM model  

```python
input_data[NUMERIC_COLS] = scaler.transform(input_data[NUMERIC_COLS])
prediction = model.predict(input_data)[0]
```

</details>


## ⚠️ Challenges & Solutions

<details>
<summary><strong>Serialization & Scaler Errors</strong></summary>

- Scaler was incorrectly deserialized during deployment  
- Fixed by saving and loading artifacts using `joblib` with absolute paths  

</details>

<details>
<summary><strong>Feature Name Mismatch During Inference</strong></summary>

- Scaler trained only on numerical features  
- Resolved by scaling **only numeric columns** and leaving one-hot features untouched  

</details>

<details>
<summary><strong>Environment Conflicts</strong></summary>

- Training and Streamlit ran in different Python environments  
- Fixed by isolating the virtual environment and running:
  ```bash
  python -m streamlit run app.py
  ```
</details>






## 🎯 Key Takeaways

- ML deployment depends on **pipeline consistency**
- Feature preprocessing must match training exactly
- Environment isolation and correct serialization are critical
- Real-world ML involves extensive debugging beyond model training

---

## 👩‍💻 Author

**Khushi Goyal**  
B.Tech Computer Science (2022–2026)

This project demonstrates end-to-end ML engineering, debugging, and deployment skills suitable for real-world applications and technical interviews.




