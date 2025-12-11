# 🫀 Heart Disease Prediction using Machine Learning

This project predicts the risk of **Heart Disease** based on clinical features.  
The model is deployed using **Streamlit**, and the trained model along with the scaler is saved using **joblib (.pkl)**.



## 📌 Dataset Overview

The dataset contains **918 rows** and **12 columns** including patient details and medical attributes.

### 🔹 Features Used
| Feature | Description |
|--------|-------------|
| Age | Age of patient |
| Sex | M = Male, F = Female |
| ChestPainType | ATA, NAP, ASY, TA |
| RestingBP | Resting blood pressure |
| Cholesterol | Serum cholesterol (mg/dl) |
| FastingBS | Fasting blood sugar (1 = Yes) |
| RestingECG | Normal, LVH, ST |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Y = Yes, N = No |
| Oldpeak | ST depression |
| ST_Slope | Up, Flat, Down |
| **HeartDisease** | Target (1 = Disease, 0 = No Disease) |

---


---

## 🔧 Tech Stack
- Python
- Pandas, NumPy
- Scikit-Learn
- Joblib (for saving `.pkl` model)
- Streamlit (for deployment)
- Matplotlib, Seaborn

---

## 🛠 Data Preprocessing Steps
- One-Hot Encoding for categorical columns:
  - Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope  
- Standard Scaling for numerical columns:
  - Age, RestingBP, Cholesterol, MaxHR, Oldpeak  
- Ensured feature names match during prediction  
- Train-test split (80/20)  
- Saved model and scaler using joblib  

---

## 🤖 Machine Learning Models Comparison

The following models were trained and evaluated using Accuracy and F1-Score.

| Model                | Accuracy | F1 Score |
|----------------------|----------|----------|
| KNN                  | 0.8548   | 0.8750   |
| Logistic Regression  | 0.8647   | 0.8805   |
| Naive Bayes          | 0.8548   | 0.8706   |
| Decision Tree        | 0.7855   | 0.8148   |
| SVM                  | 0.8548   | 0.8750   |

---

## 🏆 Best Performing Model

**Logistic Regression** achieved the **highest accuracy (0.8647)** and the **best F1-score (0.8805)**.

Therefore, **Logistic Regression was selected as the final model** and saved as:

- `heart_model.pkl` (model)
- `scaler.pkl` (StandardScaler)

These files are loaded in `app.py` using **joblib** for predictions inside the Streamlit app.

---

## 💾 Saving & Loading Model (Joblib)

### Save Model
```python
import joblib
joblib.dump(model, "heart_model.pkl")
joblib.dump(scaler, "scaler.pkl")
