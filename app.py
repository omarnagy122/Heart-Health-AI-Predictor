import streamlit as st
import pandas as pd
import joblib
import numpy as np

# إعدادات الصفحة (أيقونة وعنوان)
st.set_page_config(page_title="Heart Health AI", page_icon="❤️", layout="wide")

# تحميل الموديل والـ Preprocessors
@st.cache_resource
def load_artifact():
    return joblib.load('best_heart_disease_model.joblib')

artifact = load_artifact()
model = artifact['model']
prep = artifact['preprocessors']

# CSS عشان نخلي الشكل أحلى
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #ff4b4b; color: white; }
    .prediction-box { padding: 20px; border-radius: 15px; text-align: center; font-size: 24px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("❤️ Heart Disease Prediction System")
st.write("استخدم التقنيات الذكية لتقييم حالة القلب بناءً على بياناتك الطبية.")

# تقسيم الشاشة لأعمدة
col1, col2, col3 = st.columns([1, 1, 1], gap="large")

with col1:
    st.subheader("📋 Personal Info")
    age = st.slider("Age", 1, 100, 45)
    sex = st.radio("Gender", ["Male", "Female"], horizontal=True)
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"])
    trestbps = st.number_input("Resting BP (mm Hg)", 80, 200, 120)

with col2:
    st.subheader("🏥 Clinical Data")
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.toggle("Fasting Blood Sugar > 120 mg/dl")
    restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Wave", "LV Hypertrophy"])
    thalach = st.slider("Max Heart Rate", 60, 220, 150)

with col3:
    st.subheader("🩺 Advanced Tests")
    exang = st.toggle("Exercise Induced Angina")
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of ST", ["Upsloping", "Flat", "Downsloping"])
    ca = st.selectbox("Major Vessels (0-4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversable Defect"])

# تحويل المدخلات لـ DataFrame
input_data = pd.DataFrame([{
    'age': age,
    'sex': sex.lower(),
    'chest_pain_type': cp.lower(),
    'resting_blood_pressure': trestbps,
    'cholestoral': chol,
    'fasting_blood_sugar': fbs,
    'rest_ecg': restecg.lower(),
    'Max_heart_rate': thalach,
    'exercise_induced_angina': exang,
    'oldpeak': oldpeak,
    'slope': slope.capitalize(),
    'vessels_colored_by_flourosopy': str(ca),
    'thalassemia': thal.lower()
}])

st.divider()

if st.button("Analyze Health Status"):
    # تنفيذ الـ Preprocessing (نفس خطوات الـ Pipeline)
    # 1. Binary Encoding
    bi_cols = ['sex', 'fasting_blood_sugar', 'exercise_induced_angina']
    input_bi = prep['binary'].transform(input_data[bi_cols])
    df = pd.concat([input_data.drop(columns=bi_cols), input_bi], axis=1)

    # 2. Label Encoding
    le_cols = ['slope']
    for col in le_cols:
        df[col] = prep['labels'][col].transform(df[col])
    df['vessels_colored_by_flourosopy'] = int(ca)

    # 3. One-Hot Encoding
    oh_cols = ['chest_pain_type', 'rest_ecg', 'thalassemia']
    input_oh = prep['onehot'].transform(df[oh_cols])
    oh_feat = prep['onehot'].get_feature_names_out(oh_cols)
    df = pd.concat([df.drop(columns=oh_cols), pd.DataFrame(input_oh, columns=oh_feat)], axis=1)

    # 4. Scaling
    num_cols = ['age', 'resting_blood_pressure', 'cholestoral', 'Max_heart_rate', 'oldpeak']
    df[num_cols] = prep['scaler'].transform(df[num_cols])

    # 5. Feature Match & PCA
    df = df[prep['final_columns']]
    df_pca = prep['pca'].transform(df)

    # Prediction
    pred = model.predict(df_pca)[0]
    prob = model.predict_proba(df_pca)[0][1]

    # عرض النتيجة بشكل شيك
    if pred == 1:
        st.markdown(f'<div class="prediction-box" style="background-color: #ffdce0; color: #ff4b4b;">⚠️ Risk Detected: {prob*100:.1f}% Probability</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="prediction-box" style="background-color: #d4edda; color: #155724;">✅ Low Risk: {prob*100:.1f}% Probability</div>', unsafe_allow_html=True)

    # عرض الماتريكس اللي طلبتها
    st.info(f"Model used: {model.__class__.__name__} | Model Performance: High Precision")