import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 18, 65, 30)

education = st.sidebar.selectbox("Education Level", [
    "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"
])

occupation = st.sidebar.selectbox("Job Role", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv", "Armed-Forces"
])

hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)

experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# 🔧 ADDED: Encoding mappings
education_map = {
    "Bachelors":13,
    "Masters":14,
    "PhD":16,
    "HS-grad":9,
    "Assoc":12,
    "Some-college":10
}

occupation_map = {
    "Tech-support":0,
    "Craft-repair":1,
    "Other-service":2,
    "Sales":3,
    "Exec-managerial":4,
    "Prof-specialty":5,
    "Handlers-cleaners":6,
    "Machine-op-inspct":7,
    "Adm-clerical":8,
    "Farming-fishing":9,
    "Transport-moving":10,
    "Priv-house-serv":11,
    "Protective-serv":12,
    "Armed-Forces":13
}

# 🔧 ADDED: Convert UI text to encoded values
educational_num = education_map[education]
occupation_encoded = occupation_map[occupation]

# 🔧 CHANGED: Input dataframe must match training features
input_df = pd.DataFrame({
    'age':[age],
    'workclass':[1],              # 🔧 ADDED default value
    'marital-status':[1],         # 🔧 ADDED default value
    'occupation':[occupation_encoded],
    'relationship':[1],           # 🔧 ADDED default value
    'race':[1],                   # 🔧 ADDED default value
    'gender':[1],                 # 🔧 ADDED default value
    'educational-num':[educational_num],
    'capital-gain':[0],           # 🔧 ADDED default value
    'hours-per-week':[hours_per_week],
    'native-country':[1]          # 🔧 ADDED default value
})

st.write("### 🔎 Input Data")
st.write(input_df)

if st.button("Predict Salary Class"):

    prediction = model.predict(input_df)

    # 🔧 CHANGED: clearer output
    if prediction[0] == 1:
        st.success("💰 Employee likely earns >50K")
    else:
        st.success("📉 Employee likely earns ≤50K")

st.markdown("---")
st.markdown("#### 📂 Batch Prediction")

uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:

    batch_data = pd.read_csv(uploaded_file)

    st.write("Uploaded data preview:", batch_data.head())

    batch_preds = model.predict(batch_data)

    batch_data['PredictedClass'] = batch_preds

    st.write("✅ Predictions:")
    st.write(batch_data.head())

    csv = batch_data.to_csv(index=False).encode('utf-8')

    st.download_button("Download Predictions CSV", csv,
                       file_name='predicted_classes.csv',
                       mime='text/csv')
