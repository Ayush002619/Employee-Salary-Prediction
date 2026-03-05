import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼")

st.title("💼 Employee Salary Classification")
st.write("Predict whether an employee earns >50K or ≤50K")

st.sidebar.header("Enter Employee Details")

age = st.sidebar.slider("Age",18,65,30)

education = st.sidebar.selectbox("Education Level",
["Bachelors","Masters","PhD","HS-grad","Assoc","Some-college"])

occupation = st.sidebar.selectbox("Job Role",
["Tech-support","Craft-repair","Other-service","Sales",
"Exec-managerial","Prof-specialty","Handlers-cleaners",
"Machine-op-inspct","Adm-clerical","Farming-fishing",
"Transport-moving","Priv-house-serv","Protective-serv","Armed-Forces"])

hours_per_week = st.sidebar.slider("Hours per week",1,80,40)

experience = st.sidebar.slider("Years of Experience",0,40,5)


# Encoding maps
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


educational_num = education_map[education]
occupation_encoded = occupation_map[occupation]


# Build dataframe EXACTLY like training data
input_df = pd.DataFrame({

"age":[age],
"workclass":[1],
"fnlwgt":[100000],
"education":[1],
"educational-num":[educational_num],
"marital-status":[1],
"occupation":[occupation_encoded],
"relationship":[1],
"race":[1],
"gender":[1],
"capital-gain":[0],
"capital-loss":[0],
"hours-per-week":[hours_per_week],
"native-country":[1]

})


st.write("### Input Data")
st.write(input_df)


if st.button("Predict Salary"):

    # ensure correct column order
    input_df = input_df[model.feature_names_in_]

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.success("💰 Employee likely earns >50K")
    else:
        st.success("📉 Employee likely earns ≤50K")


# Batch prediction
st.markdown("---")
st.subheader("Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV",type="csv")

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    predictions = model.predict(data)

    data["PredictedClass"] = predictions

    st.write(data.head())

    csv = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Predictions",
        csv,
        "predictions.csv",
        "text/csv"
    )
