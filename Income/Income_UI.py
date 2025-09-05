import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st


pipeline = joblib.load("model_xgb.pkl")


st.title("Welcome 😊")
st.subheader("Let's predict your future expenses")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age",1,100,40)
    gender = st.selectbox("Gender",['Male','Female','Other'])
    education = st.selectbox("Education",['Bachelor', 'Master', 'PhD', 'High School'])
    marital_Status = st.selectbox("Marital Status",['Single', 'Married', 'Divorced', 'Widowed'])
    job_type = st.selectbox("Job Type",['Education', 'Unemployed', 'Healthcare', 'Construction', 'Finance','IT'])
    region = st.selectbox("Region",['North', 'East', 'South', 'West'])
    expenses = st.number_input("Expenses", 0, 100000, 10000)

with col2:
    income = st.number_input("Current Income", 0, 100000, 20000)
    saving = st.number_input("Saving", -100000, 100000, 0)
    credit_score = st.number_input("Credit Score",0,1000,300)
    year_of_experience = st.number_input("Experience year",0,60,30)
    loan_amount = st.number_input("Load Amouont",0,2500000,250000)
    load_approval = st.selectbox("Load Approval",[0,1])
    default_risk = st.selectbox("Default Risk",[0,1])

if st.button("Predict"):
    input_data = {
        'Age':age,
        'Gender': gender,
        'Education': education,
        'Marital_Status': marital_Status,
        'Job_Type': job_type,
        'Region': region,
        'Income': income,
        'Expenses': expenses,
        'Savings': saving,
        'Credit_Score': credit_score,
        'Years_of_Experience' : year_of_experience,
        'Loan_Amount': loan_amount,
        'Loan_Approval': load_approval,
        'Default_Risk': default_risk,
        }
    
    input_df = pd.DataFrame([input_data])
    
    income_predict = pipeline.predict(input_df)[0]

    st.success(f"💰 Estimated Future Income: **${income_predict:,.2f}**")


