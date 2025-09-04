# app.py

import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
pipeline = joblib.load("model_pipeline.pkl")

st.set_page_config(page_title="Salary Prediction App", layout="centered")
st.title("💼 Salary Prediction App")
st.subheader("Enter your information to estimate your salary:")

# Input form
age = st.slider("Age", 18, 100, 30)

col1, col2 = st.columns(2)

with col1:
    education_level = st.selectbox("Education Level", ['none', 'high_school', 'bachelor', 'master', 'phd'])
    years_experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)
    job_role = st.selectbox("Job Role", ['manager', 'support', 'consultant', 'researcher', 'analyst',
                                         'intern', 'designer', 'sales', 'developer'])
    certifications = st.number_input("Number of Certifications", 0, 10, 2)
    hours_per_week = st.number_input("Hours Worked per Week", 1, 100, 40)

with col2:
    performance_score = st.number_input("Performance Score", 0, 100, 50)
    communication_skill = st.number_input("Communication Skill", 1, 100, 50)
    leadership_skill = st.number_input("Leadership Skill", 1, 100, 50)
    industry = st.selectbox("Industry", ['manufacturing', 'healthcare', 'freelance', 'IT', 'marketing',
                                         'education', 'law', 'finance'])
    location = st.selectbox("Work Location", ['remote', 'suburban', 'urban', 'rural'])

# Predict button
if st.button("Predict Salary"):
    input_data = {
        "age": age,
        "years_experience": years_experience,
        "certifications": certifications,
        "hours_per_week": hours_per_week,
        "performance_score": performance_score,
        "communication_skill": communication_skill,
        "leadership_skill": leadership_skill,
        "education_level": education_level,
        "job_role": job_role,
        "industry": industry,
        "location": location
    }

    input_df = pd.DataFrame([input_data])

    # Predict salary
    salary_pred = pipeline.predict(input_df)[0]

    st.success(f"💰 Estimated Salary: **${salary_pred:,.2f}**")
   