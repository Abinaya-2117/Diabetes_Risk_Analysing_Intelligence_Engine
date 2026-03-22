import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the brain and translator
model = joblib.load('diabetes_svm_model.pkl')
scaler = joblib.load('data_scaler.pkl')

st.title("🩺 Diabetes Risk Predictor")

def get_user_input():
    # We group inputs to make the UI clean
    col1, col2 = st.columns(2)
    
    with col1:
        high_bp = st.selectbox("High Blood Pressure", [0, 1])
        high_chol = st.selectbox("High Cholesterol", [0, 1])
        chol_check = st.selectbox("Cholesterol Check in 5yrs", [0, 1])
        bmi = st.number_input("BMI", 10, 60, 25)
        smoker = st.selectbox("Smoker (at least 100 cigs)", [0, 1])
        stroke = st.selectbox("Ever had a Stroke?", [0, 1])
        heart_dis = st.selectbox("Heart Disease/Attack History", [0, 1])
        phys_act = st.selectbox("Physical Activity in past 30 days", [0, 1])
        fruits = st.selectbox("Eat Fruit 1+ times per day", [0, 1])
        veggies = st.selectbox("Eat Veggies 1+ times per day", [0, 1])
        hvy_alc = st.selectbox("Heavy Alcohol Consumption", [0, 1])

    with col2:
        any_hc = st.selectbox("Any Healthcare Coverage", [0, 1])
        no_doc = st.selectbox("No Doctor due to Cost", [0, 1])
        gen_hlth = st.slider("General Health (1:Excel - 5:Poor)", 1, 5, 3)
        ment_hlth = st.slider("Days of poor Mental Health (0-30)", 0, 30, 0)
        phys_hlth = st.slider("Days of poor Physical Health (0-30)", 0, 30, 0)
        diff_walk = st.selectbox("Difficulty Walking/Climbing Stairs", [0, 1])
        sex = st.selectbox("Sex (0:Female, 1:Male)", [0, 1])
        age = st.slider("Age Category (1-13)", 1, 13, 7)
        edu = st.slider("Education Level (1-6)", 1, 6, 4)
        income = st.slider("Income Scale (1-8)", 1, 8, 5)

    # Dictionary must match X_train column names and order EXACTLY
    features = {
        'HighBP': high_bp, 'HighChol': high_chol, 'CholCheck': chol_check,
        'BMI': bmi, 'Smoker': smoker, 'Stroke': stroke,
        'HeartDiseaseorAttack': heart_dis, 'PhysActivity': phys_act,
        'Fruits': fruits, 'Veggies': veggies, 'HvyAlcoholConsump': hvy_alc,
        'AnyHealthcare': any_hc, 'NoDocbcCost': no_doc, 'GenHlth': gen_hlth,
        'MentHlth': ment_hlth, 'PhysHlth': phys_hlth, 'DiffWalk': diff_walk,
        'Sex': sex, 'Age': age, 'Education': edu, 'Income': income
    }
    return pd.DataFrame([features])

input_df = get_user_input()

if st.button("Predict Diabetes Risk"):
    # Scaling is the "secret sauce"
    scaled_data = scaler.transform(input_df)
    prediction = model.predict(scaled_data)
    
    results = {0: "Healthy", 1: "Pre-diabetic", 2: "Diabetic"}
    st.subheader(f"Result: {results[prediction[0]]}")