import streamlit as st
import pandas as pd
from source.pipeline.predict_pipeline import CustomData,PredictPipeline

st.set_page_config(page_title="Student Score Predictor",layout="centered")
st.title("Student Performance Prediction App")
st.markdown("Enter Student details below to predict math score")

with st.form ("prediction_form"):
    gender=st.selectbox("Gender",["male","female"])
    race_ethnicity=st.selectbox("Race/Ethnicity",["group A","group B","group C","group D","group E"])
    parental_level_of_education=st.selectbox("Parental level of Education",["some high school", "high school", "some college",
         "associate's degree", "bachelor's degree", "master's degree"])
    lunch=st.selectbox("Lunch",["standard","free/reduced"])
    test_preparation_course=st.selectbox("Test Preparation Course",["none","completed"])
    reading_score=st.slider("Reading Score",0,100,50)
    writing_score=st.slider("Writing Score",0,100,50)

    submit=st.form_submit_button("Predict")

if submit:
    #create data object and predict
    data=CustomData(
        gender=gender,
        race_ethnicity=race_ethnicity,
        parental_level_of_education=parental_level_of_education,
        lunch=lunch,
        test_preparation_course=test_preparation_course,
        reading_score=reading_score,
        writing_score=writing_score
    )
    
    pred_df=data.get_data_as_dataframe()
    pipeline=PredictPipeline()
    result=pipeline.predict(pred_df)

    st.success(f" 🎯Predicted Math Score: **{result[0]:2f}**")
