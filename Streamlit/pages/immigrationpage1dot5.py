# Import libraries
import shap
import pandas as pd
import numpy as np
import pickle
import sklearn
import warnings

from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')
import streamlit as st

if "input_features" in st.session_state:
    input_df_encoded1 = st.session_state["input_features"]
else:
    st.error("No input data found. Please go back and run the prediction first.")

def updatePage():
    st.rerun()


st.title('Feature Analysis')
st.divider()

with st.spinner('Analyzing features... Please wait'):
    shap_pickle = open('Streamlit/pages/shap_xgb.pkl', 'rb')
    shapmodel = pickle.load(shap_pickle)
    shap_pickle.close()
    shap_values = shapmodel.shap_values(input_df_encoded1)

    # Display the SHAP values and feature analysis
    st.subheader('SHAP Feature Analysis')
    st.markdown(
      "A SHAP (SHapley Additive exPlanations) waterfall plot visualizes the contribution of each feature to a prediction. Here's a concise explanation of what the values on a SHAP waterfall plot mean:")
    st.markdown(
      "- Base Value: The starting point of the plot, E[f(X)], represents the model's average prediction across all samples or a reference prediction.")
    st.markdown(
      "- Feature Contributions: Each step in the waterfall plot shows how individual features contribute to moving the prediction from the base value to the final predicted value.")
    st.markdown(
      "- Direction of Impact: <span style='color:#ff0051'><b>Positive</b></span> contributions (upward steps) indicate features that <span style='color:#ff0051'><b>increase the model's prediction</b></span>. <span style='color:#008bfb'><b>Negative</b></span> contributions (downward steps) indicate features that <span style='color:#008bfb'><b>decrease the model's prediction</b></span>.",
      unsafe_allow_html=True)
    st.markdown(
      "- Cumulative Impact: The cumulative effect of feature contributions leads to the final predicted value.")
    st.markdown(
      "- Interpretation by Features: Each waterfall plot is specific to a sample, showing how features influence the model's prediction for that specific value.")
    # Identify country-related features
    country_columns = [col for col in input_df_encoded1.columns if col.startswith("COUNTRY_OF_CITIZENSHIP_")]
    state_columns = [col for col in input_df_encoded1.columns if col.startswith("JOB_INFO_WORK_STATE_")]
    NAICS_columns = [col for col in input_df_encoded1.columns if col.startswith("NAICS_")]
    pay_level_columns = [col for col in input_df_encoded1.columns if col.startswith("PW_LEVEL_9089_")]
    user_education_columns = [col for col in input_df_encoded1.columns if col.startswith("FOREIGN_WORKER_INFO_EDUCATION_")]
    job_experience_columns = [col for col in input_df_encoded1.columns if col.startswith("JOB_INFO_EXPERIENCE_")]
    class_columns = [col for col in input_df_encoded1.columns if col.startswith("CLASS_OF_ADMISSION_")]
    job_education_columns = [col for col in input_df_encoded1.columns if col.startswith("JOB_INFO_EDUCATION_")]
    job_training_columns = [col for col in input_df_encoded1.columns if col.startswith("JOB_INFO_TRAINING_")]
    job_foreign_edu_columns = [col for col in input_df_encoded1.columns if col.startswith("JOB_INFO_FOREIGN_ED_")]
    job_layoff_columns = [col for col in input_df_encoded1.columns if col.startswith("RI_LAYOFF_IN_PAST_SIX_MONTHS_")]
    job_req_exp_columns = [col for col in input_df_encoded1.columns if col.startswith("FW_INFO_REQ_EXPERIENCE_")]

    all_columns = country_columns + state_columns + NAICS_columns + pay_level_columns + user_education_columns + job_experience_columns + class_columns + job_education_columns + job_training_columns + job_foreign_edu_columns + job_layoff_columns + job_req_exp_columns

    # Sum SHAP values for all country-related columns
    country_shap_score = sum(shap_values[0][input_df_encoded1.columns.get_loc(col)] for col in country_columns)
    state_shap_score = sum(shap_values[0][input_df_encoded1.columns.get_loc(col)] for col in state_columns)
    NAICS_shap_score = sum(shap_values[0][input_df_encoded1.columns.get_loc(col)] for col in NAICS_columns)
    pay_level_shap_score = sum(shap_values[0][input_df_encoded1.columns.get_loc(col)] for col in pay_level_columns)
    user_education_shap_score = sum(shap_values[0][input_df_encoded1.columns.get_loc(col)] for col in user_education_columns)
    job_experience_shap_score = sum(shap_values[0][input_df_encoded1.columns.get_loc(col)] for col in job_experience_columns)
    class_shap_score = sum(shap_values[0][input_df_encoded1.columns.get_loc(col)] for col in class_columns)
    job_education_shap_score = sum(shap_values[0][input_df_encoded1.columns.get_loc(col)] for col in job_education_columns)
    job_training_shap_score = sum(shap_values[0][input_df_encoded1.columns.get_loc(col)] for col in job_training_columns)
    job_foreign_edu_shap_score = sum(shap_values[0][input_df_encoded1.columns.get_loc(col)] for col in job_foreign_edu_columns)
    job_layoff_shap_score = sum(shap_values[0][input_df_encoded1.columns.get_loc(col)] for col in job_layoff_columns)
    job_req_exp_shap_score = sum(shap_values[0][input_df_encoded1.columns.get_loc(col)] for col in job_req_exp_columns)

    all_names = ["COUNTRY_OF_CITIZENSHIP_TOTAL", "JOB_INFO_WORK_STATE_TOTAL", "NAICS_TOTAL", "PW_LEVEL_9089_TOTAL", "FOREIGN_WORKER_INFO_EDUCATION_TOTAL", "JOB_INFO_EXPERIENCE_TOTAL", "CLASS_OF_ADMISSION_TOTAL", "JOB_INFO_EDUCATION_TOTAL", "JOB_INFO_TRAINING_TOTAL", "JOB_INFO_FOREIGN_ED_TOTAL", "RI_LAYOFF_IN_PAST_SIX_MONTHS_TOTAL", "FW_INFO_REQ_EXPERIENCE_TOTAL"]

    # Create a new feature list excluding individual country columns
    filtered_feature_names = ([col for col in input_df_encoded1.columns if col not in all_columns] +
                            all_names)

    # Create new data and SHAP values arrays
    filtered_shap_values = np.array(
      [shap_values[0][input_df_encoded1.columns.get_loc(col)] for col in filtered_feature_names if
       col not in all_names] + [country_shap_score] + [state_shap_score] + [NAICS_shap_score] + [pay_level_shap_score] + [user_education_shap_score] + [job_experience_shap_score] + [class_shap_score] + [job_education_shap_score] + [job_training_shap_score] + [job_foreign_edu_shap_score] + [job_layoff_shap_score] + [job_req_exp_shap_score])

    filtered_data = np.array([input_df_encoded1.iloc[0][col] for col in filtered_feature_names if
                            col not in all_names] + [
                               sum(input_df_encoded1.iloc[0][col] for col in country_columns)] + [
                               sum(input_df_encoded1.iloc[0][col] for col in state_columns)] + [
                               sum(input_df_encoded1.iloc[0][col] for col in NAICS_columns)] + [
                               sum(input_df_encoded1.iloc[0][col] for col in pay_level_columns)] + [
                               sum(input_df_encoded1.iloc[0][col] for col in user_education_columns)] + [
                               sum(input_df_encoded1.iloc[0][col] for col in job_experience_columns)] + [
                               sum(input_df_encoded1.iloc[0][col] for col in class_columns)] + [
                               sum(input_df_encoded1.iloc[0][col] for col in job_education_columns)] + [
                               sum(input_df_encoded1.iloc[0][col] for col in job_training_columns)] + [
                               sum(input_df_encoded1.iloc[0][col] for col in job_foreign_edu_columns)] + [
                               sum(input_df_encoded1.iloc[0][col] for col in job_layoff_columns)] + [
                               sum(input_df_encoded1.iloc[0][col] for col in job_req_exp_columns)])
    all_names = ["", "", "", "", "", "", "", "", "", "FW_INFO_REQ_EXPERIENCE_TOTAL"]

    # Mapping of feature names to more user-friendly names
    feature_name_mapping = {

        "COUNTRY_OF_CITIZENSHIP_TOTAL": "Country of Citizenship",
        "JOB_INFO_WORK_STATE_TOTAL": "Job State",
        "NAICS_TOTAL": "NAICS Code",
        "PW_LEVEL_9089_TOTAL": "Wage Level",
        "FOREIGN_WORKER_INFO_EDUCATION_TOTAL": "Worker Education Level",
        "JOB_INFO_EXPERIENCE_TOTAL": "Job Experience Required?",
        "CLASS_OF_ADMISSION_TOTAL": "Class of Admission",
        "JOB_INFO_EDUCATION_TOTAL": "Required Education",
        "JOB_INFO_TRAINING_TOTAL": "Required Training?",
        "JOB_INFO_FOREIGN_ED_TOTAL": "Foreign Education Accepted?",
        "RI_LAYOFF_IN_PAST_SIX_MONTHS_TOTAL": "Layoff in Last 6 Months",
        "FW_INFO_REQ_EXPERIENCE_TOTAL": "Do You Have the Experience Required?",
        "EMPLOYER_NUM_EMPLOYEES": "Number of Employees",
        "FW_INFO_YR_REL_EDU_COMPLETED": "Year Education Completed",
    }

    # Apply the mapping to the filtered feature names
    filtered_feature_names = [feature_name_mapping.get(col, col) for col in filtered_feature_names]


    # Create a new SHAP Explanation object
    expl = shap.Explanation(
      values=filtered_shap_values,
      base_values=shapmodel.expected_value,
      data=filtered_data,
      feature_names=filtered_feature_names
    )

    # Create the waterfall plot
    tab = st.empty()
    with tab:
        # Adjust figure size using matplotlib
        shap.plots.waterfall(expl, show=False, max_display=20)
        st.pyplot()


