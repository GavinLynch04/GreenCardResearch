# Import libraries
import shap
import pandas as pd
import numpy as np
import pickle
import sklearn
import warnings
warnings.filterwarnings('ignore')
import streamlit as st

if "input_df_encoded1" in st.session_state:
    input_df_encoded1 = st.session_state["input_df_encoded1"]
else:
    st.error("No input data found. Please go back and run the prediction first.")


st.title('Feature Analysis')
st.divider()
'''FIGURE OUT HOW TO GET INPUT TO THIS FILE'''
shap_pickle = open('Streamlit/pages/shap.pkl', 'rb')
shapmodel = pickle.load(shap_pickle)
shap_pickle.close()
print("Loaded Explainer")
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

# Sum SHAP values for all country-related columns
country_shap_score = sum(shap_values[0][input_df_encoded1.columns.get_loc(col)] for col in country_columns)
state_shap_score = sum(shap_values[0][input_df_encoded1.columns.get_loc(col)] for col in state_columns)

# Create a new feature list excluding individual country columns
filtered_feature_names = ([col for col in input_df_encoded1.columns if col not in country_columns and col not in state_columns] +
                        ["COUNTRY_OF_CITIZENSHIP_TOTAL"] + ["JOB_INFO_WORK_STATE_TOTAL"])

# Create new data and SHAP values arrays
filtered_shap_values = np.array(
  [shap_values[0][input_df_encoded1.columns.get_loc(col)] for col in filtered_feature_names if
   col != "COUNTRY_OF_CITIZENSHIP_TOTAL" and col != "JOB_INFO_WORK_STATE_TOTAL"] + [country_shap_score] + [state_shap_score])
filtered_data = np.array([input_df_encoded1.iloc[0][col] for col in filtered_feature_names if
                        col != "COUNTRY_OF_CITIZENSHIP_TOTAL" and col != "JOB_INFO_WORK_STATE_TOTAL"] + [
                           sum(input_df_encoded1.iloc[0][col] for col in country_columns)] + [
                           sum(input_df_encoded1.iloc[0][col] for col in state_columns)])

# Create a new SHAP Explanation object
expl = shap.Explanation(
  values=filtered_shap_values,
  base_values=shapmodel.expected_value,
  data=filtered_data,
  feature_names=filtered_feature_names
)

# Display the waterfall plot
tab = st.empty()
with tab:
  shap.plots.waterfall(expl, show=False)
  st.pyplot()


