# Import libraries
import shap
import pandas as pd
import numpy as np
import pickle
import sklearn
import warnings


from matplotlib import pyplot as plt
from shap import initjs

warnings.filterwarnings('ignore')
import streamlit as st
st.set_page_config(layout="wide")

if "input_features" in st.session_state:
    input_df_encoded1 = st.session_state["input_features"]
else:
    st.error("No input data found. Please go back and run the prediction first.")

def updatePage():
    st.rerun()

def filter_shap_values(shap_values, feature_names, data, threshold=0.083333): #1 month threshold
    mask = np.abs(shap_values) >= threshold
    filtered_shap_values = shap_values[mask]
    filtered_feature_names = [feature_names[i] for i in range(len(feature_names)) if mask[i]]
    filtered_data = data[mask]
    return filtered_shap_values, filtered_feature_names, filtered_data

import shap
import matplotlib.pyplot as plt
import numpy as np

tooltip_text = (
        "A SHAP (SHapley Additive exPlanations) waterfall plot visualizes the contribution of each feature to a prediction.\n\n"
        "- **Base Value**: The starting point of the plot, E[f(X)], represents the model's average prediction across all samples or a reference prediction.\n\n"
        "- **Feature Contributions**: Each step in the waterfall plot shows how individual features contribute to moving the prediction from the base value to the final predicted value.\n\n"
        "- **Direction of Impact**: Positive contributions (upward steps) indicate features that increase the model's prediction. Negative contributions (downward steps) indicate features that decrease the model's prediction.\n\n"
        "- **Cumulative Impact**: The cumulative effect of feature contributions leads to the final predicted value.\n\n"
        "- **Interpretation by Features**: Each waterfall plot is specific to a sample, showing how features influence the model's prediction for that specific value."
    )
st.title('Feature Analysis', help=tooltip_text)
st.divider()

with st.spinner('Analyzing features... Please wait'):
    shap_pickle = open('Streamlit/pages/shap_xgb.pkl', 'rb')
    shapmodel = pickle.load(shap_pickle)
    shap_pickle.close()
    shap_values = shapmodel.shap_values(input_df_encoded1)

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

    filtered_data = np.array(
        [input_df_encoded1.iloc[0][col] for col in filtered_feature_names if col not in all_names] +
        [
            next((col.split("_")[-1] for col in category if input_df_encoded1.iloc[0][col] == 1), "None")
            for category in [
            country_columns, state_columns, NAICS_columns, pay_level_columns,
            user_education_columns, job_experience_columns, class_columns,
            job_education_columns, job_training_columns, job_foreign_edu_columns,
            job_layoff_columns, job_req_exp_columns
        ]
        ]
    )

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
        "PW_AMOUNT_9089": "Payment Amount",
    }

    # Apply the mapping to the filtered feature names
    filtered_feature_names = [feature_name_mapping.get(col, col) for col in filtered_feature_names]

    # Filter out any values that have less than 1 month of effect on prediction
    filtered_shap_values, filtered_feature_names, filtered_data = filter_shap_values(
        filtered_shap_values,
        filtered_feature_names,
        filtered_data
    )


    def wrap_feature_names(feature_names, max_length=15):
        wrapped_names = []
        for name in feature_names:
            chunks = name.split(" ")
            wrapped_names.append("\n".join(chunks))
        return wrapped_names


    # Example usage:
    wrapped_feature_names = wrap_feature_names(filtered_feature_names)

    # Create a new SHAP Explanation object
    expl = shap.Explanation(
      values=filtered_shap_values,
      base_values=shapmodel.expected_value,
      data=filtered_data,
      feature_names=wrapped_feature_names
    )

    def show_force_plot(expl):
        shap.plots.force(expl, matplotlib=True)
        st.pyplot(plt.gcf())

    show_force_plot(expl)

    import numpy as np
    import matplotlib.pyplot as plt
    import streamlit as st


    def plot_custom_waterfall(shap_values, feature_values, feature_names, base_value, final_prediction):
        # Sort features by absolute SHAP value impact
        sorted_indices = np.argsort(np.abs(shap_values))[::-1]
        shap_values = np.array(shap_values)[sorted_indices]
        feature_values = np.array(feature_values)[sorted_indices]
        feature_names = np.array(feature_names)[sorted_indices]

        cumulative_values = np.concatenate([[base_value], base_value + np.cumsum(shap_values)])

        colors = ['red' if val > 0 else 'green' for val in shap_values]

        fig, ax = plt.subplots(figsize=(10, 6))

        for i in range(len(shap_values)):
            ax.barh(i, shap_values[i], left=cumulative_values[i], color=colors[i], alpha=0.8)
            ax.text(cumulative_values[i] + shap_values[i] / 2, i,
                    f"{feature_names[i]} = {feature_values[i]} ({shap_values[i]:.3f} years)",
                    ha='center', va='center', fontsize=10, color='black', fontweight='bold')

        ax.axvline(base_value, color='black', linestyle='--', label=f"Base Value: {base_value:.3f} years")
        ax.axvline(final_prediction, color='blue', linestyle='-', label=f"Final Prediction: {final_prediction:.3f} years")

        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels([])
        ax.set_xlabel("SHAP Value Contribution to Prediction")
        ax.set_xlim(0, 15)
        ax.set_title("Custom SHAP Waterfall Plot")
        ax.legend()
        plt.grid(axis='x', linestyle='--', alpha=0.5)

        st.pyplot(fig)


    import shap
    import streamlit as st
    import matplotlib.pyplot as plt


    plot_custom_waterfall(
        shap_values=filtered_shap_values,
        feature_values=filtered_data,
        feature_names=filtered_feature_names,
        base_value=shapmodel.expected_value,
        final_prediction=st.session_state.prediction
    )

    import streamlit as st
    import plotly.graph_objects as go


    def plot_gauge_chart(base_value, final_prediction):
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=final_prediction,
            delta={'reference': base_value, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge={'axis': {'range': [min(0, base_value - 0.5), max(1, final_prediction + 0.5)]}},
            title={"text": "Final Prediction Impact"}
        ))
        st.plotly_chart(fig)

        fig.add_annotation(
            text=f"Baseline: hi",
            x=0.5, y=-0.2, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14)
        )

        fig.add_annotation(
            text=f"Predicted: hi",
            x=0.5, y=-0.3, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14, color="blue")
        )

    print()

    def plot_feature_gauges(features, shap_values):
        for feature, shap_value in zip(features, shap_values):
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=shap_value,
                delta={'reference': shapmodel.expected_value, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                gauge={'axis': {'range': [0, 15]}},
                title={"text": feature}
            ))
            st.plotly_chart(fig)


    # Streamlit App
    st.title("Prediction Impact Gauge Chart")
    base_value = shapmodel.expected_value
    final_prediction = st.session_state['prediction']
    plot_gauge_chart(base_value, final_prediction)

    print(filtered_data)

    plot_feature_gauges(filtered_feature_names, filtered_shap_values)




