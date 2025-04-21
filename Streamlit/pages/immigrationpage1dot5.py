# Import libraries
import shap
import pandas as pd
import numpy as np
import pickle
import sklearn
import warnings

warnings.filterwarnings('ignore')
import streamlit as st
st.set_page_config(layout="wide")
input_df_encoded1 = None

if "input_features" in st.session_state:
    input_df_encoded1 = st.session_state["input_features"]
else:
    st.error("No input data found. Please go to the 'Enter Your Info' page on the left hand side and run the prediction first.")

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
if input_df_encoded1 is not None:
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


        def format_years_months(decimal_years):
            """Converts decimal years into a 'X years, Y months' string."""
            if pd.isna(decimal_years) or decimal_years == 0:
                return "0 years, 0 months"

            sign = "-" if decimal_years < 0 else "+"
            abs_years = abs(decimal_years)
            years = int(abs_years)
            remaining_decimal = abs_years - years
            months = round(remaining_decimal * 12)

            if months == 12:
                years += 1
                months = 0

            year_str = f"{years} year" if years == 1 else f"{years} years"
            month_str = f"{months} month" if months == 1 else f"{months} months"

            if years > 0 and months > 0:
                return f"{sign} {year_str}, {month_str}"
            elif years > 0:
                return f"{sign} {year_str}"
            elif months > 0:
                return f"{sign} {month_str}"
            elif abs_years > 0:
                return f"{sign} <1 month"
            else:
                return "0 years, 0 months"


        def style_shap_value(shap_value, max_abs_shap):
            """
            Applies background color styling based on SHAP value sign and magnitude,
            adjusting text color for better contrast.
            """
            if max_abs_shap == 0 or pd.isna(shap_value) or shap_value == 0:
                return ''

            alpha = min(1.0, 0.15 + 0.70 * (abs(shap_value) / max_abs_shap))

            text_color = '#333333'

            if shap_value > 0:
                background_color = f'rgba(255, 180, 180, {alpha:.2f})'
            else:  # shap_value < 0
                background_color = f'rgba(180, 255, 180, {alpha:.2f})'

            return f'background-color: {background_color}; color: {text_color};'


        def custom_table(shap_values, feature_values, feature_names, base_value, final_prediction):
            """
            Displays a styled table in Streamlit showing feature contributions.

            Args:
                shap_values (list or np.array): SHAP values for a single prediction.
                feature_values (list or np.array): Corresponding feature values.
                feature_names (list or np.array): Names of the features.
                base_value (float): The base value (expected value) from SHAP.
                final_prediction (float): The final prediction output by the model.
            """
            if not isinstance(shap_values, np.ndarray):
                shap_values = np.array(shap_values)
            if not isinstance(feature_values, np.ndarray):
                feature_values = np.array(feature_values)
            if not isinstance(feature_names, np.ndarray):
                feature_names = np.array(feature_names)

            sorted_indices = np.argsort(np.abs(shap_values))[::-1]
            shap_values_sorted = shap_values[sorted_indices]
            feature_values_sorted = feature_values[sorted_indices]
            feature_names_sorted = feature_names[sorted_indices]

            df = pd.DataFrame({
                'Feature Name': feature_names_sorted,
                'Feature Value': feature_values_sorted,
                'SHAP Value (Decimal)': shap_values_sorted
            })

            df['Impact on Prediction (Years, Months)'] = df['SHAP Value (Decimal)'].apply(format_years_months)

            max_abs_shap = df['SHAP Value (Decimal)'].abs().max()

            def apply_styling(row):
                style_for_impact_col = style_shap_value(row['SHAP Value (Decimal)'], max_abs_shap)
                return ['', '', '', style_for_impact_col]
            styled_df_object = df.style.apply(apply_styling, axis=1)

            st.subheader("Feature Contribution Details")

            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Average Prediction (Base Value)", value=f"{base_value:.2f} years")
            with col2:
                st.metric(label="Final Prediction", value=f"{final_prediction:.2f} years",
                          delta=f"{(final_prediction - base_value):.2f} years")

            st.write("The table below shows how each feature's value pushes the prediction away from the average.")
            st.caption(
                "Features are sorted by the magnitude of their impact. Red indicates the feature increased the predicted time, Green indicates it decreased the time.")
            st.dataframe(
                styled_df_object.hide(axis="columns", subset=['SHAP Value (Decimal)']),
                hide_index=True,
                use_container_width=True
            )


        custom_table(
            shap_values=filtered_shap_values,
            feature_values=filtered_data,
            feature_names=filtered_feature_names,
            base_value=shapmodel.expected_value,
            final_prediction=st.session_state.prediction
        )
        with st.expander("Show Feature Contribution Details (Force Plot)"):
            st.write("""
                This plot illustrates the contribution of each feature to the final prediction
                compared to the baseline (average prediction). Features pushing the prediction
                higher are shown in red, and those pushing it lower are in blue.
            """)

            show_force_plot(expl)