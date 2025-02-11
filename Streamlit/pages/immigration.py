# Import libraries
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import sklearn
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def decimal_to_years_months(decimal_years):
    years = int(decimal_years)  # Extract the whole number of years
    months = round((decimal_years - years) * 12)  # Convert remaining fraction to months
    if months == 12:
        return str(years + 1) + " years", "0 months"
    elif months == 1 and years == 1:
        months = str(months) + " month"
        years = str(years) + " year"
    elif years == 1 and months != 1:
        years = str(years) + " year"
        months = str(months) + " months"
    elif years != 1 and months == 1:
        years = str(years) + " years"
        months = str(months) + " month"
    else:
        months = str(months) + " months"
        years = str(years) + " years"

    return years, months

st.title('Machine Learning Application')
st.divider()
st.write('Upload your own file or use the following form to get started.')

# Reading the pickle file that we created before
ada_pickle = open('Streamlit/pages/adaboost_model.pkl','rb')
ada = pickle.load(ada_pickle)
ada_pickle.close()

# Option 1: Asking users to input their data as a file
immigration_file = st.file_uploader('Upload your own application data in csv format.')

# Display an example dataset and prompt the user 
# to submit the data in the required format.
st.write("Please ensure that your data adheres to this specific format:")

# Cache the dataframe so it's only loaded once
@st.cache_data
def load_data(filename):
  df = pd.read_csv(filename)
  return df

data_format = load_data('Data/Data Sets/fullData.csv')
st.dataframe(data_format.head(2), hide_index = True)

with st.form('user_inputs'): 
  st.markdown("<h1 style='text-align: center; color: grey;'>JOB/EMPLOYER INFO</h1>", unsafe_allow_html=True)
  # List of U.S. states
  us_states = [
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware',
    'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky',
    'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri',
    'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina',
    'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
    'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'
  ]
  us_states_uppercase = [state.upper() for state in us_states]
  JOB_INFO_WORK_STATE = st.selectbox('Employer State',options=['ALABAMA', 'ALASKA', 'ARIZONA', 'ARKANSAS', 'CALIFORNIA', 'COLORADO', 'CONNECTICUT', 'DELAWARE', 'DISTRICT OF COLUMBIA', 'FLORIDA', 'GEORGIA', 'GUAM', 'HAWAII', 'IDAHO', 'ILLINOIS', 'INDIANA', 'IOWA', 'KANSAS', 'KENTUCKY', 'LOUISIANA', 'MAINE', 'MARYLAND', 'MASSACHUSETTS', 'MASSACHUSETTS', 'MH', 'MICHIGAN', 'MINNESOTA', 'MISSISSIPPI', 'MISSOURI', 'MONTANA', 'NEBRASKA', 'NEVADA', 'NEW HAMPSHIRE', 'NEW JERSEY', 'NEW MEXICO', 'NEW YORK', 'NORTH CAROLINA', 'NORTH DAKOTA', 'NORTHERN MARIANA ISLANDS', 'OHIO', 'OKLAHOMA', 'OREGON', 'PENNSYLVANIA', 'PUERTO RICO', 'RHODE ISLAND', 'SOUTH CAROLINA', 'SOUTH DAKOTA', 'TENNESSEE', 'TEXAS', 'UTAH', 'VERMONT', 'VIRGIN ISLANDS', 'VIRGINIA', 'WASHINGTON', 'WEST VIRGINIA', 'WISCONSIN', 'WYOMING'])
  PW_LEVEL_9089 = st.selectbox('Prevailing Wage Level',options=['Level I','Level II','Level III','Level IV'])
  PW_AMOUNT_9089 = st.number_input("Prevailing Wage Amount (Annual salary)",min_value=0)
  NAICS = st.selectbox('Employer NAICS Code (If unknown, refer to FAQs page for NAICS Code table.)',options=['11','21','22','23','31','32','33','42','44','45','48','49','51','52','53',
                                                      '54','55','56','61','62','71','72','81','92'])
  JOB_INFO_EDUCATION = st.selectbox('Education level required by job:',options=["Master's", "Bachelor's", 'Doctorate', 'Other', 'High School', "Associate's"])
  JOB_INFO_EXPERIENCE = st.radio('Job experience required?',options=['Y','N'])
  JOB_INFO_EXPERIENCE_NUM_MONTHS = st.number_input("If experience is required, how many months? If none is required, enter '0'.",min_value=0)
  JOB_INFO_TRAINING = st.radio('Does the job provide training?',options=['Y','N'])
  JOB_INFO_FOREIGN_ED = st.radio('Is a foreign educational equivalent acceptable for the job?',options=['Y','N'])
  RI_LAYOFF_IN_PAST_SIX_MONTHS = st.radio('Has the employer had any layoffs in the occupation of intended employment within the past six months?', options=['Y','N'])
  EMPLOYER_NUM_EMPLOYEES = st.number_input('Employer number of employees:',min_value = 0)
  st.markdown("<h1 style='text-align: center; color: grey;'>USER INFO</h1>", unsafe_allow_html=True)
  COUNTRY_OF_CITIZENSHIP = st.selectbox('Country of Citizenship',options=['AFGHANISTAN', 'ALBANIA', 'ALGERIA', 'ANDORRA', 'ANGOLA', 
                                                                          'ANTIGUA AND BARBUDA', 'ARGENTINA', 'ARMENIA', 'ARUBA', 'AUSTRALIA', 
                                                                          'AUSTRIA', 'AZERBAIJAN', 'BAHAMAS', 'BAHRAIN', 'BANGLADESH', 'BARBADOS', 
                                                                          'BELARUS', 'BELGIUM', 'BELIZE', 'BENIN', 'BERMUDA', 'BHUTAN', 'BOLIVIA', 
                                                                          'BOSNIA AND HERZEGOVINA', 'BOTSWANA', 'BRAZIL', 'BRITISH VIRGIN ISLANDS', 
                                                                          'BRUNEI', 'BULGARIA', 'BURKINA FASO', 'BURMA (MYANMAR)', 'BURUNDI', 'CAMBODIA', 
                                                                          'CAMEROON', 'CANADA', 'CAPE VERDE', 'CAYMAN ISLANDS', 'CHAD', 'CHILE', 'CHINA', 
                                                                          'COLOMBIA', 'COMOROS', 'COSTA RICA', "COTE d'IVOIRE", 'CROATIA', 'CUBA', 'CURACAO', 'CYPRUS',
                                                                          'CZECH REPUBLIC', 'DEMOCRATIC REPUBLIC OF CONGO', 'DENMARK', 'DOMINICA', 
                                                                          'DOMINICAN REPUBLIC', 'ECUADOR', 'EGYPT', 'EL SALVADOR', 'ERITREA', 'ESTONIA', 
                                                                          'ETHIOPIA', 'FIJI', 'FINLAND', 'FRANCE', 'GABON', 'GAMBIA', 'GEORGIA', 'GERMANY', 
                                                                          'GHANA', 'GIBRALTAR', 'GREECE', 'GRENADA', 'GUATEMALA', 'GUINEA', 'GUINEA-BISSAU', 
                                                                          'GUYANA', 'HAITI', 'HONDURAS', 'HONG KONG', 'HUNGARY', 'ICELAND', 'INDIA', 'INDONESIA', 
                                                                          'IRAN', 'IRAQ', 'IRELAND', 'ISRAEL', 'ITALY', 'IVORY COAST', 'JAMAICA', 'JAPAN', 
                                                                          'JORDAN', 'KAZAKHSTAN', 'KENYA', 'KOSOVO', 'KUWAIT', 'KYRGYZSTAN', 'LAOS', 'LATVIA', 
                                                                          'LEBANON', 'LESOTHO', 'LIBERIA', 'LIBYA', 'LIECHTENSTEIN', 'LITHUANIA', 'LUXEMBOURG', 'MACAU',
                                                                          'MACEDONIA', 'MADAGASCAR', 'MALAWI', 'MALAYSIA', 'MALDIVES', 'MALI', 'MALTA', 
                                                                          'MARSHALL ISLANDS', 'MAURITANIA', 'MAURITIUS', 'MEXICO', 'MICRONESIA', 'MOLDOVA', 
                                                                          'MONACO', 'MONGOLIA', 'MONTENEGRO', 'MONTSERRAT', 'MOROCCO', 'MOZAMBIQUE', 'NAMIBIA', 
                                                                          'NEPAL', 'NETHERLANDS', 'NEW ZEALAND', 'NICARAGUA', 'NIGER', 'NIGERIA', 'NORWAY', 
                                                                          'OMAN', 'PAKISTAN', 'PALESTINE', 'PALESTINIAN TERRITORIES', 'PANAMA', 'PARAGUAY', 
                                                                          'PERU', 'PHILIPPINES', 'POLAND', 'PORTUGAL', 'QATAR', 'REPUBLIC OF CONGO', 'ROMANIA', 
                                                                          'RUSSIA', 'RWANDA', 'SAINT VINCENT AND THE GRENADINES', 'SAUDI ARABIA', 'SENEGAL', 
                                                                          'SERBIA', 'SERBIA AND MONTENEGRO', 'SEYCHELLES', 'SIERRA LEONE', 'SINGAPORE', 'SLOVAKIA', 
                                                                          'SLOVENIA', 'SOMALIA', 'SOUTH AFRICA', 'SOUTH KOREA', 'SOUTH SUDAN', 'SPAIN', 'SRI LANKA', 
                                                                          'ST KITTS AND NEVIS', 'ST LUCIA', 'ST VINCENT', 'SUDAN', 'SURINAME', 'SWAZILAND', 'SWEDEN', 
                                                                          'SWITZERLAND', 'SYRIA', 'TAIWAN', 'TAJIKISTAN', 'TANZANIA', 'THAILAND', 'TOGO', 
                                                                          'TRINIDAD AND TOBAGO', 'TUNISIA', 'TURKEY', 'TURKMENISTAN', 'TURKS AND CAICOS ISLANDS', 
                                                                          'UGANDA', 'UKRAINE', 'UNITED ARAB EMIRATES', 'UNITED KINGDOM', 'UNITED STATES OF AMERICA', 
                                                                          'URUGUAY', 'UZBEKISTAN', 'VENEZUELA', 'VIETNAM', 'YEMEN', 'ZAMBIA', 'ZIMBABWE'])
  CLASS_OF_ADMISSION = st.selectbox('Class of Admission',options=['H-1B', 'L-1', 'F-1', 'H-4', 'Not in USA', 'O-1', 'J-2', 'J-1', 'B-2', 'Parolee', 'L-2', 'E-2', 'TN',
                                                                   'B-1', 'G-4', 'P-1', 'H-1B1', 'EWI', 'F-2', 'E-3', 'H-2B', 'TPS', 'C-1', 'H-1A', 'I', 'P-3', 'H-3', 'A1/A2', 
                                                                   'E-1', 'H-2A', 'R-1', 'C-3', 'VWT', 'M-1', 'G-5', 'TD', 'V-2', 'T-2', 'D-1', 'T-1', 'CW-1', 'G-1', 'A-3', 'O-2', 
                                                                   'N', 'P-4', 'V-1', 'M-2', 'D-2', 'R-2', 'VWB', 'O-3', 'K-4', 'U-1', 'Q'])
  FOREIGN_WORKER_INFO_EDUCATION = st.selectbox('Education Level',options=['High School',"Associate's","Bachelor's","Master's",'Doctorate','Other'])
  FW_INFO_YR_REL_EDU_COMPLETED = st.number_input('What year was your highest education completed?', min_value = 1925)
  FW_INFO_REQ_EXPERIENCE = st.radio("If experience is required, do you have the required experience? (If none is required, select 'A'.)",options=['Y','N','A'])
  ml_model = 'AdaBoost'
  submit = st.form_submit_button()

if immigration_file is None and submit:
  original_df = pd.read_csv('Data/Data Sets/fullData.csv') # Original data to create ML model
  original_df.rename(columns={'2_NAICS': 'NAICS'}, inplace=True)
  original_df['NAICS'] = original_df['NAICS'].astype(str)
  original_df['JOB_INFO_WORK_STATE'].replace({'MASSACHUSETTES': 'MASSACHUSETTS', 'MH': 'MARSHALL ISLANDS'}, inplace=True)
  original_df['COUNTRY_OF_CITIZENSHIP'].replace({'IVORY COAST': "COTE d'IVOIRE", 'NETHERLANDS ANTILLES': 'NETHERLANDS'},
                                       inplace=True)
  original_df = original_df[~original_df['COUNTRY_OF_CITIZENSHIP'].isin(['SOVIET UNION', 'UNITED STATES OF AMERICA', 'KIRIBATI', 'SAO TOME AND PRINCIPE', 'SINT MAARTEN'])]
  original_df = original_df[~original_df['JOB_INFO_WORK_STATE'].isin(['FEDERATED STATES OF MICRONESIA', 'MARSHALL ISLANDS'])]
  original_df.dropna(axis=0, how='any', inplace=True)
  original_df.drop(['DECISION_DATE', 'PW_UNIT_OF_PAY_9089', 'CASE_RECEIVED_DATE'], axis=1, inplace=True)


  # Concatenate two dataframes together along rows (axis = 0)
  combined_df1 = original_df.copy()

  # Define the new row as a dictionary to ensure it matches the column names
  new_row = {
      'NAICS': NAICS,
      'PW_LEVEL_9089': PW_LEVEL_9089,
      'PW_AMOUNT_9089': PW_AMOUNT_9089,
      'JOB_INFO_WORK_STATE': JOB_INFO_WORK_STATE,
      'COUNTRY_OF_CITIZENSHIP': COUNTRY_OF_CITIZENSHIP,
      'CLASS_OF_ADMISSION': CLASS_OF_ADMISSION,
      'EMPLOYER_NUM_EMPLOYEES': EMPLOYER_NUM_EMPLOYEES,
      'JOB_INFO_EDUCATION': JOB_INFO_EDUCATION,
      'JOB_INFO_TRAINING': JOB_INFO_TRAINING,
      'JOB_INFO_EXPERIENCE': JOB_INFO_EXPERIENCE,
      'JOB_INFO_EXPERIENCE_NUM_MONTHS': JOB_INFO_EXPERIENCE_NUM_MONTHS,
      'JOB_INFO_FOREIGN_ED': JOB_INFO_FOREIGN_ED,
      'RI_LAYOFF_IN_PAST_SIX_MONTHS': RI_LAYOFF_IN_PAST_SIX_MONTHS,
      'FOREIGN_WORKER_INFO_EDUCATION': FOREIGN_WORKER_INFO_EDUCATION,
      'FW_INFO_YR_REL_EDU_COMPLETED': FW_INFO_YR_REL_EDU_COMPLETED,
      'FW_INFO_REQ_EXPERIENCE': FW_INFO_REQ_EXPERIENCE
  }

  # Append the new row to the DataFrame
  new_row_df = pd.DataFrame([new_row])
  # Concatenate the new row to the original DataFrame
  combined_df1 = pd.concat([combined_df1, new_row_df], ignore_index=True)
  # Number of rows in original dataframe
  original_rows1 = original_df.shape[0]
  # Create dummies for the combined dataframe
  combined_df1['NAICS'] = combined_df1['NAICS'].astype(str)
  cat_var = ['NAICS', 'PW_LEVEL_9089', 'JOB_INFO_WORK_STATE', 'COUNTRY_OF_CITIZENSHIP', 'FOREIGN_WORKER_INFO_EDUCATION',
             'JOB_INFO_EXPERIENCE', 'CLASS_OF_ADMISSION', 'JOB_INFO_EDUCATION', 'JOB_INFO_TRAINING',
             'JOB_INFO_FOREIGN_ED', 'RI_LAYOFF_IN_PAST_SIX_MONTHS', 'FW_INFO_REQ_EXPERIENCE']

  combined_df_encoded1 = pd.get_dummies(combined_df1,columns = cat_var)

  # Split data into original and user input dataframes using row index
  original_df_encoded1 = combined_df_encoded1[:original_rows1]
  input_df_encoded1 = combined_df_encoded1.tail(1)

  # Using predict() with new data provided by the user
  new_prediction3 = ada.predict(input_df_encoded1)

  # Probabilities
  def create_probability_df(probabilities, class_names):
      # Create the DataFrame
      prob_df = pd.DataFrame(probabilities, columns=class_names)
      # Sort each row by descending probability
      prob_df = prob_df.apply(lambda x: x.sort_values(ascending=False), axis=1)
      # Function to highlight the maximum probability in each row
      def highlight_max(s):
          is_max = s == s.max()
          return ['background-color: limegreen' if v else '' for v in is_max]
      # Apply the highlighting
      prob_df = prob_df.style.apply(highlight_max, axis=1)
      return prob_df

  st.subheader("Predicted Waiting Time")

  if ml_model == 'AdaBoost':
      mapie_pickle = open('Streamlit/pages/mapie.pkl', 'rb')
      mapie = pickle.load(mapie_pickle)
      mapie_pickle.close()

      # For regression, you would use the prediction directly
      new_prediction3 = ada.predict(input_df_encoded1)[0] if ada.predict(input_df_encoded1) else 'No prediction'
      y_pred, y_pis = mapie.predict(input_df_encoded1, alpha=0.1)

      if y_pis[0,0,0] <= 0:
          low_year, low_month = "0 years", "0 months"
      else:
          low_year, low_month = decimal_to_years_months(y_pis[0,0,0])
      high_year, high_month = decimal_to_years_months(y_pis[0,1,0])

      # Display the prediction value
      prediction_text = (
          f'Our model predicts with 90% confidence that your waiting time will be between {low_year}, {low_month} and {high_year}, {high_month}.')

      st.markdown(prediction_text, unsafe_allow_html=True)

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



elif submit:
    original_df = pd.read_csv('Data/Data Sets/fullData.csv')  # Original data to create ML model
    original_df.rename(columns={'2_NAICS': 'NAICS'}, inplace=True)
    original_df['NAICS'] = original_df['NAICS'].astype(str)
    original_df['JOB_INFO_WORK_STATE'].replace({'MASSACHUSETTES': 'MASSACHUSETTS', 'MH': 'MARSHALL ISLANDS'},
                                               inplace=True)
    original_df['COUNTRY_OF_CITIZENSHIP'].replace(
        {'IVORY COAST': "COTE d'IVOIRE", 'NETHERLANDS ANTILLES': 'NETHERLANDS'},
        inplace=True)
    original_df = original_df[~original_df['COUNTRY_OF_CITIZENSHIP'].isin(['SOVIET UNION', 'UNITED STATES OF AMERICA'])]
    original_df.dropna(axis=0, how='any', inplace=True)
    original_df.drop(['DECISION_DATE', 'PW_UNIT_OF_PAY_9089', 'CASE_RECEIVED_DATE'], axis=1, inplace=True)
    user_df = pd.read_csv(immigration_file)
    first_row = user_df.iloc[0]

    # Convert the Series to a dictionary
    user_dict = first_row.to_dict()

    # Create variables with the same names as the columns
    for column_name, value in user_dict.items():
        exec(f"{column_name} = {repr(value)}")

    # Concatenate two dataframes together along rows (axis = 0)
    combined_df1 = original_df.copy()
    combined_df1.loc[len(combined_df1)] = [NAICS,PW_LEVEL_9089,PW_AMOUNT_9089,JOB_INFO_WORK_STATE,COUNTRY_OF_CITIZENSHIP,CLASS_OF_ADMISSION,EMPLOYER_NUM_EMPLOYEES,JOB_INFO_EDUCATION,
                                           JOB_INFO_TRAINING,JOB_INFO_EXPERIENCE,JOB_INFO_EXPERIENCE_NUM_MONTHS,JOB_INFO_FOREIGN_ED,RI_LAYOFF_IN_PAST_SIX_MONTHS,
                                           FOREIGN_WORKER_INFO_EDUCATION,FW_INFO_YR_REL_EDU_COMPLETED,FW_INFO_REQ_EXPERIENCE]

    # Number of rows in original dataframe
    original_rows1 = original_df.shape[0]
    # Create dummies for the combined dataframe
    combined_df1['NAICS'] = combined_df1['NAICS'].astype(str)
    combined_df_encoded1 = pd.get_dummies(combined_df1,columns=[
    'NAICS', 'PW_LEVEL_9089', 'JOB_INFO_WORK_STATE', 'COUNTRY_OF_CITIZENSHIP', 
    'CLASS_OF_ADMISSION', 'EMPLOYER_NUM_EMPLOYEES', 'JOB_INFO_EDUCATION', 
    'JOB_INFO_TRAINING', 'JOB_INFO_EXPERIENCE', 'JOB_INFO_EXPERIENCE_NUM_MONTHS', 
    'JOB_INFO_FOREIGN_ED', 'RI_LAYOFF_IN_PAST_SIX_MONTHS', 
    'FOREIGN_WORKER_INFO_EDUCATION', 'FW_INFO_YR_REL_EDU_COMPLETED', 
    'FW_INFO_REQ_EXPERIENCE'], dummy_na=False)


    # Split data into original and user input dataframes using row index
    original_df_encoded1 = combined_df_encoded1[:original_rows1]
    input_df_encoded1 = combined_df_encoded1.tail(1)

    # Using predict() with new data provided by the user
    new_prediction3 = ada.predict(input_df_encoded1)

    # Probabilities
    def create_probability_df(probabilities, class_names):
        # Create the DataFrame
        prob_df = pd.DataFrame(probabilities, columns=class_names)
        # Sort each row by descending probability
        prob_df = prob_df.apply(lambda x: x.sort_values(ascending=False), axis=1)
        # Function to highlight the maximum probability in each row
        def highlight_max(s):
            is_max = s == s.max()
            return ['background-color: limegreen' if v else '' for v in is_max]
        # Apply the highlighting
        prob_df = prob_df.style.apply(highlight_max, axis=1)
        return prob_df
    
    st.subheader("Predicted Waiting Time") 


    if ml_model == 'AdaBoost':
        new_prediction3 = new_prediction3[0] if new_prediction3 else 'No prediction'
        probabilities = ada.predict_proba(input_df_encoded1)
        max_prob = max(probabilities[0]) * 100
        class_names = ada.classes_
        prob_df_ada = create_probability_df(probabilities, class_names)
        prediction_text = (
          f'<span style="color:red; font-weight: bold;">{new_prediction3}</span> '
          f'with <span style="color:red; font-weight: bold;">{max_prob:.2f}%</span> probability.')
        st.markdown(prediction_text, unsafe_allow_html=True)
        st.markdown('Prediction Probabilities')
        st.dataframe(prob_df_ada, hide_index = True)

