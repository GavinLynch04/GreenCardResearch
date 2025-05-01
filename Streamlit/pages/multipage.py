import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import warnings

from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')


# --------------------
# Helper Functions
# --------------------
def load_models():
    with open('Streamlit/pages/xgb_model.pkl', 'rb') as f:
        xgb = pickle.load(f)
    with open('Streamlit/pages/mapieXGB.pkl', 'rb') as f:
        mapie = pickle.load(f)
    with open('Streamlit/pages/shap_xgb.pkl', 'rb') as f:
        shap_model = pickle.load(f)
    return xgb, mapie, shap_model


def decimal_to_years_months(decimal_years):
    years = int(decimal_years)
    months = round((decimal_years - years) * 12)
    if months == 12:
        return str(years + 1) + " years", "0 months"
    elif months == 1 and years == 1:
        return str(years) + " year", str(months) + " month"
    elif years == 1 and months != 1:
        return str(years) + " year", str(months) + " months"
    elif years != 1 and months == 1:
        return str(years) + " years", str(months) + " month"
    else:
        return str(years) + " years", str(months) + " months"


# --------------------
# Page Functions
# --------------------
def job_info_page1():
    st.title("Job/Employer Information")
    JOB_INFO_WORK_STATE = st.selectbox('Employer State',
                                       options=['ALABAMA', 'ALASKA', 'ARIZONA', 'ARKANSAS', 'CALIFORNIA', 'COLORADO',
                                                'CONNECTICUT', 'DELAWARE', 'DISTRICT OF COLUMBIA', 'FLORIDA', 'GEORGIA',
                                                'GUAM', 'HAWAII', 'IDAHO', 'ILLINOIS', 'INDIANA', 'IOWA', 'KANSAS',
                                                'KENTUCKY', 'LOUISIANA', 'MAINE', 'MARYLAND', 'MASSACHUSETTS',
                                                'MASSACHUSETTS', 'MH', 'MICHIGAN', 'MINNESOTA', 'MISSISSIPPI',
                                                'MISSOURI', 'MONTANA', 'NEBRASKA', 'NEVADA', 'NEW HAMPSHIRE',
                                                'NEW JERSEY', 'NEW MEXICO', 'NEW YORK', 'NORTH CAROLINA',
                                                'NORTH DAKOTA', 'NORTHERN MARIANA ISLANDS', 'OHIO', 'OKLAHOMA',
                                                'OREGON', 'PENNSYLVANIA', 'PUERTO RICO', 'RHODE ISLAND',
                                                'SOUTH CAROLINA', 'SOUTH DAKOTA', 'TENNESSEE', 'TEXAS', 'UTAH',
                                                'VERMONT', 'VIRGIN ISLANDS', 'VIRGINIA', 'WASHINGTON', 'WEST VIRGINIA',
                                                'WISCONSIN', 'WYOMING'])
    PW_LEVEL_9089 = st.selectbox('Prevailing Wage Level', options=['Level I', 'Level II', 'Level III', 'Level IV'])
    PW_AMOUNT_9089 = st.number_input("Prevailing Wage Amount (Annual salary)", min_value=0)
    NAICS_MAP = {
        "11": "Agriculture, Forestry, Fishing and Hunting",
        "21": "Mining, Quarrying, and Oil and Gas Extraction",
        "22": "Utilities",
        "23": "Construction",
        "31": "Manufacturing",
        "32": "Manufacturing",
        "33": "Manufacturing",
        "42": "Wholesale Trade",
        "44": "Retail Trade",
        "45": "Retail Trade",
        "48": "Transportation and Warehousing",
        "49": "Transportation and Warehousing",
        "51": "Information",
        "52": "Finance and Insurance",
        "53": "Real Estate and Rental and Leasing",
        "54": "Professional, Scientific, and Technical Services",
        "55": "Management of Companies and Enterprises",
        "56": "Administrative and Support and Waste Management and Remediation Services",
        "61": "Educational Services",
        "62": "Health Care and Social Assistance",
        "71": "Arts, Entertainment, and Recreation",
        "72": "Accommodation and Food Services",
        "81": "Other Services (except Public Administration)",
        "92": "Public Administration"
    }

    # Create formatted options for the selectbox: "CODE - DESCRIPTION"
    NAICS_OPTIONS_FORMATTED = [f"{code} - {desc}" for code, desc in NAICS_MAP.items()]
    selected_naics_formatted = st.selectbox(
        'Employer NAICS Code',
        options=NAICS_OPTIONS_FORMATTED
    )
    JOB_INFO_EDUCATION = st.selectbox('Education level required by job:',
                                      options=["Master's", "Bachelor's", 'Doctorate', 'Other', 'High School',
                                               "Associate's"])

    if st.button("Next: Continue to Job Info"):
        # Store job info in session state
        naics_code_only = None
        if selected_naics_formatted:
            naics_code_only = selected_naics_formatted.split(" - ")[0]
        st.session_state.job_info = {
            "JOB_INFO_WORK_STATE": JOB_INFO_WORK_STATE,
            "PW_LEVEL_9089": PW_LEVEL_9089,
            "PW_AMOUNT_9089": PW_AMOUNT_9089,
            "NAICS": naics_code_only,
            "JOB_INFO_EDUCATION": JOB_INFO_EDUCATION,
        }
        st.session_state.page = "job_info2"
        st.rerun()


def job_info_page2():
    st.title("Job/Employer Information")

    JOB_INFO_EXPERIENCE = st.radio('Job experience required?', options=['Y', 'N'])
    JOB_INFO_EXPERIENCE_NUM_MONTHS = st.number_input(
        "If experience is required, how many months? If none is required, enter '0'.", min_value=0)
    JOB_INFO_TRAINING = st.radio('Does the job provide training?', options=['Y', 'N'])
    JOB_INFO_FOREIGN_ED = st.radio('Is a foreign educational equivalent acceptable for the job?', options=['Y', 'N'])
    RI_LAYOFF_IN_PAST_SIX_MONTHS = st.radio(
        'Has the employer had any layoffs in the occupation of intended employment within the past six months?',
        options=['Y', 'N'])
    EMPLOYER_NUM_EMPLOYEES = st.number_input('Employer number of employees:', min_value=0)

    if st.button("Next: Continue to User Info"):
        # Store job info in session state
        st.session_state.job_info.update({
            "JOB_INFO_EXPERIENCE": JOB_INFO_EXPERIENCE,
            "JOB_INFO_EXPERIENCE_NUM_MONTHS": JOB_INFO_EXPERIENCE_NUM_MONTHS,
            "JOB_INFO_TRAINING": JOB_INFO_TRAINING,
            "JOB_INFO_FOREIGN_ED": JOB_INFO_FOREIGN_ED,
            "RI_LAYOFF_IN_PAST_SIX_MONTHS": RI_LAYOFF_IN_PAST_SIX_MONTHS,
            "EMPLOYER_NUM_EMPLOYEES": EMPLOYER_NUM_EMPLOYEES,
        })
        st.session_state.page = "user_info"
        st.rerun()

def user_info_page():
    st.title("User Information")
    country = st.selectbox("Country of Citizenship", options=['AFGHANISTAN', 'ALBANIA', 'ALGERIA', 'ANDORRA', 'ANGOLA',
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

    class_admission = st.selectbox("Class of Admission", options=[
        'H-1B', 'L-1', 'F-1', 'H-4', 'Not in USA', 'O-1', 'J-2', 'J-1', 'B-2', 'Parolee', 'L-2', 'E-2', 'TN',
        'B-1', 'G-4', 'P-1', 'H-1B1', 'EWI', 'F-2', 'E-3', 'H-2B', 'TPS'
    ])
    foreign_edu = st.selectbox("Foreign Worker Education Level",
                               options=['High School', "Associate's", "Bachelor's", "Master's", 'Doctorate', 'Other'])
    year_completed = st.number_input("Year your highest education was completed", min_value=1925)
    req_experience = st.radio("Do you have the required experience?", options=['Y', 'N', 'A'])

    if st.button("Submit and Predict"):
        # Store user info in session state
        st.session_state.user_info = {
            "COUNTRY_OF_CITIZENSHIP": country,
            "CLASS_OF_ADMISSION": class_admission,
            "FOREIGN_WORKER_INFO_EDUCATION": foreign_edu,
            "FW_INFO_YR_REL_EDU_COMPLETED": year_completed,
            "FW_INFO_REQ_EXPERIENCE": req_experience
        }
        st.session_state.page = "prediction"
        st.rerun()
    if st.button("Back: Go to Previous Page"):
        st.session_state.page = "job_info"
        st.rerun()


def prediction_page():
    st.title("Prediction Results")
    ada, mapie, shap_model = load_models()

    if 'prediction' not in st.session_state:
        with st.spinner("Predicting..."):
            original_df = pd.read_csv('Data/Data Sets/fullData.csv')  # Original data to create ML model
            original_df.rename(columns={'2_NAICS': 'NAICS'}, inplace=True)
            original_df['NAICS'] = original_df['NAICS'].astype(str)
            original_df['JOB_INFO_WORK_STATE'].replace({'MASSACHUSETTES': 'MASSACHUSETTS', 'MH': 'MARSHALL ISLANDS'},
                                                       inplace=True)
            original_df['COUNTRY_OF_CITIZENSHIP'].replace(
                {'IVORY COAST': "COTE d'IVOIRE", 'NETHERLANDS ANTILLES': 'NETHERLANDS'},
                inplace=True)
            original_df = original_df[~original_df['COUNTRY_OF_CITIZENSHIP'].isin(
                ['SOVIET UNION', 'UNITED STATES OF AMERICA', 'KIRIBATI', 'SAO TOME AND PRINCIPE', 'SINT MAARTEN'])]
            original_df = original_df[
                ~original_df['JOB_INFO_WORK_STATE'].isin(['FEDERATED STATES OF MICRONESIA', 'MARSHALL ISLANDS'])]
            original_df.dropna(axis=0, how='any', inplace=True)
            original_df.drop(['DECISION_DATE', 'PW_UNIT_OF_PAY_9089', 'CASE_RECEIVED_DATE'], axis=1, inplace=True)

            # Concatenate two dataframes together along rows (axis = 0)
            combined_df1 = original_df.copy()

            # Define the new row as a dictionary to ensure it matches the column names
            new_row = {
                'NAICS': st.session_state.job_info.get("NAICS"),
                'PW_LEVEL_9089': st.session_state.job_info.get("PW_LEVEL_9089"),
                'PW_AMOUNT_9089': st.session_state.job_info.get("PW_AMOUNT_9089"),
                'JOB_INFO_WORK_STATE': st.session_state.job_info.get("JOB_INFO_WORK_STATE"),
                'COUNTRY_OF_CITIZENSHIP': st.session_state.user_info.get("COUNTRY_OF_CITIZENSHIP"),
                'CLASS_OF_ADMISSION': st.session_state.user_info.get("CLASS_OF_ADMISSION"),
                'EMPLOYER_NUM_EMPLOYEES': st.session_state.job_info.get("EMPLOYER_NUM_EMPLOYEES"),
                'JOB_INFO_EDUCATION': st.session_state.job_info.get("JOB_INFO_EDUCATION"),
                'JOB_INFO_TRAINING': st.session_state.job_info.get("JOB_INFO_TRAINING"),
                'JOB_INFO_EXPERIENCE': st.session_state.job_info.get("JOB_INFO_EXPERIENCE"),
                'JOB_INFO_EXPERIENCE_NUM_MONTHS': st.session_state.job_info.get("JOB_INFO_EXPERIENCE_NUM_MONTHS"),
                'JOB_INFO_FOREIGN_ED': st.session_state.job_info.get("JOB_INFO_FOREIGN_ED"),
                'RI_LAYOFF_IN_PAST_SIX_MONTHS': st.session_state.job_info.get("RI_LAYOFF_IN_PAST_SIX_MONTHS"),
                'FOREIGN_WORKER_INFO_EDUCATION': st.session_state.user_info.get("FOREIGN_WORKER_INFO_EDUCATION"),
                'FW_INFO_YR_REL_EDU_COMPLETED': st.session_state.user_info.get("FW_INFO_YR_REL_EDU_COMPLETED"),
                'FW_INFO_REQ_EXPERIENCE': st.session_state.user_info.get("FW_INFO_REQ_EXPERIENCE")
            }

            # Append the new row to the DataFrame
            new_row_df = pd.DataFrame([new_row])
            # Concatenate the new row to the original DataFrame
            combined_df1 = pd.concat([combined_df1, new_row_df], ignore_index=True)
            # Number of rows in original dataframe
            original_rows1 = original_df.shape[0]
            # Create dummies for the combined dataframe
            combined_df1['NAICS'] = combined_df1['NAICS'].astype(str)
            combined_df1['PW_AMOUNT_9089'] = pd.to_numeric(combined_df1['PW_AMOUNT_9089'], errors='coerce')
            cat_var = ['NAICS', 'PW_LEVEL_9089', 'JOB_INFO_WORK_STATE', 'COUNTRY_OF_CITIZENSHIP',
                       'FOREIGN_WORKER_INFO_EDUCATION',
                       'JOB_INFO_EXPERIENCE', 'CLASS_OF_ADMISSION', 'JOB_INFO_EDUCATION', 'JOB_INFO_TRAINING',
                       'JOB_INFO_FOREIGN_ED', 'RI_LAYOFF_IN_PAST_SIX_MONTHS', 'FW_INFO_REQ_EXPERIENCE']

            combined_df_encoded1 = pd.get_dummies(combined_df1, columns=cat_var)

            # Split data into original and user input dataframes using row index
            original_df_encoded1 = combined_df_encoded1[:original_rows1]
            input_df_encoded1 = combined_df_encoded1.tail(1)

            st.session_state['input_features'] = input_df_encoded1
            st.session_state['confidence'] = 90  # Default confidence
            pred = ada.predict(input_df_encoded1)

            y_pred, y_pis = mapie.predict(input_df_encoded1, alpha=abs((st.session_state['confidence']/100)-1))

            st.session_state['prediction'] = pred[0]
            st.session_state['low'] = y_pis[0, 0, 0] if y_pis[0, 0, 0] > 0 else 0
            st.session_state['high'] = y_pis[0, 1, 0]


    predict_year, predict_month = decimal_to_years_months(st.session_state['prediction'])
    low_year, low_month = decimal_to_years_months(st.session_state['low'])
    high_year, high_month = decimal_to_years_months(st.session_state['high'])

    st.markdown(f"""
        <div style="text-align: center; margin-top: 20px; margin-bottom: 20px;">
            <p style="font-size: 18px; margin-bottom: 5px;">Our model predicts:</p>
            <p style="font-size: 32px; color: green; font-weight: bold; margin: 10px 0;">
                {predict_year}, {predict_month}
            </p>
            <p style="font-size: 18px; margin-top: 5px;">
                With {st.session_state['confidence']}% confidence, the wait time will be between
                <br><b>{low_year}, {low_month}</b> and <b>{high_year}, {high_month}</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)
    confidence = st.slider("Adjust the confidence interval?", 0, 100, st.session_state['confidence'], help="A confidence interval is a statistical range used to express the uncertainty or variability in a prediction. It provides a range of values within which we expect the true value to lie, with a specified level of confidence. For example, a 95% confidence interval suggests that, if the same experiment or analysis were repeated multiple times, 95% of the resulting intervals would contain the true value. A narrower confidence interval indicates greater precision in the prediction, while a wider interval reflects greater uncertainty. This allows for a more informed interpretation of the prediction's reliability.")

    # Example data
    lower_bound = st.session_state['low']
    upper_bound = st.session_state['high']
    final_prediction = st.session_state['prediction']
    max_time = 12
    fig, ax = plt.subplots(figsize=(8, 1))

    ax.fill_between([lower_bound, upper_bound], 0, 1, color='green', alpha=0.3, label="Confidence Interval")
    ax.axvline(final_prediction, color='red', lw=3, label='Final Prediction')

    ax.set_xlim(0, max_time)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Waiting Time (years)')
    ax.set_yticks([])
    ax.legend()
    st.pyplot(fig)

    col1, col2, col3 = st.columns([3, 2.5, 1])

    with col1:
        # 'Back' button on the left
        if st.button("Back: Reinput Information"):
            st.session_state.page = "job_info"
            del st.session_state['prediction']
            st.rerun()

    with col3:
        if st.button("Predict"):
            with st.spinner("Predicting..."):
                st.session_state['confidence'] = confidence
                y_pred, y_pis = mapie.predict(st.session_state['input_features'],
                                              alpha=abs((st.session_state['confidence'] / 100) - 1))
                st.session_state['low'] = y_pis[0, 0, 0] if y_pis[0, 0, 0] > 0 else 0
                st.session_state['high'] = y_pis[0, 1, 0]
                st.rerun()

    with col2:
        # 'Analyze Features' button in the center
        analyze_button = st.button("Analyze Features")
        if analyze_button:
            filename = 'Streamlit/pages/immigrationpage1dot5.py'
            st.switch_page("./pages/immigrationpage1dot5.py")
            with open(filename) as f:
                exec(f.read())
            updatePage()
            st.rerun()

    # Custom CSS for making the 'Analyze Features' button green and adjusting text style
    st.markdown("""
        <style>
            .css-1aumxhk {  /* Class for "Analyze Features" button */
                color: green;
                font-size: 20px;
                font-weight: bold;
            }
        </style>
    """, unsafe_allow_html=True)


# --------------------
# Main Navigation Logic
# --------------------
if "page" not in st.session_state:
    st.session_state.page = "job_info"

# Render the page based on the session state
if st.session_state.page == "job_info":
    job_info_page1()
if st.session_state.page == "job_info2":
    job_info_page2()
elif st.session_state.page == "user_info":
    user_info_page()
elif st.session_state.page == "prediction":
    prediction_page()
