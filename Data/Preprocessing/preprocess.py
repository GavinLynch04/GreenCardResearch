import pandas as pd
import torch

def preprocess():
    df_time = pd.read_csv("../Data/Data Sets/Processing Time Data.csv")
    df_time = df_time.loc[:94]
    df_time = df_time.astype(int, errors='ignore')
    df_time['MONTH_YEAR'] = pd.to_datetime(df_time['YEAR'].astype(int).astype(str) + '-' + df_time['MONTH'].astype(int).astype(str), format='%Y-%m')
    df_time['CHINA_DATE'] = pd.to_datetime(df_time['CHINA YEAR'].astype(int).astype(str) + '-' + df_time['CHINA MONTH'].astype(int).astype(str), format='%Y-%m')
    df_time['INDIA_DATE'] = pd.to_datetime(df_time['INDIA YEAR'].astype(int).astype(str) + '-' + df_time['INDIA MONTH'].astype(int).astype(str), format='%Y-%m')
    df_time['MEXICO_DATE'] = pd.to_datetime(df_time['MEXICO YEAR'].astype(int).astype(str) + '-' + df_time['MEXICO MONTH'].astype(int).astype(str), format='%Y-%m')
    df_time['PHILIPPINES_DATE'] = pd.to_datetime(df_time['PHILIPPINES YEAR'].astype(int).astype(str) + '-' + df_time['PHILIPPINES MONTH'].astype(int).astype(str), format='%Y-%m')
    df_time['ALL_DATE'] = pd.to_datetime(df_time['ALL YEAR'].astype(int).astype(str) + '-' + df_time['ALL MONTH'].astype(int).astype(str), format='%Y-%m')
    df_time = df_time.drop(columns=['MONTH','YEAR','CHINA MONTH','CHINA YEAR','INDIA MONTH','INDIA YEAR','MEXICO MONTH','MEXICO YEAR','PHILIPPINES MONTH','PHILIPPINES YEAR','ALL YEAR','ALL MONTH'])

    '''df = pd.read_csv('../Data/Data Sets/PERM_Data.csv')
    df2 = pd.read_csv('../Data/Data Sets/newData.csv')'''
    df = pd.read_csv('../Data/Data Sets/fullData.csv')

    # Data cleaning
    '''columns_to_drop = ['CASE_NO', 'CASE_STATUS', 'PW_SOC_CODE', 'JOB_INFO_WORK_CITY', 'PW_DETERM_DATE', 'PW_EXPIRE_DATE', 'JOB_INFO_MAJOR', 'JI_OFFERED_TO_SEC_J_FOREIGN_WORKER', 'RECR_INFO_PROFESSIONAL_OCC', 'FOREIGN_WORKER_INFO_MAJOR', 'CASE_RECEIVED_MONTH','CASE_RECEIVED_YEAR']
    df.drop(columns=columns_to_drop, inplace=True)'''

    conversion_factors = {
        'Bi-Weekly': 26,  # Assuming 52 weeks in a year and bi-weekly pay
        'Hour': 2080,     # Assuming 40 hours per week and 52 weeks in a year
        'Month': 12,      # Monthly pay
        'Week': 52        # Weekly pay
    }

    # Convert salaries to annual salary where unit of pay is not 'Year'
    for index, row in df.iterrows():
        if row['PW_UNIT_OF_PAY_9089'] != 'Year':
            conversion_factor = conversion_factors.get(row['PW_UNIT_OF_PAY_9089'])
            if conversion_factor:
                df.at[index, 'PW_AMOUNT_9089'] *= conversion_factor

    df.drop(columns=['PW_UNIT_OF_PAY_9089'], inplace=True)
    # Convert 'PW_AMOUNT_9089' column to numeric
    df['PW_AMOUNT_9089'] = pd.to_numeric(df['PW_AMOUNT_9089'], errors='coerce')
    # Identify values under 1000 in the 'PW_AMOUNT_9089' column
    under_1000 = df['PW_AMOUNT_9089'] < 1000
    # Multiply values under 1000 by 1000
    df.loc[under_1000, 'PW_AMOUNT_9089'] *= 1000
    # Replace incorrect values in the 'JOB_INFO_WORK_STATE' column
    df['JOB_INFO_WORK_STATE'].replace({'MASSACHUSETTES': 'MASSACHUSETTS', 'MH': 'MARSHALL ISLANDS'}, inplace=True)
    # Replace incorrect values in the 'COUNTRY_OF_CITIZENSHIP' column
    df['COUNTRY_OF_CITIZENSHIP'].replace({'IVORY COAST': "COTE d'IVOIRE", 'NETHERLANDS ANTILLES': 'NETHERLANDS'}, inplace=True)
    # Drop rows with specified values in the 'COUNTRY_OF_CITIZENSHIP' column
    df = df[~df['COUNTRY_OF_CITIZENSHIP'].isin(['SOVIET UNION', 'UNITED STATES OF AMERICA'])]
    df.dropna(axis=0, how='any', inplace=True)

    df['CASE_RECEIVED_DATE'] = pd.to_datetime(df['CASE_RECEIVED_DATE'], errors = 'coerce')
    df['DECISION_DATE'] = pd.to_datetime(df['DECISION_DATE'], errors = 'coerce')

    # Calculate the number of months between the two dates
    df['MONTHS_TO_DECISION'] = (df['DECISION_DATE'].dt.year - df['CASE_RECEIVED_DATE'].dt.year) * 12 + (df['DECISION_DATE'].dt.month - df['CASE_RECEIVED_DATE'].dt.month)

    df['DECISION_DATE'] = pd.to_datetime(df['DECISION_DATE']).dt.to_period('M').dt.to_timestamp()
    df['CASE_RECEIVED_DATE'] = pd.to_datetime(df['CASE_RECEIVED_DATE']).dt.to_period('M').dt.to_timestamp()
    df = df.reset_index(drop=True)

    df_china = df[df['COUNTRY_OF_CITIZENSHIP'] == 'CHINA']
    df_china = df_china.reset_index(drop=True)
    df_india = df[df['COUNTRY_OF_CITIZENSHIP'] == 'INDIA']
    df_india = df_india.reset_index(drop=True)
    df_mexico = df[df['COUNTRY_OF_CITIZENSHIP'] == 'MEXICO']
    df_mexico = df_mexico.reset_index(drop=True)
    df_philippines = df[df['COUNTRY_OF_CITIZENSHIP'] == 'PHILIPPINES']
    df_philippines = df_philippines.reset_index(drop=True)

    countries_to_exclude = ['INDIA', 'CHINA', 'MEXICO', 'PHILIPPINES']
    df_all = df[~df['COUNTRY_OF_CITIZENSHIP'].isin(countries_to_exclude)]
    df_all = df_all.reset_index(drop=True)

    df_china = pd.merge(df_time, df_china, left_on='MONTH_YEAR', right_on='CASE_RECEIVED_DATE', how='inner', suffixes=('', '_CHINA'))
    df_india = pd.merge(df_time, df_india, left_on='MONTH_YEAR', right_on='CASE_RECEIVED_DATE', how='inner', suffixes=('', '_INDIA'))
    df_mexico = pd.merge(df_time, df_mexico, left_on='MONTH_YEAR', right_on='CASE_RECEIVED_DATE', how='inner', suffixes=('', '_MEXICO'))
    df_philippines = pd.merge(df_time, df_philippines, left_on='MONTH_YEAR', right_on='CASE_RECEIVED_DATE', how='inner', suffixes=('', '_PHILIPPINES'))
    df_all = pd.merge(df_time, df_all, left_on='MONTH_YEAR', right_on='CASE_RECEIVED_DATE', how='inner', suffixes=('', '_ALL'))

    # Calculate waiting time in months
    df_china['WAITING_TIME'] = (df_china['DECISION_DATE'] - df_china['CHINA_DATE']).dt.days // 30
    df_india['WAITING_TIME'] = (df_india['DECISION_DATE'] - df_india['INDIA_DATE']).dt.days // 30
    df_mexico['WAITING_TIME'] = (df_mexico['DECISION_DATE'] - df_mexico['MEXICO_DATE']).dt.days // 30
    df_philippines['WAITING_TIME'] = (df_philippines['DECISION_DATE'] - df_philippines['PHILIPPINES_DATE']).dt.days // 30
    df_all['WAITING_TIME'] = (df_all['DECISION_DATE'] - df_all['ALL_DATE']).dt.days // 30

    df_china = df_china.drop(columns=['MONTH_YEAR','INDIA_DATE','MEXICO_DATE','PHILIPPINES_DATE','ALL_DATE'])
    df_india = df_india.drop(columns=['MONTH_YEAR','CHINA_DATE','MEXICO_DATE','PHILIPPINES_DATE','ALL_DATE'])
    df_mexico = df_mexico.drop(columns=['MONTH_YEAR','CHINA_DATE','INDIA_DATE','PHILIPPINES_DATE','ALL_DATE'])
    df_philippines = df_philippines.drop(columns=['MONTH_YEAR','CHINA_DATE','INDIA_DATE','MEXICO_DATE','ALL_DATE'])
    df_all = df_all.drop(columns=['MONTH_YEAR','CHINA_DATE','INDIA_DATE','MEXICO_DATE','PHILIPPINES_DATE'])

    df_china.dropna(subset=['WAITING_TIME'], inplace=True)
    df_india.dropna(subset=['WAITING_TIME'], inplace=True)
    df_mexico.dropna(subset=['WAITING_TIME'], inplace=True)
    df_philippines.dropna(subset=['WAITING_TIME'], inplace=True)
    df_all.dropna(subset=['WAITING_TIME'], inplace=True)

    df_combined = pd.concat([df_china, df_india, df_mexico, df_philippines, df_all], ignore_index=True)
    df_combined.rename(columns={'2_NAICS': 'NAICS'}, inplace=True)

    # Define waiting time ranges in months
    waiting_time_ranges = [(0, 30), (30, 60), (60, 120), (120, float('inf'))]  # float('inf') for 10+ years

    # Define waiting time labels in years
    waiting_time_labels = ['0-2.5 years', '2.5-5 years', '5-10 years', '>10 years']

    # Function to categorize waiting time into ranges
    def categorize_waiting_time(waiting_time):
        for i, (start, end) in enumerate(waiting_time_ranges):
            if start <= waiting_time < end:
                return waiting_time_labels[i]
        return waiting_time_labels[-1]

    # Apply categorization to waiting time and assign to y
    df_combined['WAITING_TIME_RANGE'] = df_combined['WAITING_TIME'].apply(categorize_waiting_time)
    #For classification
    '''y = df_combined['WAITING_TIME_RANGE']'''
    #For regression
    y = df_combined['WAITING_TIME'] / 12

    # Prepare data for training
    X = df_combined.drop(columns=['MONTHS_TO_DECISION', 'WAITING_TIME', 'DECISION_DATE', 'CASE_RECEIVED_DATE', 'CHINA_DATE', 'INDIA_DATE','MEXICO_DATE','PHILIPPINES_DATE','ALL_DATE', 'WAITING_TIME_RANGE'])
    cat_var = ['NAICS', 'PW_LEVEL_9089', 'JOB_INFO_WORK_STATE', 'COUNTRY_OF_CITIZENSHIP', 'FOREIGN_WORKER_INFO_EDUCATION', 'JOB_INFO_EXPERIENCE', 'CLASS_OF_ADMISSION', 'JOB_INFO_EDUCATION', 'JOB_INFO_TRAINING', 'JOB_INFO_FOREIGN_ED', 'RI_LAYOFF_IN_PAST_SIX_MONTHS', 'FW_INFO_REQ_EXPERIENCE']
    X_encoded = pd.get_dummies(X, columns=cat_var, dtype='int8')

    return X_encoded, y


import pandas as pd
import numpy as np # Import numpy for vectorized operations
# import torch # Keep torch import if used elsewhere

def preprocess_optimized():
    # --- 1. Load and Preprocess df_time ---
    # Specify dtypes for potentially faster loading and less memory
    time_dtypes = {
        'MONTH': 'int16', 'YEAR': 'int16',
        'CHINA MONTH': 'int16', 'CHINA YEAR': 'int16',
        'INDIA MONTH': 'int16', 'INDIA YEAR': 'int16',
        'MEXICO MONTH': 'int16', 'MEXICO YEAR': 'int16',
        'PHILIPPINES MONTH': 'int16', 'PHILIPPINES YEAR': 'int16',
        'ALL MONTH': 'int16', 'ALL YEAR': 'int16',
    }
    df_time = pd.read_csv(
        "../Data/Data Sets/Processing Time Data.csv",
        dtype=time_dtypes,
        nrows=95 # Read only necessary rows (0 to 94)
    )
    # df_time = df_time.loc[:94] # Already handled by nrows

    # Combine date creation - slightly cleaner
    def create_date(year_col, month_col):
        # Using .zfill(2) handles single-digit months correctly
        return pd.to_datetime(
            df_time[year_col].astype(str) + '-' + df_time[month_col].astype(str).str.zfill(2),
            format='%Y-%m',
            errors='coerce' # Handle potential errors gracefully
        )

    df_time['MONTH_YEAR'] = create_date('YEAR', 'MONTH')
    df_time['CHINA_DATE'] = create_date('CHINA YEAR', 'CHINA MONTH')
    df_time['INDIA_DATE'] = create_date('INDIA YEAR', 'INDIA MONTH')
    df_time['MEXICO_DATE'] = create_date('MEXICO YEAR', 'MEXICO MONTH')
    df_time['PHILIPPINES_DATE'] = create_date('PHILIPPINES YEAR', 'PHILIPPINES MONTH')
    df_time['ALL_DATE'] = create_date('ALL YEAR', 'ALL MONTH')

    # Drop original date components earlier
    cols_to_drop_time = [
        'MONTH','YEAR','CHINA MONTH','CHINA YEAR','INDIA MONTH','INDIA YEAR',
        'MEXICO MONTH','MEXICO YEAR','PHILIPPINES MONTH','PHILIPPINES YEAR',
        'ALL YEAR','ALL MONTH'
    ]
    df_time = df_time.drop(columns=cols_to_drop_time)
    # Drop rows if MONTH_YEAR failed conversion
    df_time.dropna(subset=['MONTH_YEAR'], inplace=True)


    # --- 2. Load and Clean Main DataFrame (df) ---
    # Consider specifying dtypes here too if known, especially for categorical/int
    # Example: dtypes = {'NAICS': 'str', 'PW_AMOUNT_9089': 'float64', ...}
    # df = pd.read_csv('../Data/Data Sets/fullData.csv', dtype=dtypes)
    df = pd.read_csv('../Data/Data Sets/fullData.csv')

    # Drop unused columns early (if the commented list is accurate)
    # columns_to_drop_main = ['CASE_NO', 'CASE_STATUS', 'PW_SOC_CODE', ...]
    # df.drop(columns=columns_to_drop_main, inplace=True, errors='ignore')

    # --- Vectorized Salary Conversion (MUCH FASTER) ---
    conversion_factors = {
        'Bi-Weekly': 26,
        'Hour': 2080,
        'Month': 12,
        'Week': 52,
        'Year': 1 # Include 'Year' for completeness
    }
    # Ensure PW_AMOUNT is numeric first
    df['PW_AMOUNT_9089'] = pd.to_numeric(df['PW_AMOUNT_9089'], errors='coerce')

    # Map conversion factors, fill missing/non-matching units with 1 (no change)
    unit_map = df['PW_UNIT_OF_PAY_9089'].map(conversion_factors).fillna(1)
    df['PW_AMOUNT_9089'] *= unit_map

    # Handle values under 1000 (vectorized)
    under_1000_mask = (df['PW_AMOUNT_9089'] < 1000) & (df['PW_AMOUNT_9089'].notna())
    df.loc[under_1000_mask, 'PW_AMOUNT_9089'] *= 1000

    # Drop unit column after conversion
    df.drop(columns=['PW_UNIT_OF_PAY_9089'], inplace=True)

    # --- Other Cleaning Steps (mostly unchanged, already efficient) ---
    state_replace = {'MASSACHUSETTES': 'MASSACHUSETTS', 'MH': 'MARSHALL ISLANDS'}
    country_replace = {'IVORY COAST': "COTE d'IVOIRE", 'NETHERLANDS ANTILLES': 'NETHERLANDS'}
    df['JOB_INFO_WORK_STATE'].replace(state_replace, inplace=True)
    df['COUNTRY_OF_CITIZENSHIP'].replace(country_replace, inplace=True)

    countries_to_drop = ['SOVIET UNION', 'UNITED STATES OF AMERICA']
    df = df[~df['COUNTRY_OF_CITIZENSHIP'].isin(countries_to_drop)]

    # --- Date Conversion and Alignment ---
    df['CASE_RECEIVED_DATE'] = pd.to_datetime(df['CASE_RECEIVED_DATE'], errors='coerce')
    df['DECISION_DATE'] = pd.to_datetime(df['DECISION_DATE'], errors='coerce')

    # Create a month-start date column for merging
    df['CASE_RECEIVED_MONTH_START'] = df['CASE_RECEIVED_DATE'].dt.to_period('M').dt.to_timestamp()

    # --- Drop NA before Merge (Crucial Columns) ---
    # Drop rows where key merge/calculation columns are missing
    critical_cols_na = [
        'CASE_RECEIVED_DATE', 'DECISION_DATE', 'CASE_RECEIVED_MONTH_START',
        'COUNTRY_OF_CITIZENSHIP', 'PW_AMOUNT_9089' # Add others if critical
    ]
    df.dropna(subset=critical_cols_na, inplace=True)
    # Consider dropping based on all columns if that was the original intent:
    # df.dropna(axis=0, how='any', inplace=True) # Original behaviour

    # --- 3. Single Merge ---
    df_merged = pd.merge(
        df,
        df_time,
        left_on='CASE_RECEIVED_MONTH_START',
        right_on='MONTH_YEAR',
        how='inner' # Keep only rows that match a processing time entry
    )
    # We can drop merge keys if no longer needed
    # df_merged.drop(columns=['MONTH_YEAR', 'CASE_RECEIVED_MONTH_START'], inplace=True)


    # --- 4. Vectorized Waiting Time Calculation ---
    # Define conditions based on country
    cond_china = df_merged['COUNTRY_OF_CITIZENSHIP'] == 'CHINA'
    cond_india = df_merged['COUNTRY_OF_CITIZENSHIP'] == 'INDIA'
    cond_mexico = df_merged['COUNTRY_OF_CITIZENSHIP'] == 'MEXICO'
    cond_philippines = df_merged['COUNTRY_OF_CITIZENSHIP'] == 'PHILIPPINES'

    # Define corresponding date columns from df_time
    date_choices = [
        df_merged['CHINA_DATE'],
        df_merged['INDIA_DATE'],
        df_merged['MEXICO_DATE'],
        df_merged['PHILIPPINES_DATE'],
    ]

    # Use np.select to pick the correct date based on country
    # Default to 'ALL_DATE' if none of the specific countries match
    correct_processing_date = np.select(
        [cond_china, cond_india, cond_mexico, cond_philippines],
        date_choices,
        default=df_merged['ALL_DATE']
    )

    # Calculate waiting time in approximate months
    # Ensure the result of np.select is treated as datetime
    df_merged['WAITING_TIME'] = (df_merged['DECISION_DATE'] - pd.to_datetime(correct_processing_date)).dt.days // 30

    # Drop rows where waiting time calculation failed (e.g., date issues)
    df_merged.dropna(subset=['WAITING_TIME'], inplace=True)

    # --- 5. Prepare Target Variable (y) and Features (X) ---

    # Regression Target
    y = df_merged['WAITING_TIME'] / 12

    # Classification Target (Optional - uncomment if needed)
    # waiting_time_ranges = [(0, 30), (30, 60), (60, 120), (120, float('inf'))]
    # waiting_time_labels = ['0-2.5 years', '2.5-5 years', '5-10 years', '>10 years']
    # def categorize_waiting_time(waiting_time):
    #     for i, (start, end) in enumerate(waiting_time_ranges):
    #         if start <= waiting_time < end:
    #             return waiting_time_labels[i]
    #     return waiting_time_labels[-1] # Handle edge case or NaN if necessary
    # df_merged['WAITING_TIME_RANGE'] = df_merged['WAITING_TIME'].apply(categorize_waiting_time)
    # y_classification = df_merged['WAITING_TIME_RANGE']


    # Define columns to drop for features X
    # Includes intermediate dates, merge keys, target, etc.
    columns_to_drop_final = [
        'WAITING_TIME', 'DECISION_DATE', 'CASE_RECEIVED_DATE',
        'CASE_RECEIVED_MONTH_START', 'MONTH_YEAR', # Merge keys
        'CHINA_DATE', 'INDIA_DATE', 'MEXICO_DATE', 'PHILIPPINES_DATE', 'ALL_DATE', # Dates from df_time
        # 'WAITING_TIME_RANGE', # Drop if classification target isn't needed elsewhere
        # 'MONTHS_TO_DECISION' # This wasn't calculated in this version
    ]
    X = df_merged.drop(columns=columns_to_drop_final, errors='ignore')

    # Rename NAICS column if it exists after merge (check actual name)
    if '2_NAICS' in X.columns:
         X.rename(columns={'2_NAICS': 'NAICS'}, inplace=True)
    elif 'NAICS_x' in X.columns: # Handle potential merge suffixes
         X.rename(columns={'NAICS_x': 'NAICS'}, inplace=True)

    # --- 6. One-Hot Encode ---
    cat_var = [
        'NAICS', 'PW_LEVEL_9089', 'JOB_INFO_WORK_STATE', 'COUNTRY_OF_CITIZENSHIP',
        'FOREIGN_WORKER_INFO_EDUCATION', 'JOB_INFO_EXPERIENCE', 'CLASS_OF_ADMISSION',
        'JOB_INFO_EDUCATION', 'JOB_INFO_TRAINING', 'JOB_INFO_FOREIGN_ED',
        'RI_LAYOFF_IN_PAST_SIX_MONTHS', 'FW_INFO_REQ_EXPERIENCE'
    ]
    # Ensure only columns present in X are used for encoding
    cat_var_present = [col for col in cat_var if col in X.columns]

    # Use int8 for memory efficiency. sparse=True could save more memory
    # but might require changes in downstream model handling (PyTorch usually needs dense)
    X_encoded = pd.get_dummies(X, columns=cat_var_present, dtype='int8') # sparse=False (default)

    print(f"Optimized preprocessing complete.")
    print(f"X_encoded shape: {X_encoded.shape}")
    print(f"X_encoded memory usage: {X_encoded.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    print(f"y length: {len(y)}")

    return X_encoded, y

