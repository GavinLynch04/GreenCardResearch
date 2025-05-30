import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, CatBoostRegressor

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
    #X_encoded = pd.get_dummies(X, columns=cat_var, dtype='int8')

    return X, y, cat_var

X, y, cat = preprocess()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)

model = CatBoostRegressor(
    iterations=700,          # Number of boosting iterations
    learning_rate=0.1,       # Learning rate
    depth=16,                 # Depth of trees
    cat_features=cat,  # Specify categorical features
    verbose=1,              # Print progress every 1 iterations
    loss_function="RMSE",
    task_type="GPU",
    border_count=512,
)
'''from sklearn.model_selection import GridSearchCV

param_grid = {
    'depth': [6, 8, 10, 12, 14],
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [300, 500, 700],
    'l2_leaf_reg': [3, 5, 7],
    'border_count': [32, 64, 128],
    'task_type': ['GPU'],
    'auto_class_weights': ['Balanced'],
    'loss_function': ['MultiClass'],
}

grid_search = GridSearchCV(
    estimator=CatBoostClassifier(cat_features=cat, verbose=0),
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=3,
)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)'''

model.fit(X_train, y_train, eval_set=(X_test, y_test), plot=True)

y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

print(f"Test Set Root Mean Squared Error (RMSE): {root_mean_squared_error(y_test, y_pred):.4f}")
print(f"Test Set R-squared (R²): {r2_score(y_test, y_pred):.4f}")
print(f"Train Set Root Mean Squared Error (RMSE): {root_mean_squared_error(y_train, y_pred_train):.4f}")
print(f"Train Set R-squared (R²): {r2_score(y_train, y_pred_train):.4f}")