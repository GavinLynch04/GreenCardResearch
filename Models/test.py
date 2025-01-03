import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


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

df = pd.read_csv('../Data/Data Sets/PERM_Data.csv')

# Data cleaning
columns_to_drop = ['CASE_NO', 'CASE_STATUS', 'PW_SOC_CODE', 'JOB_INFO_WORK_CITY', 'PW_DETERM_DATE', 'PW_EXPIRE_DATE', 'JOB_INFO_MAJOR', 'JI_OFFERED_TO_SEC_J_FOREIGN_WORKER', 'RECR_INFO_PROFESSIONAL_OCC', 'FOREIGN_WORKER_INFO_MAJOR', 'CASE_RECEIVED_MONTH','CASE_RECEIVED_YEAR']
df.drop(columns=columns_to_drop, inplace=True)
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
y = df_combined['WAITING_TIME_RANGE']
print("finished preprocessing")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Prepare data for training
X = df_combined.drop(columns=['MONTHS_TO_DECISION', 'WAITING_TIME', 'DECISION_DATE', 'CASE_RECEIVED_DATE', 'CHINA_DATE', 'INDIA_DATE','MEXICO_DATE','PHILIPPINES_DATE','ALL_DATE', 'WAITING_TIME_RANGE'])
cat_var = ['NAICS', 'PW_LEVEL_9089', 'JOB_INFO_WORK_STATE', 'COUNTRY_OF_CITIZENSHIP', 'FOREIGN_WORKER_INFO_EDUCATION', 'JOB_INFO_EXPERIENCE', 'CLASS_OF_ADMISSION', 'JOB_INFO_EDUCATION', 'JOB_INFO_TRAINING', 'JOB_INFO_FOREIGN_ED', 'RI_LAYOFF_IN_PAST_SIX_MONTHS', 'FW_INFO_REQ_EXPERIENCE']
X_encoded = pd.get_dummies(X, columns=cat_var)
category_mapping = {
    "0-2.5 years": 0,
    "2.5-5 years": 1,
    "5-10 years": 2,
    ">10 years": 3
}
y = np.vectorize(category_mapping.get)(y)
X_encoded = X_encoded.astype(int)

# Standardize features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_encoded)

X_tensor = torch.tensor(X_normalized, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

# Split data into training and testing sets
train_X, test_X, train_y, test_y = train_test_split(X_tensor, y_tensor, test_size=0.25, random_state=27)


# Define the neural network
class ClassificationNN(nn.Module):
    def __init__(self, input_size):
        super(ClassificationNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 4)  # 4 output classes for 4-class classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        return x


# Model, loss, and optimizer
input_size = train_X.shape[1]
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import KFold
import numpy as np

# K-Fold Cross Validation
k_folds = 5  # Number of folds
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Cross-validation accuracies
# Cross-validation accuracies
cv_accuracies = []
epochs = 50
batch_size = 100

# Initialize accumulators for metrics
precision_scores = []
recall_scores = []
f1_scores = []
accuracy_scores = []

# Loop over the K-folds
for fold, (train_idx, val_idx) in enumerate(kf.split(train_X)):
    print(f"Training fold {fold + 1}/{k_folds}")

    # Create train and validation datasets
    train_fold_X, val_fold_X = train_X[train_idx], train_X[val_idx]
    train_fold_y, val_fold_y = train_y[train_idx], train_y[val_idx]

    # Initialize model
    model = ClassificationNN(input_size).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(train_fold_X.size(0))  # Shuffle training data

        for i in range(0, train_fold_X.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_X, batch_y = train_fold_X[indices], train_fold_y[indices]

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_X)

            loss = criterion(outputs, batch_y.long())  # Ensure targets are long for CrossEntropyLoss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    # Validation loop
    model.eval()
    with torch.no_grad():
        val_outputs = model(val_fold_X)
        _, predicted = torch.max(val_outputs, 1)  # Get class predictions
        val_loss = criterion(val_outputs, val_fold_y.long())

        # Calculate metrics
        accuracy = accuracy_score(val_fold_y.cpu(), predicted.cpu())
        precision = precision_score(val_fold_y.cpu(), predicted.cpu(), average='weighted')
        recall = recall_score(val_fold_y.cpu(), predicted.cpu(), average='weighted')
        f1 = f1_score(val_fold_y.cpu(), predicted.cpu(), average='weighted')

        # Store metrics
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        print(
            f"Fold {fold + 1} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

# Calculate average metrics
avg_accuracy = np.mean(accuracy_scores)
avg_precision = np.mean(precision_scores)
avg_recall = np.mean(recall_scores)
avg_f1 = np.mean(f1_scores)

print(f"\nCross-validation results:")
print(f"Average Accuracy: {avg_accuracy:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average F1-Score: {avg_f1:.4f}")
