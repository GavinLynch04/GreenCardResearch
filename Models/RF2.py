import pickle
import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from Data.Preprocessing.preprocess import *
import time
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold

X, y = preprocess()
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=27)

'''feature_names = train_X.columns.to_numpy()
print(feature_names)'''

'''tree_reg = RandomForestRegressor(random_state=9, n_estimators=300, n_jobs=-1, min_samples_split=4, min_samples_leaf=1, max_depth=60)
tree_reg.fit(train_X, train_y)'''

with open("bestRF.pkl", "rb") as f:
    tree_reg = pickle.load(f)

import shap

# 1) Create a TreeExplainer for your RF model
explainer = shap.TreeExplainer(tree_reg)
print("here1")
# 2) Compute SHAP values for your test set
#    shap_values will be an array of shape (n_samples, n_features)
shap_values = explainer.shap_values(test_X)
print("here2")
# 3) Convert to a DataFrame for easier handling
shap_df = pd.DataFrame(shap_values, columns=test_X.columns)
print("here3")
# 4) Compute the *mean* positive and negative contributions per feature
#    (we mask the sign so that positive shap_values>0 count towards positive,
#     and shap_values<0 towards negative)
pos_contrib = shap_df.clip(lower=0).mean(axis=0)
neg_contrib = shap_df.clip(upper=0).mean(axis=0)

# 5) Find the top-k features
k = 5
top_pos = pos_contrib.nlargest(k)
top_neg = neg_contrib.nsmallest(k)  # most-negative

print("Top positive contributors (increase predicted wait):")
print(top_pos)

print("\nTop negative contributors (decrease predicted wait):")
print(top_neg)



'''start_predict = time.time()
test_predictions = tree_reg.predict(test_X)
train_pred = tree_reg.predict(train_X)
end_predict = time.time()
print(f"Prediction time on test set: {end_predict - start_predict:.4f}s")

test_rmse = root_mean_squared_error(test_y, test_predictions)
test_r2 = r2_score(test_y, test_predictions)
train_rmse = root_mean_squared_error(train_y, train_pred)
train_r2 = r2_score(train_y, train_pred)

print("\n--- Best Model Evaluation on Test Set ---")
print(f"Test Set Root Mean Squared Error (RMSE): {test_rmse:.4f}")
print(f"Test Set R-squared (R²): {test_r2:.4f}")
print(f"Train Set Root Mean Squared Error (RMSE): {train_rmse:.4f}")
print(f"Train Set R-squared (R²): {train_r2:.4f}")

importances   = tree_reg.feature_importances_

top_idx     = np.argmax(importances)
top_feature = feature_names[top_idx]
feature_names = np.delete(feature_names, top_idx)
importances  = np.delete(importances, top_idx)
print(f"Highest‐impact feature: {top_feature}")

# 2) Drop that column
train_X = train_X.drop(columns=[top_feature])

readable_names = {
    "NAICS":                        "NAICS Code",
    "PW_LEVEL_9089":               "Prevailing Wage Level",
    "PW_AMOUNT_9089":              "Prevailing Wage Amount (USD)",
    "JOB_INFO_WORK_STATE":         "Work State",
    "COUNTRY_OF_CITIZENSHIP":      "Country of Citizenship",
    "CLASS_OF_ADMISSION":          "Class of Admission",
    "EMPLOYER_NUM_EMPLOYEES":      "Number of Employees",
    "JOB_INFO_EDUCATION":          "Required Education Level",
    "JOB_INFO_TRAINING":           "Required Training",
    "JOB_INFO_EXPERIENCE":         "Required Experience",
    "JOB_INFO_EXPERIENCE_NUM_MONTHS": "Experience Duration (Months)",
    "JOB_INFO_FOREIGN_ED":         "Foreign Education Level",
    "RI_LAYOFF_IN_PAST_SIX_MONTHS":"Layoff in Past 6 Months",
    "FOREIGN_WORKER_INFO_EDUCATION":"Foreign Worker Education Level",
    "FW_INFO_YR_REL_EDU_COMPLETED":"Year of Education Completion",
    "FW_INFO_REQ_EXPERIENCE":      "Required Experience"
}
train_X.rename(columns=readable_names, inplace=True)
feature_names = train_X.columns.to_numpy()

# Sort indices for plotting
indices = np.argsort(importances)

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(indices)), importances[indices])
ax.set_yticks(range(len(indices)))
ax.set_yticklabels(feature_names[indices])
ax.set_xlabel("Feature Importance", fontsize=14)
plt.savefig("featureImportance.svg", format="svg", bbox_inches="tight")

plt.tight_layout()
plt.show()'''


