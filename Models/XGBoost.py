import pickle
import time

import joblib
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import shap
from sklearn.metrics import classification_report, precision_score, recall_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from mapie.regression import MapieRegressor
from xgboost import XGBClassifier, XGBRegressor

from Data.Preprocessing.preprocess import *
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

X, y = preprocess()
print(y)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=207)

bestXGB = XGBRegressor(
    max_depth=6,                    # Controls tree depth, 3-10 is typical
    learning_rate=0.05,              # Lower learning rate for better generalization
    n_estimators=500,                 # More trees for lower learning rate
    gamma=0.1,                        # Regularization to avoid overfitting
    min_child_weight=3,               # Minimum sum of instance weight (analogous to min_samples_split)
    subsample=0.8,                    # Randomly sample 80% of data for each tree
    colsample_bytree=0.8,             # Randomly sample 80% of features for each tree
    reg_alpha=0.1,                    # L1 regularization (sparse weight penalty)
    reg_lambda=1,                     # L2 regularization (weight shrinkage)
    random_state=9,
    nthread=8
)
start = time.time()
xgb = bestXGB.fit(train_X, train_y)
stop = time.time()
print(f"Training time: {stop - start}s")

with open("../Streamlit/pages/xgb_model.pkl", "wb") as file:
    pickle.dump(xgb, file)

y_pred = xgb.predict(test_X)
mae = mean_absolute_error(test_y, y_pred)
print(f"MAE: {mae}")
explainer = shap.TreeExplainer(bestXGB)
with open("../Streamlit/pages/shap_xgb.pkl", "wb") as file:
    pickle.dump(explainer, file)

'''shap_pickle = open('../Streamlit/pages/xgboost_model.pkl', 'rb')
xgb = pickle.load(shap_pickle)
shap_pickle.close()'''

print("end")
mapie = MapieRegressor(estimator=xgb, n_jobs=-1)  # 90% confidence interval
mapie.fit(train_X, train_y)
with open("../Streamlit/pages/mapieXGB.pkl", "wb") as file:
    pickle.dump(mapie, file)

print("here")
sample_indices = test_X.sample(n=1000, random_state=42).index  # Take random 1000 rows, get their indices
test_X = test_X.loc[sample_indices]                      # Subset features
test_y = test_y.loc[sample_indices]                      # Subset labels (use loc to match index)

print("here2")
y_pred, y_pis = mapie.predict(test_X, alpha=0.1)

print("here3")
print(test_y)
correct = (test_y >= y_pis[:, 0, 0]) & (test_y <= y_pis[:, 1, 0])  # True if within the interval
accuracy = np.mean(correct)

# Calculate custom precision and recall
# Convert predictions into "correct" and "incorrect" categories
binary_y_pred = correct.astype(int)  # 1 for correct, 0 for incorrect
binary_y_true = np.ones_like(test_y)  # True labels are all "correct" (1)

# Calculate precision and recall based on binary values
precision = precision_score(binary_y_true, binary_y_pred)
recall = recall_score(binary_y_true, binary_y_pred)

# Display the results
print(f"Custom Accuracy (within 90% confidence): {accuracy:.2f}")
print(f"Custom Precision: {precision:.2f}")
print(f"Custom Recall: {recall:.2f}")
