import pickle
import time

import joblib
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import shap
from sklearn.metrics import classification_report, precision_score, recall_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from mapie.regression import MapieRegressor
from Data.Preprocessing.preprocess import *
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


start = time.time()
X, y = preprocess()
end = time.time()


train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=207)

bestAda = AdaBoostRegressor(estimator =  DecisionTreeRegressor(max_depth=68, min_samples_leaf=2, min_samples_split=13, random_state=9), n_estimators=50, learning_rate=0.01, random_state = 9)
start = time.time()
ada = bestAda.fit(train_X, train_y)
stop = time.time()
print(f"Training time: {stop - start}s")
y_pred = bestAda.predict(test_X)
test_mse = mean_squared_error(test_y, y_pred)
test_r2 = r2_score(test_y, y_pred)

print("\n--- Old Best Model Evaluation on Test Set ---")
print(f"Test Set Mean Squared Error (MSE): {test_mse:.4f}")
print(f"Test Set R-squared (R²): {test_r2:.4f}")


base_estimator = DecisionTreeRegressor(random_state=9)

# 2. Define the main estimator (AdaBoost)
ada_reg = AdaBoostRegressor(estimator=base_estimator, random_state=9)

# 3. Define the parameter grid to search
#    Note: Parameters for the base_estimator are prefixed with 'estimator__'
#    Adjust these ranges based on your computational budget and prior knowledge
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5, 1.0],
    'estimator__max_depth': [5, 10, 20, None], # None means nodes expand until pure or min_samples
    'estimator__min_samples_split': [2, 5, 10, 13],
    'estimator__min_samples_leaf': [1, 2, 5]
}

# 4. Define the Cross-Validation strategy
#    For regression, KFold is typically used. StratifiedKFold is for classification.
#    Using shuffle=True is generally recommended for KFold.
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

# 5. Instantiate GridSearchCV
#    n_jobs=-1 uses all available CPU cores.
#    verbose=2 shows progress.
#    scoring='r2' (default for regressors) or 'neg_mean_squared_error' are common
grid_search = GridSearchCV(
    estimator=ada_reg,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring='r2', # Or 'neg_mean_squared_error', etc.
    n_jobs=-1,
    verbose=2
)

# --- Run Grid Search ---
print("Starting GridSearchCV...")
start_grid_search = time.time()
grid_search.fit(train_X, train_y)
end_grid_search = time.time()
print(f"GridSearchCV Training time: {end_grid_search - start_grid_search:.4f}s")

# --- Results ---
print("\n--- Grid Search Results ---")
print(f"Best Parameters found: {grid_search.best_params_}")
print(f"Best Cross-Validation R² score: {grid_search.best_score_:.4f}")

# --- Evaluate Best Model on Test Set ---
best_ada_model = grid_search.best_estimator_ # This is the model with the best params

test_predictions = best_ada_model.predict(test_X)
test_mse = mean_squared_error(test_y, test_predictions)
test_r2 = r2_score(test_y, test_predictions)

print("\n--- Best Model Evaluation on Test Set ---")
print(f"Test Set Mean Squared Error (MSE): {test_mse:.4f}")
print(f"Test Set R-squared (R²): {test_r2:.4f}")



'''mapie = MapieRegressor(estimator=ada, n_jobs=-1)  # 90% confidence interval
mapie.fit(train_X, train_y)
with open("../Streamlit/pages/mapie.pkl", "wb") as file:
    pickle.dump(mapie, file)
test_X = test_X[:1000]  # Use only 1,000 samples
test_y = test_y[:1000]
y_pred, y_pis = mapie.predict(test_X, alpha=0.1)

correct = (test_y >= y_pis[:, 0]) & (test_y <= y_pis[:, 1])  # True if within the interval
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
print(f"Custom Recall: {recall:.2f}")'''