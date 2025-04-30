import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import time
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from Data.Preprocessing.preprocess import *

X, y = preprocess()
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=27)

mtry_fraction = 0.5

num_predictors = len(train_X.columns)
mtry = int(np.ceil(mtry_fraction * num_predictors))

bestClassForest = RandomForestRegressor(max_depth=105, min_samples_leaf=2, min_samples_split=2,
                      random_state=9, n_jobs=-1)

bestXGB = XGBRegressor (
max_depth=15, learning_rate=0.07, n_estimators=1000, gamma=0.1, min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
reg_alpha=0.1, reg_lambda=1,
random_state=9, nthread=8)

potential_xgb = XGBRegressor(
    # Core Boosting Parameters
    n_estimators=1500,         # Start high, use early stopping
    learning_rate=0.03,        # Relatively low learning rate for robustness
    objective='reg:squarederror', # Standard for regression

    # Tree Complexity & Regularization (Inspired by RF)
    max_depth=10,              # Moderately deep for XGBoost (RF's 105 is extreme)
                               # XGB doesn't usually need *that* deep due to boosting
    min_child_weight=1,        # Low value, allows splits on smaller groups (like RF's min_samples)
    gamma=0.2,                 # Minimum loss reduction for split (slight regularization)

    # Subsampling (Adds Randomness like RF Bagging)
    subsample=0.8,             # Use 80% of data per tree
    colsample_bytree=0.7,      # Use 70% of features per tree

    # Regularization (L1/L2)
    reg_alpha=0.01,            # Small L1 regularization
    reg_lambda=1.0,            # Standard L2 regularization

    # Other Parameters
    random_state=9,            # For reproducibility (match RF if desired)
    n_jobs=-1                  # Use all available CPU cores
)

potential_xgb.fit(train_X, train_y)
y_pred3 = potential_xgb.predict(test_X)

start = time.time()            # Start Time
forest = bestClassForest.fit(train_X, train_y)
stop = time.time()             # End Time
print(f"Training time: {stop - start}s")
forest2 = bestXGB.fit(train_X, train_y)
y_pred = bestClassForest.predict(test_X)
y_pred2 = bestXGB.predict(test_X)


print("R2 for old best: " + str(r2_score(test_y, y_pred)))
print("RMSE for old best: " + str(root_mean_squared_error(test_y, y_pred)))
print("R2 for best XGB: " + str(r2_score(test_y, y_pred2)))
print("RMSE for best XGB: " + str(root_mean_squared_error(test_y, y_pred2)))
print("R2 for new XGB: " + str(r2_score(test_y, y_pred3)))
print("RMSE for new XGB: " + str(root_mean_squared_error(test_y, y_pred3)))

tree_reg = RandomForestRegressor(random_state=9)

param_grid = {
    'max_depth': [None], # None allows full growth, start limited
    'n_estimators': [100, 200, 300],
    'min_samples_split': [30], # Increase minimums for large N
    'min_samples_leaf': [3],   # Increase minimums for large N
    'max_features': [None] # 'sqrt', fraction, or all features
    # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error'] # If you want to test criteria
}

# 3. Define the Cross-Validation strategy
#    KFold is standard for regression. Shuffle is recommended.
#    Using 3 or 5 folds is common. 3 folds will be faster.
cv_strategy = KFold(n_splits=3, shuffle=True, random_state=42)

# 4. Instantiate GridSearchCV
#    n_jobs=-1 uses all available CPU cores. verbose=2 shows progress.
#    Scoring: 'r2' or 'neg_mean_squared_error' are typical for regression.
print("\nSetting up GridSearchCV for DecisionTreeRegressor...")
print(f"Parameter grid: {param_grid}")
print(f"CV strategy: {cv_strategy}")

grid_search = GridSearchCV(
    estimator=tree_reg,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1, # USE ALL AVAILABLE CORES!
    verbose=2  # Print progress updates
)

# --- Run Grid Search ---
print("\nStarting GridSearchCV... This may take a significant amount of time!")
start_grid_search = time.time()
grid_search.fit(train_X, train_y)
end_grid_search = time.time()
print(f"GridSearchCV Training time: {end_grid_search - start_grid_search:.4f}s")

# --- Results ---
print("\n--- Grid Search Results ---")
print(f"Best Parameters found: {grid_search.best_params_}")
print(f"Best Cross-Validation RMSE score: {grid_search.best_score_:.4f}")

# --- Evaluate Best Model on Test Set ---
print("\nEvaluating the best model found by GridSearchCV on the test set...")
best_tree_model = grid_search.best_estimator_ # The model refit with best params

start_predict = time.time()
test_predictions = best_tree_model.predict(test_X)
end_predict = time.time()
print(f"Prediction time on test set: {end_predict - start_predict:.4f}s")

test_rmse = root_mean_squared_error(test_y, test_predictions)
test_r2 = r2_score(test_y, test_predictions)

print("\n--- Best Model Evaluation on Test Set ---")
print(f"Test Set Root Mean Squared Error (RMSE): {test_rmse:.4f}")
print(f"Test Set R-squared (R²): {test_r2:.4f}")

print("R2 for old best: " + str(r2_score(test_y, y_pred)))
print("RMSE for old best: " + str(root_mean_squared_error(test_y, y_pred)))
print("R2 for old best2: " + str(r2_score(test_y, y_pred2)))
print("RMSE for old best2: " + str(root_mean_squared_error(test_y, y_pred2)))