import pickle
import time

from skimage.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from mapie.regression import MapieRegressor
from tensorflow.python.ops.losses.losses_impl import mean_squared_error
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


y_pred = xgb.predict(test_X)

r2_score = r2_score(test_y, y_pred)
mae = mean_squared_error(test_y, y_pred)
print(f"MAE: {mae}")
print(f"R2: {r2_score}")

print("\nSetting up base XGBoost Regressor with GPU support...")
base_xgb = XGBRegressor(
    objective='reg:squarederror',
    random_state=9,

    # Parameters from your original model that we *might* fix or tune:
    subsample=0.8,                # Keep fixed for this example grid
    colsample_bytree=0.8,         # Keep fixed for this example grid
    gamma=0.1,                    # Keep fixed for this example grid
    reg_alpha=0.1,                # Keep fixed for this example grid
    reg_lambda=1                  # Keep fixed for this example grid
)

# 2. Define the parameter grid for GridSearchCV
#    Focus on key tuning parameters around your initial values.
#    Keep the grid relatively small initially due to dataset size.
#    Expand later if needed and computationally feasible.
param_grid = {
    'max_depth': [15, 20, 25, 30],             # Test depths around your original 6
    'learning_rate': [0.07, 0.09, 0.11], # Test rates around your original 0.05
    'n_estimators': [600],      # Test estimator counts around your 500
    'min_child_weight': [1]    # Test values around your original 3
    # Add more parameters here if desired, e.g.:
    # 'subsample': [0.7, 0.8],
    # 'colsample_bytree': [0.7, 0.8],
    # 'gamma': [0.05, 0.1, 0.2],
}
print(f"Parameter grid for GridSearchCV: {param_grid}")

# 3. Define the Cross-Validation strategy
#    KFold is standard for regression. Shuffle is recommended.
#    Using 3 folds is faster for large datasets.
cv_strategy = KFold(n_splits=3, shuffle=True, random_state=42)
print(f"CV strategy: {cv_strategy}")

# 4. Instantiate GridSearchCV
#    n_jobs=-1 might not be as critical if GPU does the heavy lifting,
#    but doesn't hurt. verbose=2 shows progress.
grid_search = GridSearchCV(
    estimator=base_xgb,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring='r2',       # Or 'neg_mean_absolute_error', 'neg_mean_squared_error'
    n_jobs=-1,          # Use available cores for data handling/prep if needed
    verbose=2           # Print progress updates
)

# --- Run Grid Search ---
print("\nStarting GridSearchCV with XGBoost (GPU)... This may take time.")
start_grid_search = time.time()
grid_search.fit(train_X, train_y)
end_grid_search = time.time()
print(f"GridSearchCV Training time: {end_grid_search - start_grid_search:.4f}s")

# --- Results ---
print("\n--- Grid Search Results ---")
print(f"Best Parameters found: {grid_search.best_params_}")
print(f"Best Cross-Validation R² score: {grid_search.best_score_:.4f}")

# --- Evaluate Best Model on Test Set ---
print("\nEvaluating the best model found by GridSearchCV on the test set...")
best_xgb_model = grid_search.best_estimator_ # The model refit with best params

start_predict = time.time()
y_pred = best_xgb_model.predict(test_X)
end_predict = time.time()
print(f"Prediction time on test set: {end_predict - start_predict:.4f}s")

# Use the imported functions directly
final_r2 = r2_score(test_y, y_pred)
final_mse = mean_squared_error(test_y, y_pred) # Also calculate MSE

print("\n--- Best Model Evaluation on Test Set ---")
print(f"Test Set R-squared (R²): {final_r2:.4f}")
print(f"Test Set Mean Squared Error (MSE): {final_mse:.4f}")