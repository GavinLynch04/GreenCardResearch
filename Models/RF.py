import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from Data.Preprocessing.preprocess import *

X, y = preprocess()
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=27)

mtry_fraction = 0.5

num_predictors = len(train_X.columns)
mtry = int(np.ceil(mtry_fraction * num_predictors))

bestClassForest = RandomForestRegressor(max_depth=105, min_samples_leaf=2, min_samples_split=2,
                      random_state=9, n_jobs=-1)

bestClassForest2 = RandomForestRegressor(max_features=mtry, max_depth=105, min_samples_leaf=2, min_samples_split=2, random_state=9, n_jobs=-1)

start = time.time()            # Start Time
forest = bestClassForest.fit(train_X, train_y)
stop = time.time()             # End Time
print(f"Training time: {stop - start}s")
forest2 = bestClassForest2.fit(train_X, train_y)
y_pred = bestClassForest.predict(test_X)
y_pred2 = bestClassForest2.predict(test_X)
import time
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold

print("R2 for old best: " + str(r2_score(test_y, y_pred)))
print("RMSE for old best: " + str(root_mean_squared_error(test_y, y_pred)))
print("R2 for old best2: " + str(r2_score(test_y, y_pred2)))
print("RMSE for old best2: " + str(root_mean_squared_error(test_y, y_pred2)))


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