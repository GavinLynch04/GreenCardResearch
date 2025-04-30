import time

import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor

pd.set_option('display.max_columns', None)
from Data.Preprocessing.preprocess import *

X, y = preprocess()
print(X.shape)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=27)


bestClassTree = DecisionTreeRegressor(max_depth=None, max_features=None, min_samples_leaf=5, min_samples_split=50,
                      random_state=9)
bestClassTree.fit(train_X, train_y)
y_pred = bestClassTree.predict(test_X)
y_pred_train = bestClassTree.predict(train_X)
print("R2 for old best: " + str(r2_score(test_y, y_pred)))
print("RMSE for old best: " + str(root_mean_squared_error(test_y, y_pred)))
print("R2 for train set: " + str(r2_score(train_y, y_pred_train)))
print("RMSE for train set: " + str(root_mean_squared_error(train_y, y_pred_train)))

tree_reg = DecisionTreeRegressor(random_state=9)

# 2. Define the parameter grid for DecisionTreeRegressor
#    Adjust ranges based on dataset size (800k x 350) and computational resources.
#    Higher min_samples values help prevent overfitting on large datasets.
#    Consider limiting max_depth initially.
param_grid = {
    'max_depth': [10, 20, 30, None], # None allows full growth, start limited
    'min_samples_split': [10, 50, 100, 200], # Increase minimums for large N
    'min_samples_leaf': [5, 25, 50, 100],   # Increase minimums for large N
    'max_features': ['sqrt', 0.5, None] # 'sqrt', fraction, or all features
    # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error'] # If you want to test criteria
}

# 3. Define the Cross-Validation strategy
#    KFold is standard for regression. Shuffle is recommended.
#    Using 3 or 5 folds is common. 3 folds will be faster.
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

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
    n_jobs=-1,
    verbose=2
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
print(f"Best Cross-Validation R² score: {grid_search.best_score_:.4f}")

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