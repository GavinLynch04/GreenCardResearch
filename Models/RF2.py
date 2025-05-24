import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from Data.Preprocessing.preprocess import *
import time
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold

X, y = preprocess()
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=27)

tree_reg = RandomForestRegressor(random_state=9, n_estimators=100, n_jobs=-1, min_samples_split=4, min_samples_leaf=1, max_depth=60)
tree_reg.fit(train_X, train_y)
start_predict = time.time()
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
