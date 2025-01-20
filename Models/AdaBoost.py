import time

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from Data.Preprocessing.preprocess import *
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

X, y = preprocess()
print(y)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=207)

bestAda = AdaBoostRegressor(estimator =  DecisionTreeRegressor(max_depth=68, min_samples_leaf=2, min_samples_split=13, random_state=9), n_estimators=50, learning_rate=0.01, random_state = 9)

start = time.time()
ada = bestAda.fit(train_X, train_y)
stop = time.time()
print(f"Training time: {stop - start}s")

y_pred = ada.predict(test_X)
'''print("Classification Report - \n",
      classification_report(test_y, y_pred))'''

import numpy as np
from sklearn.metrics import recall_score, precision_score

# Define a tolerance for "correct" predictions (e.g., ±1 year)
tolerance = 1.0

# Calculate if predictions are within tolerance
correct = np.abs(y_pred - test_y) <= tolerance

# Calculate custom accuracy
accuracy = np.mean(correct)

# Calculate custom precision and recall
# Convert predictions into "correct" and "incorrect" categories
binary_y_pred = correct.astype(int)  # 1 for correct, 0 for incorrect
binary_y_true = np.ones_like(test_y)  # True labels are all "correct" (1)

# Calculate precision and recall based on binary values
precision = precision_score(binary_y_true, binary_y_pred)
recall = recall_score(binary_y_true, binary_y_pred)

# Display the results
print(f"Custom Accuracy (within ±{tolerance} years): {accuracy:.2f}")
print(f"Custom Precision: {precision:.2f}")
print(f"Custom Recall: {recall:.2f}")
