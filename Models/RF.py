import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from Data.Preprocessing.preprocess import *

X, y = preprocess()
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=27)

mtry_fraction = 0.5

num_predictors = len(train_X.columns)
mtry = int(np.ceil(mtry_fraction * num_predictors))

bestClassForest = RandomForestClassifier(max_depth=105, min_samples_leaf=2, min_samples_split=2,
                      random_state=9)

start = time.time()            # Start Time
forest = bestClassForest.fit(train_X, train_y)
stop = time.time()             # End Time
print(f"Training time: {stop - start}s")

y_pred = forest.predict(test_X)
print("Classification Report - \n",
      classification_report(test_y, y_pred))