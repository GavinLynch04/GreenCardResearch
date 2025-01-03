import time

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
pd.set_option('display.max_columns', None)

from Data.Preprocessing.preprocess import *

X, y = preprocess()
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=27)

print(train_X.columns)

bestClassTree = DecisionTreeClassifier(max_depth=68, max_features=None, min_samples_leaf=2, min_samples_split=13,
                      random_state=9)

start = time.time()            # Start Time
forest = bestClassTree.fit(train_X, train_y)
stop = time.time()             # End Time
print(f"Training time: {stop - start}s")

y_pred = forest.predict(test_X)
print("Classification Report - \n",
      classification_report(test_y, y_pred))