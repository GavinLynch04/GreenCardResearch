import time

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from Data.Preprocessing.preprocess import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

X, y = preprocess()

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=27)


bestAda = AdaBoostClassifier(estimator =  DecisionTreeClassifier(max_depth=68, min_samples_leaf=2, min_samples_split=13, random_state=9), n_estimators=50, learning_rate=0.01, random_state = 9)

start = time.time()
ada = bestAda.fit(train_X, train_y)
stop = time.time()
print(f"Training time: {stop - start}s")

y_pred = ada.predict(test_X)
print("Classification Report - \n",
      classification_report(test_y, y_pred))