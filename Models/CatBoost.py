from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from Data.Preprocessing.preprocess import *
from catboost import CatBoostClassifier

X, y, cat = preprocess()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)

model = CatBoostClassifier(
    iterations=500,          # Number of boosting iterations
    learning_rate=0.1,       # Learning rate
    depth=12,                 # Depth of trees
    cat_features=cat,  # Specify categorical features
    verbose=1               # Print progress every 1 iterations
)

model.fit(X_train, y_train, eval_set=(X_test, y_test), plot=True)

y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
