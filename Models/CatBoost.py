from sklearn.metrics import accuracy_score, classification_report, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from Data.Preprocessing.preprocess import *
from catboost import CatBoostClassifier, CatBoostRegressor

X, y, cat = preprocess()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)

model = CatBoostRegressor(
    iterations=700,          # Number of boosting iterations
    learning_rate=0.1,       # Learning rate
    depth=20,                 # Depth of trees
    cat_features=cat,  # Specify categorical features
    verbose=1,              # Print progress every 1 iterations
    loss_function="RMSE",
    task_type="GPU",
    border_count=128,
)
'''from sklearn.model_selection import GridSearchCV

param_grid = {
    'depth': [6, 8, 10, 12, 14],
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [300, 500, 700],
    'l2_leaf_reg': [3, 5, 7],
    'border_count': [32, 64, 128],
    'task_type': ['GPU'],
    'auto_class_weights': ['Balanced'],
    'loss_function': ['MultiClass'],
}

grid_search = GridSearchCV(
    estimator=CatBoostClassifier(cat_features=cat, verbose=0),
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=3,
)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)'''

model.fit(X_train, y_train, eval_set=(X_test, y_test), plot=True)

y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)


print("R2 Test:", r2_score(y_test, y_pred))
print("RMSE Test: " + root_mean_squared_error(y_test, y_pred))
print("R2 Train:", r2_score(y_train, y_pred_train))
print("RMSE Train: " + root_mean_squared_error(y_train, y_pred_train))