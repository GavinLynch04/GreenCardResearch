import pickle
import time

import joblib
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import shap
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split
from mapie.regression import MapieRegressor
from Data.Preprocessing.preprocess import *
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

X, y = preprocess()
print(y)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=207)

bestAda = AdaBoostRegressor(estimator =  DecisionTreeRegressor(max_depth=68, min_samples_leaf=2, min_samples_split=13, random_state=9), n_estimators=50, learning_rate=0.01, random_state = 9)
'''
start = time.time()
ada = bestAda.fit(train_X, train_y)
stop = time.time()
print(f"Training time: {stop - start}s")'''

shap_pickle = open('../Streamlit/pages/adaboost_model.pkl', 'rb')
ada = pickle.load(shap_pickle)
shap_pickle.close()


'''
explainer = LimeTabularExplainer(
    training_data=train_X.values,  # Train data (should be in numpy format)
    training_labels=train_y.values,  # Labels (should be in numpy format)
    mode="regression",  # Use "regression" for a regression task
    feature_names=train_X.columns.tolist(),  # Feature names
)

# Select an instance from the test set to explain (for example, the first sample)
instance = test_X.iloc[29].values  # Adjust index for a specific test instance

# Get the explanation for the selected instance
explanation = explainer.explain_instance(instance, ada.predict)

# Show the explanation (in the notebook or a browser window)
html = explanation.as_html()
with open("lime_explanation.html", "w") as f:
    f.write(html)'''
mapie = MapieRegressor(estimator=ada, n_jobs=-1)  # 90% confidence interval
mapie.fit(train_X, train_y)
with open("../Streamlit/pages/mapie.pkl", "wb") as file:
    pickle.dump(mapie, file)
test_X = test_X[:1000]  # Use only 1,000 samples
test_y = test_y[:1000]
y_pred, y_pis = mapie.predict(test_X, alpha=0.1)

correct = (test_y >= y_pis[:, 0]) & (test_y <= y_pis[:, 1])  # True if within the interval
accuracy = np.mean(correct)

# Calculate custom precision and recall
# Convert predictions into "correct" and "incorrect" categories
binary_y_pred = correct.astype(int)  # 1 for correct, 0 for incorrect
binary_y_true = np.ones_like(test_y)  # True labels are all "correct" (1)

# Calculate precision and recall based on binary values
precision = precision_score(binary_y_true, binary_y_pred)
recall = recall_score(binary_y_true, binary_y_pred)

# Display the results
print(f"Custom Accuracy (within 90% confidence): {accuracy:.2f}")
print(f"Custom Precision: {precision:.2f}")
print(f"Custom Recall: {recall:.2f}")
'''
import matplotlib.pyplot as plt
import numpy as np

#%%
importance = ada.feature_importances_

feature_imp = pd.DataFrame(list(zip(train_X.columns, importance)),
               columns = ['Feature', 'Importance'])

feature_imp = feature_imp.sort_values('Importance', ascending = False).reset_index(drop = True)

# Define categories and their respective feature lists
categories = {
    'COUNTRY_OF_CITIZENSHIP': [],
    'STATE': [],
    'PW_LEVEL': [],
    'FOREIGN_WORKER_INFO_EDUCATION': [],
    'FW_INFO_REQ_EXPERIENCE': [],
    'NAICS': [],
    'CLASS_OF': [],
    'JOB_INFO_EDUCATION': [],
    'JOB_INFO_TRAINING': [],
    'JOB_INFO_FOREIGN_ED': [],
    'RI_LAYOFF': [],
    'JOB_INFO_EXPERIENCE_NUM_MONTHS': [],
    'FW_INFO_YR_REL_EDU_COMPLETED': [],
    'EMPLOYER_NUM_EMPLOYEES': [],
    'JOB_INFO_EXPERIENCE': [],
    'PW_AMOUNT_9089': []
}

# Iterate over categories to populate the feature lists
for category in categories.keys():
    categories[category] = feature_imp[feature_imp['Feature'].str.contains(category)]['Feature'].tolist()

# Create a color map for categories
color_map = {
    'COUNTRY_OF_CITIZENSHIP': 'maroon',
    'STATE': 'red',
    'PW_LEVEL': 'darkorange',
    'FOREIGN_WORKER_INFO_EDUCATION': 'gold',
    'FW_INFO_REQ_EXPERIENCE': 'yellow',
    'NAICS': 'yellowgreen',
    'JOB_INFO_EXPERIENCE': 'green',
    'CLASS_OF': 'springgreen',
    'JOB_INFO_EDUCATION': 'lightseagreen',
    'JOB_INFO_TRAINING': 'deepskyblue',
    'JOB_INFO_FOREIGN_ED': 'royalblue',
    'RI_LAYOFF': 'navy',
    'JOB_INFO_EXPERIENCE_NUM_MONTHS': 'mediumpurple',
    'FW_INFO_YR_REL_EDU_COMPLETED': 'darkorchid',
    'EMPLOYER_NUM_EMPLOYEES': 'magenta',
    'PW_AMOUNT_9089': 'pink'
}

# Color each feature based on its category
colors = []
for feature in feature_imp['Feature']:
    for category, feature_list in categories.items():
        if feature in feature_list:
            colors.append(color_map[category])
            break
    else:
        colors.append('gray')  # default color for features not belonging to any category

# Selecting features with specific threshold for importance values
feature_imp_nonzero = feature_imp[feature_imp['Importance'] > 0.0005]

display_names = {
    'COUNTRY_OF_CITIZENSHIP': 'Country of Citizenship',
    'STATE': 'Work State',
    'PW_LEVEL': 'Pay Level',
    'FOREIGN_WORKER_INFO_EDUCATION': 'Worker Education Level',
    'FW_INFO_REQ_EXPERIENCE': 'Worker had Required Experience?',
    'NAICS': 'NAICS Code',
    'JOB_INFO_EXPERIENCE': 'Job Experience Required?',
    'CLASS_OF': 'Class of Admission',
    'JOB_INFO_EDUCATION': 'Job Education Level Requirement',
    'JOB_INFO_TRAINING': 'Job Training Offered?',
    'JOB_INFO_FOREIGN_ED': 'Job Foreign Education Equivalent Acceptable?',
    'RI_LAYOFF': 'Employer Recent Layoff?',
    'JOB_INFO_EXPERIENCE_NUM_MONTHS': 'Job Experience (Months)',
    'FW_INFO_YR_REL_EDU_COMPLETED': 'Year Highest Education Completed',
    'EMPLOYER_NUM_EMPLOYEES': 'Employer Number of Employees',
    'PW_AMOUNT_9089': 'Pay Amount'
}

# Bar plot with colored bars
plt.figure(figsize=(10, 6), dpi=100)
bars = plt.barh(feature_imp_nonzero['Feature'], feature_imp_nonzero['Importance'], color=colors)

# Additional plot configurations
plt.xlabel("Importance", fontsize=14)
plt.ylabel("Input Feature", fontsize=14)
plt.title("Feature Importance", fontsize=16)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Add legend for categories
legend_handles = [plt.Rectangle((0,0),1,1, color=color) for color in color_map.values()]
plt.legend(legend_handles, [display_names.get(category, category) for category in color_map.keys()], loc='upper right')
plt.show()'''