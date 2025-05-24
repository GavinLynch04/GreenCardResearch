from mapie.metrics import regression_coverage_score
from sklearn.metrics import root_mean_squared_error
import pickle
from Data.Preprocessing.preprocess import preprocess
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

def downcast_df(df):
    for col in df.columns:
        if df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')  # downcasts to int32 or smaller
        elif df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')    # downcasts to float32 or smaller
    return df


X, y = preprocess()


X = downcast_df(X)
print(X.info())
y = y.astype(np.float32)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


base_model = RandomForestRegressor(max_depth=130, min_samples_leaf=1, min_samples_split=4, n_estimators=300,
                      random_state=9, n_jobs=-1)
base_model.fit(x_train, y_train)
y_pred = base_model.predict(x_test)
print(root_mean_squared_error(y_test, y_pred))

with open("rf_model.pkl", "wb") as f:
    pickle.dump(base_model, f)
'''
import numpy as np

from mapie.regression import MapieRegressor
from mapie.metrics import regression_coverage_score

mapie = MapieRegressor(estimator = base_model, # Prediction model to use
                       n_jobs = -1,
                       random_state = 42)

print(x_train.memory_usage(deep=True).sum() / 1024**2, "MB")
print(y_train.memory_usage(deep=True) / 1024**2, "MB")



# Fit mapie regressor on training data
mapie.fit(x_train, y_train)
with open("mapie_model.pkl", "wb") as f:
    pickle.dump(mapie, f)

alpha = 0.1 # For 90% confidence level

n_samples = 2000

# Evenly spaced indices across the full dataset
indices = np.linspace(0, len(x_test) - 1, n_samples, dtype=int)

# Select the subset
x_subset = x_test.iloc[indices].astype(np.float32)
y_subset = y_test.iloc[indices].astype(np.float32)  # if you want the corresponding labels

# Use mapie.predict() to get predicted values and intervals
y_test_pred, y_test_pis = mapie.predict(x_subset, alpha = alpha)

predictions = y_subset.to_frame()
predictions.columns = ['Actual Value']
predictions["Predicted Value"] = y_test_pred.round()
predictions["Lower Value"] = y_test_pis[:, 0].round()
predictions["Upper Value"] = y_test_pis[:, 1].round()

coverage = regression_coverage_score(y_subset,           # Actual values
                                     y_test_pis[:, 0], # Lower bound of prediction intervals
                                     y_test_pis[:, 1]) # Upper bound of prediction intervals

coverage_percentage = coverage * 100
print(f"Coverage: {coverage_percentage:.2f}%")


# Import necessary library for setting up the plot format
import matplotlib as mpl

# Sort the predictions by 'Actual Value' for better visualization and reset the index
sorted_predictions = predictions.sort_values(by=['Actual Value']).reset_index(drop=True)

# Create a figure and axis object with specified size and resolution
fig, ax = plt.subplots(figsize=(25, 10), dpi=250)

# Plot the actual values with green dots
plt.plot(sorted_predictions["Actual Value"], 'go', markersize=4, label="Actual Value")

# Fill the area between the lower and upper bounds of the prediction intervals with semi-transparent green color
plt.fill_between(np.arange(len(sorted_predictions)),
                 sorted_predictions["Lower Value"],
                 sorted_predictions["Upper Value"],
                 alpha=0.2, color="green", label="Prediction Interval")

# Set font size for x and y ticks
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Format y-axis to show values with commas as thousand separators
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

# Set the limit for the x-axis to cover the range of samples
plt.xlim([0, len(sorted_predictions)])

# Label the x-axis and y-axis with appropriate font size
plt.xlabel("Samples", fontsize=20)
plt.ylabel("Target", fontsize=20)

# Add a title to the plot, including the coverage percentage, with bold formatting
plt.title(f"Prediction Intervals and Coverage: {coverage_percentage:.2f}%", fontsize=25, fontweight="bold")

# Add a legend to the plot, placed in the upper left, with specified font size
plt.legend(loc="upper left", fontsize=20)

# Save the plot as a PDF file with tight layout
plt.savefig("prediction_intervals_coverage.svg", format="svg", bbox_inches="tight")

# Display the plot
plt.show();'''