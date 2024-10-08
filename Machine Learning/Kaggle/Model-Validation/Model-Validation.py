import pandas as pd# type: ignore

melbourne_file_path = "/Users/dustinmeyer/Documents/Github/ML_Learning/Machine Learning/Kaggle/Model-Validation/Input/melb_data.csv"
melbourne_data = pd.read_csv(melbourne_file_path)

filtered_melbourne_data = melbourne_data.dropna(axis=0)

y = filtered_melbourne_data.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']

X = filtered_melbourne_data[melbourne_features]

from sklearn.tree import DecisionTreeRegressor # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import mean_absolute_error # type: ignore
"""
Training on data and testing with same data it was trained on


melbourne_model = DecisionTreeRegressor()

melbourne_model.fit(X,y)



# Predicts the home prices based on the features that we gave it
predicted_home_prices = melbourne_model.predict(X)

# Calculates the mean_absolute_error
# error = abs(actual - predicated)
print(mean_absolute_error(y, predicted_home_prices))
"""

# Train the model with train_X data and run predication with val_X data
# Give the features of train_y and run validation testing with val_y

train_X, val_X, train_y, val_y = train_test_split(X,y, random_state=0)

melbourne_model = DecisionTreeRegressor(random_state=0)

# Train the model on X data and y features
melbourne_model.fit(train_X, train_y)

# Make the predications on new data that it hasn't seen before
val_predications = melbourne_model.predict(val_X)

# See how it is very inaccurate
print(mean_absolute_error(val_y, val_predications))
