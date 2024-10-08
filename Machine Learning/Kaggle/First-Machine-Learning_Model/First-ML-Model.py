import pandas as pd
from sklearn.tree import DecisionTreeRegressor # Used to create models

melbourne_file_path = 'C:/Users/Dusti/OneDrive/Documents/Machine Learning/Kaggle/First-Machine-Learning_Model/Input/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 

# Prints the columns in the data set
# print(melbourne_data.columns) # Uncomment to see the columns of the data set

# Drops the values that are missing
melbourne_data = melbourne_data.dropna(axis=0)


# y is the prediction target
# .column_name gives us the column with all the rows values
y = melbourne_data.Price


# Features are used to make perdictions
# We must choose these features some more is good and some times less is good
# Current example is to predict the price
# X is all the features we need from our data set
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

# print(X.describe()) # Uncomment to see the summary of the features

# Shows us the first few rows of the data 
# print(X.head()) # Uncomment to see the first few rows of the data set

# Define model. Specify random_state number to get the same result everytime
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
# X = Features
# y = prediction target
melbourne_model.fit(X,y)

print("Making Predictions for the following 5 Houses:")
print(X.head())
print("The predictions are ...")
# Predict the prices for the house that were provided
print(melbourne_model.predict(X.head()))

# The predictions of the model was exactly right for the price
print(melbourne_data.Price.head())
