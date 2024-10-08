# Pandas is used to create tables like excel
import pandas as pd


# Save filepath to variable for easier access
melbourne_file_path = 'C:\Users\Dusti\OneDrive\Documents\Machine Learning\Kaggle\Basic-Data-Exploration\Input\melb_data.csv'

# Reads the data and stores in in DataFrame titled Melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path)

# Prints summary of the data in melbourne_data
print(melbourne_data.describe())

