import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer


melbourne_data = pd.read_csv("/Users/dustinmeyer/Documents/Github/ML_Learning/Machine Learning/Kaggle-Intermdiate/Input/melb_data.csv")

y = melbourne_data.Price

melb_predictors = melbourne_data.drop(['Price'], axis=1)
X = melb_predictors.select_dtypes(exclude=['object'])


train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


# Remove columns with missing values Approach

# Creates a list with all the columns with missing values
cols_with_missing = [col for col in train_X.columns if train_X[col].isnull().any()]

# Drop columns in training and validation data
# Removes the rows with the missing value in the column in both the trained and new set of data
reduced_X_train = train_X.drop(cols_with_missing, axis=1)
reduced_X_valid = val_X.drop(cols_with_missing, axis=1)

print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, train_y, val_y))
print()

# Imputation approach
my_imputer = SimpleImputer()

# Fit the data to the model but transforms the columns with missing values to the mean
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(train_X)) 

# Transforms the columns with missing values in the new data set with the mean
imputed_X_valid = pd.DataFrame(my_imputer.transform(val_X))

# Imputation removed column names; put them back
imputed_X_train.columns = train_X.columns
imputed_X_valid.columns = val_X.columns

print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, train_y, val_y))
print()

# Adding columns stating the values are artifical approach 3

# Make copy to avoid changing original data (when imputing)
X_train_plus = train_X.copy()
X_valid_plus = val_X.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    # Adds the columns new column stating if the value was missing and was replace with dummy value
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print("MAE from Approach 3 (An Extension to Imputation):")
print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, train_y, val_y))






