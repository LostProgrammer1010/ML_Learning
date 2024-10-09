# Model Validation
- Comparing predications to target values in training data does not prove the accuracy of your data
- Summarize the models data into a understandable way
- Looking through a list of 10,000 pridication values and target values would take for ever
- Many types of metrics for summarizing model quality


# Mean Absolute Error (MAE)
- Predication Error: error=actual-predicated
- Take abs value of each error to work with positve numbers
- Taking the average of the error for all the predication error calculated
- Basically states that our predications are off by about X

## Example
House Cost: $150K
Predicated Value: $100K
Error: $50K

## "In-Sample" Scores
- Using single sample of house for building and evaluation of the model

### Problem
- If there is a feature of a home that does not affect the home price but it just happen that that feature is on expensive homes it will start associate expensive homes with that feature that has no effect on the price
- When adding new data to the model with its current pattern it will be very inaccurate

### Solution
- Train the model on data set but for the predication test it with new data
- Train model with data and see the accuracy with data it hasn't seen before
- Validation data: Data that the model was not trained on