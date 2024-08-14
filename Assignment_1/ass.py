import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file
path = '/Users/lizgokhvat/Desktop/Projects/Intro_To_Data_Science_Projects/Assignment_1/'
df = pd.read_csv(path + 'insurData.csv')

# Print the number of rows and columns
print('\nNumber of rows and columns in the data set: ', df.shape)
print('')

# # Display the first few rows of the dataset
# df.head()

# df['Intercept'] = 1
# df['charges'] = df['charges'] / 1000
# # Convert categorical data to numeric using one-hot encoding
# df = pd.get_dummies(df, columns=['sex', 'region'], drop_first=True)
# # Convert 'Smoker' binary variable to numeric
# df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

# # Display the first few rows of the processed dataset
# df.head()

#Adding a column of 1's
df.insert(loc=0, column="s", value=1)
#Defining the charges in thousands( dividing by 1000)
df['charges'] = df['charges'].apply(lambda x: x / 1000)

#One-hot encoding
df = pd.get_dummies(df)

X = df[['s', 'age', 'sex_male', 'sex_female', 'bmi', 'children', 'smoker_yes', 'smoker_no',  'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest']]
y = df[['charges']]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Define the feature columns and target column
X = df.drop(columns=['charges'])
y = df['charges']

# Function to train the model and evaluate it
def train_and_evaluate(X, y):
    # Split the data into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict on the training set and test set
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate the mean squared error
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    return train_mse, test_mse

# Run 5 experiments
train_mses = []
test_mses = []
for _ in range(5):
    train_mse, test_mse = train_and_evaluate(X, y)
    train_mses.append(train_mse)
    test_mses.append(test_mse)
    print(f"Train MSE: {train_mse}, Test MSE: {test_mse}")

# Calculate the average MSE for train and test sets
avg_train_mse = np.mean(train_mses)
avg_test_mse = np.mean(test_mses)
print(f"Average Train MSE: {avg_train_mse}, Average Test MSE: {avg_test_mse}")
