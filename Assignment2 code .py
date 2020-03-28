# Muhammad Huzaifa Abbasi
# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset.csv (Gender,  Age, brain weight, head size)
dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, 2:3].values
y = dataset.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set (using 1:4 as Test Size)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'salmon')
plt.plot(X_train, regressor.predict(X_train), color = 'navy')
plt.title('Plot-1 \n brain weight from head size (Training set)')
plt.xlabel('Head Size')
plt.ylabel('Brain weight')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'b')
plt.plot(X_train, regressor.predict(X_train), color = 'peru')
plt.title('Plot-2 \n brain weight from head size (Test set)')
plt.xlabel('Head Size')
plt.ylabel('Brain weight')
plt.show()

