# Muhammad  Huzaifa Abbasi
#ML Assignment3 Task4

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Assignment3 housing price.csv')
X = dataset.iloc[:, 0:1].values
Y = dataset.iloc[:, 1].values

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, Y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

# Visualising the Polynomial Regression results
plt.scatter(X, Y, color = 'y')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'r')
plt.title('Housing price')
plt.ylabel('Price')
plt.xlabel('Housing ID')
plt.show()

# Predicting a new result with Polynomial Regression

y_pred_1 = lin_reg_2.predict(poly_reg.fit_transform([[1500]]))
print('Predicted Price for ID 1500 :')
print(y_pred_1)
y_pred_1 = lin_reg_2.predict(poly_reg.fit_transform([[2907]]))
print('Predicted Price for ID 2907')
print(y_pred_1)
y_pred_1 = lin_reg_2.predict(poly_reg.fit_transform([[3000]]))
print('Predicted Price for ID 3000')
print(y_pred_1)
