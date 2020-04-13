# Muhammad  Huzaifa Abbasi
#ML Assignment3 Task4

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Assignment3 monthlyexp vs incom.csv')
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
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'c')
plt.title('Experience Vs Income')
plt.ylabel('Income $')
plt.xlabel('Months Experience')
plt.show()

# Predicting a new result with Polynomial Regression

y_pred_1 = lin_reg_2.predict(poly_reg.fit_transform([[20]]))
print('Predicted income for 20 month Experience :')
print(y_pred_1)
y_pred_1 = lin_reg_2.predict(poly_reg.fit_transform([[2]]))
print('Predicted income for 2 month Experience :')
print(y_pred_1)

