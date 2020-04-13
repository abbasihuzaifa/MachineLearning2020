# Muhammad  Huzaifa Abbasi
#ML Assignment3 Task3

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Assignment3 global_co2.csv')
X = dataset.iloc[:, 0:1].values
Y = dataset.iloc[:, 1].values

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, Y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

# Visualising the Polynomial Regression results
plt.scatter(X, Y, color = 'c')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'r')
plt.title('CO2 Production')
plt.xlabel('Year')
plt.ylabel('Total CO2')
plt.show()

# Predicting a new result with Polynomial Regression

y_pred_1 = lin_reg_2.predict(poly_reg.fit_transform([[2011]]))
print('Predicted CO2 production in 2011:')
print(y_pred_1)
y_pred_1 = lin_reg_2.predict(poly_reg.fit_transform([[2012]]))
print('Predicted CO2 production in 2012:')
print(y_pred_1)
y_pred_1 = lin_reg_2.predict(poly_reg.fit_transform([[2013]]))
print('Predicted CO2 production in 2013:')
print(y_pred_1)
