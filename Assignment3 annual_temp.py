# Muhammad  Huzaifa Abbasi
#ML Assignment3 Task2

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Assignment3 annual_temp.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

    #for GCAG
X1 = dataset.loc[(dataset.Source=="GCAG"),['Source','Year','Mean']]
XGCAG= X1.iloc[:,1:2].values
YGCAG= X1.iloc[:,2].values

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(XGCAG, YGCAG)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(XGCAG)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, YGCAG)

# Visualising the Polynomial Regression results
plt.scatter(XGCAG, YGCAG, color = 'b')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'g')
plt.title('Annual Temperature GCAG')
plt.xlabel('Year')
plt.ylabel('Mean Temp')
plt.show()

# Predicting a new result with Polynomial Regression
y_pred= lin_reg_2.predict(poly_reg.fit_transform([[2016]]))
print('Predicted Temp of GCAG in 2016:')
print(y_pred)
y_pred_1 = lin_reg_2.predict(poly_reg.fit_transform([[2017]]))
print('Predicted Temp of GCAG in 2017:')
print(y_pred_1)

    #FOR GISTEMP
X1 = dataset.loc[(dataset.Source=="GISTEMP"),['Source','Year','Mean']]
XGISTEMP= X1.iloc[:,1:2].values
YGISTEMP= X1.iloc[:,2].values

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(XGISTEMP, YGISTEMP)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(XGISTEMP)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, YGISTEMP)

# Visualising the Polynomial Regression results
plt.scatter(XGISTEMP, YGISTEMP, color = 'b')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'g')
plt.title('Annual Temperature GISTEMP')
plt.xlabel('Year')
plt.ylabel('Mean Temp')
plt.show()

# Predicting Avg Annual Temperaature 
y_pred= lin_reg_2.predict(poly_reg.fit_transform([[2016]]))
print('Predicted Temp of GISTEMP in 2016:')
print(y_pred)
y_pred_1 = lin_reg_2.predict(poly_reg.fit_transform([[2017]]))
print('Predicted Temp of GISTEMP in 2017:')
print(y_pred_1)