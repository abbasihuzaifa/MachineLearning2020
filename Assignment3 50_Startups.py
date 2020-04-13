# Muhammad  Huzaifa Abbasi
#ML Assignment3 Task1

# Importing the libraries
import numpy as np
import pandas as pd
 
dataset = pd.read_csv('Assignment3 50_Startups.csv')

    #For California
C =dataset.loc[dataset.State=='California',:]
C1 = C.iloc[:, -1].values
C2 = np.arange(17).reshape(-1, 1)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
C2_train, C2_test, C1_train, C1_test = train_test_split(C2, C1, test_size = 0.25, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(C2_train, C1_train)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
C2_poly = poly_reg.fit_transform(C2)
poly_reg.fit(C2_poly,C1)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(C2_poly, C1)


# predicting and Visualising the linear results
predicted = poly_reg.fit_transform([[5]])
CaliPredict = lin_reg_2.predict(predicted)
print('Prediction for a California Startup:')
print(CaliPredict)

        #For Florida
F =dataset.loc[dataset.State=='Florida',:]
F1 = F.iloc[:, -1].values
F2 = np.arange(16).reshape(-1, 1)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
F2_train, F2_test, F1_train, F1_test = train_test_split(F2, F1, test_size = 1/5, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(F2_train, F1_train)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
F2_poly = poly_reg.fit_transform(F2)
poly_reg.fit(F2_poly,F1)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(F2_poly, F1)

# predicting and Visualising the linear results
FloriPredict = lin_reg_2.predict(predicted)
print('Prediction for a Florida Startup:') 
print(FloriPredict)

if FloriPredict > CaliPredict:
  print("Predicted Max Profit of Florida is Greater then that of California")
if FloriPredict < CaliPredict:
      print("Predicted Max Profit of California is Greater then that of Florida")
  