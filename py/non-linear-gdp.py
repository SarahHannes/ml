#-----
# GDP Predictor using Non-Linear Regression (Logistic)
# Author: Sarah H
# Date: 20 Feb 2021
#-----

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


def sigmoid(x, Beta_1, Beta_2):
    """
     :param x: independent variable values
     :param Beta_1: optimized parameter beta1
     :param Beta_2: optimized parameter beta2
     :return: predicted y values calculated based on parameters Beta_1 and Beta_2
     """
    y = 1 / (1 + np.exp(-Beta_1 * (x - Beta_2)))
    return y


# data import
df = pd.read_csv("china_gdp.csv")
x_data, y_data = (df["Year"].values, df["Value"].values)

# data normalization
xdata = x_data / max(x_data)
ydata = y_data / max(y_data)

# create a mask to select random rows
msk = np.random.rand(len(df)) < 0.8

# split the dataset (80% for training, 20% for testing) using normalized data
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y = ydata[~msk]

# get parameters (curve_fit from scipy returns 2 optimized parameters in popt)
popt, pcov = curve_fit(sigmoid, train_x, train_y)
print('beta1: ', popt[0])
print('beta2: ', popt[1])

# get predicted value using sigmoid function
test_y_hat = sigmoid(test_x, popt[0], popt[1])

# calc accuracy score
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))  # or mean residual error
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat, test_y))  # best possible score is 1.0. Score can be negative because the model can be arbitrarily worse
