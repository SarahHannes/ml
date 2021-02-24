#-----
# Student's Final Grade Predictor using Simple Linear Regression
# Author: Sarah H
# Date: 19 Feb 2021
#-----

## Import libraries ##
from sklearn import linear_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

### Step 1: Data Pre-processing ###
## Read data
df = pd.read_csv("../input/studentperformance/student-appended-mat-por.csv")

## Summarize data
df.describe()

## Separating X variable candidates and Y variable into a new df ##
cdf = df[['absences', 'G1', 'G2', 'G3']]

## Observing the distribution of all variables through histogram ##
viz = cdf[['absences', 'G1', 'G2', 'G3']]
fig = plt.figure(figsize = (15,10))
axes = fig.gca() # get current axes
viz.hist(ax=axes)
plt.show()

## Melting/ unpivoting cdf for FacetGrid ##
melted = cdf.melt(id_vars=['G3'], value_vars=['absences', 'G1', 'G2'])

## Plotting Scatterplot for all X candidates against Y (using FacetGrid) ##
g = sns.FacetGrid(melted, col='variable', sharex=False, sharey='row', hue='variable', height=5, margin_titles=True) # hue=changes the color based on variable
g.map(sns.scatterplot, 'value', 'G3', edgecolor='w')
g.fig.subplots_adjust(wspace=.02, hspace=.02)

# changing the title font size for all subplots
g.set_titles(size=25)
# to change xlim for second and third subplot
g.set(xlim=(0, None))
# to dynamically change the xlabel (to reflect its title)
for i in range(3):
    xlabel = g.axes[0, i].get_title() # get the title from each subplot
    xlabel_trunc = xlabel[11:] # truncating the titles :)
    g.axes[0,i].set_xlabel(xlabel_trunc) # setting it to xlabel

### Step 2: Choose the independent variable, X (the variable which we use to try to predict the Final Grade) ###
# From scatterplots, we have identified possible X candidates: G1 (Option 1) and G2 (Option 2)

### Step 3: Split the dataset into Test and Train sets for Option 1 ###

## Option 1: using G1 as X
# create a mask to select random rows
msk = np.random.rand(len(df)) < 0.8

# split the dataset (80% training, 20% testing)
train = cdf[msk]
test = cdf[~msk]

# viewing train data distribution
plt.scatter(train.G1, train.G3, color='orange')
plt.xlabel('1st Grade')
plt.ylabel('Final Grade')
plt.title('1st Grade against Final Grade')
plt.show()

### Step 4: Fit the Linear Model of Option 1 ###
## Modeling
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['G1']])  # selecting a column from train (G1) as X
train_y = np.asanyarray(train[['G3']])  # selecting a column from train (G3) as Y
regr.fit(train_x, train_y) # fitting the linear regression model

## Viewing the coefficients
print('𝜃1, Coefficients:', regr.coef_)
print('𝜃0, Intercept:', regr.intercept_)

theta1 = regr.coef_
theta0 = regr.intercept_

## Plot the best fit line over the data
plt.scatter(train.G1, train.G3, color='orange') # plot scatterplot
plt.plot(train_x, theta0+theta1[0][0]*train_x, '-k')
plt.xlabel('1st Grade')
plt.ylabel('Final Grade')
plt.title('1st Grade against Final Grade')

### Step 5: Evaluate our prediction model of Option 1 ###

## Evaluation
test_x = np.asanyarray(test[['G1']])
test_y = np.asanyarray(test[['G3']])
test_y_hat = regr.predict(test_x)

print('Mean absolute error: %.2f' % np.mean(np.absolute(test_y_hat - test_y)))
print('Residual sum of squares (MSE): %.2f' % np.mean((test_y_hat - test_y)**2))
print('R2-score: %.2f' % r2_score(test_y_hat, test_y))

# For option 1, we get only 0.42 accuracy score - so not that good..

### Step 3: Split the dataset into Test and Train sets of Option 2 ###

## Option 2: using G2 as X
# we have split the test and train dataset previously when doing it for G1
# so we can straight away proceed..
# viewing train data distribution
plt.scatter(train.G2, train.G3, color='green')
plt.xlabel('2nd Grade')
plt.ylabel('Final Grade')
plt.title('2nd Grade against Final Grade')
plt.show()

### Step 4: Fit the Linear Model of Option 2 ###

# Modeling
regrG2 = linear_model.LinearRegression()
train_xG2 = np.asanyarray(train[['G2']])  # selecting a column from train (G1) as X
train_yG2 = np.asanyarray(train[['G3']])  # selecting a column from train (G3) as Y
regrG2.fit(train_xG2, train_yG2) # fitting the linear regression model

theta1G2 = regrG2.coef_
theta0G2 = regrG2.intercept_

# Viewing the coefficients
print('𝜃1, Coefficients:', theta1G2)
print('𝜃0, Intercept:', theta0G2)

# Plot the best fit line over the data
plt.scatter(train.G2, train.G3, color='green') # plot scatterplot
plt.plot(train_xG2, theta0G2+theta1G2[0][0]*train_xG2, '-k')
plt.xlabel('2nd Grade')
plt.ylabel('Final Grade')
plt.title('2nd Grade against Final Grade')

### Step 5: Evaluate our prediction model of Option 1 ###

# Evaluation
test_xG2 = np.asanyarray(test[['G2']])
test_yG2 = np.asanyarray(test[['G3']])
test_y_hatG2 = regrG2.predict(test_xG2)

print('Mean absolute error: %.2f' % np.mean(np.absolute(test_y_hatG2 - test_yG2)))
print('Residual sum of squares (MSE): %.2f' % np.mean((test_y_hatG2 - test_yG2)**2))
print('R2-score: %.2f' % r2_score(test_y_hatG2, test_yG2))

# 0.83 accuracy score - better than using x=G1! yay!