# -----
# Logistic Regression Classification using Iris Dataset
# Author: Sarah H
# Date: 27 Feb 2021
# -----

# Import Libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Read data
df = pd.read_csv("../input/iris/Iris.csv")
df = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"]]

# Transforming Species column into dummy variable
le_species = LabelEncoder()
le_species.fit(["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
df[['Species']] = le_species.transform(df[['Species']])

# using Non-normalized (raw) data ----------------------------

# Define independent variable, X
X = np.asanyarray(df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]])

# Define dependent variable, y
y = np.asanyarray(df['Species'])

# Split test/ train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Build model
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)

# Using model to predict
yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)
print(yhat)
print(yhat_prob)

# Evaluation using Jaccard Index
print('average=None', metrics.jaccard_score(y_test, yhat, average=None))
print('micro', metrics.jaccard_score(y_test, yhat, average='micro'))
print('macro', metrics.jaccard_score(y_test, yhat, average='macro'))
print('weighted', metrics.jaccard_score(y_test, yhat, average='weighted'))

# Evaluation using confusion matrix
cm = metrics.confusion_matrix(y_test, yhat)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
disp.plot()

# Evaluation using confusion matrix (normalize=true -> return probability over true label (row))
cm_true = metrics.confusion_matrix(y_test, yhat, normalize='true')
disp_true = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_true, display_labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
disp_true.plot()

# Evaluation using confusion matrix (normalize=pred -> return probability over predicted(col))
cm_pred = metrics.confusion_matrix(y_test, yhat, normalize='pred')
disp_pred = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_pred, display_labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
disp_pred.plot()

# Evaluation using confusion matrix (normalize=all -> return probability over row and col)
cm_pred = metrics.confusion_matrix(y_test, yhat, normalize='all')
disp_pred = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_pred, display_labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
disp_pred.plot()

# Get Classification Report
target_name = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
print(metrics.classification_report(y_test, yhat, target_names=target_name))

# Evaluation using Logloss
print(metrics.log_loss(y_test, yhat_prob)) # 0.863

# using Normalized data ----------------------------

normdf = df.copy()

# Define independent variable, X
normX = np.asanyarray(normdf[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]])

# Define dependent variable, y
normy = np.asanyarray(normdf['Species'])

# X variables preprocessing
normX = preprocessing.StandardScaler().fit(normX).transform(normX)
normX

# Split test/ train sets
normX_train, normX_test, normy_train, normy_test = train_test_split(normX, normy, test_size=0.2, random_state=0)

# Build model
LRnorm = LogisticRegression(C=0.01, solver='liblinear').fit(normX_train, normy_train)

# Using model to predict
yhatnorm = LRnorm.predict(normX_test)
yhat_probnorm = LRnorm.predict_proba(normX_test)
print(yhatnorm)
print(yhat_probnorm)

# Evaluation using Jaccard Index
print('average=None', metrics.jaccard_score(normy_test, yhatnorm, average=None))
print('micro', metrics.jaccard_score(normy_test, yhatnorm, average='micro'))
print('macro', metrics.jaccard_score(normy_test, yhatnorm, average='macro'))
print('weighted', metrics.jaccard_score(normy_test, yhatnorm, average='weighted'))

# Evaluation using confusion matrix
cm_norm = metrics.confusion_matrix(normy_test, yhatnorm)
dispnorm = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
dispnorm.plot()

# Evaluation using confusion matrix (normalize=true -> return probability over true label (row))
cm_norm_true = metrics.confusion_matrix(normy_test, yhatnorm, normalize='true')
dispnorm_true = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_norm_true, display_labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
dispnorm_true.plot()

# Evaluation using confusion matrix (normalize=pred -> return probability over predicted(col))
cm_norm_pred = metrics.confusion_matrix(normy_test, yhatnorm, normalize='pred')
dispnorm_pred = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_norm_pred, display_labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
dispnorm_pred.plot()

# Evaluation using confusion matrix (normalize=all -> return probability over row and col)
cm_norm_pred = metrics.confusion_matrix(normy_test, yhatnorm, normalize='all')
dispnorm_pred = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_norm_pred, display_labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
dispnorm_pred.plot()

# Get Classification Report
target_name = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
print(metrics.classification_report(normy_test, yhatnorm, target_names=target_name))

# Evaluation using Logloss
print(metrics.log_loss(normy_test, yhat_probnorm)) # 0.855
