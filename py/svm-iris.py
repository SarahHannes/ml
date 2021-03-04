#-----
# SVM using Iris Dataset
# Author: Sarah H
# Date: 4 Mar 2021
#-----

# Classification of Versicolor and Virginica using various SVM kernels

# Import libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn import svm
from sklearn import metrics

# Read data
df = pd.read_csv("../input/iris/Iris.csv")
df = df[['SepalLengthCm', 'SepalWidthCm', 'Species']]

# Transforming species column into dummy variable
le_species = preprocessing.LabelEncoder()
le_species.fit(["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
df[['Species']] = le_species.transform(df[['Species']])

# Selecting only 2 classes (of target variable)
df = df[df.Species.isin([1, 2])]

# Define independent variables, X
X = np.asanyarray(df[['SepalLengthCm', 'SepalWidthCm']])

# Define dependent variable, y
y = np.asanyarray(df['Species'])

# Split into train/test sets using stratifiedShuffleSplit to get statified randomized folds
sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=1)
for train_index, test_index in sss.split(X,y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# Store classifiers instances (Create model and fit)
classifiers = {
    'linear': svm.SVC(kernel='linear').fit(X_train, y_train),
    'poly': svm.SVC(kernel='poly').fit(X_train, y_train),
    'rbf': svm.SVC(kernel='rbf').fit(X_train, y_train),
    'sigmoid': svm.SVC(kernel='sigmoid').fit(X_train, y_train)
}

n_classifiers = len(classifiers)

for index, (name, classifier) in enumerate(classifiers.items()):
    
    # Predict using classifier
    yhat_classifier = classifier.predict(X_test)
    
    # Evaluate using f1 score
    print(metrics.f1_score(y_test, yhat_classifier, average='weighted'))
    
    # Evaluate using Jaccard Index
    print(f'Jaccard Index (average = None) for {name}: {metrics.jaccard_score(y_test, yhat_classifier, average=None)}')
    print(f'Jaccard Index (micro) for {name}: {metrics.jaccard_score(y_test, yhat_classifier, average="micro")}')
    print(f'Jaccard Index (macro) for {name}: {metrics.jaccard_score(y_test, yhat_classifier, average="macro")}')
    print(f'Jaccard Index (weighted) for {name}: {metrics.jaccard_score(y_test, yhat_classifier, average="weighted")}')
    
    # Get Accuracy Score
    accuracy = metrics.accuracy_score(y_test, yhat_classifier)
    
    # Evaluate using confusion matrix
    cm_classifier = metrics.confusion_matrix(y_test, yhat_classifier)
    disp_cm_classifier = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_classifier, display_labels=["Iris-Versicolor", "Iris-Virginica"])
    disp_cm_classifier.plot()
    plt.title(f'SVM {name} kernel\nAccuracy: {accuracy}')
    
    # Get Classification Report
    target_name = ["Iris-Versicolor", "Iris-Virginica"]
    print(metrics.classification_report(y_test, yhat_classifier, target_names=target_name))