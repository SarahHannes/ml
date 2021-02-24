#-----
# Decision Tree using Iris Dataset
# Author: Sarah H
# Date: 24 Feb 2021
#-----

# Import Libraries
import numpy as np 
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import graphviz

# Read data
iris = pd.read_csv("../input/iris/Iris.csv")

# Get the shape of the data
shape = iris.shape
print('\nDataFrame Shape :', shape)
print('\nNumber of rows :', shape[0])
print('\nNumber of columns :', shape[1])

# Assign X variables
X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm' ]].values

# Assign target variable, y
y = iris['Species']

# Splitting dataset into train/test set of size 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create an instance of DecisionTreeClassifier, using information gain(criterion=entropy) as accuracy measurement
# max_depth=default of None (nodes will be expanded until all leaves are pure)
irisTree = DecisionTreeClassifier(criterion='entropy')

# Train irisTree
irisTree.fit(X_train, y_train)

# Predict using the modelled tree
predict = irisTree.predict(X_test)

# Visual Comparison
print(predict[0:5])
print(y_test[0:5])

# Evaluation
print("DecisionTree's Accuracy:", metrics.accuracy_score(y_test, predict))

# Visualization
# tree.plot_tree(irisTree)
featureNames = iris.columns[1:-1]
targetNames = iris["Species"].unique().tolist()
dot_data = tree.export_graphviz(irisTree, out_file=None, feature_names=featureNames,class_names=targetNames, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)  
graph 

# Print graph as pdf file
graph.render("iris")