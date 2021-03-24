#-----
# DBSCAN using Iris Dataset
# Author: Sarah H
# Date: 24 Mar 2021
#-----

# Unsupervised Learning using DBSCAn
# Evaluation using Homogeneity, Completeness, V-measure, Adjusted Rand Index, Adjusted Mutual Information, Silhouette Coefficient

# Load libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import metrics

# Load data
iris = pd.read_csv("../input/iris/Iris.csv")

# Define independent variables, X
X = iris[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]

# Data visualization
colors={'Iris-setosa': 'red', 'Iris-versicolor': 'blue', 'Iris-virginica': 'green'}
plt.scatter(iris["SepalLengthCm"], iris["SepalWidthCm"], c=iris["Species"].map(colors), label=colors)
plt.title('Iris: labels_true')
plt.xlabel('Sepal Length cm')
plt.ylabel('Sepal Width cm')

red_patch = mpatches.Patch(color='red', label='Setosa')
blue_patch = mpatches.Patch(color='blue', label='Versicolor')
green_patch = mpatches.Patch(color='green', label='Virginica')
plt.legend(handles=[red_patch, blue_patch, green_patch])
plt.show()

# (Preprocessing) Normalization
Xnorm = X.values  # returns a numpy array
min_max_scaler = MinMaxScaler()
X_mtx = min_max_scaler.fit_transform(Xnorm)

# Replace values
X = pd.DataFrame(X_mtx, columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])

# Create model

model2 = DBSCAN(eps=0.2, min_samples=3).fit(X_mtx)
labels2 = model2.labels_
unique_labels2 = set(labels2)

# create colors for the clusters
colors2 = plt.cm.Spectral(np.linspace(0,1,len(unique_labels2)))

# Distinguishing outliers (Outliers will be False)
# model.core_sample_indices_ get the indices of the core samples
iris_mask2 = np.zeros_like(model2.labels_, dtype=bool)
iris_mask2[model2.core_sample_indices_] = True

# plot the points with colors
for lab, col in zip(unique_labels2, colors2):
    if lab == -1:
        col = 'k'
    
    class_member_mask2 = (labels2 == lab)
    
    # plot datapoints that are clustered
    xy2 = X[class_member_mask2 & iris_mask2]
    plt.scatter(xy2.SepalLengthCm, xy2.SepalWidthCm, s=50, c=[col], marker=u'o', alpha=0.5)
    
    # plot the outliers
    xy2 = iris[class_member_mask2 & ~iris_mask2]
    plt.scatter(xy2.SepalLengthCm, xy2.SepalWidthCm, s=50, c=[col], marker=u'o', alpha=0.5)
    
plt.title("DBSCAN: labels_pred\nEpsilon: 0.2, Min Samples:3")
plt.show()

# Evaluation
# Ref: https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py

# Transforming species column (label) into dummy variable
le_species = LabelEncoder()
le_species.fit(["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
labels_true = le_species.transform(iris[['Species']]) # returns an array

labels_pred = model2.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels_pred)) - (1 if -1 in labels_pred else 0)
n_noise_ = list(labels_pred).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels_pred))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels_pred))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels_pred))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels_pred))
print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels_pred))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels_pred))