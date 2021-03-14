#-----
# KMeans using Iris Dataset
# Author: Sarah H
# Date: 14 Mar 2021
#-----

# Unsupervised Learning Clustering using KMeans
# Evaluation & Optimization using Rand Index and Elbow Method

# Import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn import metrics # for Rand Index
from scipy.spatial.distance import cdist # for Euclidean Distance
import matplotlib.pyplot as plt

# Load data
iris = pd.read_csv("../input/iris/Iris.csv")

# Preprocessing
# drop Id and Species columns
df = iris.drop(['Id', 'Species'], axis=1)

# Transforming species column (label) into dummy variable
le_species = LabelEncoder()
le_species.fit(["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
labels_true = le_species.transform(iris[['Species']])
labels_true

# Data Normalization
# convert the df into numpy array using pandas
X = df.to_numpy()

# Replace NaN with zero using numpy
X = np.nan_to_num(X)

# Standardization
# Method A
#fitx = StandardScaler().fit(X)
#clus = fitx.transform(X)

# Method B
#clus = StandardScaler().fit(X).transform(X)

# Method C
clus = StandardScaler().fit_transform(X)

# Train & Evaluation
# Train model and evaluate using Rand Index
randIndex = []
K = range(1, 10)

for k in K:
    
    # Fit
    km = KMeans(init='k-means++', n_clusters=k, n_init=10)
    km.fit(X)
    labels_pred_loop = km.labels_
    
    # Measure
    score = metrics.adjusted_rand_score(labels_true, labels_pred_loop)
    randIndex.append(score)
    print(f'k={k}, score: {score}')

# Visualization
plt.plot(K, randIndex)
plt.title('Evaluation: Rand Index (Higher = better)')
plt.xlabel('K Clusters')
plt.ylabel('Rand Index')

# Optimization using Elbow Method
# ref: https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/

# distortion = avg of the squared distances from the cluster centers of the respective clusters. using euclidean distance.
# inertia = sum of squared instances of samples to their closest cluster center

distortions = []
inertias = []
K = range(1, 10)

for k in K:
    
    # Fit
    km2 = KMeans(init='k-means++', n_clusters=k, n_init=10)
    km2Model = km2.fit(X)
    labels_pred_loop2 = km2.labels_
    
    # Calculate distortions
    centroid = km2Model.cluster_centers_ # returns ndarray of coordinates of cluster centers
    euc_dist = cdist(X, centroid, 'euclidean')
    distort = sum(np.min(euc_dist, axis=1))/X.shape[0]
    distortions.append(distort)
    
    # Get inertias
    inertias.append(km2Model.inertia_)
    

# Visualization
plt.plot(K, distortions)
plt.title('(Optimization) Finding Optimal K value:\nElbow Method using Distortions')
plt.xlabel('K Clusters')
plt.ylabel('Distortion')
plt.show()


plt.plot(K, inertias)
plt.title('(Optimization) Finding Optimal K value:\nElbow Method using Inertias')
plt.xlabel('K Clusters')
plt.ylabel('Inertia')
plt.show()