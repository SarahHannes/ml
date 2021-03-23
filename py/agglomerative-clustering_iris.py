#-----
# Agglomerative Hierarchical Clustering using Iris Dataset
# Author: Sarah H
# Date: 23 Mar 2021
#-----

# Unsupervised Learning using Agglomerative Hierarchical Clustering
# Evaluation using sklearn's homogeneity_score, adjusted_mutual_info_score, normalized_mutual_info_score, v_measure_score

# Import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import homogeneity_score, adjusted_mutual_info_score, normalized_mutual_info_score, v_measure_score

# Load data
iris = pd.read_csv("../input/iris/Iris.csv")

# Define independent variables, X
X = iris[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]

# Preprocessing (Normalization)
# MinMaxScaler transforms features by scaling each feature to a given range (default is 0,1)
Xnorm = X.values # returns a numpy array
min_max_scaler = MinMaxScaler() # initialize a preprocessing obj
X_mtx = min_max_scaler.fit_transform(Xnorm)

# Plot hierarchical dendrogram using Scipy
# Ref: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and the plot the dendogram
    
    # Create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0]) # get numpy ndarray of shape = number of rows in children_
    n_samples = len(model.labels_) # get the number of clusters
    
    for i, merge in enumerate(model.children_): # i=index, merge=the 2 index rows of bivariate merges?(not sure)
        current_count = 0
        for child_idx in merge: # merge returns a list of 2 elements
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    
    # Plot the dendrogram
    dendrogram(linkage_matrix, **kwargs)

# Clustering using sklearn
dist_matrix = euclidean_distances(X_mtx, X_mtx)

# Initialize Agglomerative Clustering object and fit the hierarchical clustering from features
# set distance_threshold=0 to get a full tree
agg2 = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(X_mtx)

# Plotting dendrogram
plt.title('Hierarchical Clustering Dendrogram')
plot_dendrogram(agg2, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

# Adding the cluster labels to a new col in df
iris['cluster_'] = agg2.labels_

# Looking at the characteristic of each cluster
agg2_iris = iris.groupby(['cluster_', 'Species'])["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"].mean()

# Visualization (Scatterplot)
n_cluster = max(agg2.labels_) + 1
colors = cm.rainbow(np.linspace(0, 1, n_cluster))
cluster_labels = list(range(0, n_cluster))

for color, label in zip(colors, cluster_labels):
    subset = iris[iris.cluster_ == label]
    for i in subset.index:
        plt.text(subset.SepalLengthCm[i], subset.PetalLengthCm[i], str(subset['Id'][i]), rotation=25)
    plt.scatter(subset.SepalLengthCm, subset.PetalLengthCm, c=color, label='cluster' + str(label), alpha=0.5)

plt.legend()
plt.title('Clusters')
plt.xlabel('Sepal Length cm')
plt.ylabel('Petal Length cm')

# Evaluation
# Transforming species column (label) into dummy variable
le_species = LabelEncoder()
le_species.fit(["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
labels_true = le_species.transform(iris[['Species']]) # returns an array
labels_true

# Transform df containing only cluster_ column to numpy array
labels_pred = iris['cluster_'].to_numpy()

# All 4 evaluation methods below are independent of the absolute values of the labels (permutation of the cluster label values wont change the score)

# A clustering result satisfies homogeneity if all of its clusters contain only data points which are members of a single class
homo_score = homogeneity_score(labels_true, labels_pred)

# Normalized Mutual Information (NMI) is a normalization of the Mutual Information (MI) score to scale the results between 0 (no mutual information) and 1 (perfect correlation)
mutual_info_score = adjusted_mutual_info_score(labels_true, labels_pred)

# Adjusted Mutual Information (AMI) is an adjustment of the Mutual Information (MI) score to account for chance
normalized_info_score = normalized_mutual_info_score(labels_true, labels_pred)

# V-measure is the harmonic mean between homogeneity and completeness
# identical normalized_mutual_info_score with the 'arithmetic' option for averaging
v_measure_score = v_measure_score(labels_true, labels_pred)

print('Evaluation:')
print(f'Homogeneity Score: {homo_score:.3f}')
print(f'Adjusted Mutual Info Score: {mutual_info_score:.3f}')
print(f'Normalized Mutual Info Score: {normalized_info_score:.3f}')
print(f'V-measure Score: {v_measure_score:.3f}')
