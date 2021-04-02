# Table of Contents
- [Supervised Learning](https://github.com/SarahHannes/ml/blob/main/README.md#Supervised-Learning)
  * [Regression](https://github.com/SarahHannes/ml/blob/main/README.md#Regression)
    + [Simple Linear Regression](https://github.com/SarahHannes/ml/blob/main/README.md#Simple-Linear-Regression)
    + [Non Linear Regression](https://github.com/SarahHannes/ml/blob/main/README.md#Non-Linear-Regression)
  * [Classification](https://github.com/SarahHannes/ml/blob/main/README.md#Classification)
    + [Decision Tree](https://github.com/SarahHannes/ml/blob/main/README.md#Decision-Tree)
    + [K-Nearest Neighbors](https://github.com/SarahHannes/ml/blob/main/README.md#K-Nearest-Neighbors)
    + [Logistic Regression](https://github.com/SarahHannes/ml/blob/main/README.md#Logistic-Regression)
    + [Support Vector Machine](https://github.com/SarahHannes/ml/blob/main/README.md#Support-Vector-Machine)
- [Unsupervised Learning](https://github.com/SarahHannes/ml/blob/main/README.md#Unsupervised-Learning)
  * [Clustering](https://github.com/SarahHannes/ml/blob/main/README.md#Clustering)
    + [K-Means](https://github.com/SarahHannes/ml/blob/main/README.md#K-Means)
    + [Agglomerative Hierarchical](https://github.com/SarahHannes/ml/blob/main/README.md#Agglomerative-Hierarchical)
    + [DBSCAN](https://github.com/SarahHannes/ml/blob/main/README.md#DBSCAN)
  * [Recommender Systems](https://github.com/SarahHannes/ml/blob/main/README.md#Recommender-Systems)
    + [Content-based](https://github.com/SarahHannes/ml/blob/main/README.md#Content-based)


<!-- toc -->
## Supervised Learning
### Regression 
#### Simple Linear Regression
- <a href="/py/simple-linear-regression_grades.py">Student's Final Grade Predictor</a>
<br>Uses <a href="https://archive.ics.uci.edu/ml/datasets/Student+Performance">
Student Performance Data Set</a> from UCI ML Repository.

#### Non-Linear Regression
- <a href="/py/non-linear-regression_gdp.py">GDP Predictor</a>
<br>Uses yearly GDP data. Can be obtained from various sources such as <a href="https://data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG">The World Bank</a>.

### Classification
#### Decision Tree
- <a href="/py/decisiontree_iris.py">Iris Species Decision Tree</a>
<br>Uses iris dataset
<br>Model Accuracy: 0.977
<br><a href="/py/decisiontree_iris.py"><img src="/plot/decisiontree_iris.jpg" width="500"></a>

#### K-Nearest Neighbors
- <a href="/py/knn_iris.py">Iris Species Identification using KNN</a>
<br>Uses iris dataset
<br>The best accuracy was with 1.0 with k = 8
<br><a href="/py/knn_iris.py"><img src="/plot/knn_iris_scatter.jpg" width="250"></a>
<a href="/py/knn_iris.py"><img src="/plot/knn_iris_accuracy.jpg" width="300"></a>

#### Logistic Regression
- <a href="/py/logistic-regression-classification_iris.py">Iris Species Identification using Logistic Regression</a>
<br>Uses iris dataset

<br>Prediction using Raw Data

```
Logloss = 0.863

Classification Report
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        11
Iris-versicolor       0.00      0.00      0.00        13
 Iris-virginica       0.32      1.00      0.48         6

       accuracy                           0.57        30
      macro avg       0.44      0.67      0.49        30
   weighted avg       0.43      0.57      0.46        30
```
Confusion Matrix
<br><a href="/py/logistic-regression-classification_iris.py"><img src="/plot/logistic-regression_iris_cm_raw-data.jpg" width="950"></a>

<br>Prediction using Normalized Data
```
Logloss = 0.855

Classification Report
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        11
Iris-versicolor       1.00      0.23      0.38        13
 Iris-virginica       0.38      1.00      0.55         6

       accuracy                           0.67        30
      macro avg       0.79      0.74      0.64        30
   weighted avg       0.88      0.67      0.64        30
```
Confusion Matrix
<br><a href="/py/logistic-regression-classification_iris.py"><img src="/plot/logistic-regression_iris_cm_normalized-data.jpg" width="950"></a>

#### Support Vector Machine
- <a href="/py/svm_iris.py">Iris Species Identification using various SVM kernels</a>
<br>Uses iris dataset
<br>Best accuracy obtained using SVM linear kernel with 0.85 accuracy score

linear Classifcation Report
```
                 precision    recall  f1-score   support

Iris-Versicolor       0.82      0.90      0.86        10
 Iris-Virginica       0.89      0.80      0.84        10

       accuracy                           0.85        20
      macro avg       0.85      0.85      0.85        20
   weighted avg       0.85      0.85      0.85        20
```

poly Classification Report
```
                 precision    recall  f1-score   support

Iris-Versicolor       0.75      0.90      0.82        10
 Iris-Virginica       0.88      0.70      0.78        10

       accuracy                           0.80        20
      macro avg       0.81      0.80      0.80        20
   weighted avg       0.81      0.80      0.80        20
```

rbf Classification Report
```
                 precision    recall  f1-score   support

Iris-Versicolor       0.75      0.90      0.82        10
 Iris-Virginica       0.88      0.70      0.78        10

       accuracy                           0.80        20
      macro avg       0.81      0.80      0.80        20
   weighted avg       0.81      0.80      0.80        20
```

Sigmoid Classification Report
```
                 precision    recall  f1-score   support

Iris-Versicolor       0.50      1.00      0.67        10
 Iris-Virginica       0.00      0.00      0.00        10

       accuracy                           0.50        20
      macro avg       0.25      0.50      0.33        20
   weighted avg       0.25      0.50      0.33        20
```
Confusion Matrix
<br><a href="/py/svm_iris.py"><img src="/plot/svm_iris_cm.jpg" width="800"></a>

## Unsupervised Learning
### Clustering
#### K-Means
- <a href="/py/kmeans_iris.py">Iris Species Clustering</a>
<br>Uses iris dataset
<br> k = 3 clusters gives the best Rand Index score at 0.73
<br> This evaluation method is possible since the original label (Species column) was retained as `label_true`, and comparison were made between `label_pred` and `label_true` using rand index.
<br><a href="/py/kmeans_iris.py"><img src="/plot/kmeans_iris_rand-index.jpg" width="250"></a>

Optimization using elbow methods were also performed using both distortion and inertia.
<br> Both methods confirm the best cluster is k = 3.
<br><a href="/py/kmeans_iris.py"><img src="/plot/kmeans_iris_elbow-method.jpg" width="800"></a>

#### Agglomerative Hierarchical
- <a href="/py/agglomerative-clustering_iris.py">Iris Species Clustering</a>
<br>Uses iris dataset
<br><a href="/py/agglomerative-clustering_iris.py"><img src="/plot/agglomerative-clustering_iris_scatter.jpg" width="350"></a>
<a href="/py/agglomerative-clustering_iris.py"><img src="/plot/agglomerative-clustering_iris_dendrogram.jpg" width="333"></a>

`iris.groupby(['cluster_', 'Species'])["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"].mean()`
```
 	                         SepalLengthCm 	SepalWidthCm 	PetalLengthCm 	PetalWidthCm
cluster_ 	Species
0 	       Iris-setosa 	    5.006000 	3.418000 	1.464000 	0.244000

1 	       Iris-versicolor 	6.700000 	3.000000 	5.000000 	1.700000
               Iris-virginica 	6.893939 	3.118182 	5.806061 	2.133333

2 	       Iris-versicolor 	5.920408 	2.765306 	4.244898 	1.318367
               Iris-virginica 	5.994118 	2.694118 	5.058824 	1.817647
```

```
Evaluation using Species column as ground truth:
Homogeneity Score: 0.744
Adjusted Mutual Info Score: 0.753
Normalized Mutual Info Score: 0.756
V-measure Score: 0.756
```

#### DBSCAN
- <a href="/py/dbscan_iris.py">Iris Species Clustering</a>
<br>Uses iris dataset
<br><a href="/py/dbscan_iris.py"><img src="/plot/dbscan_iris_scatter_labels_true.jpg" width="350"></a>
<a href="/py/dbscan_iris.py"><img src="/plot/dbscan_iris_scatter_labels_pred.jpg" width="333"></a>

```
Evaluation using Species column as ground truth:
Estimated number of clusters: 2
Estimated number of noise points: 3
Homogeneity: 0.576
Completeness: 0.877
V-measure: 0.696
Adjusted Rand Index: 0.554
Adjusted Mutual Info: 0.690
Silhouette Coefficient: 0.555
```

### Recommender Systems
#### Content-based
- <a href="/py/content-based_restaurant.py">Recomendating restaurants based on user past rating history</a>
- Selected feature is the cuisine type
- Datasets as provided <a href="datasets/content-based_restaurant/">here</a>. (Click <a href="https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data">here</a> to navigate to original source)
- Future improvement:
  - using knn to classify restaurant by cuisine type and use it as ground truth for evaluation
  - adding/ incorporating other rating criterias to get a more solid user profile (only food_rating was considered in the existing built model)
  - Somehow, all of the recommended placeID obtained from the model is not in geospatial2.csv file provided from the source (which I had assumed to contain all of the restaurants info).. Still unsure if this is a bug..
- Output: df containig topN of recommended placeID & its weighted recommendation score for the specified userID
```
get_recommendation("U1138")
```
```
Rcuisine 	total_by_place
placeID 	
132774 	7
135099 	6
135098 	4
135103 	4
135097 	4
```

Full credit belongs to its source. Thank you IBM for providing free education.
