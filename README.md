# Machine Learning Models

ML models made while progressing through the <a href="https://cognitiveclass.ai/courses/machine-learning-with-python">Machine Learning with Python course</a> by IBM.

## Regression 
<b> Simple Linear Regression </b>
- <a href="/py/simple-linear-regression_grades.py">Student's Final Grade Predictor</a>
<br>Uses <a href="https://archive.ics.uci.edu/ml/datasets/Student+Performance">
Student Performance Data Set</a> from UCI ML Repository.

<b> Non-Linear Regression </b>
- <a href="/py/non-linear-regression_gdp.py">GDP Predictor</a>
<br>Uses yearly GDP data. Can be obtained from various sources such as <a href="https://data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG">The World Bank</a>.

## Classification
<b> Decision Tree </b>
- <a href="/py/decisiontree_iris.py">Iris Species Decision Tree</a>
<br>Uses iris dataset
<br>Model Accuracy: 0.977
<br><a href="/py/decisiontree_iris.py"><img src="/plot/decisiontree-iris.jpg" width="500"></a>

<b> K-Nearest Neighbors </b>
- <a href="/py/knn_iris.py">Iris Species Identification using KNN</a>
<br>Uses iris dataset
<br>The best accuracy was with 1.0 with k = 8
<br><a href="/py/knn_iris.py"><img src="/plot/knn-iris-scatter.jpg" width="250"></a>
<a href="/py/knn_iris.py"><img src="/plot/knn-iris-accuracy.jpg" width="300"></a>

<b> Logistic Regression </b>
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
<br><a href="/py/logistic-regression-classification_iris.py"><img src="/plot/logistic-regr-iris-cmraw.jpg" width="950"></a>

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
<br><a href="/py/logistic-regression-classification_iris.py"><img src="/plot/logistic-regr-iris-cmnorm.jpg" width="950"></a>

<b> Support Vector Machine </b>
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
<br><a href="/py/svm_iris.py"><img src="/plot/svm-iris-cm.jpg" width="800"></a>

Full credit belongs to its source. Thank you IBM for providing free education.
