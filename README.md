# Machine Learning Models

ML models made while progressing through the <a href="https://cognitiveclass.ai/courses/machine-learning-with-python">Machine Learning with Python course</a> by IBM.

## Regression 
<b> Simple Linear Regression </b>
- <a href="/py/simple-linear-grades.py">Student's Final Grade Predictor</a>
<br>Uses <a href="https://archive.ics.uci.edu/ml/datasets/Student+Performance">
Student Performance Data Set</a> from UCI ML Repository.

<b> Non-Linear Regression </b>
- <a href="/py/non-linear-gdp.py">GDP Predictor</a>
<br>Uses yearly GDP data. Can be obtained from various sources such as <a href="https://data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG">The World Bank</a>.

## Classification
<b> Decision Tree </b>
- <a href="/decisiontree-iris.py">Iris Species Decision Tree</a>
<br>Uses iris dataset
<br>Model Accuracy: 0.977
<br><a href="/py/decisiontree-iris.py"><img src="/plot/decisiontree-iris.jpg" width="500"></a>

<b> K-Nearest Neighbors </b>
- <a href="/py/knn-iris.py">Iris Species Identification using KNN</a>
<br>Uses iris dataset
<br>The best accuracy was with 1.0 with k = 8
<br><a href="/py/knn-iris.py"><img src="/plot/knn-iris-scatter.jpg" width="250"></a>
<a href="/py/knn-iris.py"><img src="/plot/knn-iris-accuracy.jpg" width="300"></a>

<b> Logistic Regression </b>
- <a href="/py/logistic-regr-classification-iris.py">Iris Species Identification using Logistic Regression</a>
<br>Uses iris dataset

<br>Raw Data

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
<br><a href="/py/logistic-regr-classification-iris.py"><img src="/plot/logistic-regr-iris-cmraw.jpg" width="950"></a>

<br>Normalized Data
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
<br><a href="/py/logistic-regr-classification-iris.py"><img src="/plot/logistic-regr-iris-cmnorm.jpg" width="950"></a>


Full credit belongs to its source. Thank you IBM for providing free education.
