# -----
# K-Nearest Neighbors using Iris Dataset
# Author: Sarah H
# Date: 24 Feb 2021
# -----

# Import Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Read data
df = pd.read_csv("../input/iris/Iris.csv")

# Visualization
colors={'Iris-setosa': 'red', 'Iris-versicolor': 'blue', 'Iris-virginica': 'green'}
plt.scatter(df["SepalLengthCm"], df["PetalLengthCm"], c=df["Species"].map(colors), label=colors)
plt.title('Iris')
plt.xlabel('Sepal Length cm')
plt.ylabel('Petal Length cm')

red_patch = mpatches.Patch(color='red', label='Setosa')
blue_patch = mpatches.Patch(color='blue', label='Versicolor')
green_patch = mpatches.Patch(color='green', label='Virginica')
plt.legend(handles=[red_patch, blue_patch, green_patch])
plt.show()

# Assign X and y variables
x2 = df[['SepalLengthCm', 'PetalLengthCm']].values
y = df['Species'].values

# Split train/test sets
X_train2, X_test2, y_train2, y_test2 = train_test_split(x2, y, test_size=0.2, random_state=1)

# Create an instance of KNeighborsClassifier and train model (for k in range 1-10)
Ks = 10
mean_acc2 = np.zeros((Ks-1))
std_acc2 = np.zeros((Ks-1))
for n in range(1, Ks):
    #Train Model and Predict  
    neigh2 = KNeighborsClassifier(n_neighbors=n).fit(X_train2, y_train2)
    yhat2 = neigh2.predict(X_test2)
    mean_acc2[n-1] = metrics.accuracy_score(y_test2, yhat2)
    std_acc2[n-1] = np.std(yhat2 == y_test2)/np.sqrt(yhat2.shape[0])

mean_acc2

# Plot model accuracy for multiple Ks
plt.plot(range(1,Ks),mean_acc2,'g')
plt.fill_between(range(1,Ks),mean_acc2 - 1 * std_acc2,mean_acc2 + 1 * std_acc2, alpha=0.10)
plt.legend(('Accuracy', '+/- 3xstd'))
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()
print("The best accuracy was with", mean_acc2.max(), "with k=", mean_acc2.argmax()+1)