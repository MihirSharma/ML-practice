import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

iris_dataset = load_iris()

# show dataset: Uncomment 2 lines below to see Feature names and their values

# print(iris_dataset["feature_names"])
# print(iris_dataset["data"])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = 0)  # split data into training and testing sets


iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset["feature_names"])

# plot scatter matrix
# grr = pd.plotting.scatter_matrix(iris_dataframe, marker = 'o', c =y_train, figsize = (15,15), hist_kwds= {'bins':20})
# plt.show()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(knn.score(X_test, y_test))
