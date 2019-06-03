from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

cancer_dataset = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer_dataset['data'], cancer_dataset['target'], stratify = cancer_dataset['target'] ,random_state = 66)

training_accuracy = []
test_accuracy = []

neighbor_settings = range(1,11)

for n_neighbors in neighbor_settings:
    clf = KNeighborsClassifier(n_neighbors= n_neighbors)
    clf.fit(X_train,y_train)
    training_accuracy.append(clf.score(X_train,y_train))
    test_accuracy.append(clf.score(X_test, y_test))
    
plt.plot(neighbor_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbor_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Number of neighbors")
plt.legend()
plt.show()