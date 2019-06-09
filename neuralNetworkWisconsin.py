import mglearn
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X_train, X_test, y_train,  y_test = train_test_split(cancer.data, cancer.target, random_state =0 )
mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)
print(mlp.score(X_train, y_train))
print(mlp.score(X_test, y_test))

mean_on_train = X_train.mean(axis = 0)
std_on_train = X_train.std(axis = 0)

X_train_scaled = (X_train - mean_on_train)/ std_on_train
X_test_scaled = (X_test - mean_on_train)/ std_on_train

mlp_scaled = MLPClassifier( max_iter=1000, random_state=0, alpha = 1, hidden_layer_sizes=(100, 10))
mlp_scaled.fit(X_train_scaled, y_train)

print("\n")
print(mlp_scaled.score(X_train_scaled, y_train))
print(mlp_scaled.score(X_test_scaled, y_test))

plt.figure(figsize=(20,5))
plt.imshow(mlp_scaled.coefs_[0], interpolation='none', cmap= 'viridis')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel("columns in weight matrix")
plt.ylabel("Input features")
plt.colorbar()

plt.figure(num = 2, figsize=(2,10))
plt.margins(x = -0.49, y = -0.49)
plt.imshow(mlp_scaled.coefs_[1], interpolation='none', cmap= 'viridis', origin = 'upper', aspect='auto')
plt.yticks(range(100))
plt.xlabel("weights")
plt.ylabel("hidden layer 1 node")
plt.colorbar()

plt.figure(num = 3, figsize=(2,10))
plt.margins(x = -0.49, y = -0.49)
plt.imshow(mlp_scaled.coefs_[2], interpolation='none', cmap= 'viridis', origin = 'upper', aspect='auto')
plt.yticks(range(10))
plt.xlabel("weights")
plt.ylabel("hidden layer 2 node")
plt.colorbar()



plt.show()
