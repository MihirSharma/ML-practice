from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from mglearn.datasets import load_extended_boston
from sklearn.model_selection import train_test_split
import numpy as np

print("\n\n")

X,y = load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 0)

lr = LinearRegression().fit(X_train,y_train)

print("linear regression score: training set = " + str(lr.score(X_train,y_train)))
print("linear regression score: testing set = " + str(lr.score(X_test,y_test)))

print("\n\n ----------------------------------------------------------------------------------------------------------------------- \n\n")

for i in range(-4,1):
    ridge = Ridge(alpha = 10**i).fit(X_train, y_train)
    print("ridge regression score: training set (alpha = " +  str(10**i) +") = " + str(ridge.score(X_train,y_train)))
    print("ridge regression score: testing set (alpha = " +  str(10**i) +") = " + str(ridge.score(X_test,y_test)))
    print("\n")

print("\n\n ----------------------------------------------------------------------------------------------------------------------- \n\n")

for i in range(-4,1):
    lasso = Lasso(alpha= 10**i, max_iter= 100000).fit(X_train, y_train)
    print("lasso score: training set (max iter = 100000 , alpha = " +  str(10**i) +") = " + str(lasso.score(X_train,y_train)))
    print("lasso score: testing set (max iter = 100000 , alpha = " +  str(10**i) +") = " + str(lasso.score(X_test,y_test)))
    print("number of features used = " + str(np.sum(lasso.coef_ != 0)))
    print("\n")