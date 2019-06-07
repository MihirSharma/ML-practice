from sklearn.datasets import make_blobs
import mglearn
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from mpl_toolkits.mplot3d import Axes3D, axes3d
import numpy as np

X,y = make_blobs(centers= 4, random_state= 8)
y = y%2
mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.xlabel("Feature 0")
plt.ylabel("feature 1")

linear_svm = LinearSVC(max_iter=10000, verbose=True).fit(X,y)
mglearn.plots.plot_2d_separator(linear_svm, X)

X_new = np.hstack([X, X[:, 1:] ** 2])

mask = y == 0

linear_svm_3d = LinearSVC(max_iter= 4000).fit(X_new, y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_

figure = plt.figure()
ax = Axes3D(figure, elev=-152, azim= -26)
xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)

XX, YY = np.meshgrid(xx, yy)
ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
ax.plot_surface(XX, YY, ZZ, rstride = 8, cstride = 8, alpha = 0.3, shade = True) 
ax.scatter(X_new[mask,0], X_new[mask, 1], X_new[mask, 2], c = 'b', cmap = mglearn.cm2, s = 60)
ax.scatter(X_new[~mask,0], X_new[~mask, 1], X_new[~mask, 2], c = 'r',marker = '^', cmap = mglearn.cm2, s = 60)
ax.set_xlabel("feature 0")
ax.set_ylabel("feature 1")
ax.set_zlabel("feature 1 ** 2")


figure2 = plt.figure()
ZZ = YY ** 2
dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
plt.contourf(XX, YY, dec.reshape(XX.shape), levels = [dec.min(), 0, dec.max()], cmap = mglearn.cm2, alpha = 0.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

plt.show()
