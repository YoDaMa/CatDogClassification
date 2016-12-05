from time import time

import numpy as np
import scipy.io as sio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn import svm

data = sio.loadmat('TwoSpirals.mat')
print(sio.whosmat('TwoSpirals.mat'))

X = data['X']
y = data['y']
print(X.shape)
print(y.shape)

# X = preprocessing.scale(X)
print("Scaled X Mean: {}".format(X.mean(axis=0)))
print("Scaled X Variance: {}".format(X.std(axis=0)))


mpl.style.use('ggplot')

fig = plt.figure()
# plt.grid(False)
ax = fig.add_subplot(111)
ax.scatter(X[:,0],X[:,1],s=100,c=y[:,0],cmap='jet')
ax.set_ylabel('X[:,1]')
ax.set_xlabel('X[:,0]')
ax.set_title('Visualization of Data (Scaled)')
plt.show()
print('Not linearly seperable')


# t0 = time()
# param_grid = {'C': [.001,.01,1,1e2,1e3, 5e3, 1e4, 5e4, 1e5, 1e6],
#               'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1,1], }
# clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), param_grid)
# clf = clf.fit(X, y[:, 0])
# print("done in %0.3fs" % (time() - t0))
# print("Best estimator found by grid search:")
# print(clf.best_estimator_)


C = .01
clf = svm.SVC(C=C)
clf.fit(X, y[:, 0])
# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
#    max_iter=-1, probability=False, random_state=None, shrinking=True,
#    tol=0.001, verbose=False)

scores = cross_val_score(clf, X, y[:, 0], cv=5)
print(scores)
print("Accuracy: % 0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

X_train, X_test, y_train, y_test = train_test_split(X, y[:, 0],
                                                    test_size=0.4, random_state=0)
clf1 = svm.SVC(kernel='rbf',C=C)
clf1 = clf1.fit(X_train,y_train)
clf1.score(X_test,y_test)
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test,y_pred))

temp = np.zeros((2,2))
kf = KFold(n_splits=5)
for train, test in kf.split(X):
    clf1 = svm.SVC(kernel='rbf', C=C)
    clf1 = clf1.fit(X_train, y_train)
    clf1.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    temp = np.add(temp,1/5*confusion_matrix(y_test,y_pred))

    print(confusion_matrix(y_test,y_pred))


# clf2 = svm.SVC(kernel='poly',C=C,degree=5)
# clf2 = clf2.fit(X, y[:, 0])
# scores1 = cross_val_score(clf2, X, y[:, 0], cv=5)
# print("Accuracy: % 0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))

h = .02
x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))



plt.subplot(211)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], s =200, c=y, cmap=plt.cm.coolwarm)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title('SVC with Gaussian (RBF) Kernel')


# plt.subplot(212)
# Z = clf2.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
# plt.scatter(X[:, 0], X[:, 1],s =200, c=y, cmap=plt.cm.coolwarm)
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.xticks(())
# plt.yticks(())
# plt.title('SVC with Polynomial (Degree 5) Kernel')



plt.show()
