import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

if __name__ == "__main__":

    irisData = datasets.load_iris()
    X = irisData["data"][:, (2,3)]
    Y = (irisData["target"] == 2).astype('int')

    svmClf = Pipeline([
        ("scaler", StandardScaler()),
        ("LinearSVC", LinearSVC(C = 1, loss = "hinge"))
    ])

    svmClf.fit(X, Y)

    # kernel trick
    coef = 1.0
    X, y = make_moons(n_samples=100, noise = 0.15)
    svmClfNonLin = Pipeline([
        ("scaler", StandardScaler()),
        ("svmClf", SVC(C = 5, kernel="poly", coef0=coef,degree=3))
    ])
    svmClfNonLin.fit(X,y)

    # Plot decision boundary
    h = .02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the label for every point in the mesh grid
    Z = svmClfNonLin.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=100, cmap=plt.cm.coolwarm)
    plt.title(f'SVM Classifier with Polynomial Kernel (coef0={coef})')
    plt.show()

    pause = 1
