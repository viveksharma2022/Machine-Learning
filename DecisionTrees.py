import numpy as np
import scipy as sp
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

if __name__ == "__main__":
    iris = load_iris()
    X = iris.data[:,2:]
    y = iris.target

    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(X,y)

    predProbs = clf.predict_proba(X)
    print(predProbs)

    export_graphviz(clf, 'decisionTree.out', feature_names= iris.feature_names[2:], class_names=iris.target_names, rounded = True, filled= True)
