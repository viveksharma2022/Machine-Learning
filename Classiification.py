import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

if __name__ == "__main__":
    mnist = fetch_openml('mnist_784', version=1)

    X,y = mnist["data"], mnist["target"]

    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

    # Train binary classifier for number 5
    yTrain5 = (y_train == 5)
    yTest5 = (y_test == 5)

    randomF = RandomForestClassifier(random_state=42)
    yProbsPredict = cross_val_predict(randomF, X_train, yTrain5, cv = 3, method="predict_proba")
    yScoresForest = yProbsPredict[:,0]

    fprForest, tprForest, thresholdsForest = roc_curve(yProbsPredict, yScoresForest)

    # multiclass prediction

    yTrainLarge = y_train.astype('int') > 7
    yTrainOdd = y_train.astype('int') % 2 == 0
    yLabelMulti = np.c_[yTrainLarge, yTrainOdd]

    knnClf = KNeighborsClassifier()
    knnClf.fit(X_train, yLabelMulti)

    yTrainKnnPred = cross_val_predict(knnClf, X_train, yLabelMulti, cv=3)

    print(f1_score(yLabelMulti, yTrainKnnPred, average="macro"))


