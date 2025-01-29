import numpy as np
from sklearn.datasets import load_iris
from svmClassifier import PlotDecisionBoundary
from sklearn.datasets import make_moons
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X, y = make_moons(n_samples=100, noise = 0.25)

    adaClf = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=200,
                                algorithm ="SAMME", learning_rate=0.5)
    adaClf.fit(X,y)
    PlotDecisionBoundary(adaClf, X, y, "AdaBoost")

    XTrain, XVal, YTrain, YVal = train_test_split(X,y)

    gbrt = GradientBoostingClassifier(max_depth=2, n_estimators=120)
    gbrt.fit(XTrain,YTrain)

    errors = [mean_squared_error(YVal, ypred) for ypred in gbrt.staged_predict(XVal)]
    bestEstimators = np.argmin(errors) + 1

    plt.figure()
    plt.plot(errors)
    plt.title("Errors")
    plt.show()

    gbrt = GradientBoostingClassifier(max_depth=2, n_estimators=bestEstimators)
    gbrt.fit(XTrain,YTrain)
    plt.plot(YVal, '.')
    plt.plot(gbrt.predict(XVal), '-')
    plt.show()


