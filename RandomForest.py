import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from svmClassifier import PlotDecisionBoundary
from sklearn.datasets import make_moons

if __name__ == "__main__":
    iris = load_iris()
    X = iris.data[:, 2:] # petal length and width
    y = iris.target

    logClf = LogisticRegression()
    rForest = RandomForestClassifier()
    svmClf = SVC()

    votingClf = VotingClassifier(estimators=
                                 [("lr", logClf),
                                  ("rF", rForest),
                                  ("svm", svmClf)], voting='hard')
    votingClf.fit(X, y)

    PlotDecisionBoundary(votingClf, X, y, "Ensemble")

    X, y = make_moons(n_samples=100, noise = 0.15)

    bgClf = BaggingClassifier(estimator=DecisionTreeClassifier(min_samples_split=0.05),n_estimators=500, max_samples=50,bootstrap=True, n_jobs = 1, oob_score=True)
    bgClf.fit(X,y)

    PlotDecisionBoundary(bgClf, X, y, "Random forest")
    print(bgClf.oob_score_)