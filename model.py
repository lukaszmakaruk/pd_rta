from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import pickle


class Perceptron:

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.5, 1, 0)


iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

df['target'] = iris.target

df = df.loc[df.target != 2, ['sepal length (cm)', 'petal length (cm)', 'target']]

pcp = Perceptron(n_iter=10)

pcp.fit(df.iloc[:, :2].values, df.target.values)

with open("RTA_Model_pickle.pkl", 'wb') as f:
    pickle.dump(pcp, f)

