import numpy as np


class Kernel:
    def __init__(self, vectors, gamma, coef0, degree, kernel_name):
        self.coef0 = coef0
        self.gamma = gamma
        self.degree = degree
        self.vectors = vectors
        self.func = self.get_kernel_func(kernel_name=kernel_name)

    def linear(self, feature_vectors) -> np.array:
        return np.dot(feature_vectors, self.vectors.T)

    def poly(self, features) -> np.array:
        return np.array([pow((self.gamma * k) + self.coef0, self.degree) for k in np.dot(self.vectors, features)])

    def rbf(self, features) -> np.array:
        return np.array(
            [np.exp(-self.gamma * k) for k in [np.sum(np.power(vector - features, 2)) for vector in self.vectors]])

    def sigmoid(self, features) -> np.array:
        return np.array([np.tanh((self.gamma * k) + self.coef0) for k in np.dot(self.vectors, features)])

    def get_kernel_func(self, kernel_name):
        kernel_func_switcher = {'linear': self.linear, 'poly': self.poly, 'rbf': self.rbf, 'sigmoid': self.sigmoid}
        return kernel_func_switcher[kernel_name]

    def __call__(self, features):
        return self.func(features)
