import numpy as np
from smart.ops import SealOps
from smart.seal_matrix import CipherMatrix


class SealKernel:
    def __init__(self, vectors, gamma, coef0, degree, kernel_name, seal_ops: SealOps):
        self.coef0 = coef0
        self.gamma = gamma
        self.degree = degree
        self.vectors = vectors
        self.seal_ops = seal_ops
        self.func = self.get_kernel_func(kernel_name=kernel_name)

    def linear(self, features: CipherMatrix):
        # Instead of vectors * features^T we calculate features * vectors^T and get W^T
        return self.seal_ops.dot_matrix_with_plain_matrix_transpose(features, self.vectors)

    def poly(self, features: CipherMatrix):
        return [pow((self.gamma * k) + self.coef0, self.degree) for k in np.dot(self.vectors, features)]

    def rbf(self, features: CipherMatrix):
        return [np.exp(-self.gamma * k) for k in [np.sum(np.power(vector - features, 2)) for vector in self.vectors]]

    def sigmoid(self, features: CipherMatrix):
        return [np.tanh((self.gamma * k) + self.coef0) for k in np.dot(self.vectors, features)]

    def get_kernel_func(self, kernel_name):
        kernel_func_switcher = {'linear': self.linear, 'poly': self.poly, 'rbf': self.rbf, 'sigmoid': self.sigmoid}
        return kernel_func_switcher[kernel_name]

    def __call__(self, features):
        return self.func(features)
