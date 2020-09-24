import numpy as np

from smart.ops import SealOps
from svm.kernels import Kernel
from svm.svm_generator import GeneratedSVC


class GeneratedSimSVC(GeneratedSVC):
    def __init__(self, n_classes, classes, vectors, n_weights, coefficients, intercepts, weights, kernel, gamma, degree,
                 coef0, seal_ops: SealOps):
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.weights = weights
        self.intercepts = intercepts
        self.coefficients = coefficients
        self.classes = classes
        self.n_classes = n_classes
        self.n_support_vectors = n_weights
        self.vectors = vectors
        self.kernel = Kernel(vectors=vectors, gamma=gamma, degree=degree, coef0=coef0, kernel_name=kernel)
        self.seal_ops = seal_ops

    def decision_function(self, starts, ends, kernels: np.array):
        decisions = []
        d = 0

        for i in range(self.n_support_vectors):
            for j in range(i + 1, self.n_support_vectors):
                tmp = kernels[starts[j]: ends[j]].dot(self.coefficients[i][starts[j]: ends[j]])
                tmp += kernels[starts[i]: ends[i]].dot(self.coefficients[j - 1][starts[i]: ends[i]])

                decisions += [tmp + self.intercepts[d]]
                d += 1

        return np.array(decisions)

    def predict(self, features):
        kernels = self.kernel(features=features)
        enc_kernels = self.seal_ops.encrypt(kernels)
        self.seal_ops.add_plain(enc_kernels, np.zeros((enc_kernels.rows, enc_kernels.cols)))
        self.seal_ops.multiply_plain(enc_kernels, np.ones((enc_kernels.rows, enc_kernels.cols)))
        kernels = self.seal_ops.decrypt(enc_kernels).to_numpy_array()

        starts, ends = self.get_starts_ends()

        if self.n_classes == 2:
            decision = kernels[starts[1]: ends[1]].dot(self.coefficients[0][starts[1]: ends[1]])
            decision += kernels[starts[0]: ends[0]].dot(self.coefficients[0][starts[0]: ends[0]])

            decision += self.intercepts[0]
            return decision > 0
        else:
            decisions = self.decision_function(starts=starts, ends=ends, kernels=kernels)

            votes = []
            d = 0
            for i in range(self.n_support_vectors):
                for j in range(i + 1, self.n_support_vectors):
                    votes += [i if decisions[d] > 0 else j]
                    d += 1

            amounts = [0 for i in range(self.n_classes)]
            for i in range(len(amounts)):
                amounts[votes[i]] += 1

            return self.classes[np.argmax(amounts)]
