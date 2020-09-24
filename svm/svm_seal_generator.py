import numpy as np
from seal import Ciphertext

from smart.ops import SealOps
from smart.seal_matrix import CipherMatrix
from svm.seal_kernels import SealKernel
from svm.svm_generator import GeneratedSVC


class GeneratedSealSVC(GeneratedSVC):
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
        self.seal_ops = seal_ops
        self.kernel = SealKernel(vectors=vectors, gamma=gamma, degree=degree, coef0=coef0, kernel_name=kernel,
                                 seal_ops=self.seal_ops)

    def decision_function(self, starts, ends, kernels: Ciphertext):
        decisions = []
        d = 0

        for i in range(self.n_support_vectors):
            for j in range(i + 1, self.n_support_vectors):
                k = self.seal_ops.get_vector_range(kernels, 0, 6)
                tmp = kernels[starts[j]: ends[j]].dot(self.coefficients[i][starts[j]: ends[j]])
                tmp += kernels[starts[i]: ends[i]].dot(self.coefficients[j - 1][starts[i]: ends[i]])

                decisions += [tmp + self.intercepts[d]]
                d += 1

        return np.array(decisions)

    def predict(self, features: CipherMatrix):
        kernels = self.kernel(features=features)
        kernels = self.seal_ops.decrypt(kernels).to_numpy_array()
        print(f'decrypted {kernels}')

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
