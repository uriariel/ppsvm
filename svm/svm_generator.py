from sklearn.svm import SVC

from smart.ops import SealOps
from svm.svm_seal_generator import GeneratedSealSVC
from svm.svm_sim_generator import GeneratedSimSVC


class GeneratedSVC:
    def get_starts_ends(self):
        starts = [0]
        for i in range(1, self.n_support_vectors):
            start = 0
            for j in range(i):
                start += self.weights[j]
            starts += [start]

        ends = []

        for i in range(self.n_support_vectors):
            ends += [self.weights[i] + starts[i]]

        return starts, ends


def generate_seal_svc(svc: SVC, seal_ops: SealOps, sim=False):
    vectors = svc.support_vectors_
    kernel = svc.kernel
    n_features = len(vectors[0])
    weights = svc.n_support_
    n_weights = len(weights)
    classes = svc.classes_
    n_classes = len(svc.classes_)
    coefficients = svc.dual_coef_
    intercepts = svc.intercept_
    gamma = 1. / n_features if svc.gamma == 'auto' else svc.gamma
    degree = svc.degree
    coef0 = svc.coef0
    gen_svc_type = GeneratedSimSVC if sim else GeneratedSealSVC
    return gen_svc_type(vectors=vectors, coefficients=coefficients, intercepts=intercepts, weights=weights,
                        kernel=kernel, gamma=gamma, coef0=coef0, degree=degree, n_classes=n_classes,
                        n_weights=n_weights, classes=classes, seal_ops=seal_ops)
