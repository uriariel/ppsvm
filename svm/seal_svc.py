from sklearn.svm import SVC

from smart.seal_matrix import CipherMatrix
from svm.svm_generator import generate_seal_svc


class SealSVC:
    def __init__(self, seal_ops, *args, **kwargs):
        self.svc = SVC(*args, **kwargs)
        self.seal_svc = None
        self.seal_ops = seal_ops

    def fit(self, X, y):
        self.svc.fit(X=X, y=y)
        self.seal_svc = generate_seal_svc(svc=self.svc, seal_ops=self.seal_ops)

    def predict(self, X: CipherMatrix):
        # raise Exception('Batch prediction isn\'t supported yet')
        return self.seal_svc.predict(X)
