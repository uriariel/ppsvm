from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from tests.helpers import scale_data, classify_array, load_credit
from smart.ops import SealOps
from svm.seal_svc import SealSVC


def svm(scaling=False, check=False):
    X, y = load_credit()
    import random
    seed = random.randint(0, 100)
    print(seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001, random_state=seed)
    if scaling:
        scale_data(X_train)
        scale_data(X_test)
    seal_ops = SealOps.with_env()
    kernel = 'linear'
    gamma = 1.0
    if check:
        s = SVC(kernel=kernel, gamma=gamma)
        s.fit(X=X_train, y=y_train)
        y_pred = s.predict(X_test)
    else:
        s = SealSVC(seal_ops=seal_ops, kernel=kernel, gamma=gamma)
        s.fit(X=X_train, y=y_train)
        y_pred = classify_array(s, X_test)
    print(classification_report(y_test, y_pred))


def main():
    svm()


if __name__ == '__main__':
    main()
