import numpy as np
from seal import Plaintext, DoubleVector

from smart.ops import SealOps


def test_seal_env_running():
    n = np.ones((3, 3))
    s = SealOps.with_env()
    m = s.encrypt(n)
    x = s.get_vector_range(m[0], 1, 2)
    p = Plaintext()
    p1 = DoubleVector()
    s.decryptor.decrypt(x, p)
    s.encoder.decode(p, p1)
    print(p1)


if __name__ == '__main__':
    test_seal_env_running()
