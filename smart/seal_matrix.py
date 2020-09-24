import numpy as np
from seal import Ciphertext

from smart.seal_vector import Vector


class CipherMatrix:
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.data = tuple(Ciphertext() for i in range(rows))

    def __setitem__(self, key, value: Ciphertext):
        self.data = self.data[:key] + (value,) + self.data[key + 1:]

    def __getitem__(self, item) -> Ciphertext:
        return self.data[item]


class Matrix:
    def __init__(self, rows: int, cols: int, data=None):
        self.rows = rows
        self.cols = cols
        if data is not None:
            self.data = tuple(Vector(list(data[i])) for i in range(rows))
        else:
            self.data = tuple(Vector() for _i in range(rows))

    @classmethod
    def from_numpy_array(cls, array: np.array):
        if array.ndim > 2:
            raise ValueError('array is not a 2D matrix')
        elif array.ndim == 1:
            array = array.reshape(1, -1)

        rows = array.shape[0]
        cols = array.shape[1]

        return cls(rows=rows, cols=cols, data=array)

    def to_numpy_array(self):
        if self.rows == 1:
            return np.array(self.data[0])
        else:
            return tuple(np.array(v) for v in self.data)

    def __setitem__(self, key, value: list):
        self.data = self.data[:key] + (value,) + self.data[key + 1:]

    def __getitem__(self, item):
        return self.data[item]

    def __repr__(self):
        return 'Matrix'.center(26 + 3 * self.cols, '-') + '\n' + '\n'.join(
            [repr(vec[:self.cols]) for vec in self.data]) + '\n' + ''.center(26 + 3 * self.cols, '-')
