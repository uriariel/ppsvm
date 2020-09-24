from seal import DoubleVector

PRINTING_PRECISION = 5


class Vector(DoubleVector):
    def __repr__(self):
        return str([round(element, PRINTING_PRECISION) for element in self])
