import numpy as np

class Dual():
    def __init__(self, a, b):
        self.real = a
        self.dual = b

    def __add__(self, other):
        try:
            return Dual(self.real + other.real, self.dual + other.dual)
        except AttributeError:
            return Dual(self.real + other, self.dual)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        try:
            return Dual(self.real * other.real, self.real*other.dual + other.real*self.dual)
        except AttributeError:
            return Dual(self.real*other, self.dual*other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __eq__(self, other):
        return self.real == other.real and self.dual == other.dual

    def __repr__(self):
        return "REAL: {}\nDual: {}".format(self.real, self.dual)

    def __str__(self):
        return "{} + e{}".format(self.real, self.dual)

