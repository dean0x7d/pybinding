import copy
import numpy as np

__all__ = ['FuzzySet']


class FuzzySet:
    def __init__(self, iterable=None, rtol=1.e-3, atol=1.e-5):
        self.data = []
        self.rtol = rtol
        self.atol = atol

        if iterable:
            for item in iterable:
                self.add(item)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __contains__(self, item):
        return any(np.allclose(item, x, rtol=self.rtol, atol=self.atol) for x in self.data)

    def __iadd__(self, other):
        for item in other:
            self.add(item)
        return self

    def __add__(self, other):
        if isinstance(other, FuzzySet):
            ret = copy.copy(self)
            ret += other
            return ret
        else:
            return copy.copy(self)

    def __radd__(self, other):
        return self + other

    def add(self, item):
        if item not in self:
            self.data.append(item)
