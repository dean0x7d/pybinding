from .. import _cpp
from scipy.sparse import csr_matrix


class SparseMatrix:
    def __init__(self, impl: _cpp.SparseURef):
        self.impl = impl

    def __getstate__(self):
        return self.arrays, self.shape

    def __setstate__(self, state):
        self.impl = csr_matrix(*state)

    @property
    def arrays(self):
        return self.impl.data, self.impl.indices, self.impl.indptr

    @property
    def shape(self):
        return self.impl.shape

    @property
    def nnz(self):
        return self.impl.data.size

    @property
    def csr_data(self):
        return self.arrays, self.shape

    def tocsr(self):
        return csr_matrix(*self.csr_data, copy=False)

    def tocoo(self):
        return self.tocsr().tocoo(copy=False)

    def triplets(self):
        row_boundaries = zip(self.impl.indptr[:-1], self.impl.indptr[1:])
        for i, (start, end) in enumerate(row_boundaries):
            for idx in range(start, end):
                yield i, self.impl.indices[idx], self.impl.data[idx]
