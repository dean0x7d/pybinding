import _pybinding


class SparseMatrix:
    def __init__(self, impl: _pybinding.SparseURef):
        self.impl = impl

    @property
    def data_pack(self):
        return self.impl.values, self.impl.inner_indices, self.impl.outer_starts

    @property
    def shape(self):
        return self.impl.rows, self.impl.cols

    @property
    def nnz(self):
        return self.impl.values.size

    def tocsr(self):
        from scipy.sparse import csr_matrix
        return csr_matrix(self.data_pack, shape=self.shape, copy=False)

    def tocoo(self):
        return self.tocsr().tocoo()

    def indices(self):
        row_boundaries = zip(self.impl.outer_starts[:-1], self.impl.outer_starts[1:])
        for i, (start, end) in enumerate(row_boundaries):
            for idx in range(start, end):
                yield i, self.impl.inner_indices[idx]

    def triplets(self):
        row_boundaries = zip(self.impl.outer_starts[:-1], self.impl.outer_starts[1:])
        for i, (start, end) in enumerate(row_boundaries):
            for idx in range(start, end):
                yield i, self.impl.inner_indices[idx], self.impl.values[idx]
