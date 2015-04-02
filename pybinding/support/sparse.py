import _pybinding


class SparseMatrix(_pybinding.SparseURef):
    @property
    def data_pack(self):
        return self.values, self.inner_indices, self.outer_starts

    @property
    def shape(self):
        return self.rows, self.cols

    @property
    def nnz(self):
        return self.values.size

    def tocsr(self):
        from scipy.sparse import csr_matrix
        return csr_matrix(self.data_pack, shape=self.shape, copy=False)

    def tocoo(self):
        return self.tocsr().tocoo()

    def indices(self):
        for i, (start, end) in enumerate(zip(self.outer_starts[:-1], self.outer_starts[1:])):
            for idx in range(start, end):
                yield i, self.inner_indices[idx]

    def triplets(self):
        for i, (start, end) in enumerate(zip(self.outer_starts[:-1], self.outer_starts[1:])):
            for idx in range(start, end):
                yield i, self.inner_indices[idx], self.values[idx]
