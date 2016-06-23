import numpy as np
from scipy.sparse import csr_matrix


class AliasArray(np.ndarray):
    """An ndarray with a mapping of values to user-friendly names -- see example

    This ndarray subclass enables comparing sub_id and hop_id arrays directly with
    their friendly string identifiers. The mapping parameter translates sublattice
    or hopping names into their number IDs.

    Only the `==` and `!=` operators are overloaded to handle the aliases.

    Examples
    --------
    >>> a = AliasArray([0, 1, 0], mapping={'A': 0, 'B': 1})
    >>> list(a == 0)
    [True, False, True]
    >>> list(a == 'A')
    [True, False, True]
    >>> list(a != 'A')
    [False, True, False]
    """
    def __new__(cls, array, mapping):
        obj = np.asarray(array).view(cls)
        obj.mapping = mapping
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.mapping = getattr(obj, 'mapping', None)

    def _alias(self, key):
        if isinstance(key, str):
            return self.mapping[key]
        else:
            return key

    def __eq__(self, other):
        return super().__eq__(self._alias(other))

    def __ne__(self, other):
        return super().__ne__(self._alias(other))


# noinspection PyAbstractClass
class AliasCSRMatrix(csr_matrix):
    """Same as :class:`AliasArray` but for a CSR matrix

    Examples
    --------
    >>> from scipy.sparse import spdiags
    >>> m = AliasCSRMatrix(spdiags([1, 2, 1], [0], 3, 3), mapping={'A': 1, 'B': 2})
    >>> list(m.data == 'A')
    [True, False, True]
    >>> list(m.tocoo().data == 'A')
    [True, False, True]
    >>> list(m[:2].data == 'A')
    [True, False]
    """
    def __init__(self, *args, **kwargs):
        mapping = kwargs.pop('mapping', {})
        if not mapping:
            mapping = getattr(args[0], 'mapping', {})

        super().__init__(*args, **kwargs)
        self.data = AliasArray(self.data, mapping)

    @property
    def format(self):
        return 'csr'

    @format.setter
    def format(self, _):
        pass

    @property
    def mapping(self):
        return self.data.mapping

    def tocoo(self, *args, **kwargs):
        coo = super().tocoo(*args, **kwargs)
        coo.data = AliasArray(coo.data, mapping=self.mapping)
        return coo

    def __getitem__(self, item):
        result = super().__getitem__(item)
        if getattr(result, 'format', '') == 'csr':
            return AliasCSRMatrix(result, mapping=self.mapping)
        else:
            return result
