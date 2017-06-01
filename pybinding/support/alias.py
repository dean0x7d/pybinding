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
    >>> a = AliasArray([0, 1, 0], mapping={"A": 0, "B": 1})
    >>> list(a == 0)
    [True, False, True]
    >>> list(a == "A")
    [True, False, True]
    >>> list(a != "A")
    [False, True, False]
    >>> a = AliasArray([0, 1, 0, 2], mapping={"A|1": 0, "B": 1, "A|2": 2})
    >>> list(a == "A")
    [True, False, True, True]
    >>> list(a != "A")
    [False, True, False, False]
    >>> import pickle
    >>> s = pickle.dumps(a)
    >>> a2 = pickle.loads(s)
    >>> list(a2 == "A")
    [True, False, True, True]
    """
    def __new__(cls, array, mapping):
        obj = np.asarray(array).view(cls)
        obj.mapping = {SplitName(k): v for k, v in mapping.items()}
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.mapping = getattr(obj, "mapping", None)

    def _mapped_eq(self, other):
        if other in self.mapping:
            return super().__eq__(self.mapping[other])
        else:
            result = np.zeros(len(self), dtype=np.bool)
            for k, v in self.mapping.items():
                if k == other:
                    result = np.logical_or(result, super().__eq__(v))
            return result

    def __eq__(self, other):
        if isinstance(other, str):
            return self._mapped_eq(other)
        else:
            return super().__eq__(other)

    def __ne__(self, other):
        if isinstance(other, str):
            return np.logical_not(self._mapped_eq(other))
        else:
            return super().__ne__(other)

    def __reduce__(self):
        r = super().__reduce__()
        state = r[2] + (self.mapping,)
        return r[0], r[1], state

    def __setstate__(self, state):
        self.mapping = state[-1]
        super().__setstate__(state[:-1])


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
    >>> import pickle
    >>> s = pickle.dumps(m)
    >>> m2 = pickle.loads(s)
    >>> list(m2.data == 'A')
    [True, False, True]
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


class AliasIndex:
    """An all-or-nothing array index based on equality with a specific value

    The `==` and `!=` operators are overloaded to return a lazy array which is either
    all `True` or all `False`. See the examples below. This is useful for modifiers
    where the each call gets arrays with the same sub_id/hop_id for all elements.
    Instead of passing an `AliasArray` with `.size` identical element, `AliasIndex`
    does the same all-or-nothing indexing.

    Examples
    --------
    >>> l = np.array([1, 2, 3])
    >>> ai = AliasIndex("A", len(l))
    >>> list(l[ai == "A"])
    [1, 2, 3]
    >>> list(l[ai == "B"])
    []
    >>> list(l[ai != "A"])
    []
    >>> list(l[ai != "B"])
    [1, 2, 3]
    >>> np.logical_and([True, False, True], ai == "A")
    array([ True, False,  True], dtype=bool)
    >>> np.logical_and([True, False, True], ai != "A")
    array([False, False, False], dtype=bool)
    >>> bool(ai == "A")
    True
    >>> bool(ai != "A")
    False
    >>> str(ai)
    'A'
    >>> hash(ai) == hash("A")
    True
    >>> int(ai.eye)
    1
    >>> np.allclose(AliasIndex("A", 1, (2, 2)).eye, np.eye(2))
    True
    """
    class LazyArray:
        def __init__(self, value, shape):
            self.value = value
            self.shape = shape

        def __bool__(self):
            return bool(self.value)

        def __array__(self):
            return np.full(self.shape, self.value)

    def __init__(self, name, shape, orbs=(1, 1)):
        self.name = name
        self.shape = shape
        self.orbs = orbs

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self.LazyArray(self.name == other, self.shape)

    def __ne__(self, other):
        return self.LazyArray(self.name != other, self.shape)

    def __hash__(self):
        return hash(self.name)

    @property
    def eye(self):
        return np.eye(*self.orbs)


class SplitName(str):
    """String subclass with special support for strings of the form "first|second"

    Operators `==` and `!=` are overloaded to return `True` even if only the first part matches.

    Examples
    --------
    >>> s = SplitName("first|second")
    >>> s == "first|second"
    True
    >>> s != "first|second"
    False
    >>> s == "first"
    True
    >>> s != "first"
    False
    >>> s == "second"
    False
    >>> s != "second"
    True
    """
    @property
    def first(self):
        return self.split("|")[0]

    def __eq__(self, other):
        return super().__eq__(other) or self.first == other

    def __ne__(self, other):
        return super().__ne__(other) and self.first != other

    def __hash__(self):
        return super().__hash__()
