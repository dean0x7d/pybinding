from functools import singledispatch, update_wrapper

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

import pybinding as pb


def _assertdispatch(func):
    """Adapted `@singledispatch` for custom assertions

    * Works with methods instead of functions
    * Keeps track of the data structure via context stack
    * Detects objects which can be used with `pb.save` and `pb.load`
    """
    dispatcher = singledispatch(func)

    def wrapper(self, actual, expected, context=None):
        if context is not None:
            self.stack.append(context)

        is_pb_savable = any(hasattr(actual, s) for s in ['__getstate__', '__getinitargs__'])
        kind = pb.save if is_pb_savable else actual.__class__
        dispatcher.dispatch(kind)(self, actual, expected)

        if context is not None and self.stack:
            self.stack.pop()

    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper


class FuzzyReport:
    """Explains failed fuzzy_equal asserts

    For example:

        actual =   array([3, 1, 7, 2, 9])
        expected = array([3, 5, 7, 4, 6])
        assert pytest.fuzzy_equal(actual, expected)

            @_assert.register(np.ndarray)
            def _(self, actual, expected):
        >       assert FuzzyReport(actual, self.rtol, self.atol) == expected
        E       assert failed on 3 of 5 values: 60 percent
        E         diff indices: [1, 3, 4]
        E         actual:   [1, 2, 9]
        E         expected: [5, 4, 6]
    """
    def __init__(self, value, rtol, atol):
        self.value = value
        self.rtol = rtol
        self.atol = atol
        self.explanation = []

    def __eq__(self, other):
        actual, expected = map(np.atleast_1d, (self.value, other))
        if actual.shape != expected.shape:
            self.explanation = [
                "failed on shape mismatch",
                "actual:   {}".format(actual.shape),
                "expected: {}".format(expected.shape),
            ]
            return False

        isclose = np.isclose(actual, expected, self.rtol, self.atol)
        if np.all(isclose):
            return True

        notclose = np.logical_not(isclose)
        num_failed = np.sum(notclose)
        a = actual[notclose]
        b = expected[notclose]
        self.explanation = [
            "failed on {} of {} values: {:.0f} percent".format(num_failed, actual.size,
                                                               100 * num_failed / actual.size),
            "actual:   {}".format(a),
            "expected: {}".format(b),
            "diff indices: {}".format([idx[0] if idx.size == 1 else list(idx)
                                       for idx in np.argwhere(notclose)]),
            "abs diff: {}".format(abs(a - b)),
            "rel diff: {}".format(abs(a - b) / abs(b)),
        ]
        return False


class FuzzyEqual:
    """Recursively compares structures of ndarrays using np.isclose() comparison

    The `stack` attribute shows the structure depth at a given assert.
    """
    def __init__(self, actual, expected, rtol=1e-05, atol=1e-08):
        self.actual = actual
        self.expected = expected
        self.rtol = rtol
        self.atol = atol
        self.stack = []

    def __bool__(self):
        self._assert(self.actual, self.expected)
        return True

    def __repr__(self):
        return ''.join(self.stack)

    @_assertdispatch
    def _assert(self, actual, expected):
        assert actual == expected

    @_assert.register(np.ndarray)
    def _(self, actual, expected):
        assert FuzzyReport(actual, self.rtol, self.atol) == expected

    @_assert.register(csr_matrix)
    def _(self, actual, expected):
        for s in ['shape', 'data', 'indices', 'indptr']:
            self._assert(getattr(actual, s), getattr(expected, s), context=".{}".format(s))

    @_assert.register(coo_matrix)
    def _(self, actual, expected):
        for s in ['shape', 'data', 'row', 'col']:
            self._assert(getattr(actual, s), getattr(expected, s), context=".{}".format(s))

    @_assert.register(tuple)
    @_assert.register(list)
    def _(self, actual, expected):
        assert len(actual) == len(expected)
        for index, (a, b) in enumerate(zip(actual, expected)):
            self._assert(a, b, context="[{}]".format(index))

    @_assert.register(dict)
    def _(self, actual, expected):
        assert sorted(actual) == sorted(expected)
        for key in actual:
            self._assert(actual[key], expected[key], context="['{}']".format(key))

    @_assert.register(pb.save)
    def _(self, actual, expected):
        specials = [s for s in ['__getstate__', '__getinitargs__'] if hasattr(actual, s)]
        for s in specials:
            self._assert(getattr(actual, s)(), getattr(expected, s)(), context="{}()".format(s))
