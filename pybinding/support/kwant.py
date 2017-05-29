"""Kwant compatibility layer"""
import warnings
import numpy as np

try:
    from kwant.system import FiniteSystem, InfiniteSystem
    kwant_installed = True
except ImportError:
    FiniteSystem = InfiniteSystem = object
    kwant_installed = False


def _warn_if_not_empty(args, params):
    if args or params:
        warnings.warn(
            "Additional `args/params` are ignored because pybinding's Hamiltonian is immutable. "
            "Complete the model with all parameters before calling the `tokwant()` conversion.",
            stacklevel=3
        )


class Graph:
    """Mock `kwant.graph.CGraph`

    Only the `num_nodes` attribute seems to be required, at least for `smatrix`.
    """
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes


class KwantFiniteSystem(FiniteSystem):
    """Mock 'kwant.system.FiniteSystem`

    Mostly complete, some features of `hamiltonian_submatrix` are not supported,
    however it seems to work well for `smatrix`.
    """
    def __init__(self, pb_model):
        self.pb_model = pb_model
        self.graph = Graph(pb_model.system.num_sites)
        self._pos = np.array(pb_model.system.positions[:pb_model.lattice.ndim]).T
        self.leads = [KwantInfiniteSystem(l) for l in pb_model.leads]
        self.lead_interfaces = [l.indices for l in pb_model.leads]

    def pos(self, index):
        return self._pos[index]

    def hamiltonian(self, i, j, *args, params=None):
        _warn_if_not_empty(args, params)
        return self.pb_model.hamiltonian[i, j]

    def hamiltonian_submatrix(self, args=(), to_sites=None, from_sites=None,
                              sparse=False, return_norb=False, *, params=None):
        if to_sites is not None or from_sites is not None:
            raise RuntimeError("The `to_sites` and `from_sites` arguments are not supported")
        _warn_if_not_empty(args, params)

        ham = self.pb_model.hamiltonian
        matrix = ham.tocoo() if sparse else ham.todense()
        if not return_norb:
            return matrix
        else:
            subs = self.pb_model.system.sublattices
            norb = self.pb_model.system.impl.compressed_sublattices.orbital_counts
            to_norb = norb[subs]
            from_norb = to_norb
            return matrix, to_norb, from_norb


class KwantInfiniteSystem(InfiniteSystem):
    """Mock 'kwant.system.InfiniteSystem`

    Should completely reproduce all features.
    """
    def __init__(self, pb_lead):
        self.h0 = pb_lead.h0
        self.h1 = pb_lead.h1

    def hamiltonian(self, i, j, *args, params=None):
        _warn_if_not_empty(args, params)
        return self.h0[i, j]

    def cell_hamiltonian(self, args=(), sparse=False, *, params=None):
        _warn_if_not_empty(args, params)
        return self.h0.tocoo() if sparse else self.h0.todense()

    def inter_cell_hopping(self, args=(), sparse=False, *, params=None):
        _warn_if_not_empty(args, params)
        return self.h1.tocoo() if sparse else self.h1.todense()


def tokwant(model):
    """Return a finalized kwant system constructed from a pybinding Model

    Parameters
    ----------
    model : Model

    Returns
    -------
    kwant.system.FiniteSystem
    """
    if not kwant_installed:
        raise ImportError("Can't convert to kwant format if kwant isn't installed")
    return KwantFiniteSystem(model)
