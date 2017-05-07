from abc import ABCMeta, abstractmethod
from collections import namedtuple

import numpy as np
from numpy import ma
from scipy.sparse import csr_matrix, coo_matrix


def _slice_csr_matrix(csr, idx):
    """Return a slice of a CSR matrix matching the given indices (applied to both rows and cols"""
    from copy import copy

    m = copy(csr)  # shallow copy
    m.data = m.data.copy()
    m.data += 1  # increment by 1 to preserve zeroes when slicing to preserve all data, even zeros
    m = m[idx][:, idx]
    m.data -= 1
    return m

Positions = namedtuple("Positions", "x y z")
Positions.__doc__ = """
Named tuple of arrays

Attributes
----------
x, y, z : array_like
    1D arrays of Cartesian coordinates
"""


class AbstractSites(metaclass=ABCMeta):
    """Abstract interface for site position and family ID storage"""

    @property
    @abstractmethod
    def x(self) -> np.ndarray:
        """1D array of coordinates"""

    @property
    @abstractmethod
    def y(self) -> np.ndarray:
        """1D array of coordinates"""

    @property
    @abstractmethod
    def z(self) -> np.ndarray:
        """1D array of coordinates"""

    @property
    @abstractmethod
    def ids(self) -> np.ndarray:
        """Site family identifies. Multiple sites can share the same ID, 
        e.g. sites which belong to the same sublattice."""

    @abstractmethod
    def __getitem__(self, item):
        """Matches numpy indexing behavior and applies it to all attributes"""

    def __len__(self):
        """Total number of sites"""
        return self.x.size

    @property
    def size(self) -> int:
        """Total number of sites"""
        return self.x.size

    @property
    def positions(self) -> Positions:
        """Named tuple of x, y, z positions"""
        return Positions(self.x, self.y, self.z)

    @property
    def xyz(self) -> np.ndarray:
        """Return a new array with shape=(N, 3). Convenient, but slow for big systems."""
        return np.array(self.positions).T

    def distances(self, target_position):
        """Return the distances of all sites from the target position

        Parameters
        ----------
        target_position : array_like

        Examples
        --------
        >>> sites = Sites(([0, 1, 1.1], [0, 0, 0], [0, 0, 0]), [0, 1, 0])
        >>> np.allclose(sites.distances([1, 0, 0]), [1, 0, 0.1])
        True
        """
        target_position = np.atleast_1d(target_position)
        ndim = len(target_position)
        positions = np.stack(self.positions[:ndim], axis=1)
        return np.linalg.norm(positions - target_position, axis=1)

    def find_nearest(self, target_position, target_site_family=""):
        """Return the index of the position nearest the target

        Parameters
        ----------
        target_position : array_like
        target_site_family : Optional[str]
            Look for a specific sublattice site. By default any will do.

        Returns
        -------
        int

        Examples
        --------
        >>> sites = Sites(([0, 1, 1.1], [0, 0, 0], [0, 0, 0]), [0, 1, 0])
        >>> sites.find_nearest([1, 0, 0])
        1
        >>> sites.find_nearest([1, 0, 0], target_site_family=0)
        2
        """
        distances = self.distances(target_position)
        if target_site_family == "":
            return np.argmin(distances)
        else:
            return ma.argmin(ma.array(distances, mask=(self.ids != target_site_family)))

    def argsort_nearest(self, target_position, target_site_family=None):
        """Return an ndarray of site indices, sorted by distance from the target

        Parameters
        ----------
        target_position : array_like
        target_site_family : int
            Look for a specific sublattice site. By default any will do.

        Returns
        -------
        np.ndarray

        Examples
        --------
        >>> sites = Sites(([0, 1, 1.1], [0, 0, 0], [0, 0, 0]), [0, 1, 0])
        >>> np.all(sites.argsort_nearest([1, 0, 0]) == [1, 2, 0])
        True
        >>> np.all(sites.argsort_nearest([1, 0, 0], target_site_family=0) == [2, 0, 1])
        True
        """
        distances = self.distances(target_position)
        if target_site_family is None:
            return np.argsort(distances)
        else:
            return ma.argsort(ma.array(distances, mask=(self.ids != target_site_family)))


class Sites(AbstractSites):
    """Reference implementation of :class:`AbstractSites`"""

    def __init__(self, positions, ids=None):
        self._x, self._y, self._z = np.atleast_1d(tuple(positions))
        if ids is not None:
            self._ids = np.atleast_1d(ids)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def ids(self):
        return getattr(self, "_ids", np.zeros_like(self._x, dtype=np.int8))

    def __getitem__(self, item):
        return self.__class__([v[item] for v in self.positions], self._ids[item])


class AbstractHoppings(metaclass=ABCMeta):
    """Abstract hopping storage with conversion and slicing functionality"""

    def __len__(self):
        """Total number of hoppings"""
        return self.nnz

    @property
    @abstractmethod
    def nnz(self):
        """Total number of hoppings"""

    @abstractmethod
    def tocsr(self) -> csr_matrix:
        """Return hoppings as :class:`~scipy.sparse.csr_matrix`"""

    @abstractmethod
    def tocoo(self) -> coo_matrix:
        """Return hoppings as :class:`~scipy.sparse.coo_matrix`"""

    @abstractmethod
    def __getitem__(self, item):
        """Matches numpy indexing behavior and applies it to hoppings"""


class Hoppings(AbstractHoppings):
    """Reference implementation which stores a CSR matrix internally"""

    def __init__(self, hoppings):
        self._csr = hoppings.tocsr()

    @property
    def nnz(self):
        return self._csr.nnz

    def tocsr(self) -> csr_matrix:
        return self._csr

    def tocoo(self) -> coo_matrix:
        return self._csr.tocoo()

    def __getitem__(self, idx):
        return self.__class__(_slice_csr_matrix(self._csr, idx))


class Boundary:
    """Describes a boundary between translation units of an infinite periodic system"""

    def __init__(self, shift, hoppings):
        self._shift = shift
        self._hoppings = hoppings

    @property
    def shift(self) -> np.ndarray:
        """The coordinate difference between the destination translation unit and the original"""
        return self._shift

    @property
    def hoppings(self) -> AbstractHoppings:
        """Hopping between the destination translation unit and the original"""
        return self._hoppings

    def __getitem__(self, item):
        """Matches numpy indexing behavior and applies it to hoppings"""
        return self.__class__(self.shift, self.hoppings[item])
