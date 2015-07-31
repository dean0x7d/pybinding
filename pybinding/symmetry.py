from . import _cpp


def translational(v1=None, v2=None, v3=None):
    """
    Simple translational symmetry.

    Parameters
    ----------
    v1, v2, v3 : float
        Length (in nanometers) of a translation in one of the primitive vector directions.
        Special values:
            0 - automatically sets the minimal translation length for the lattice
            None - no translational symmetry in this direction
    """
    if any(v is not None for v in (v1, v2, v3)):
        lengths = tuple((v if v is not None else -1) for v in (v1, v2, v3))
    else:
        lengths = 0,

    return _cpp.Translational(lengths)
