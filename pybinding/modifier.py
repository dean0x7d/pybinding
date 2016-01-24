"""Modifier decorators

Used to create functions which express some feature of a tight-binding model,
such as various fields, defects or geometric deformations.
"""
import inspect
import numpy as np

from . import _cpp
from .support.inspect import get_call_signature

__all__ = ['site_state_modifier', 'site_position_modifier', 'onsite_energy_modifier',
           'hopping_energy_modifier', 'constant_potential']


def _check_modifier_spec(func, keywords):
    """Make sure the arguments are specified correctly

    Parameters
    ----------
    func : callable
        The function which is to become a modifier.
    keywords : list
        Used to check that `func` arguments are correct.
    """
    argnames = inspect.signature(func).parameters.keys()
    unexpected = ", ".join([name for name in argnames if name not in keywords])
    if unexpected:
        expected = ", ".join(keywords)
        raise RuntimeError("Unexpected argument(s) in modifier: {unexpected}\n"
                           "Arguments must be any of: {expected}".format(**locals()))


def _check_modifier_return(modifier, num_arguments, num_return, maybe_complex):
    """Make sure the modifier returns the correct type and size

    Parameters
    ----------
    modifier
        The modifier being checked.
    num_arguments : int
        Expected number of modifier arguments.
    num_return : int
        Expected number of return values.
    maybe_complex : bool
        The modifier may return a complex result even if the input is real.
    """
    in_shape = 10,
    in_data = np.random.rand(*in_shape).astype(np.float16)

    try:
        out_data = modifier.apply(*(in_data,) * num_arguments)
    except AttributeError as e:
        if "astype" in str(e):  # known issue
            raise RuntimeError("Modifier must return numpy.ndarray")
        else:  # unknown issue
            raise

    out_data = out_data if isinstance(out_data, tuple) else (out_data,)
    if len(out_data) != num_return:
        raise RuntimeError("Modifier expected to return {} ndarray(s), "
                           "but got {}".format(num_return, len(out_data)))
    if any(v.shape != in_shape for v in out_data):
        raise RuntimeError("Modifier must return the same shape ndarray as the arguments")

    if not maybe_complex and modifier.is_complex():
        raise RuntimeError("This modifier must not return complex values")


def _make_modifier(func, kind, keywords, num_return=1, maybe_complex=False):
    """Turn a regular function into a modifier of the desired kind

    Parameters
    ----------
    func : callable
        The function which is to become a modifier.
    kind : object
        Modifier base class.
    keywords : str
        String of comma separated names: the expected arguments of a modifier function.
    num_return : int
        Expected number of return values.
    maybe_complex : bool
        The modifier may return a complex result even if the input is real.

    Returns
    -------
    Modifier
    """
    keywords = [word.strip() for word in keywords.split(",")]
    _check_modifier_spec(func, keywords)

    class Modifier(kind):
        argnames = tuple(inspect.signature(func).parameters.keys())
        try:
            callsig = get_call_signature(up=3)
        except IndexError:
            callsig = get_call_signature(up=0)
            callsig.function = func

        def __str__(self):
            return str(self.callsig)

        def __repr__(self):
            return repr(self.callsig)

        def __call__(self, *args, **kwargs):
            return func(*args, **kwargs)

        def apply(self, *args):
            # only pass the requested arguments to func
            named_args = {name: value for name, value in zip(keywords, args)
                          if name in self.argnames}
            ret = func(**named_args)

            def cast_dtype(v):
                return v.astype(args[0].dtype, casting='same_kind', copy=False)

            try:  # cast output array to same element type as the input
                if isinstance(ret, tuple):
                    return tuple(map(cast_dtype, ret))
                else:
                    return cast_dtype(ret)
            except TypeError:
                return ret

        def is_complex(self):
            ret = self.apply(np.ones(1), *(np.zeros(1) for _ in keywords[1:]))
            return np.iscomplexobj(ret)

    modifier = Modifier()
    _check_modifier_return(modifier, len(keywords), num_return, maybe_complex)
    return modifier


def site_state_modifier(func):
    """Modify the state (valid or invalid) of lattice sites, e.g. to create vacancies

    Parameters
    ----------
    func : callable
        The function parameters must be a combination of any number of the following:

        state : ndarray of bool
            Indicates if a lattice site is valid. Invalid sites will be removed from
            the model after all modifiers have been applied.
        x, y, z : ndarray
            Lattice site position.
        sub_id : ndarray of int
            Sublattice ID. Can be checked for equality with `lattice[sublattice_name]`.

        Modifier returns:

        ndarray
            A modified `state` argument or an `ndarray` of the same dtype and shape.

    Returns
    -------
    Modifier
    """
    return _make_modifier(func, _cpp.SiteStateModifier, keywords="state, x, y, z, sub_id")


def site_position_modifier(func):
    """Modify the position of lattice sites, e.g. to apply geometric deformations

    Parameters
    ----------
    func : callable
        The function parameters must be a combination of any number of the following:

        x, y, z : ndarray
            Lattice site position.
        sub_id : ndarray of int
            Sublattice ID. Can be checked for equality with `lattice[sublattice_name]`.

        Modifier returns:

        tuple of ndarray
            Modified 'x, y, z' arguments or 3 `ndarray` objects of the same dtype and shape.

    Returns
    -------
    Modifier
    """
    return _make_modifier(func, _cpp.PositionModifier, keywords="x, y, z, sub_id", num_return=3)


def onsite_energy_modifier(func):
    """Modify the onsite energy, e.g. to apply an electric field

    Parameters
    ----------
    func : callable
        The function parameters must be a combination of any number of the following:

        energy : ndarray
            The onsite energy.
        x, y, z : ndarray
            Lattice site position.
        sub_id : ndarray of int
            Sublattice ID. Can be checked for equality with `lattice[sublattice_name]`.

        Modifier returns:

        ndarray
            A modified `potential` argument or an `ndarray` of the same dtype and shape.

    Returns
    -------
    Modifier
    """
    return _make_modifier(func, _cpp.OnsiteModifier, keywords="energy, x, y, z, sub_id")


def hopping_energy_modifier(func):
    """Modify the hopping energy, e.g. to apply a magnetic field

    Parameters
    ----------
    func : callable
        The function parameters must be a combination of any number of the following:

        energy : ndarray
            The hopping energy between two sites.
        x1, y1, z1, x2, y2, z2 : ndarray
            Positions of the two lattice sites connected by the hopping parameter.
        hop_id : ndarray of int
            Hopping ID. Check for equality with `lattice(hopping_name)`.

        Modifier returns:

        ndarray
            A modified `hopping` argument or an `ndarray` of the same dtype and shape.

    Returns
    -------
    Modifier
    """
    return _make_modifier(func, _cpp.HoppingModifier, maybe_complex=True,
                          keywords="energy, x1, y1, z1, x2, y2, z2, hop_id")


def constant_potential(magnitude):
    """Apply a constant onsite energy to every lattice site

    Parameters
    ----------
    magnitude : float
        In units of eV.
    """
    @onsite_energy_modifier
    def function(energy):
        return energy + magnitude

    return function
