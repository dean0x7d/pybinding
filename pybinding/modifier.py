"""Modifier function decorators

Used to create functions which express some feature of a tight-binding model,
such as various fields, defects or geometric deformations.
"""
import inspect
import functools
from collections import defaultdict

import numpy as np

from . import _cpp
from .system import Sites
from .support.inspect import get_call_signature
from .support.alias import AliasArray
from .utils.misc import decorator_decorator

__all__ = ['constant_potential', 'force_double_precision', 'hopping_energy_modifier',
           'hopping_generator', 'onsite_energy_modifier', 'site_position_modifier',
           'site_state_modifier']


def _make_alias_array(obj):
    if isinstance(obj, (_cpp.SubIdRef, _cpp.HopIdRef)):
        return AliasArray(obj.ids, obj.name_map)
    else:
        return obj


def _process_modifier_args(args, keywords, requested_argnames):
    """Return only the requested modifier arguments

    Also process any special args like 'sub_id', 'hop_id' and 'sites'.
    """
    kwargs = dict(zip(keywords, args))
    if 'sub_id' in requested_argnames or 'sites' in requested_argnames:
        kwargs['sub_id'] = _make_alias_array(kwargs['sub_id'])
    if 'hop_id' in requested_argnames:
        kwargs['hop_id'] = _make_alias_array(kwargs['hop_id'])

    requested_kwargs = {name: value for name, value in kwargs.items()
                        if name in requested_argnames}

    if 'sites' in requested_argnames:
        requested_kwargs['sites'] = Sites((kwargs[k] for k in ('x', 'y', 'z')), kwargs['sub_id'])

    return requested_kwargs


def _check_modifier_spec(func, keywords, has_sites=False):
    """Make sure the arguments are specified correctly

    Parameters
    ----------
    func : callable
        The function which is to become a modifier.
    keywords : list
        Used to check that `func` arguments are correct.
    has_sites : bool
        Check for 'site' argument.
    """
    argnames = inspect.signature(func).parameters.keys()
    if has_sites:
        keywords += ["sites"]
    unexpected = ", ".join([name for name in argnames if name not in keywords])
    if unexpected:
        expected = ", ".join(keywords)
        raise RuntimeError("Unexpected argument(s) in modifier: {unexpected}\n"
                           "Arguments must be any of: {expected}".format(**locals()))


def _check_modifier_return(func, num_arguments, num_return, can_be_complex):
    """Make sure the modifier returns the correct type and size

    Parameters
    ----------
    func : callable
        The function which is to become a modifier.
    num_arguments : int
        Expected number of modifier arguments.
    num_return : int
        Expected number of return values.
    can_be_complex : bool
        Is this modifier allowed to have a complex return value.
    """
    in_shape = 10,
    dummy_input = [AliasArray(np.ones(in_shape), defaultdict(int))] * num_arguments

    try:
        out_data = func(*dummy_input)
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

    is_complex = any(np.iscomplexobj(v) for v in out_data)
    if is_complex and not can_be_complex:
        raise RuntimeError("This modifier must not return complex values")

    return is_complex


def _make_modifier(func, kind, init, keywords, has_sites=True, num_return=1, can_be_complex=False):
    """Turn a regular function into a modifier of the desired kind

    Parameters
    ----------
    func : callable
        The function which is to become a modifier.
    kind : object
        Modifier base class.
    init : dict
        Initializer kwargs for the Modifier base class.
    keywords : str
        String of comma separated names: the expected arguments of a modifier function.
    has_sites : bool
        Arguments may include the :class:`Sites` helper.
    num_return : int
        Expected number of return values.
    can_be_complex : bool
        The modifier may return a complex result even if the input is real.

    Returns
    -------
    Modifier
    """
    keywords = [word.strip() for word in keywords.split(",")]
    _check_modifier_spec(func, keywords, has_sites)
    requested_argnames = tuple(inspect.signature(func).parameters.keys())

    def apply_func(*args):
        requested_kwargs = _process_modifier_args(args, keywords, requested_argnames)
        ret = func(**requested_kwargs)

        def cast_dtype(v):
            return v.astype(args[0].dtype, casting='same_kind', copy=False)

        try:  # cast output array to same element type as the input
            if isinstance(ret, tuple):
                return tuple(map(cast_dtype, ret))
            else:
                return cast_dtype(ret)
        except TypeError:
            return ret

    is_complex = _check_modifier_return(apply_func, len(keywords), num_return, can_be_complex)
    if can_be_complex:
        init.update(is_complex=is_complex)

    class Modifier(kind):
        callsig = getattr(func, 'callsig', None)
        if not callsig:
            callsig = get_call_signature()
            callsig.function = func

        def __init__(self):
            # noinspection PyArgumentList
            super().__init__(apply_func, **init)
            self.apply = apply_func

        def __str__(self):
            return str(self.callsig)

        def __repr__(self):
            return repr(self.callsig)

        def __call__(self, *args, **kwargs):
            return func(*args, **kwargs)

    return Modifier()


@decorator_decorator
def site_state_modifier(min_neighbors=0):
    """Modify the state (valid or invalid) of lattice sites, e.g.\  to create vacancies

    Parameters
    ----------
    min_neighbors : int
        After modification, remove dangling sites with less than this number of neighbors.

    Notes
    -----
    The function parameters must be a combination of any number of the following:

    state : ndarray of bool
        Indicates if a lattice site is valid. Invalid sites will be removed from
        the model after all modifiers have been applied.
    x, y, z : ndarray
        Lattice site position.
    sub_id : ndarray
        Sublattice identifier: Can be checked for equality with sublattice names
        specified in :class:`.Lattice`. For example, `state[sub_id == 'A'] = False`
        will invalidate only sites on sublattice A.
    sites : :class:`.Sites`
        Helper object. Can be used instead of `x, y, z, sub_id`. See :class:`.Sites`.

    The function must return:

    ndarray
        A modified `state` argument or an `ndarray` of the same dtype and shape.

    Examples
    --------
    ::

        def vacancy(position, radius):
            @pb.site_state_modifier
            def f(state, x, y):
                x0, y0 = position
                state[(x-x0)**2 + (y-y0)**2 < radius**2] = False
                return state
            return f

        model = pb.Model(
            ... # lattice, shape, etc.
            vacancy(position=[0, 0], radius=0.1)
        )
    """
    return functools.partial(_make_modifier, kind=_cpp.SiteStateModifier,
                             init=dict(min_neighbors=min_neighbors),
                             keywords="state, x, y, z, sub_id")


@decorator_decorator
def site_position_modifier():
    """Modify the position of lattice sites, e.g.\  to apply geometric deformations

    Notes
    -----
    The function parameters must be a combination of any number of the following:

    x, y, z : ndarray
        Lattice site position.
    sub_id : ndarray of int
        Sublattice identifier: can be checked for equality with sublattice names
        specified in :class:`.Lattice`. For example, `x[sub_id == 'A'] += 0.1` will
        only displace sites on sublattice A.
    sites : :class:`.Sites`
        Helper object. Can be used instead of `x, y, z, sub_id`. See :class:`.Sites`.

    The function must return:

    tuple of ndarray
        Modified 'x, y, z' arguments or 3 `ndarray` objects of the same dtype and shape.

    Examples
    --------
    ::

        def triaxial_displacement(c):
            @pb.site_position_modifier
            def displacement(x, y, z):
                ux = 2*c * x*y
                uy = c * (x**2 - y**2)
                return x + ux, y + uy, z
            return displacement

        model = pb.Model(
            ... # lattice, shape, etc.
            triaxial_displacement(c=0.15)
        )
    """
    return functools.partial(_make_modifier, kind=_cpp.PositionModifier, init={},
                             keywords="x, y, z, sub_id", num_return=3)


@decorator_decorator
def onsite_energy_modifier(double=False):
    """Modify the onsite energy, e.g.\  to apply an electric field

    Parameters
    ----------
    double : bool
        Requires the model to use double precision floating point values.
        Defaults to single precision otherwise.

    Notes
    -----
    The function parameters must be a combination of any number of the following:

    energy : ndarray
        The onsite energy.
    x, y, z : ndarray
        Lattice site position.
    sub_id : ndarray of int
        Sublattice identifier: can be checked for equality with sublattice names
        specified in :class:`.Lattice`. For example, `energy[sub_id == 'A'] = 0`
        will set the onsite energy only for sublattice A sites.
    sites : :class:`.Sites`
        Helper object. Can be used instead of `x, y, z, sub_id`. See :class:`.Sites`.

    The function must return:

    ndarray
        A modified `potential` argument or an `ndarray` of the same dtype and shape.

    Examples
    --------
    ::

        def wavy(a, b):
            @pb.onsite_energy_modifier
            def f(x, y):
                return np.sin(a * x)**2 + np.cos(b * y)**2
            return f

        model = pb.Model(
            ... # lattice, shape, etc.
            wavy(a=0.6, b=0.9)
        )
    """
    return functools.partial(_make_modifier, kind=_cpp.OnsiteModifier,
                             init=dict(is_double=double),
                             keywords="energy, x, y, z, sub_id")


@decorator_decorator
def hopping_energy_modifier(double=False):
    """Modify the hopping energy, e.g.\  to apply a magnetic field

    Parameters
    ----------
    double : bool
        Requires the model to use double precision floating point values.
        Defaults to single precision otherwise.

    Notes
    -----
    The function parameters must be a combination of any number of the following:

    energy : ndarray
        The hopping energy between two sites.
    x1, y1, z1, x2, y2, z2 : ndarray
        Positions of the two lattice sites connected by the hopping parameter.
    hop_id : ndarray of int
        Hopping identifier: can be checked for equality with hopping names specified
        in :class:`.Lattice`. For example, `energy[hop_id == 't_nn'] *= 1.1` will only
        modify the energy of the hopping family named `t_nn`.

    The function must return:

    ndarray
        A modified `hopping` argument or an `ndarray` of the same dtype and shape.

    Examples
    --------
    ::

        def constant_magnetic_field(B):
            @pb.hopping_energy_modifier
            def f(energy, x1, y1, x2, y2):
                y = 0.5 * (y1 + y2) * 1e-9
                peierls = B * y * (x1 - x2) * 1e-9
                return energy * np.exp(1j * 2*pi/phi0 * peierls)
            return f

        model = pb.Model(
            ... # lattice, shape, etc.
            constant_magnetic_field(B=10)
        )
    """
    return functools.partial(_make_modifier, kind=_cpp.HoppingModifier,
                             init=dict(is_double=double), can_be_complex=True, has_sites=False,
                             keywords="energy, x1, y1, z1, x2, y2, z2, hop_id")


def constant_potential(magnitude):
    """Apply a constant onsite energy to every lattice site

    Parameters
    ----------
    magnitude : float
        In units of eV.
    """
    @onsite_energy_modifier
    def f(energy):
        return energy + magnitude
    return f


def force_double_precision():
    """Forces the model to use double precision even if no other modifier requires it"""
    @onsite_energy_modifier(double=True)
    def f(energy):
        return energy
    return f


def _make_generator(func, kind, name, energy, keywords):
    """Turn a regular function into a generator of the desired kind

    Parameters
    ----------
    func : callable
        The function which is to become a modifier.
    kind : object
        Modifier base class.
    keywords : str
        String of comma separated names: the expected arguments of a modifier function.
    """
    keywords = [word.strip() for word in keywords.split(",")]
    _check_modifier_spec(func, keywords)
    requested_argnames = tuple(inspect.signature(func).parameters.keys())

    def generator_func(*args):
        requested_kwargs = _process_modifier_args(args, keywords, requested_argnames)
        return func(**requested_kwargs)

    class Generator(kind):
        callsig = getattr(func, 'callsig', None)
        if not callsig:
            callsig = get_call_signature()
            callsig.function = func

        def __init__(self):
            # noinspection PyArgumentList
            super().__init__(name, energy, generator_func)

        def __str__(self):
            return str(self.callsig)

        def __repr__(self):
            return repr(self.callsig)

        def __call__(self, *args, **kwargs):
            return func(*args, **kwargs)

    return Generator()


@decorator_decorator
def hopping_generator(name, energy):
    """Introduce a new hopping family (with a new `hop_id`) via a list of index pairs

    This can be used to create new hoppings independent of the main :class:`Lattice` definition.
    It's especially useful for creating additional local hoppings, e.g. to model defects.

    Parameters
    ----------
    name : string
        Friendly identifier for the new hopping family.
    energy : Union[float, complex]
        Base hopping energy value.

    Notes
    -----
    The function parameters must be a combination of any number of the following:

    x, y, z : np.ndarray
        Lattice site position.
    sub_id : np.ndarray
        Sublattice identifier: can be checked for equality with sublattice names
        specified in :class:`.Lattice`.

    The function must return:

    Tuple[np.ndarray, np.ndarray]
        Arrays of index pairs which form the new hoppings.
    """
    return functools.partial(_make_generator, kind=_cpp.HoppingGenerator,
                             name=name, energy=energy, keywords="x, y, z, sub_id")
