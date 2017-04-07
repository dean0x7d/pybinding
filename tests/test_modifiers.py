import pytest

import numpy as np
import pybinding as pb
from pybinding.repository import graphene


one, zero = np.ones(1), np.zeros(1)
complex_one = np.ones(1, dtype=np.complex64)


def build_model(*params):
    model = pb.Model(graphene.monolayer(), *params)
    model.report()
    return model


def test_modifier_function_signature():
    pb.onsite_energy_modifier(lambda energy: energy)
    with pytest.raises(RuntimeError) as excinfo:
        pb.onsite_energy_modifier(lambda this_is_unexpected: None)
    assert "Unexpected argument" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        pb.onsite_energy_modifier(lambda energy, x, y, z, w: None)
    assert "Unexpected argument" in str(excinfo.value)


@pb.site_state_modifier
def global_mod(state):
    return np.ones_like(state)


def test_callsig():
    assert "global_mod()" == str(global_mod)
    assert "global_mod()" == repr(global_mod)

    @pb.site_state_modifier
    def local_mod(state):
        return np.ones_like(state)
    assert "test_callsig()" == str(local_mod)
    assert "test_callsig()" == repr(local_mod)

    def wrapped_mod(a, b):
        @pb.site_state_modifier
        def actual_mod(state):
            return np.ones_like(state) * a * b
        return actual_mod
    assert "wrapped_mod(a=1, b=8)" == str(wrapped_mod(1, 8))
    assert "test_callsig.<locals>.wrapped_mod(a=1, b=8)" == repr(wrapped_mod(1, 8))


def test_type_errors():
    """Modifier return values should be arrays and satisfy a few dtype criteria"""
    build_model(pb.onsite_energy_modifier(lambda energy: energy + 1))
    with pytest.raises(TypeError) as excinfo:
        build_model(pb.onsite_energy_modifier(lambda: 1))
    assert "Modifiers must return ndarray(s)" in str(excinfo.value)

    build_model(pb.site_position_modifier(lambda x, y, z: (x, y, z)))
    with pytest.raises(TypeError) as excinfo:
        build_model(pb.site_position_modifier(lambda x, y, z: (x, y)))
    assert "expected to return 3 ndarray(s), but got 2" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        build_model(pb.onsite_energy_modifier(lambda: (one, one)))
    assert "expected to return 1 ndarray(s), but got 2" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        build_model(pb.onsite_energy_modifier(lambda x: np.zeros(x.size // 2)))
    assert "must return the same size" in str(excinfo.value)

    def complex_ones(energy):
        return np.ones_like(energy, dtype=np.complex128)

    build_model(pb.hopping_energy_modifier(complex_ones))
    build_model(pb.onsite_energy_modifier(complex_ones))

    with pytest.raises(TypeError) as excinfo:
        build_model(pb.site_position_modifier(lambda x: (np.ones_like(x, dtype=np.complex128),)*3))
    assert "'complex128', but expected same kind as 'float32'" in str(excinfo.value)


def test_cast():
    dtypes = [np.float32, np.float64, np.complex64, np.complex128]

    @pb.hopping_energy_modifier
    def float32_out(energy):
        return np.ones_like(energy, dtype=np.float32)

    for dtype in dtypes:
        assert float32_out.apply(np.ones(1, dtype=dtype), zero, zero, zero).dtype == dtype

    @pb.hopping_energy_modifier
    def float64_out(energy):
        return np.ones_like(energy, dtype=np.float64)

    for dtype in dtypes:
        assert float64_out.apply(np.ones(1, dtype=dtype), zero, zero, zero).dtype == dtype

    @pb.hopping_energy_modifier
    def complex64_out(energy):
        return np.ones_like(energy, dtype=np.complex64)

    for dtype, result in zip(dtypes, [np.complex64, np.complex64, np.complex64, np.complex128]):
        assert complex64_out.apply(np.ones(1, dtype=dtype), zero, zero, zero).dtype == result

    @pb.hopping_energy_modifier
    def complex128_out(energy):
        return np.ones_like(energy, dtype=np.complex128)

    for dtype, result in zip(dtypes, [np.complex128, np.complex128, np.complex64, np.complex128]):
        assert complex128_out.apply(np.ones(1, dtype=dtype), zero, zero, zero).dtype == result


def test_site_state():
    @pb.site_state_modifier
    def mod(state):
        return np.ones_like(state)
    assert np.all(mod(zero))
    assert np.all(mod.apply(zero, one, one, one, one))

    capture = {}

    @pb.site_state_modifier
    def check_args(state, x, y, z, sub_id, sites):
        capture[sub_id] = [v.copy() for v in (state, x, y, z)]
        capture[sub_id].append(sites.argsort_nearest([0, graphene.a_cc / 2]))
        return state

    def assert_state(name, **kwargs):
        state, x, y, z, nearest = capture[name]
        assert state.shape == kwargs["shape"]
        assert pytest.fuzzy_equal(state, kwargs["state"])
        assert x.shape == kwargs["shape"]
        assert pytest.fuzzy_equal(x, kwargs["x"])
        assert pytest.fuzzy_equal(y, kwargs["y"])
        assert pytest.fuzzy_equal(z, kwargs["z"])
        assert np.all(nearest == kwargs["nearest"])

    model = build_model(check_args, pb.primitive(1, 2))
    assert model.hamiltonian.dtype == np.float32

    assert_state("A", shape=(2,), state=[True] * 2, nearest=[1, 0],
                 x=[-graphene.a / 2, 0], y=[-2 * graphene.a_cc, -graphene.a_cc / 2], z=[0, 0])
    assert_state("B", shape=(2,), state=[True] * 2, nearest=[1, 0],
                 x=[-graphene.a / 2, 0], y=[-graphene.a_cc, graphene.a_cc / 2], z=[0, 0])

    @pb.site_state_modifier
    def remove_site(state):
        state[0] = False
        return state

    model = build_model(remove_site, pb.primitive(2, 2))
    assert model.system.num_sites == 6

    @pb.site_state_modifier(min_neighbors=1)
    def remove_dangling(state, sub_id):
        state[sub_id == "A"] = False
        return state

    with pytest.raises(RuntimeError) as excinfo:
        build_model(remove_dangling, pb.primitive(3, 3))
    assert "0 sites" in str(excinfo.value)


def test_site_position():
    @pb.site_position_modifier
    def mod(x, y, z):
        return x + 1, y + 1, z + 1
    assert (one,) * 3 == mod(zero, zero, zero)
    assert (one,) * 3 == mod.apply(zero, zero, zero, one)

    capture = {}

    @pb.site_position_modifier
    def check_args(x, y, z, sub_id, sites):
        capture[sub_id] = [v.copy() for v in (x, y, z)]
        capture[sub_id].append(sites.argsort_nearest([0, graphene.a_cc / 2]))
        return x, y, z

    def assert_positions(name, **kwargs):
        x, y, z, nearest = capture[name]
        assert x.shape == kwargs["shape"]
        assert y.shape == kwargs["shape"]
        assert z.shape == kwargs["shape"]
        assert pytest.fuzzy_equal(x, kwargs["x"])
        assert pytest.fuzzy_equal(y, kwargs["y"])
        assert pytest.fuzzy_equal(z, kwargs["z"])
        assert np.all(nearest == kwargs["nearest"])

    model = build_model(check_args, pb.primitive(1, 2))
    assert model.hamiltonian.dtype == np.float32

    assert_positions("A", shape=(2,), nearest=[1, 0],
                     x=[-graphene.a / 2, 0], y=[-2 * graphene.a_cc, -graphene.a_cc / 2], z=[0, 0])
    assert_positions("B", shape=(2,), nearest=[1, 0],
                     x=[-graphene.a / 2, 0], y=[-graphene.a_cc, graphene.a_cc / 2], z=[0, 0])

    @pb.site_position_modifier
    def shift(x, y, z):
        return x + 1, y + 1, z + 1

    model = build_model(check_args, shift)
    assert pytest.fuzzy_equal(model.system.x, [1, 1])
    assert pytest.fuzzy_equal(model.system.y, [1 - graphene.a_cc / 2, 1 + graphene.a_cc / 2])
    assert pytest.fuzzy_equal(model.system.z, [1, 1])


def test_onsite():
    @pb.onsite_energy_modifier
    def mod(energy):
        return energy + 2
    assert np.all(2 == mod(zero))
    assert np.all(2 == mod.apply(zero, zero, zero, zero, one))

    capture = {}

    @pb.onsite_energy_modifier
    def check_args(energy, x, y, z, sub_id, sites):
        capture[sub_id] = [v.copy() for v in (energy, x, y, z)]
        capture[sub_id].append(sites.argsort_nearest([0, graphene.a_cc / 2]))
        return energy

    model = build_model(check_args)
    assert model.hamiltonian.dtype == np.float32

    energy, x, y, z, nearest = capture["A"]
    assert np.allclose(energy, 0)
    assert np.allclose(x, 0)
    assert np.allclose(y, -graphene.a_cc / 2)
    assert np.allclose(z, 0)
    assert np.all(nearest == [0])

    energy, x, y, z, nearest = capture["B"]
    assert np.allclose(energy, 0)
    assert np.allclose(x, 0)
    assert np.isclose(y, graphene.a_cc / 2)
    assert np.allclose(z, 0)
    assert np.all(nearest == [0])

    @pb.onsite_energy_modifier(is_double=True)
    def make_double(energy):
        return energy

    model = build_model(make_double)
    assert model.hamiltonian.dtype == np.float64


def test_hopping_energy():
    @pb.hopping_energy_modifier
    def mod(energy):
        return energy * 2
    assert np.all(2 == mod(one))
    assert np.all(2 == mod.apply(one, zero, zero, zero, zero, zero, zero, zero))

    capture = {}

    @pb.hopping_energy_modifier
    def check_args(energy, hop_id, x1, y1, z1, x2, y2, z2):
        capture[hop_id] = [v.copy() for v in (energy, x1, y1, z1, x2, y2, z2)]
        return energy

    model = build_model(check_args)
    assert model.hamiltonian.dtype == np.float32

    energy, x1, y1, z1, x2, y2, z2 = capture["t"]
    assert np.allclose(energy, graphene.t)
    assert np.allclose(x1, 0)
    assert np.allclose(y1, -graphene.a_cc / 2)
    assert np.allclose(z1, 0)
    assert np.allclose(x2, 0)
    assert np.allclose(y2, graphene.a_cc / 2)
    assert np.allclose(z2, 0)

    @pb.hopping_energy_modifier(is_double=True)
    def make_double(energy):
        return energy

    model = build_model(make_double)
    assert model.hamiltonian.dtype == np.float64

    @pb.hopping_energy_modifier
    def make_complex(energy):
        return energy * 1j

    model = build_model(make_complex)
    assert model.hamiltonian.dtype == np.complex64

    @pb.hopping_energy_modifier(is_double=True)
    def make_complex_double(energy):
        return energy * 1j

    model = build_model(make_complex_double)
    assert model.hamiltonian.dtype == np.complex128


def test_hopping_generator():
    """Generated next-nearest hoppings should produce the same result as the builtin lattice"""
    from scipy.spatial import cKDTree

    @pb.hopping_generator("tnn_test", energy=graphene.t_nn)
    def next_nearest(x, y, z):
        pos = np.stack([x, y, z], axis=1)
        dmin = graphene.a * 0.95
        dmax = graphene.a * 1.05
        kdtree = cKDTree(pos)
        coo = kdtree.sparse_distance_matrix(kdtree, dmax).tocoo()
        idx = coo.data > dmin
        return coo.row[idx], coo.col[idx]

    @pb.onsite_energy_modifier
    def onsite_offset(energy):
        return energy + 3 * graphene.t_nn

    model = pb.Model(graphene.monolayer(), next_nearest, onsite_offset, graphene.hexagon_ac(1))
    expected = pb.Model(graphene.monolayer(2), graphene.hexagon_ac(1))
    assert pytest.fuzzy_equal(model.hamiltonian, expected.hamiltonian)


def test_wrapper_return():
    """Make sure the wrapper return type conversion is working"""
    @pb.hopping_energy_modifier
    def mul(energy):
        """Returning a non-contiguous view will force the wrapper to create a copy"""
        energy = np.concatenate([energy, energy])
        energy *= 3
        return energy[::2]

    lattice = pb.Lattice([1, 0])
    lattice.add_sublattices(("A", [0, 0]), ("B", [0, 0]))
    lattice.add_one_hopping([0], "A", "B", 1.0)

    model = pb.Model(lattice, mul, pb.primitive(2))
    assert pytest.fuzzy_equal(model.hamiltonian.data, [3, 3, 3, 3])


def test_invalid_return():
    @pb.onsite_energy_modifier
    def mod_inf(energy):
        return np.ones_like(energy) * np.inf

    with pytest.raises(RuntimeError) as excinfo:
        build_model(mod_inf)
    assert "NaN or INF" in str(excinfo.value)

    @pb.onsite_energy_modifier
    def mod_nan(energy):
        return np.ones_like(energy) * np.NaN

    with pytest.raises(RuntimeError) as excinfo:
        build_model(mod_nan)
    assert "NaN or INF" in str(excinfo.value)


def test_mutability():
    """Only modifier return arguments should be mutable"""
    @pb.onsite_energy_modifier
    def mod_energy(energy):
        """The return energy is writable"""
        energy += 1
        return energy

    assert build_model(mod_energy)

    @pb.onsite_energy_modifier
    def mod_x(energy, x):
        """Arguments are read-only"""
        x += 1
        return energy

    with pytest.raises(ValueError) as excinfo:
        build_model(mod_x)
    assert "read-only" in str(excinfo.value)


def test_multiorbital_onsite():
    def multi_orbital_lattice():
        lat = pb.Lattice([1, 0], [0, 1])

        tau_z = np.array([[1, 0],
                          [0, -1]])
        tau_x = np.array([[0, 1],
                          [1, 0]])
        lat.add_sublattices(("A", [0,   0], tau_z + 2 * tau_x),
                            ("B", [0, 0.1], 0.5),
                            ("C", [0, 0.2], [1, 2, 3]))
        lat.add_hoppings(([0, -1], "A", "A", 3 * tau_z),
                         ([1,  0], "A", "A", 3 * tau_z),
                         ([0, 0], "B", "C", [[2, 3, 4]]))
        return lat

    capture = {}

    @pb.onsite_energy_modifier
    def onsite(energy, x, y, z, sub_id):
        capture[sub_id] = [v.copy() for v in (energy, x, y, z)]
        return energy

    def assert_positions(name, **expected):
        _, x, y, z = capture[name]
        assert pytest.fuzzy_equal(x.squeeze(), expected["x"])
        assert pytest.fuzzy_equal(y.squeeze(), expected["y"])
        assert pytest.fuzzy_equal(z.squeeze(), expected["z"])

    def assert_onsite(name, **expected):
        energy, *_ = capture[name]
        assert energy.shape == expected["shape"]
        for i in range(energy.shape[0]):
            assert pytest.fuzzy_equal(energy[i], expected["energy"])

    model = pb.Model(multi_orbital_lattice(), pb.rectangle(2, 1), onsite)
    assert model.system.num_sites == 6
    assert model.hamiltonian.shape[0] == 12

    assert_positions("A", x=[0, 1], y=[0, 0], z=[0, 0])
    assert_positions("B", x=[0, 1], y=[0.1, 0.1], z=[0, 0])
    assert_positions("C", x=[0, 1], y=[0.2, 0.2], z=[0, 0])

    assert_onsite("A", shape=(2, 2, 2), energy=[[1, 2],
                                                [2, -1]])
    assert_onsite("B", shape=(2,), energy=[0.5])
    assert_onsite("C", shape=(2, 3, 3), energy=[[1, 0, 0],
                                                [0, 2, 0],
                                                [0, 0, 3]])

    @pb.onsite_energy_modifier
    def onsite_mod(energy, x, y, z, sub_id):
        if sub_id == "A":
            energy += sub_id.eye * (x > -1)
        elif sub_id == "B":
            energy *= 2
        elif sub_id == "C":
            energy += [[0, 1, 0]]
        capture[sub_id] = [v.copy() for v in (energy, x, y, z)]
        return energy

    model = pb.Model(multi_orbital_lattice(), pb.rectangle(2, 1), onsite_mod)
    assert model.system.num_sites == 6
    assert model.hamiltonian.shape[0] == 12

    assert_onsite("A", shape=(2, 2, 2), energy=[[2, 2],
                                                [2, 0]])
    assert_onsite("B", shape=(2,), energy=[1])
    assert_onsite("C", shape=(2, 3, 3), energy=[[1, 1, 0],
                                                [0, 3, 0],
                                                [0, 1, 3]])


def test_multiorbital_hoppings():
    """For multi-orbital lattices, hopping modifiers get `energy` as 3D array"""
    def multi_orbital_lattice():
        lat = pb.Lattice([1, 0], [0, 1])

        tau_z = np.array([[1, 0],
                          [0, -1]])
        tau_x = np.array([[0, 1],
                          [1, 0]])
        lat.add_sublattices(("A", [-0.25, 0.0], tau_z + 2 * tau_x),
                            ("B", [0.0, 0.5], 0.5),
                            ("C", [0.25, 0.0], [1, 2, 3]))
        lat.register_hopping_energies({
            "t11": 1,
            "t1|2": 2,
            "t1|3": 3,
            "t22": 3 * tau_z,
            "t23": [[0, 1, 2],
                    [3, 4, 5]],
            "t13": [[11, 12, 13]],
        })
        lat.add_hoppings(([1,  0], "B", "B", "t11"),
                         ([0,  1], "B", "B", "t11"),
                         ([1,  1], "B", "B", "t1|2"),
                         ([1, -1], "B", "B", "t1|3"),
                         ([0,  1], "A", "A", "t22"),
                         ([0,  0], "A", "C", "t23"),
                         ([0,  0], "B", "C", "t13"))
        return lat

    capture = {}

    @pb.hopping_energy_modifier
    def hopping(energy, hop_id, x1, y1, z1, x2, y2, z2):
        capture[hop_id] = [v.copy() for v in (energy, x1, y1, z1, x2, y2, z2)]
        return energy

    def assert_positions(name, **expected):
        _, x1, y1, z1, x2, y2, z2 = capture[name]

        assert pytest.fuzzy_equal(x1.squeeze(), expected["x1"])
        assert pytest.fuzzy_equal(y1.squeeze(), expected["y1"])
        assert pytest.fuzzy_equal(z1.squeeze(), expected["z1"])
        assert pytest.fuzzy_equal(x2.squeeze(), expected["x2"])
        assert pytest.fuzzy_equal(y2.squeeze(), expected["y2"])
        assert pytest.fuzzy_equal(z1.squeeze(), expected["z2"])

    def assert_hoppings(name, **expected):
        energy, *_ = capture[name]
        assert energy.shape == expected["shape"]
        for i in range(energy.shape[0]):
            assert pytest.fuzzy_equal(energy[i], expected["energy"])

    model = pb.Model(multi_orbital_lattice(), pb.primitive(2, 2), hopping)
    assert model.system.num_sites == 12
    assert model.hamiltonian.shape[0] == 24

    assert_positions("t11",
                     x1=[-1, -1, 0, -1], y1=[-0.5, -0.5, -0.5, 0.5], z1=[0, 0, 0, 0],
                     x2=[0, -1, 0, 0], y2=[-0.5, 0.5, 0.5, 0.5], z2=[0, 0, 0, 0])
    assert_positions("t22",
                     x1=[-1.25, -0.25], y1=[-1, -1], z1=[0, 0],
                     x2=[-1.25, -0.25], y2=[0, 0], z2=[0, 0])
    assert_positions("t23",
                     x1=[-1.25, -0.25, -1.25, -0.25], y1=[-1, -1, 0, 0], z1=[0, 0, 0, 0],
                     x2=[-0.75, 0.25, -0.75, 0.25], y2=[-1, -1, 0, 0], z2=[0, 0, 0, 0])
    assert_positions("t13",
                     x1=[-1, 0, -1, 0], y1=[-0.5, -0.5, 0.5, 0.5], z1=[0, 0, 0, 0],
                     x2=[-0.75, 0.25, -0.75, 0.25], y2=[-1, -1, 0, 0], z2=[0, 0, 0, 0])

    assert_hoppings("t11", shape=(4,), energy=[1])
    assert_hoppings("t22", shape=(2, 2, 2), energy=[[3, 0],
                                                    [0, -3]])
    assert_hoppings("t23", shape=(4, 2, 3), energy=[[0, 1, 2],
                                                    [3, 4, 5]])
    assert_hoppings("t13", shape=(4, 1, 3), energy=[[11, 12, 13]])

    @pb.hopping_energy_modifier
    def hopping_mod(energy, hop_id, x1, y1, z1, x2, y2, z2):
        if hop_id == "t11":
            energy *= x1 * 0
        elif hop_id == "t1":
            energy *= 3
        elif hop_id == "t22":
            energy += [[0, 1],
                       [1, 0]]
        elif hop_id == "t23":
            energy += [[1, 0, 0]]
        elif hop_id == "t13":
            energy *= 2

        capture[hop_id] = [v.copy() for v in (energy, x1, y1, z1, x2, y2, z2)]
        return energy

    model = pb.Model(multi_orbital_lattice(), pb.primitive(2, 2), hopping_mod)
    assert model.system.num_sites == 12
    assert model.hamiltonian.shape[0] == 24

    assert_hoppings("t11", shape=(4,), energy=[0])
    assert_hoppings("t1|2", shape=(1,), energy=[6])
    assert_hoppings("t1|3", shape=(1,), energy=[9])
    assert_hoppings("t22", shape=(2, 2, 2), energy=[[3, 1],
                                                    [1, -3]])
    assert_hoppings("t23", shape=(4, 2, 3), energy=[[1, 1, 2],
                                                    [4, 4, 5]])
    assert_hoppings("t13", shape=(4, 1, 3), energy=[[22, 24, 26]])


def test_hopping_buffer():
    """The energy passed to hopping modifiers is buffered, but users should not be aware of it"""
    def lattice():
        lat = pb.Lattice([1, 0], [0, 1])

        lat.add_sublattices(("A", [0, 0], [0, 0, 0, 0]))
        lat.register_hopping_energies({
            "t44": [[ 0,  1,  2,  3],
                    [ 4,  5,  6,  7],
                    [ 8,  9, 10, 11],
                    [12, 13, 14, 15]]
        })
        lat.add_hoppings(([1, 0], "A", "A", "t44"),
                         ([0, 1], "A", "A", "t44"))
        return lat

    capture = {}

    @pb.hopping_energy_modifier
    def check_buffer(energy, hop_id):
        capture.setdefault(hop_id, [])
        capture[hop_id] += [energy.copy()]
        energy[0] = 99
        return energy

    model = pb.Model(lattice(), pb.primitive(3000, 2), check_buffer)
    assert model.system.num_sites == 6000
    assert model.hamiltonian.shape[0] == 24000

    energies = capture["t44"]
    assert len(energies) >= 2
    assert energies[0].shape == (6250, 4, 4)
    for energy in energies:
        assert np.argwhere(energy == 99).size == 0
