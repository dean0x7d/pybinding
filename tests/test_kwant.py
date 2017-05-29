import pytest

import numpy as np

import pybinding as pb
from pybinding.support.kwant import kwant_installed
from pybinding.repository import graphene, group6_tmd


if not kwant_installed:
    def test_kwant_error():
        """Raise an exception if kwant isn't installed"""
        model = pb.Model(graphene.monolayer())
        with pytest.raises(ImportError) as excinfo:
            model.tokwant()
        assert "kwant isn't installed" in str(excinfo.value)
else:
    import kwant

    # noinspection PyUnresolvedReferences
    def make_wire_with_flat_potential(width=10, lenght=2, t=1):
        def onsite(_, v):
            return 4 * t - v

        lattice = kwant.lattice.square(a=1)
        builder = kwant.Builder()
        builder[(lattice(x, y) for x in range(lenght) for y in range(width))] = onsite
        builder[lattice.neighbors()] = -t

        lead = kwant.Builder(kwant.TranslationalSymmetry([-1, 0]))
        lead[(lattice(0, y) for y in range(width))] = 4 * t
        lead[lattice.neighbors()] = -t

        builder.attach_lead(lead)
        builder.attach_lead(lead.reversed())
        return builder.finalized()


    # noinspection PyUnresolvedReferences
    def calc_transmission(system, energy, v=None):
        args = [v] if v is not None else ()
        smatrix = kwant.smatrix(system, energy, args=args)
        return smatrix.transmission(1, 0)


    def pb_model(v=0, length=2, width=10):
        def square_lattice(d, t):
            lat = pb.Lattice(a1=[d, 0], a2=[0, d])
            lat.add_sublattices(("A", [0, 0], 4 * t))
            lat.add_hoppings(([0, 1], "A", "A", -t),
                             ([1, 0], "A", "A", -t))
            return lat

        @pb.onsite_energy_modifier
        def potential_well(energy, x):
            energy[np.logical_and(x >= 0, x <= 1)] -= v
            return energy

        model = pb.Model(square_lattice(d=1, t=1), pb.rectangle(length, width), potential_well)
        model.attach_lead(-1, pb.line([0, -width / 2 - 0.1], [0, width / 2]))
        model.attach_lead(+1, pb.line([0, -width / 2 - 0.1], [0, width / 2]))
        return model


    def test_kwant():
        """Create the same model using kwant and pybinding and solve with kwant.smatrix"""
        energy = 1
        vs = np.linspace(-2, 0, 5)

        system = make_wire_with_flat_potential()
        kwant_result = np.array([calc_transmission(system, energy, v) for v in vs])
        pb_result = np.array([calc_transmission(pb_model(v).tokwant(), energy) for v in vs])

        assert pytest.fuzzy_equal(pb_result, kwant_result)


    @pytest.mark.parametrize("lattice, norb", [
        (graphene.monolayer(), 1),
        (group6_tmd.monolayer_3band("MoS2"), 3),
    ])
    def test_hamiltonian_submatrix_orbitals(lattice, norb):
        """Return the number of orbitals at each site in addition to the Hamiltonian"""
        model = pb.Model(lattice, pb.rectangle(1, 1))
        kwant_sys = model.tokwant()

        matrix, to_norb, from_norb = kwant_sys.hamiltonian_submatrix(sparse=True, return_norb=True)
        assert matrix.shape == model.hamiltonian.shape
        assert to_norb.size == model.system.num_sites
        assert from_norb.size == model.system.num_sites
        assert np.all(to_norb == norb)
        assert np.all(from_norb == norb)


    def test_hamiltonian_submatrix_sites():
        """The `to_sites` and `from_sites` arguments are not supported"""
        kwant_sys = pb.Model(graphene.monolayer(), pb.rectangle(1, 1)).tokwant()

        with pytest.raises(RuntimeError) as excinfo:
            kwant_sys.hamiltonian_submatrix(to_sites=1, from_sites=1)
        assert "not supported" in str(excinfo.value)


    def test_warnings():
        """Extra arguments and ignored by pybinding -- warn users"""
        kwant_sys = pb.Model(graphene.monolayer(), pb.rectangle(1, 1)).tokwant()
        with pytest.warns(UserWarning):
            kwant_sys.hamiltonian_submatrix(sparse=True, args=(1,))
        with pytest.warns(UserWarning):
            kwant_sys.hamiltonian_submatrix(sparse=True, params=dict(v=1))
