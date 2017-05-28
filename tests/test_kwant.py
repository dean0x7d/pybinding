import pytest

import numpy as np

import pybinding as pb
from pybinding.support.kwant import kwant_installed
from pybinding.repository import graphene


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
