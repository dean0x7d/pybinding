import pytest
import pybinding as pb
import numpy as np
from pybinding.repository import graphene

RELATIVE_TOLERANCE = 1.e-2

parameters = {
    'graphene_square': (
        dict(model=[graphene.lattice.monolayer(), pb.shape.rectangle(3)],
             feast=[(-0.1, 0.1), 10]),
        [-0.0113, -7.15e-05, -8.73e-09, 5.9e-09, 7.15e-05, 0.0113]
    )
}


def make_solver(params, name):
    model = pb.Model(*params['model'])
    make = getattr(pb.solver, 'make_{}'.format(name))
    return make(model, *params[name])


def generate_data():
    for name, (params, _) in parameters.items():
        solver = make_solver(params, 'feast')
        values = ', '.join('{:.3g}'.format(v) for v in solver.eigenvalues)
        print('{name}: [{values}]'.format(**locals()))


@pytest.mark.parametrize('params, expected', parameters.values(), ids=list(parameters.keys()))
def test_feast(params, expected):
    solver = make_solver(params, 'feast')

    assert np.allclose(solver.eigenvalues, expected, RELATIVE_TOLERANCE)
    assert solver.eigenvectors.shape == (solver.eigenvalues.size, solver.system.num_sites)

    # property lifetime test
    values, vectors, system = solver.eigenvalues, solver.eigenvectors, solver.system
    del solver
    assert np.allclose(values, expected, RELATIVE_TOLERANCE)
    assert vectors.shape == (values.size, system.num_sites)


if __name__ == '__main__':
    generate_data()
