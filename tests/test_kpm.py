import pytest
import pybinding as pb
import numpy as np
from pybinding.repository import graphene


parameters = {
    'graphene_square': (
        dict(model=[graphene.lattice.monolayer(), pb.shape.rectangle(50)],
             ldos=[np.linspace(0, 0.5, 11), 0.03, (0, 0)]),
        [0.0212, 0.0261, 0.0328, 0.0406, 0.0513, 0.0589, 0.0675, 0.0792, 0.087, 0.0968, 0.108]
    )
}


def calc_ldos(params):
    model = pb.Model(*params['model'])
    greens = pb.greens.KPM(model)
    return greens.calc_ldos(*params['ldos'])


def generate_data(show_plot=False):
    for name, (params, _) in parameters.items():
        result = calc_ldos(params)
        values = ', '.join('{:.3g}'.format(v) for v in result.ldos)
        print('{name}: [{values}]'.format(**locals()))

        if show_plot:
            import matplotlib.pyplot as plt
            result.plot()
            plt.show()


@pytest.mark.parametrize('params, expected', parameters.values(), ids=list(parameters.keys()))
def test_kpm(params, expected):
    result = calc_ldos(params)
    assert np.allclose(result.ldos, expected, 1.e-2)


if __name__ == '__main__':
    generate_data()
