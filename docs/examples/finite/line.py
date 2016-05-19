"""1D lattice chains - finite dimension are imposed using builtin `pb.line` shape"""
import pybinding as pb
import matplotlib.pyplot as plt

pb.pltutils.use_style()


def simple_chain_lattice(a=1, t=-1):
    """Very simple 1D lattice"""
    lat = pb.Lattice(a)
    lat.add_one_sublattice('A', [0, 0])
    lat.add_one_hopping(1, 'A', 'A', t)
    return lat

model = pb.Model(
    simple_chain_lattice(),
    pb.line(-3.5, 3.5)  # line start/end in nanometers
)
model.plot()
plt.show()


def trestle(a=0.2, t1=0.8 + 0.6j, t2=2):
    """A more complicated 1D lattice with 2 sublattices"""
    lat = pb.Lattice(1.3 * a)
    lat.add_sublattices(
        ('A', [0,   0], 0),
        ('B', [a/2, a], 0)
    )
    lat.add_hoppings(
        (0, 'A', 'B', t1),
        (1, 'A', 'B', t1),
        (1, 'A', 'A', t2),
        (1, 'B', 'B', t2)
    )
    lat.min_neighbors = 2
    return lat

model = pb.Model(trestle(), pb.line(-0.7, 0.7))
model.plot()
plt.show()
