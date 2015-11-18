"""Molybdenum disulfide"""
import math
import pybinding as pb


def three_band_lattice():
    """MoS2 lattice using the three-band model"""
    # TODO: this is a proof of concept for `Lattice.add_hopping_matrices()`
    # TODO: still needs to be checked for accuracy

    # lattice constant
    a = 0.319  # [nm]
    # onsite energies
    eps0 = 1.046
    eps2 = 2.104
    # hoppings
    t0 = -0.184
    t1 = 0.401
    t2 = 0.507
    t11 = 0.218
    t12 = 0.338
    t22 = 0.057
    # convenient constant
    rt3 = math.sqrt(3)

    lat = pb.Lattice(a1=[a, 0], a2=[0.5 * a, 0.5*rt3 * a])
    lat.add_sublattices(('s1', [0, 0], eps0),
                        ('s2', [0, 0], eps2),
                        ('s3', [0, 0], eps2))

    r1 = [1,  0]
    r2 = [1, -1]
    r3 = [0, -1]

    t_mat1 = [[ t0,   t1,  t2],
              [-t1,  t11, t12],
              [ t2, -t12, t22]]

    t_mat2 = [[                  t0,        0.5*t1 - 0.5*rt3*t2,       -0.5*rt3*t1 - 0.5*t2],
              [-0.5*t1 - 0.5*rt3*t2,        0.25*t11 + 0.75*t22, 0.25*rt3*(t22 - t11) - t12],
              [ 0.5*rt3*t1 - 0.5*t2, 0.25*rt3*(t22 - t11) + t12,        0.75*t11 + 0.25*t22]]

    t_mat3 = [[                 t0,        -0.5*t1 + 0.5*rt3*t2,       -0.5*rt3*t1 - 0.5*t2],
              [0.5*t1 + 0.5*rt3*t2,         0.25*t11 + 0.75*t22, 0.25*rt3*(t11 - t22) + t12],
              [0.5*rt3*t1 - 0.5*t2, -0.25*rt3*(t11 + t22) - t12,        0.75*t11 + 0.25*t22]]

    lat.add_hopping_matrices([r1, t_mat1], [r2, t_mat2], [r3, t_mat3])
    lat.min_neighbors = 2
    return lat
