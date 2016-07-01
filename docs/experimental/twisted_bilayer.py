"""Construct a circular flake of twisted bilayer graphene (arbitrary angle)"""
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

import pybinding as pb

c0 = 0.335  # [nm] graphene interlayer spacing


def two_graphene_monolayers():
    """Two individual layers of monolayer graphene without any interlayer hopping"""
    from pybinding.repository.graphene.constants import a_cc, a, t

    lat = pb.Lattice(a1=[a/2, a/2 * math.sqrt(3)], a2=[-a/2, a/2 * math.sqrt(3)])
    lat.add_sublattices(('A1', [0,  -a_cc,   0]),
                        ('B1', [0,      0,   0]),
                        ('A2', [0,      0, -c0]),
                        ('B2', [0,   a_cc, -c0]))
    lat.register_hopping_energies({'gamma0': t})
    lat.add_hoppings(
        # layer 1
        ([ 0,  0], 'A1', 'B1', 'gamma0'),
        ([ 0, -1], 'A1', 'B1', 'gamma0'),
        ([-1,  0], 'A1', 'B1', 'gamma0'),
        # layer 2
        ([ 0,  0], 'A2', 'B2', 'gamma0'),
        ([ 0, -1], 'A2', 'B2', 'gamma0'),
        ([-1,  0], 'A2', 'B2', 'gamma0'),
        # not interlayer hopping
    )
    lat.min_neighbors = 2
    return lat


def twist_layers(theta):
    """Rotate one layer and then a generate hopping between the rotated layers"""
    theta = theta / 180 * math.pi  # from degrees to radians

    @pb.site_position_modifier
    def rotate(x, y, z):
        """Rotate layer 2 by the given angle `theta`"""
        layer2 = (z < 0)
        x0 = x[layer2]
        y0 = y[layer2]
        x[layer2] = x0 * math.cos(theta) - y0 * math.sin(theta)
        y[layer2] = y0 * math.cos(theta) + x0 * math.sin(theta)
        return x, y, z

    @pb.hopping_generator('interlayer', energy=0.1)  # eV
    def interlayer_generator(x, y, z):
        """Generate hoppings for site pairs which have distance `d_min < d < d_max`"""
        positions = np.stack([x, y, z], axis=1)
        layer1 = (z == 0)
        layer2 = (z != 0)

        d_min = c0 * 0.98
        d_max = c0 * 1.1
        kdtree1 = cKDTree(positions[layer1])
        kdtree2 = cKDTree(positions[layer2])
        coo = kdtree1.sparse_distance_matrix(kdtree2, d_max, output_type='coo_matrix')

        idx = coo.data > d_min
        abs_idx1 = np.flatnonzero(layer1)
        abs_idx2 = np.flatnonzero(layer2)
        row, col = abs_idx1[coo.row[idx]], abs_idx2[coo.col[idx]]
        return row, col  # lists of site indices to connect

    @pb.hopping_energy_modifier
    def interlayer_hopping_value(energy, x1, y1, z1, x2, y2, z2, hop_id):
        """Set the value of the newly generated hoppings as a function of distance"""
        d = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        interlayer = (hop_id == 'interlayer')
        energy[interlayer] = 0.4 * c0 / d[interlayer]
        return energy

    return rotate, interlayer_generator, interlayer_hopping_value


model = pb.Model(
    two_graphene_monolayers(),
    pb.circle(radius=1.5),
    twist_layers(theta=21.798)
)
plt.figure(figsize=(6.5, 6.5))
model.plot(hopping=dict(width=1.6, cmap='auto'))
plt.title(r"$\theta$ = 21.798 $\degree$")
plt.show()


model = pb.Model(
    two_graphene_monolayers(),
    pb.circle(radius=1.5),
    twist_layers(theta=12.95)
)
plt.figure(figsize=(6.5, 6.5))
model.plot(hopping=dict(width=1.6, cmap='auto'))
plt.title(r"$\theta$ = 12.95 $\degree$")
plt.show()
