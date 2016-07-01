"""Several finite-sized systems created using builtin lattices and shapes"""
import pybinding as pb
from pybinding.repository import graphene
import matplotlib.pyplot as plt
from math import pi

pb.pltutils.use_style()

model = pb.Model(
    graphene.monolayer(),
    pb.rectangle(x=2, y=1.2)
)
model.plot()
plt.show()


model = pb.Model(
    graphene.monolayer(),
    pb.regular_polygon(num_sides=6, radius=1.4, angle=pi/6)
)
model.plot()
plt.show()


# A graphene-specific shape which guaranties armchair edges on all sides
model = pb.Model(
    graphene.bilayer(),
    graphene.hexagon_ac(side_width=1)
)
model.plot()
plt.show()
