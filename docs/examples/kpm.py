import matplotlib.pyplot as plt
import numpy as np
import pybinding as pb
from pybinding.repository import graphene

# create a new model
model = pb.Model(
    graphene.lattice.monolayer(),
    pb.shape.rectangle(x=50, y=50),  # [nm]
    pb.magnetic.constant(60),  # [T]
    pb.electric.constant(0.1),  # [eV]
    pb.greens.KPM()
)

result = pb.result.ldos_point(
    model,
    energy=np.linspace(-0.5, 0.5, 1000),  # [eV]
    broadening=0.03,  # [eV]
    position=(0, 0)  # (x, y) [nm]
)

result.plot()
plt.show()
