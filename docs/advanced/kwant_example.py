#! /usr/bin/env python3
"""Transport through a barrier

The `main()` function builds identical models in pybinding and kwant and then calculates
the transmission using `kwant.smatrix`. The results are plotted to verify that they are
identical.

The `measure_and_plot()` function compares transport calculation time for various system
sizes. Modify the `__name__ == '__main__'` section at the bottom to run this benchmark.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

import kwant
import pybinding as pb
from pybinding.repository import graphene

pb.pltutils.use_style()
pb.pltutils.set_palette("Set1", start=3)


def measure_pybinding(width, length, electron_energy, barrier_heights, plot=False):
    def potential_barrier(v0):
        @pb.onsite_energy_modifier(is_double=True)
        def function(energy, x):
            energy[np.logical_and(-length / 4 <= x, x <= length / 4)] = v0
            return energy
        return function

    def make_model(v0=0):
        model = pb.Model(
            graphene.monolayer().with_min_neighbors(1),
            pb.rectangle(length, width),
            potential_barrier(v0),
        )
        model.attach_lead(-1, pb.line([-length/2, -width/2], [-length/2, width/2]))
        model.attach_lead(+1, pb.line([ length/2, -width/2], [ length/2, width/2]))
        return model

    if plot:
        make_model().plot()
        plt.show()

    transmission = []
    for v in barrier_heights:
        smatrix = kwant.smatrix(make_model(v).tokwant(), energy=electron_energy)
        transmission.append(smatrix.transmission(1, 0))

    return transmission


def measure_kwant(width, length, electron_energy, barrier_heights, plot=False):
    def make_system():
        t = 2.8
        a_cc = 0.142
        a = a_cc * math.sqrt(3)
        graphene_lattice = kwant.lattice.general([(a, 0), (a / 2, a / 2 * math.sqrt(3))],
                                                 [(0, -a_cc / 2), (0, a_cc / 2)])

        def shape(pos):
            x, y = pos
            return -length / 2 <= x <= length / 2 and -width / 2 <= y <= width / 2

        def onsite(site, v0):
            x, _ = site.pos
            return v0 if -length / 4 <= x <= length / 4 else 0

        builder = kwant.Builder()
        builder[graphene_lattice.shape(shape, (0, 0))] = onsite
        builder[graphene_lattice.neighbors()] = -t

        def lead_shape(pos):
            x, y = pos
            return -width / 2 <= y <= width / 2

        lead = kwant.Builder(kwant.TranslationalSymmetry(graphene_lattice.vec((-1, 0))))
        lead[graphene_lattice.shape(lead_shape, (0, 0))] = 0
        lead[graphene_lattice.neighbors()] = -t
        builder.attach_lead(lead)
        builder.attach_lead(lead.reversed())
        return builder.finalized()

    system = make_system()
    if plot:
        kwant.plot(system)

    transmission = []
    for v in barrier_heights:
        smatrix = kwant.smatrix(system, energy=electron_energy, args=[v])
        transmission.append(smatrix.transmission(1, 0))

    return transmission


def main():
    """Build the same model using pybinding and kwant and verify that the results are identical"""
    width, length = 15, 15
    electron_energy = 0.25
    barrier_heights = np.linspace(0, 0.5, 100)

    with pb.utils.timed("pybinding:"):
        pb_transmission = measure_pybinding(width, length, electron_energy, barrier_heights)
    with pb.utils.timed("kwant:"):
        kwant_transmission = measure_kwant(width, length, electron_energy, barrier_heights)

    plt.plot(barrier_heights, pb_transmission, label="pybinding")
    plt.plot(barrier_heights, kwant_transmission, ls="--", label="kwant")
    plt.ylabel("transmission")
    plt.xlabel("barrier height (eV)")
    plt.axvline(electron_energy, 0, 0.5, color="gray", ls=":")
    plt.annotate("electron energy\n{} eV".format(electron_energy), (electron_energy, 0.52),
                 xycoords=("data", "axes fraction"), ha="center")
    pb.pltutils.despine()
    pb.pltutils.legend()
    plt.show()


def plot_time(sizes, times, label):
    plt.plot(sizes, times, label=label, marker='o', markersize=5, lw=2, zorder=10)
    plt.grid(True, which='major', color='gray', ls=':', alpha=0.5)
    plt.title("transmission calculation time")
    plt.xlabel("system size (nm)")
    plt.ylabel("compute time (seconds)")
    plt.xlim(0.8 * min(sizes), 1.05 * max(sizes))
    pb.pltutils.despine()
    pb.pltutils.legend(loc='upper left', reverse=True)


def measure_and_plot(sizes):
    """Measure transport calculation time

    The list of `sizes` specifies the dimensions of the scattering region in nanometers.
    """
    electron_energy = 0.25
    barrier_heights = np.linspace(0, 0.5, 100)

    print("pybinding:")
    pb_times = []
    for size in sizes:
        with pb.utils.timed() as time:
            measure_pybinding(size, size, electron_energy, barrier_heights)
        print("  {:7} <-> size = {} nm".format(str(time), size))
        pb_times.append(time.elapsed)

    print("\nkwant:")
    kwant_times = []
    for size in sizes:
        with pb.utils.timed() as time:
            measure_kwant(size, size, electron_energy, barrier_heights)
        print("  {:7} <-> size = {} nm".format(str(time), size))
        kwant_times.append(time.elapsed)

    plt.figure(figsize=(3, 2.4))
    plot_time(sizes, pb_times, label="pybinding")
    plot_time(sizes, kwant_times, label="kwant")

    filename = "kwant_example_results.png"
    plt.savefig(filename)
    print("\nDone! Results saved to file: {}".format(filename))


if __name__ == '__main__':
    # measure_and_plot(sizes=[5, 10, 15, 20, 25, 30])
    main()
