#! /usr/bin/env python3
"""Tight-binding system construction benchmark

Usage: Make sure you have all the requirements and then run this script using python3.
The considered system sizes can be changed in the `__name__ == '__main__'` section at
the bottom.

Requires Python >= 3.6 and the following packages which can be installed using pip:
  - memory_profiler
  - psutil (memory_profiler will be slow without it)

The benchmark compares the following packages. See the websites for install instructions:
  - pybinding : http://pybinding.site/
  - kwant     : http://kwant-project.org/

The benchmark constructs a circular graphene flake with a pn-junction and a constant
magnetic field. The system build time is measured from the start of the definition
to the point where the Hamiltonian matrix is fully constructed.
"""

import math
import cmath
import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import memory_usage

import kwant
import pybinding as pb
from pybinding.repository import graphene

pb.pltutils.use_style()
pb.pltutils.set_palette("Set1", start=3)


def calc_radius(num_sites, lattice=graphene.monolayer()):
    """The approximate radius of a circle which can contain the given number of lattice sites"""
    unit_area = np.linalg.norm(np.cross(*lattice.vectors)) / len(lattice.sublattices)
    return math.sqrt(num_sites * unit_area / math.pi)


def measure_pybinding(num_sites, warmup=False, plot=False):
    def circular_pn_junction(v0, radius):
        @pb.onsite_energy_modifier
        def function(energy, x, y):
            energy[(x**2 + y**2) < (radius / 2)**2] = v0
            return energy
        return function

    def make_model(radius, potential=0.2, magnetic_field=3):
        return pb.Model(
            graphene.monolayer().with_min_neighbors(2),
            pb.circle(radius),
            circular_pn_junction(potential, radius),
            graphene.constant_magnetic_field(magnetic_field)
        )

    with pb.utils.timed() as time:
        model = make_model(radius=calc_radius(num_sites))
        h = model.hamiltonian

    if not warmup:
        print("  {:7} <-> atoms = {}, non-zeros = {}".format(str(time), h.shape[0], h.nnz))
    if plot:
        model.plot()
        plt.show()
    return time.elapsed


def measure_kwant(num_sites, warmup=False, plot=False):
    def make_system(radius, potential=0.2, magnetic_field=3):
        def circle(pos):
            x, y = pos
            return x**2 + y**2 < radius**2

        def onsite(site):
            x, y = site.pos
            if x**2 + y**2 < (radius / 2)**2:
                return potential
            else:
                return 0

        def hopping(site_i, site_j):
            xi, yi = site_i.pos
            xj, yj = site_j.pos
            phi_ij = 0.5 * magnetic_field * (xi + xj) * (yi - yj)
            const = 1.519e-3  # 2*pi*e/h*[nm^2]
            return -2.8 * cmath.exp(1j * phi_ij * const)

        a_cc = 0.142
        a = a_cc * math.sqrt(3)
        graphene_lattice = kwant.lattice.general([(a, 0), (a/2, a/2 * math.sqrt(3))],
                                                 [(0, -a_cc/2), (0, a_cc/2)])
        builder = kwant.Builder()
        builder[graphene_lattice.shape(circle, (0, 0))] = onsite
        builder[graphene_lattice.neighbors()] = hopping
        builder.eradicate_dangling()
        return builder.finalized()

    with pb.utils.timed() as time:
        system = make_system(radius=calc_radius(num_sites))
        h = system.hamiltonian_submatrix(sparse=True)

    if not warmup:
        print("  {:7} <-> atoms = {}, non-zeros = {}".format(str(time), h.shape[0], h.nnz))
    if plot:
        kwant.plot(system)
    return time.elapsed


def plot_time(sizes, times, label):
    plt.plot(sizes, times, label=label, marker='o', markersize=5, lw=2, zorder=10)
    plt.grid(True, which='major', color='gray', ls=':', alpha=0.5)
    plt.loglog()
    plt.title("system build time")
    plt.xlabel("system size (number of atoms)")
    plt.ylabel("build time (seconds)")
    plt.xlim(0.7 * min(sizes), 1.2 * max(sizes))
    pb.pltutils.despine()
    pb.pltutils.legend(loc='upper left', reverse=True)


def plot_memory(memory_profile, label):
    progress = np.linspace(0, 100, len(memory_profile))
    units = "MiB"
    if max(memory_profile) > 1024:
        memory_profile = np.array(memory_profile) / 1024
        units = "GiB"
    plt.plot(progress, memory_profile, lw=2, label=label, zorder=10)
    plt.grid(True, which='major', color='gray', ls=':', alpha=0.5)
    plt.title("RAM usage over time")
    plt.xlabel("build progress (%)")
    plt.ylabel("memory ({})".format(units))
    pb.pltutils.despine()
    pb.pltutils.legend(loc='upper left', reverse=True)


def measure_and_plot(sizes, interval=0.02):
    """Measure build time and memory usage

    The list of `sizes` specifies the number atoms in the constructed systems.
    The `interval` is the memory usage sampling rate -- see `memory_profiler` package.

    The measurements are preceded by a short warmup run with a small system size. This
    takes care of submodule initialization so that it's not part of the measurement.
    """
    print("pybinding:")
    measure_pybinding(1e3, warmup=True)
    pb_memory, pb_times = memory_usage(lambda: [measure_pybinding(n) for n in sizes],
                                       interval, retval=True)
    print("\nkwant:")
    measure_kwant(1e3, warmup=True)
    kwant_memory, kwant_times = memory_usage(lambda: [measure_kwant(n) for n in sizes],
                                             interval, retval=True)

    plt.figure(figsize=(6, 2.4))
    plt.subplots_adjust(wspace=0.3)

    plt.subplot(121)
    plot_time(sizes, pb_times, label="pybinding")
    plot_time(sizes, kwant_times, label="kwant")

    plt.subplot(122)
    plot_memory(pb_memory, label="pybinding")
    plot_memory(kwant_memory, label="kwant")

    filename = "system_build_results.png"
    plt.savefig(filename)
    print("\nDone! Results saved to file: {}".format(filename))


if __name__ == '__main__':
    measure_and_plot(sizes=[1e3, 1e4, 1e5, 1e6])
