"""Tight-binding models for group 6 transition metal dichalcogenides (TMD)."""
import re
import math
import pybinding as pb


_default_3band_params = {  # from https://doi.org/10.1103/PhysRevB.88.085433
    # ->           a,  eps1,  eps2,     t0,    t1,    t2,   t11,   t12,    t22
    "MoS2":  [0.3190, 1.046, 2.104, -0.184, 0.401, 0.507, 0.218, 0.338,  0.057],
    "WS2":   [0.3191, 1.130, 2.275, -0.206, 0.567, 0.536, 0.286, 0.384, -0.061],
    "MoSe2": [0.3326, 0.919, 2.065, -0.188, 0.317, 0.456, 0.211, 0.290,  0.130],
    "WSe2":  [0.3325, 0.943, 2.179, -0.207, 0.457, 0.486, 0.263, 0.329,  0.034],
    "MoTe2": [0.3557, 0.605, 1.972, -0.169, 0.228, 0.390, 0.207, 0.239,  0.252],
    "WTe2":  [0.3560, 0.606, 2.102, -0.175, 0.342, 0.410, 0.233, 0.270,  0.190],
}


def monolayer_3band(name, override_params=None):
    """Monolayer of a group 6 TMD using the nearest-neighbor 3-band model

    Parameters
    ----------
    name : str
        Name of the TMD to model. The available options are: MoS2, WS2, MoSe2,
        WSe2, MoTe2, WTe2. The relevant tight-binding parameters for these 
        materials are given by https://doi.org/10.1103/PhysRevB.88.085433.
    override_params : Optional[dict]
        Replace or add new material parameters. The dictionary entries must 
        be in the format `"name": [a, eps1, eps2, t0, t1, t2, t11, t12, t22]`.

    Examples
    --------
    .. plot::
        :context: reset
        :alt: Molybdenum disulfide: unit cell for the nearest-neighbor 3-band model

        from pybinding.repository import group6_tmd

        group6_tmd.monolayer_3band("MoS2").plot()

    .. plot::
        :context: close-figs
        :alt: Molybdenum disulfide: 3-band model band structure

        model = pb.Model(group6_tmd.monolayer_3band("MoS2"), pb.translational_symmetry())
        solver = pb.solver.lapack(model)

        k_points = model.lattice.brillouin_zone()
        gamma = [0, 0]
        k = k_points[0]
        m = (k_points[0] + k_points[1]) / 2

        plt.figure(figsize=(6.7, 2.3))

        plt.subplot(121, title="MoS2 3-band model band structure")
        bands = solver.calc_bands(gamma, k, m, gamma)
        bands.plot(point_labels=[r"$\Gamma$", "K", "M", r"$\Gamma$"])

        plt.subplot(122, title="Band structure path in reciprocal space")
        model.lattice.plot_brillouin_zone(decorate=False)
        bands.plot_kpath(point_labels=[r"$\Gamma$", "K", "M", r"$\Gamma$"])

    .. plot::
        :context: close-figs
        :alt: Band structure of various group 6 TMDs: MoS2, WS2, MoSe2, WSe2, MoTe2, WTe2

        grid = plt.GridSpec(3, 2, hspace=0.4)
        plt.figure(figsize=(6.7, 8))

        for square, name in zip(grid, ["MoS2", "WS2", "MoSe2", "WSe2", "MoTe2", "WTe2"]):
            model = pb.Model(group6_tmd.monolayer_3band(name), pb.translational_symmetry())
            solver = pb.solver.lapack(model)

            k_points = model.lattice.brillouin_zone()
            gamma = [0, 0]
            k = k_points[0]
            m = (k_points[0] + k_points[1]) / 2

            plt.subplot(square, title=name)
            bands = solver.calc_bands(gamma, k, m, gamma)
            bands.plot(point_labels=[r"$\Gamma$", "K", "M", r"$\Gamma$"], lw=1.5)
    """
    params = _default_3band_params.copy()
    if override_params:
        params.update(override_params)

    a, eps1, eps2, t0, t1, t2, t11, t12, t22 = params[name]
    rt3 = math.sqrt(3)  # convenient constant

    lat = pb.Lattice(a1=[a, 0], a2=[1/2 * a, rt3/2 * a])

    metal_name, chalcogenide_name = re.findall("[A-Z][a-z]*", name)
    lat.add_one_sublattice(metal_name, [0, 0], [eps1, eps2, eps2])

    h1 = [[ t0, -t1,   t2],
          [ t1, t11, -t12],
          [ t2, t12,  t22]]

    h2 = [[                    t0,     1/2 * t1 + rt3/2 * t2,     rt3/2 * t1 - 1/2 * t2],
          [-1/2 * t1 + rt3/2 * t2,     1/4 * t11 + 3/4 * t22, rt3/4 * (t11 - t22) - t12],
          [-rt3/2 * t1 - 1/2 * t2, rt3/4 * (t11 - t22) + t12,     3/4 * t11 + 1/4 * t22]]

    h3 = [[                    t0,    -1/2 * t1 - rt3/2 * t2,     rt3/2 * t1 - 1/2 * t2],
          [ 1/2 * t1 - rt3/2 * t2,     1/4 * t11 + 3/4 * t22, rt3/4 * (t22 - t11) + t12],
          [-rt3/2 * t1 - 1/2 * t2, rt3/4 * (t22 - t11) - t12,     3/4 * t11 + 1/4 * t22]]

    m = metal_name
    lat.add_hoppings(([1,  0], m, m, h1),
                     ([0, -1], m, m, h2),
                     ([1, -1], m, m, h3))
    return lat
