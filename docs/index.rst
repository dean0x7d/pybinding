.. title:: Tight-binding package for Python
.. meta::
   :description: Pybinding is a Python code package for tight-binding calculations in solid state
                 physics. It can be used to construct and solver large tight-binding models.

.. container:: pbheader

    .. image:: pb.png
        :class: pblogo

    Pybinding is a scientific Python package for numerical tight-binding calculations in solid
    state physics. If you're just browsing, the :doc:`tutorial/index` section is a good place
    to start. It gives a good overview of the most important features with lots of code examples.

As a very quick example, the following code creates a triangular quantum dot of bilayer graphene
and then applies a custom asymmetric strain function:

.. plot::
    :alt: Asymmetrically strained bilayer graphene quantum dot

    import pybinding as pb
    from pybinding.repository import graphene

    def asymmetric_strain(c):
        @pb.site_position_modifier
        def displacement(x, y, z):
            ux = -c/2 * x**2 + c/3 * x + 0.1
            uy = -c*2 * x**2 + c/4 * x
            return x + ux, y + uy, z
        return displacement

    model = pb.Model(
        graphene.bilayer(),
        pb.regular_polygon(num_sides=3, radius=1.1),
        asymmetric_strain(c=0.42)
    )
    model.plot()

Within the pybinding framework, tight-binding models are assembled from logical parts which
can be mixed and matched in various ways. The package comes with a few predefined components:
crystal lattices, shapes, symmetries, defects, fields and more (like the
:func:`graphene.bilayer() <.graphene.lattice.bilayer>` lattice and the :func:`.regular_polygon`
shape shown above). Users can also define new components (just like the asymmetric strain above).
This modular approach enables the construction of arbitrary tight-binding models with clear,
easy-to-use code. Various solvers, computation routines and visualization tools are also part
of the package. See the :doc:`tutorial/index` for a walkthrough of the features.

The source code repository is `located on Github <https://github.com/dean0x7d/pybinding>`_
where you can also post any questions, comments or issues that you might have.


.. toctree::
    :hidden:
    :maxdepth: 2

    intro

.. toctree::
    :hidden:
    :titlesonly:

    benchmarks/index
    changelog


.. toctree::
    :caption: User guide
    :maxdepth: 2

    install/index
    tutorial/index
    advanced/index
    plotting/index
    examples/index


.. toctree::
    :caption: Reference docs
    :maxdepth: 2

    materials/index
    api
    experimental/index

* :ref:`genindex`


.. raw:: html

    <style type="text/css" >
        @media screen and (min-width: 990px) {
            .rst-content div.highlight-python {
                float: left;
                width: 54%;
            }
            .rst-content div.figure {
                padding-top: 16px;
                margin-left: 55%;
            }
            .rst-content p { clear: both; }

            .pblogo {
                margin: 0px 10px 8px 0px;
                float: left;
            }

            .pbheader p {
                margin-left: 220px;
                padding-top: 9%;
                clear: none;
            }
        }

        .pblogo { width: 220px; }
    </style>
