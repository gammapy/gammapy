.. _astro-population:

**************************************
Astrophysical source population models
**************************************

Introduction
============

The `gammapy.astro.population` module provides a simple framework for population
synthesis of gamma-ray sources, which is useful in the context of surveys and
population studies.

Getting started
===============

The following example illustrates how to simulate a basic catalog including a
spiral arm model.

.. testcode::

    import astropy.units as u
    from gammapy.astro.population import make_base_catalog_galactic

    max_age = 1E6 * u.yr
    SN_rate = 3. / (100. * u.yr)
    n_sources = int(max_age * SN_rate)
    table = make_base_catalog_galactic(
        n_sources=n_sources,
        rad_dis='L06',
        vel_dis='F06B',
        max_age=max_age,
        spiralarms=True,
    )

The total number of sources is determined assuming a maximum age and a supernova
rate. The table returned is an instance of `~astropy.table.Table` which
can be used for further processing. The example population with spiral-arms is
illustrated in the following plot.

.. plot:: user-guide/astro/population/plot_spiral_arms.py

Galactocentric spatial distributions
------------------------------------

Here is a comparison plot of all available radial distribution functions of the
surface density of pulsars and related objects used in literature:

.. plot:: user-guide/astro/population/plot_radial_distributions.py

TODO: add illustration of Galactocentric z-distribution model and combined (r,
z) distribution for the Besancon model.

Spiral arm models
-----------------

Two spiral arm models of the Milky way are available:
`~gammapy.astro.population.ValleeSpiral` and
`gammapy.astro.population.FaucherSpiral`

.. plot:: user-guide/astro/population/plot_spiral_arm_models.py


Velocity distributions
----------------------

Here is a comparison plot of all available velocity distribution functions:

.. plot:: user-guide/astro/population/plot_velocity_distributions.py
