.. _astro-population:

*******************************************************************
Astrophysical source population models (`gammapy.astro.population`)
*******************************************************************

.. currentmodule:: gammapy.astro.population

Introduction
============

The `gammapy.astro.population` module provides a simple framework for population synthesis of 
gamma-ray sources. 

Getting Started
===============

TODO


Radial surface density distributions
====================================
Here is a comparison plot of all available radial distribution functions of the surface density of pulsars
and related objects used in literature:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from gammapy.astro.population import radial_distributions
    from gammapy.utils.distributions import normalize
    
    max_radius = 20  # kpc
    r = np.linspace(0, max_radius, 100)
    colors = ['b', 'k', 'k', 'b', 'g', 'g']
    
    for color, key in zip(colors, radial_distributions.keys()):
        model = radial_distributions[key]()
        if model.evolved:
            linestyle = '-'
        else:
            linestyle = '--'
        label = model.__class__.__name__
        plt.plot(r, normalize(model, 0, max_radius)(r), color=color, linestyle=linestyle, label=label)
    plt.xlim(0, max_radius)
    plt.ylim(0, 0.28)
    plt.xlabel('Galactocentric Distance [kpc]')
    plt.ylabel('Normalized Surface Density [kpc^-2]')
    plt.legend(prop={'size': 10})
    plt.show()


Velocity distributions
======================
Here is a comparison plot of all available velocity distribution functions:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from gammapy.astro.population import velocity_distributions
    from gammapy.utils.distributions import normalize

    v_min, v_max = 10, 3000  # km / s
    v = np.linspace(v_min, v_max, 200)
    colors = ['b', 'k', 'g']
    
    for color, key in zip(colors, velocity_distributions.keys()):
    	model = velocity_distributions[key]()
    	label = model.__class__.__name__
    	plt.plot(v, normalize(model, v_min, v_max)(v), color=color, linestyle='-', label=label)
    
    plt.xlim(v_min, v_max)
    plt.ylim(0, 0.004)
    plt.xlabel('Velocity [km/s]')
    plt.ylabel('Probability Density [(km / s)^-1]')
    plt.semilogx()
    plt.legend(prop={'size': 10})
    plt.show()


Reference/API
=============

.. automodapi:: gammapy.astro.population
    :no-inheritance-diagram:
