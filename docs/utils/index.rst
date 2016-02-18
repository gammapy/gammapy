.. _utils:

*************************************************
Utility functions and classes (``gammapy.utils``)
*************************************************

Introduction
============

``gammapy.utils`` is a collection of utility functions that are used in many places
or don't fit in one of the other packages.

Since the various sub-modules of ``gammapy.utils`` are mostly unrelated,
they are not imported into the top-level namespace.
Here are some examples of how to import functionality from the ``gammapy.utils``
sub-modules:

.. code-block:: python

   from gammapy.utils.random import sample_sphere
   sample_sphere(size=10)

   from gammapy.utils import random
   random.sample_sphere(size=10)



.. _energy_handling_gammapy:

Energy handling in Gammapy
==========================

Basics
------

Most objects in Astronomy require an energy axis, e.g. counts spectra or
effective area tables. In general, this axis can be defined in two ways.

* As an array of energy values. E.g. the Fermi-LAT diffuse flux is given at
  certain energies and those are stored in an ENERGY FITS table extension.
  In Gammalib this is represented by GEnergy.
* As an array of energy bin edges. This is usually stored in EBOUNDS tables,
  e.g. for Fermi-LAT counts cubes. In Gammalib this is represented by GEbounds.

In Gammapy both the use cases are handled by two seperate classes:
`gammapy.utils.energy.Energy` for energy values and
`gammapy.utils.energy.EnergyBounds` for energy bin edges

Energy
------

The Energy class is a subclass of `~astropy.units.Quantity` and thus has the
same functionality plus some convenience functions for fits I/O

.. code-block:: python

    >>> from gammapy.utils.energy import Energy
    >>> energy = Energy([1,2,3], 'TeV')
    >>> hdu = energy.to_fits()
    >>> type(hdu)
    <class 'astropy.io.fits.hdu.table.BinTableHDU'>

EnergyBounds
------------

The EnergyBounds class is a subclass of Energy. Additional functions are available
e.g. to compute the bin centers

.. code-block:: python

    >>> from gammapy.utils.energy import EnergyBounds
    >>> ebounds = EnergyBounds.equal_log_spacing(1, 10, 8, 'GeV')
    >>> ebounds.size
    9
    >>> ebounds.nbins
    8
    >>> center = ebounds.log_centers
    >>> center
    <Energy [ 1.15478198, 1.53992653, 2.05352503, 2.73841963, 3.65174127,
              4.86967525, 6.49381632, 8.65964323] GeV>


Reference/API
=============
.. automodapi:: gammapy.utils.energy
    :no-inheritance-diagram:

.. automodapi:: gammapy.utils.mpl_style
    :no-inheritance-diagram:

.. automodapi:: gammapy.utils.coordinates
    :no-inheritance-diagram:

.. automodapi:: gammapy.utils.const
    :no-inheritance-diagram:

.. automodapi:: gammapy.utils.fits
    :no-inheritance-diagram:

.. automodapi:: gammapy.utils.root
    :no-inheritance-diagram:

.. automodapi:: gammapy.utils.random
    :no-inheritance-diagram:

.. automodapi:: gammapy.utils.distributions
    :no-inheritance-diagram:

.. automodapi:: gammapy.utils.pyfact
    :no-inheritance-diagram:

.. automodapi:: gammapy.utils.scripts
    :no-inheritance-diagram:

.. automodapi:: gammapy.utils.testing
    :no-inheritance-diagram:

.. automodapi:: gammapy.utils.wcs
    :no-inheritance-diagram:

.. automodapi:: gammapy.utils.axis
    :no-inheritance-diagram:
