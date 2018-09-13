.. include:: ../references.txt

.. _utils:

*****************
utils - Utilities
*****************

Introduction
============

``gammapy.utils`` is a collection of utility functions that are used in many
places or don't fit in one of the other packages.

Since the various sub-modules of ``gammapy.utils`` are mostly unrelated, they
are not imported into the top-level namespace. Here are some examples of how to
import functionality from the ``gammapy.utils`` sub-modules:

.. code-block:: python

    from gammapy.utils.random import sample_sphere
    sample_sphere(size=10)

    from gammapy.utils import random
    random.sample_sphere(size=10)

.. _time_handling:

Time handling in Gammapy
========================

See `gammapy.utils.time`.

Time format and scale
---------------------

In Gammapy, `astropy.time.Time` objects are used to represent times:

.. code-block:: python

    >>> from astropy.time import Time
    >>> Time(['1999-01-01T00:00:00.123456789', '2010-01-01T00:00:00'])
    <Time object: scale='utc' format='isot' value=['1999-01-01T00:00:00.123' '2010-01-01T00:00:00.000']>

Note that Astropy chose ``format='isot'`` and ``scale='utc'`` as default and in
Gammapy these are also the recommended format and time scale.

.. warning::

   Actually what's written here is not true. In CTA it hasn't been decided if
   times will be in ``utc`` or ``tt`` (terrestial time) format.

   Here's a reminder that this needs to be settled / updated:
   https://github.com/gammapy/gammapy/issues/284


When other time formats are needed it's easy to convert, see the :ref:`time
format section and table in the Astropy docs <astropy:time-format>`.

E.g. sometimes in astronomy the modified Julian date ``mjd`` is used and for
passing times to matplotlib for plotting the ``plot_date`` format should be
used:

.. code-block:: python

    >>> from astropy.time import Time
    >>> time = Time(['1999-01-01T00:00:00.123456789', '2010-01-01T00:00:00'])
    >>> time.mjd
    array([ 51179.00000143,  55197.        ])
    >>> time.plot_date
    array([ 729755.00000143,  733773.        ])

Converting to other time scales is also easy, see the :ref:`time scale section,
diagram and table in the Astropy docs <astropy:time-scale>`.

E.g. when converting celestial (RA/DEC) to horizontal (ALT/AZ) coordinates, the
`sidereal time <https://en.wikipedia.org/wiki/Sidereal_time>`__ is needed. This
is done automatically by `astropy.coordinates.AltAz` when the
`astropy.coordinates.AltAz.obstime` is set with a `~astropy.time.Time` object in
any scale, no need for explicit time scale transformations in Gammapy (although
if you do want to explicitly compute it, it's easy, see `here
<http://docs.astropy.org/en/latest/time/index.html#sidereal-time>`__).

The `Fermi-LAT time systems in a nutshell`_ page gives a good, brief explanation
of the differences between the relevant time scales ``UT1``, ``UTC`` and ``TT``.

.. _MET_definition:

Mission elapsed times (MET)
---------------------------

[MET]_ time references are times representing UTC seconds after a specific
origin. Each experiment may have a different MET origin that should be included
in the header of the corresponding data files. For more details see `Fermi-LAT
time systems in a nutshell`_.

It's not clear yet how to best implement METs in Gammapy, it's one of the tasks
here: https://github.com/gammapy/gammapy/issues/284

For now, we use the `gammapy.utils.time.time_ref_from_dict`,
`gammapy.utils.time.time_relative_to_ref` and `gammapy.utils.time.absolute_time` functions
to convert MET floats to `~astropy.time.Time` objects via the reference times
stored in FITS headers.

Time differences
----------------

TODO: discuss when to use `~astropy.time.TimeDelta` or `~astropy.units.Quantity`
or [MET]_ floats and where one needs to convert between those and what to watch
out for.

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

The EnergyBounds class is a subclass of Energy. Additional functions are
available e.g. to compute the bin centers

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
    :include-all-objects:

.. automodapi:: gammapy.utils.units
    :no-inheritance-diagram:
    :include-all-objects:

.. automodapi:: gammapy.utils.coordinates
    :no-inheritance-diagram:
    :include-all-objects:

.. automodapi:: gammapy.utils.table
    :no-inheritance-diagram:
    :include-all-objects:

.. automodapi:: gammapy.utils.fits
    :no-inheritance-diagram:
    :include-all-objects:

.. automodapi:: gammapy.utils.random
    :no-inheritance-diagram:
    :include-all-objects:

.. automodapi:: gammapy.utils.distributions
    :no-inheritance-diagram:
    :include-all-objects:

.. automodapi:: gammapy.utils.scripts
    :no-inheritance-diagram:
    :include-all-objects:

.. automodapi:: gammapy.utils.testing
    :no-inheritance-diagram:
    :include-all-objects:

.. automodapi:: gammapy.utils.wcs
    :no-inheritance-diagram:
    :include-all-objects:

.. automodapi:: gammapy.utils.nddata
    :no-inheritance-diagram:
    :include-all-objects:

.. automodapi:: gammapy.utils.time
    :no-inheritance-diagram:
    :include-all-objects:

.. automodapi:: gammapy.utils.fitting
    :no-inheritance-diagram:
    :include-all-objects:
