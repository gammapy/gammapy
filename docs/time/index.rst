.. _time:

********************************
Timing analysis (`gammapy.time`)
********************************

.. currentmodule:: gammapy.time

Introduction
============

`gammapy.time` contains methods for timing analysis.

At the moment there's almost no functionality here.
Any contributions welcome, e.g.

* utility functions for time computations
* lightcurve analysis functions
* pulsar periodogram
* event dead time calculations and checks for constant rate

Although where possible we shouldn't duplicate timing analysis functionality
that's already available elsewhere. In some cases it's enough to refer
gamma-ray astronomers to other packages, sometimes a convenience wrapper for
our data formats or units might make sense.

Some references (please add what you find useful

* https://github.com/matteobachetti/MaLTPyNT
* https://github.com/nanograv/PINT
* http://www.astroml.org/modules/classes.html#module-astroML.time_series
* http://www.astroml.org/book_figures/chapter10/index.html
* http://astropy.readthedocs.org/en/latest/api/astropy.stats.bayesian_blocks.html
* https://github.com/samconnolly/DELightcurveSimulation
* https://github.com/cokelaer/spectrum
* https://github.com/YSOVAR/YSOVAR
* http://nbviewer.ipython.org/github/YSOVAR/Analysis/blob/master/TimeScalesinYSOVAR.ipynb


Getting Started
===============

TODO: document


.. _time_handling:

Time handling in Gammapy
========================

Time format and scale
---------------------

In Gammapy, `astropy.time.Time` objects are used to represent times:

.. code-block:: python

    >>> from astropy.time import Time
    >>> Time(['1999-01-01T00:00:00.123456789', '2010-01-01T00:00:00'])
    <Time object: scale='utc' format='isot' value=['1999-01-01T00:00:00.123' '2010-01-01T00:00:00.000']>

Note that Astropy chose ``format='isot'`` and ``scale='utc'`` and in Gammapy these are also the
recommended format and time scale.

.. warning::

   Actually what's written here is not true.
   In CTA it hasn't been decided if times will be in `utc` or `tt` (terrestial time) format.

   Here's a reminder that this needs to be settled / updated:
   https://github.com/gammapy/gammapy/issues/284


When other time formats are needed it's easy to convert, see the
:ref:`time format section and table in the Astropy docs <astropy:time-format>`.

E.g. sometimes in astronomy the modified Julian date ``mjd`` is used and for passing times to matplotlib
for plotting the ``plot_date`` format should be used:

.. code-block:: python

    >>> from astropy.time import Time
    >>> time = Time(['1999-01-01T00:00:00.123456789', '2010-01-01T00:00:00'])
    >>> time.mjd
    array([ 51179.00000143,  55197.        ])
    >>> time.plot_date
    array([ 729755.00000143,  733773.        ])

Converting to other time scales is also easy, see the
:ref:`time scale section, diagram and table in the Astropy docs <astropy:time-scale>`.

E.g. when converting celestial (RA/DEC) to horizontal (ALT/AZ) coordinates, the
`sidereal time <https://en.wikipedia.org/wiki/Sidereal_time>`__ is needed.
This is done automatically by `astropy.coordinates.AltAz` when the
`astropy.coordinates.AltAz.obstime` is set with a `astropy.time.Time` object in any scale,
no need for explicit time scale transformations in Gammapy
(although if you do want to explicitly compute it, it's easy, see `here <http://astropy.readthedocs.org/en/latest/time/index.html#sidereal-time>`__).

The "Time Systems in a nutshell" section `here <http://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Cicerone/Cicerone_Data/Time_in_ScienceTools.html>`__
gives a good, brief explanation of the differences between the relevant time scales `UT1`, `UTC` and `TT`.

Mission elapsed times (MET)
---------------------------

It's not clear yet how to best implement METs, it's one of the tasks here:
https://github.com/gammapy/gammapy/issues/284

For now, we use the `gammapy.time.time_ref_from_dict` and `gammapy.time.time_relative_to_ref` functions
to convert MET floats to `~astropy.time.Time` objects via the reference times stored in FITS headers.

See `Cicerone: Data - Time in Fermi Data Analysis
<http://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Cicerone/Cicerone_Data/
Time_in_ScienceTools.html>`_.

Time differences
----------------

TODO: discuss when to use `astropy.time.TimeDelta` or `astropy.units.Quantity` or MET floats and
where one needs to convert between those and what to watch out for.

Reference/API
=============

.. automodapi:: gammapy.time
    :no-inheritance-diagram:
