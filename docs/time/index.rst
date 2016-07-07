.. _time:

*******************************************
Time handling and analysis (`gammapy.time`)
*******************************************

.. currentmodule:: gammapy.time

Introduction
============

`gammapy.time` contains methods for timing analysis.

At the moment there isn't a lot of functionality yet ... contributions welcome!


Getting Started
===============

.. _time-lc:

Lightcurve
----------

The `~gammapy.time.LightCurve` class can be used to read a lightcurve,
plot it and compute some summary statistics:

.. code-block:: python

    >>> from gammapy.time import LightCurve
    >>> lc = LightCurve.read('$GAMMAPY_EXTRA/todo/make/example-lightcurve.fits.gz')
    >>> lc.plot()
    >>> lc.info()


.. _time-variability:

Variability test
----------------

The `~gammapy.time.exptest` function can be used to compute the significance
of variability (compared to the null hypothesis of constant rate)
for a list of event time differences.

Here's an example how to use the `~gammapy.time.random_times` helper
function to simulate a `~astropy.time.TimeDelta` array for a given constant rate
and use `~gammapy.time.exptest` to assess the level of variability (1.3 sigma in this case,
not variable):

.. code-block:: python

    >>> from astropy.units import Quantity
    >>> from gammapy.time import random_times, exptest
    >>> rate = Quantity(10, 'Hz')
    >>> time_delta = random_times(size=100, rate=rate, return_diff=True, random_state=0)
    >>> mr = exptest(time_delta)
    >>> print(mr)
    0.11395763079


See ``examples/example_exptest.py`` for a longer example.

TODO: apply this to the 2FHL events and check which sources are variable as a nice example.

.. code-block:: python

    from gammapy.data import EventList
    from gammapy.time import exptest
    events = EventList.read('$GAMMAPY_EXTRA/datasets/fermi_2fhl/2fhl_events.fits.gz ', hdu='EVENTS')
    # TODO: cone select events for 2FHL catalog sources, compute mr for each and print 10 most variable sources

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

Note that Astropy chose ``format='isot'`` and ``scale='utc'`` as default and in
Gammapy these are also the recommended format and time scale.

.. warning::

   Actually what's written here is not true.
   In CTA it hasn't been decided if times will be in ``utc`` or ``tt`` (terrestial time) format.

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
`astropy.coordinates.AltAz.obstime` is set with a `~astropy.time.Time` object in any scale,
no need for explicit time scale transformations in Gammapy
(although if you do want to explicitly compute it, it's easy, see `here <http://docs.astropy.org/en/latest/time/index.html#sidereal-time>`__).

The "Time Systems in a nutshell" section `here <http://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Cicerone/Cicerone_Data/Time_in_ScienceTools.html>`__
gives a good, brief explanation of the differences between the relevant time scales ``UT1``, ``UTC`` and ``TT``.

.. _MET_definition:

Mission elapsed times (MET)
---------------------------

[MET]_ time references are times representing UTC seconds after a
specific origin. Each experiment may have a different MET origin
that should be included in the header of the corresponding data
files. For more details see `Cicerone: Data - Time in Fermi Data Analysis
<http://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Cicerone/Cicerone_Data/
Time_in_ScienceTools.html>`_.

It's not clear yet how to best implement METs in Gammapy, it's one
of the tasks here:
https://github.com/gammapy/gammapy/issues/284

For now, we use the `gammapy.time.time_ref_from_dict`, `gammapy.time.time_relative_to_ref`
and `gammapy.time.absolute_time` functions to convert MET floats to `~astropy.time.Time`
objects via the reference times stored in FITS headers.

Time differences
----------------

TODO: discuss when to use `~astropy.time.TimeDelta` or `~astropy.units.Quantity` or [MET]_ floats and
where one needs to convert between those and what to watch out for.

Other codes
===========

Where possible we shouldn't duplicate timing analysis functionality
that's already available elsewhere. In some cases it's enough to refer
gamma-ray astronomers to other packages, sometimes a convenience wrapper for
our data formats or units might make sense.

Some references (please add what you find useful

* https://github.com/astroML/gatspy
* https://github.com/matteobachetti/MaLTPyNT
* https://github.com/nanograv/PINT
* http://www.astroml.org/modules/classes.html#module-astroML.time_series
* http://www.astroml.org/book_figures/chapter10/index.html
* http://docs.astropy.org/en/latest/api/astropy.stats.bayesian_blocks.html
* https://github.com/samconnolly/DELightcurveSimulation
* https://github.com/cokelaer/spectrum
* https://github.com/YSOVAR/YSOVAR
* http://nbviewer.ipython.org/github/YSOVAR/Analysis/blob/master/TimeScalesinYSOVAR.ipynb


Reference/API
=============

.. automodapi:: gammapy.time
    :no-inheritance-diagram:
