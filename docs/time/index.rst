.. _time:

********************
time - Time analysis
********************

.. currentmodule:: gammapy.time

Introduction
============

`gammapy.time` contains methods for time-based analysis, e.g. from AGN, binaries
or pulsars. There is also `gammapy.utils.time`, which contains low-level helper
functions for time conversions e.g. following the

Getting Started
===============

.. _time-lc:

Lightcurve
----------

This section introduces the `~gammapy.time.LightCurve` class.

Read a table that contains a lightcurve::

    >>> from astropy.table import Table
    >>> url = 'https://github.com/gammapy/gamma-cat/raw/master/input/data/2006/2006A%2526A...460..743A/tev-000119-lc.ecsv'
    >>> table = Table.read(url, format='ascii.ecsv')

Create a ``LightCurve`` object:

    >>> from gammapy.time import LightCurve
    >>> lc = LightCurve(table)

``LightCurve`` is a simple container that stores the LC table, and provices a
few conveniences, like creating time objects and a quick-look plot:

    >>> lc.time[:2].iso
    ['2004-05-23 01:47:08.160' '2004-05-23 02:17:31.200']
    >>> lc.plot()

.. _time-variability:

Variability test
----------------

The `~gammapy.time.exptest` function can be used to compute the significance of
variability (compared to the null hypothesis of constant rate) for a list of
event time differences.

Here's an example how to use the `~gammapy.time.random_times` helper function to
simulate a `~astropy.time.TimeDelta` array for a given constant rate and use
`~gammapy.time.exptest` to assess the level of variability (0.11 sigma in this
case, not variable):

.. code-block:: python

    >>> from astropy.units import Quantity
    >>> from gammapy.time import random_times, exptest
    >>> rate = Quantity(10, 'Hz')
    >>> time_delta = random_times(size=100, rate=rate, return_diff=True, random_state=0)
    >>> mr = exptest(time_delta)
    >>> print(mr)
    0.11395763079

See ``examples/example_exptest.py`` for a longer example.

TODO: apply this to the 3FHL events and check which sources are variable as a nice example.

.. code-block:: python

    from gammapy.data import EventList
    from gammapy.time import exptest
    events = EventList.read("$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-events.fits.gz")
    # TODO: cone select events for 3FHL catalog sources, compute mr for each and print 10 most variable sources

Other codes
===========

Where possible we shouldn't duplicate timing analysis functionality that's
already available elsewhere. In some cases it's enough to refer gamma-ray
astronomers to other packages, sometimes a convenience wrapper for our data
formats or units might make sense.

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
* https://nbviewer.ipython.org/github/YSOVAR/Analysis/blob/master/TimeScalesinYSOVAR.ipynb

Using `gammapy.time`
====================

.. toctree::
   :maxdepth: 1

   period

Reference/API
=============

.. automodapi:: gammapy.time
    :no-inheritance-diagram:
    :include-all-objects:

.. automodapi:: gammapy.time.models
    :no-inheritance-diagram:
    :include-all-objects:
