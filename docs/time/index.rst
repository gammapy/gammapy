.. _time:

********************
time - Time analysis
********************

.. currentmodule:: gammapy.time

Introduction
============

`gammapy.time` contains classes and methods for time-based analysis, e.g. for AGN, binaries
or pulsars studies. The main classes are `~gammapy.time.LightCurve`, which is a container for
lightcurves, and `~gammapy.time.LightCurveEstimator`, which extracts a light curve from a list
 of datasets. A number of functions to test for variability and periodicity are available in
`~gammapy.time.variability` and `~gammapy.time.periodicity`. Finally, there is also
`gammapy.utils.time`, which contains low-level helper
functions for time conversions.

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

TODO: Add some rapid discussion of chisquare and fractional variance functions

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
